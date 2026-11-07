"""Model backends for the duel app. Each has stream(messages) -> iterator of text pieces.
Configured in models.json:  {"base": {"backend": "mlx|transformers|http|mock", ...}, "finetuned": {...}}

Common options: "label" (shown in the UI), "enable_thinking" (Qwen3 chat-template switch, default false).
Qwen3 may start its answer with an empty <think></think> block; it is removed from the stream here.
Backends that hold weights also have close(); release_memory() hands freed memory back to the system.
All MLX work runs on one thread, see MLXThread."""
import gc
import importlib.util
import json
import queue
import sys
import threading
import time
import urllib.request
from concurrent.futures import Future
from pathlib import Path


class MLXThread:
    """The one thread that does all MLX work: import, load, generate, close, clearing the cache.
    Since MLX 0.32 a stream belongs to the thread that made it, and an array that is not evaluated yet stays tied to its stream. mlx_lm leaves some arrays
    unevaluated after load() (LoRA adapter weights, the RoPE tables of some models), so using them from any other thread fails with
    "There is no Stream(cpu, N) in current thread". Jobs run in order. The thread never waits for a lock of the app, so a caller holding one cannot deadlock it."""
    def __init__(self):
        self.jobs, self.thread, self.start_lock = queue.SimpleQueue(), None, threading.Lock()

    def _loop(self):
        while True:
            job, future = self.jobs.get()
            try:
                future.set_result(job())
            except BaseException as e:                                  # a failed job never ends the thread
                cause = e
                while cause is not None:                                # the traceback would keep a half-loaded model alive
                    cause.__traceback__ = None
                    cause = cause.__cause__ or cause.__context__
                future.set_exception(e)
            job = future = None                                         # or the last job's model would stay here while the thread waits

    def submit(self, job):
        with self.start_lock:
            if self.thread is None:
                self.thread = threading.Thread(target=self._loop, name="mlx", daemon=True)
                self.thread.start()
        future = Future()
        self.jobs.put((job, future))
        return future

    def run(self, job):
        """Run job on the MLX thread and return its result, or raise its exception here."""
        return self.submit(job).result()

    def stream(self, make_pieces):
        """Run a generator on the MLX thread; each item arrives here as soon as it is written. When this ends, nothing is running on that thread for it any more."""
        pieces, stop, end = queue.SimpleQueue(), threading.Event(), object()

        def job():
            source = make_pieces()
            try:
                for piece in source:
                    pieces.put(piece)
                    if stop.is_set():
                        break
            finally:
                source.close()

        future = self.submit(job)
        future.add_done_callback(lambda _: pieces.put(end))
        error = None
        try:
            while (piece := pieces.get()) is not end:
                yield piece
        finally:
            stop.set()
            error = future.exception()
        if error:
            raise error


MLX = MLXThread()


def free_mlx():
    gc.collect()                                                        # here, so that dropped MLX arrays go on the MLX thread
    mx = sys.modules.get("mlx.core")
    if mx is not None:
        for owner in (mx, getattr(mx, "metal", None)):                  # mx.clear_cache; older MLX: mx.metal.clear_cache
            try:
                owner.clear_cache()
                break
            except Exception:
                pass


def release_memory():
    """Collect dropped models, then empty the framework caches: MLX and PyTorch MPS keep freed buffers for reuse, so without this the memory stays
    with the app. Uses only frameworks that are already imported. Call it only while no generation is running."""
    if "mlx.core" in sys.modules:
        MLX.run(free_mlx)
    else:
        gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                torch.mps.empty_cache()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def is_oom(e):
    """Out of memory, as PyTorch (MPS, CUDA), MLX or Python report it."""
    text = str(e).lower()
    return isinstance(e, MemoryError) or any(m in text for m in ("out of memory", "insufficient memory", "unable to allocate", "attempting to allocate"))


def memory_hint(what, other=None):
    """The message for the panel. The safety limit of PyTorch stays on: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 can freeze the Mac."""
    side = f"The other side's model ({other}) takes memory too. " if other else ""
    return (f"Not enough memory for {what}. {side}A bf16 8B model needs about 16 GB, so two of them do not fit. "
            "Choose the MLX 4-bit version (about 4.6 GB) or a smaller model on the other side, then pick this model again.")[:400]


def strip_think(pieces):
    """Drop a leading <think>...</think> block from a stream of text pieces."""
    buf, started = "", False
    for piece in pieces:
        if started:
            yield piece
            continue
        buf += piece
        s = buf.lstrip()
        if not s:
            continue
        if s.startswith("<think>"):
            if "</think>" in s:
                started = True
                rest = s.split("</think>", 1)[1].lstrip("\n")
                if rest:
                    yield rest
        elif "<think>".startswith(s):
            continue
        else:
            started = True
            yield buf
    if not started and buf:
        yield buf


class MockBackend:
    """No model: writes a plausible program for the first two cubes, token by token (for testing the app)."""
    def __init__(self, label="mock", broken=False, drop_def=False, **_):
        self.label, self.broken, self.drop_def = label, broken, drop_def

    def stream(self, messages, max_new_tokens=900):
        user = messages[-1]["content"]
        if "poem" in user.lower() or "weather" in user.lower():
            text = "I can only write robot programs. I see the cubes listed on the table. Please ask me to move or stack them."
        else:
            ids = [l.split(" (")[0].strip() for l in user.split("\n")[0].replace("Objects:", "").split(",")]
            a, b = ids[0], ids[1 % len(ids)]
            z = "0.20" if self.broken else "0.06"
            text = (f"def run(robot):\n    a = robot.get_object_position('{a}')\n    b = robot.get_object_position('{b}')\n"
                    "    robot.open_gripper()\n    robot.move_to(a[0], a[1], 0.20)\n    robot.move_to(a[0], a[1], 0.02)\n    robot.close_gripper()\n"
                    "    robot.move_to(a[0], a[1], 0.20)\n    robot.move_to(b[0], b[1], 0.20)\n"
                    f"    robot.move_to(b[0], b[1], {z})\n    robot.open_gripper()\n    robot.move_to(b[0], b[1], 0.20)")
        if self.drop_def:
            text = text.replace("def run(robot):", "run(robot):", 1)          # imitates the slip of the small fine-tuned model
        for i in range(0, len(text), 6):
            time.sleep(0.03)
            yield text[i:i + 6]


MLX_FILES = ["*.json", "model*.safetensors", "*.py", "tokenizer.model", "*.tiktoken", "tiktoken.model", "*.txt", "*.jsonl", "*.jinja"]      # what mlx_lm.load() fetches


def fetch_mlx_model(path):
    """Download a model before its load goes to the MLX thread, or a 5 GB download would hold up the other side's generation.
    Only a head start: if it fails, load() on the MLX thread tries again and reports the error in its own words."""
    if Path(path).exists():
        return
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(path, allow_patterns=MLX_FILES)
    except Exception as e:
        print(f"download of {path} failed ({type(e).__name__}), load will try again", flush=True)


class MLXBackend:
    """Apple Silicon: mlx-lm. path = Hugging Face id or local folder (a merged/converted model);
    adapter = optional folder with an MLX LoRA adapter (adapters.safetensors + adapter_config.json).
    Everything that touches the model runs on the MLX thread."""
    def __init__(self, path, adapter=None, label="mlx", enable_thinking=False, **_):
        self.label, self.thinking = label, enable_thinking
        fetch_mlx_model(path)

        def load_model():
            import mlx.core as mx
            from mlx_lm import load
            self.model, self.tok = load(path, adapter_path=adapter) if adapter else load(path)
            mx.eval(self.model.parameters())                            # load() leaves adapter weights unread: a bad file or no memory should show up now

        MLX.run(load_model)

    def stream(self, messages, max_new_tokens=900):
        def pieces():
            from mlx_lm import stream_generate
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.thinking)
            for r in stream_generate(self.model, self.tok, prompt=prompt, max_tokens=max_new_tokens):
                yield r.text

        yield from strip_think(MLX.stream(pieces))

    def close(self):
        def drop():
            self.model = self.tok = None

        MLX.run(drop)


def half_dtype(torch, dev):
    """bf16 on every device, never float32 (an 8B model would need 32 GB instead of 16 GB). float16 only where MPS has no bf16 (macOS before 14)."""
    if dev == "mps":
        try:
            torch.zeros(1, dtype=torch.bfloat16, device=dev)
        except Exception:
            return torch.float16
    return torch.bfloat16


def load_causal_lm(auto_model, path, dtype, dev):
    """Weights straight onto the device (device_map, needs accelerate, which peft brings along). The plain way holds a CPU copy and a device copy at once."""
    if importlib.util.find_spec("accelerate"):
        try:
            return auto_model.from_pretrained(path, dtype=dtype, device_map={"": dev})
        except Exception as e:
            if is_oom(e):
                raise
            print(f"direct load onto {dev} failed ({type(e).__name__}: {e}), loading through the CPU", flush=True)
    return auto_model.from_pretrained(path, dtype=dtype).to(dev)


class TransformersBackend:
    """CUDA / CPU / MPS through transformers; optional PEFT LoRA adapter (e.g. the QLoRA adapter trained on the H100)."""
    def __init__(self, path, adapter=None, label="transformers", device=None, enable_thinking=False, **_):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if adapter:
                import peft  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"this model needs the transformers backend, and '{e.name}' is not installed. In the mac_app folder run: "
                               ".venv/bin/pip install torch peft   then pick the model again. An MLX version of the model is much faster.") from None
        self.label, self.thinking = label, enable_thinking
        dev = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        dtype = half_dtype(torch, dev)
        self.model = load_causal_lm(AutoModelForCausalLM, path, dtype, dev)
        if next(self.model.parameters()).dtype == torch.float32:
            self.model = self.model.to(dtype)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
            release_memory()                                            # the adapter's own tensors, now merged into the weights
        self.dev = dev
        print(f"transformers: {path} on {dev}, {next(self.model.parameters()).dtype}", flush=True)

    def stream(self, messages, max_new_tokens=900):
        import torch
        from transformers import TextIteratorStreamer
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.thinking)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.dev)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        failed = []

        def generate():
            try:
                with torch.no_grad():                                   # per thread, so it has to be set here
                    self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer)
            except Exception as e:                                      # only the text: a traceback would keep the KV cache alive
                failed.append(f"{type(e).__name__}: {e}")
                streamer.end()                                          # or the reader below would wait forever

        worker = threading.Thread(target=generate, daemon=True)
        worker.start()
        try:
            yield from strip_think(streamer)
        finally:
            worker.join()                                               # the thread holds the model and the KV cache until it has really ended
            ids = streamer = None
            release_memory()
        if failed:
            raise RuntimeError(failed[0])

    def close(self):
        self.model = self.tok = None


class HTTPBackend:
    """Any OpenAI-compatible server (llama.cpp server, Ollama, LM Studio, vLLM)."""
    def __init__(self, url="http://localhost:11434/v1", model="qwen3:8b", label="http", **_):
        self.url, self.model, self.label = url.rstrip("/"), model, label

    def stream(self, messages, max_new_tokens=900):
        req = urllib.request.Request(self.url + "/chat/completions", headers={"Content-Type": "application/json"}, data=json.dumps(
            dict(model=self.model, messages=messages, stream=True, max_tokens=max_new_tokens, temperature=0)).encode())

        def pieces():
            with urllib.request.urlopen(req) as r:
                for line in r:
                    line = line.decode().strip()
                    if line.startswith("data:") and line != "data: [DONE]":
                        delta = json.loads(line[5:])["choices"][0]["delta"].get("content")
                        if delta:
                            yield delta
        yield from strip_think(pieces())


BACKENDS = {"mock": MockBackend, "mlx": MLXBackend, "transformers": TransformersBackend, "http": HTTPBackend}


def make(cfg):
    cfg = dict(cfg)
    return BACKENDS[cfg.pop("backend")](**cfg)
