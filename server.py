"""Duel app: one prompt, two models, two live MuJoCo simulators side by side.   python server.py [--port 8766] [--mock]
Standard library web server; the models run through MLX (see models.json)."""
import argparse
import base64
import gzip
import importlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sim"))
import llm_backends  # noqa: E402
import lara_assets  # noqa: E402
import models_scan  # noqa: E402
import recorder  # noqa: E402
import scene_gen  # noqa: E402

SYSTEM = (HERE / "system_prompt.md").read_text().strip()
_OBJECTS_LINE = "The user message lists the objects on the table (id and color), followed by the request. Only use these ids."
SYSTEM_SCENE = SYSTEM.replace(_OBJECTS_LINE, "The user message gives the request, followed by Scene_Info: JSON with the objects on the table (id, color, start position) and the gripper. Only use these ids.")
# The "answers" format: the system prompt of data_posv1 / data_posv2 byte for byte (system_prompt_answers.md), that is the Scene_Info prompt plus this line.
ANSWER_LINE = "If the request is only a question about the scene, do not write code and do not move the arm: answer it in one short sentence."
SYSTEM_ANSWERS = (HERE / "system_prompt_answers.md").read_text().strip()
assert SYSTEM_ANSWERS.replace(ANSWER_LINE + "\n", "") == SYSTEM_SCENE, "system_prompt_answers.md must be the Scene_Info prompt plus ANSWER_LINE"
# data_posv3: its system prompt byte for byte (system_prompt_posv3.md), that is the answers prompt plus THINK_LINE. A posv3 model trained without
# reasoning gets the same prompt without that line, which is the answers prompt again.
THINK_LINE = "Before the code, think briefly: what the scene shows, the goal, the plan, and where the gripper must turn or drop a cube."
SYSTEM_POSV3 = (HERE / "system_prompt_posv3.md").read_text().strip()
SYSTEM_POSV3_NOTHINK = SYSTEM_POSV3.replace(THINK_LINE + "\n", "")
assert SYSTEM_POSV3_NOTHINK != SYSTEM_POSV3 and SYSTEM_POSV3_NOTHINK == SYSTEM_ANSWERS, "system_prompt_posv3.md must be the answers prompt plus THINK_LINE"
# format (models_scan.PROMPT_FORMATS, documented there) -> name on the page, system prompt, Scene_Info JSON in the user message (else the Objects line)
PROMPTS = {"objects": ("Objects line", SYSTEM, False), "scene_v5": ("Scene_Info JSON", SYSTEM_SCENE, True), "answers_posv1": ("Scene_Info + answers", SYSTEM_ANSWERS, True),
           "posv3_think": ("posv3, thinking", SYSTEM_POSV3, True), "posv3_nothink": ("posv3, no thinking", SYSTEM_POSV3_NOTHINK, True)}
assert set(PROMPTS) == set(models_scan.PROMPT_FORMATS)
ANSWERING = {"answers_posv1", "posv3_think", "posv3_nothink"}         # formats whose models also answer scene questions in one sentence
EXAMPLES = [json.loads(l) for l in (HERE / "examples.jsonl").read_text().splitlines() if l.strip()] if (HERE / "examples.jsonl").exists() else []
BY_ID = {e["id"]: e for e in EXAMPLES}
DATASETS = [n for n in ("data_posv3", "data_posv2", "data_posv1", "data_posv0", "data_v5", "data_v4", "data_v3", "data_v2", "data_v1", "data_v0") if (HERE / "datasets" / f"{n}.jsonl.gz").exists()]    # newest first
DATASET_ROWS = {}                                        # name -> {id: row}, read when the ID overlay first asks for it
PROVENANCE = ("source", "family", "start_state")         # data_posv3: where a row comes from, shown in the row info
RUNS, GEN_LOCK, LOAD_LOCK = {}, threading.Lock(), threading.Lock()
BACKENDS = {"base": None, "finetuned": None}             # the only owners of the models: a model lives exactly as long as a side holds it
MODEL_KEY = {"base": None, "finetuned": None}            # what each side holds (see model_key); equal keys = one shared model
MODEL_FORMAT = {"base": "objects", "finetuned": "objects"}   # the prompt format the loaded model was trained with, a key of PROMPTS (see models_scan.prompt_format)
STATUS = {s: dict(state="empty", label="", error=None) for s in BACKENDS}
STATE = threading.Condition()                            # guards GENERATING, LOAD_SEQ and the step "all ready? then start a run" against a load
GENERATING = set()                                       # (run, side) from the start of a run until that side's model has finished writing
LOAD_SEQ = {s: 0 for s in BACKENDS}                      # newest load request per side: an older one that still waits gives up
EXPORTS, EXPORT_LOCK = {}, threading.Lock()              # Export video: job id -> state; one export renders at a time (memory)
EXPORT_DIR, EXPORT_KEEP = Path(tempfile.mkdtemp(prefix="p2r_exports_")), 6
# Read-only token shipped with the app (hf_readonly_token.txt, ignored by git). Share the app only with people you trust; a token saved in Settings or HF_TOKEN wins.
_tok_file = HERE / "hf_readonly_token.txt"
DEFAULT_TOKEN = (_tok_file.read_text().strip() or None) if _tok_file.exists() else None
HF_TOKEN = {"value": os.environ.get("HF_TOKEN") or DEFAULT_TOKEN}
if HF_TOKEN["value"]:
    os.environ["HF_TOKEN"] = HF_TOKEN["value"]
COLORS = ["red", "blue", "green", "yellow", "orange", "white", "purple", "cyan", "pink", "black", "brown", "gray"]     # the 12 colours of the dataset
TRAINED_CUBES = 6                                        # scenes of the datasets up to data_posv0 stay around 6 cubes; a model with "max_cubes" in p2r_meta.json says its own number
GRID = [(x, y) for x in (0.30, 0.407, 0.513, 0.62) for y in (-0.28, -0.168, -0.056, 0.056, 0.168, 0.28)]    # workspace of random_scene, every 10.6 cm or more
BIG_GRID = [(round(0.20 + 0.104 * i, 3), round(-0.364 + 0.104 * j, 3)) for i in range(6) for j in range(8)]     # above 24 cubes: 48 cells, 10.4 cm apart, all reachable and graspable
# Rows of these versions are checked with their own checker in sim/, copied unchanged from the dataset's tools/: check_posv1.py (shapes, chain steps, answers),
# check_posv3.py (check_posv1 plus the strict grasp rule and the outside_area constraint).
NEW_CHECK = {"data_posv1": "check_posv1", "data_posv2": "check_posv1", "data_posv3": "check_posv3"}
_check = {}


def request_of(row):
    """The request alone: rows up to data_v4 start with an "Objects:" line, newer ones are only the request (which can have several lines)."""
    up = row["user_prompt"]
    return up.split("\n", 1)[1] if up.startswith("Objects:") and "\n" in up else up


def expects_answer(sample, fmt):
    """Is a plain-text reply here an answer to a scene question? The dataset row decides. Without a row only the format can tell, and a refusal looks the same."""
    return sample.get("expected_behavior") == "answer" if sample else fmt in ANSWERING


def dataset_rows(name):
    """One bundled dataset version. Rows without an "Objects:" line get it here, because sim/check.py (identical to tools/check.py) requires it;
    tools/check_v5.py differs from it only in that rule."""
    if name not in DATASET_ROWS:
        rows = {}
        with gzip.open(HERE / "datasets" / f"{name}.jsonl.gz", "rt") as f:
            for r in map(json.loads, f):
                if name in NEW_CHECK:
                    r["_check"] = NEW_CHECK[name]                # request-only prompts, as that checker expects
                elif not r["user_prompt"].startswith("Objects:"):
                    head = ", ".join(f"{i} ({o['color']})" for i, o in r["scene"]["objects"].items())
                    r["request"], r["user_prompt"] = r["user_prompt"], f"Objects: {head}\n{r['user_prompt']}"
                rows[r["id"]] = r
        DATASET_ROWS[name] = rows
    return DATASET_ROWS[name]


def find_example(example_id, dataset=None):
    return (dataset_rows(dataset) if dataset in DATASETS else BY_ID).get(example_id)


def check_sample(sample, answer):
    name = sample.get("_check", "check")
    if name not in _check:
        _check[name] = importlib.import_module(name).check
    ok, reasons = _check[name](dict(sample, assistant_answer=answer))
    return bool(ok), list(reasons)[:4]


NUMBER_WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
NUMBERS = {w: str(i) for i, w in enumerate(NUMBER_WORDS)} | {f"twenty {w}": str(20 + i) for i, w in enumerate(NUMBER_WORDS[:10]) if i}


def check_answer(sample, text):
    """(True, False or None, reasons) for a reply to an expected_behavior "answer" row; None = the row gives no words to judge by.
    Stopgap for destination {"mode": "answer", "accept": [...], "reject": [...]} as drafted in the spec: every accept word must appear, no reject word may. An accept entry
    that is a list means any one of its words. Whole words, ignoring case, and "five" counts as "5". The stored words are all it knows: it cannot notice a stale row.
    Rows of data_posv1 / data_posv2 / data_posv3 use sim/check_posv1.py or sim/check_posv3.py instead, which derives the true answer from the scene (the words below are only the fallback).
    sim/check.py stays as it is, the rows with mode "answer" never reach it."""
    if sample.get("_check"):
        return check_sample(sample, text)
    def norm(value):
        low = re.sub(r"[\s-]+", " ", str(value).lower())
        return re.sub(r"\b(" + "|".join(sorted(NUMBERS, key=len, reverse=True)) + r")\b", lambda m: NUMBERS[m.group(1)], low)

    def said(word):
        return re.search(r"(?<!\w)" + re.escape(norm(word)) + r"(?!\w)", low) is not None

    dest, low = sample.get("destination") or {}, norm(text)
    accept, reject = ([dest[k]] if isinstance(dest.get(k), str) else list(dest.get(k) or []) for k in ("accept", "reject"))
    if not accept and not reject:
        return None, ["this row has no accept or reject words, so there is no verdict"]
    reasons = [f"missing: {w}" for w in accept if not any(said(x) for x in (w if isinstance(w, list) else [w]))] + [f"must not say: {w}" for w in reject if said(w)]
    return not reasons, reasons


def text_result(sample, fmt, text):
    """What a reply without a program counts as -> the fields of the result event. A side that had to write code stays a plain text reply, checked as before."""
    if not expects_answer(sample, fmt):
        kw = dict(answer=text.strip())
        if sample:
            kw["success"], kw["reasons"] = check_sample(sample, text.strip())
        return kw
    answer = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).replace("<|im_end|>", "").strip()
    kw = dict(answer=answer, answered=True)
    if sample:
        ok, kw["reasons"] = check_answer(sample, answer)
        if ok is not None:
            kw["success"] = ok
    return kw


def other_side(side):
    return "finetuned" if side == "base" else "base"


def model_key(cfg):
    """Everything that makes a loaded model what it is; the label is only a name. Thinking on or off counts as a different model."""
    c = {k: os.path.realpath(v) if k in ("path", "adapter") and v and os.path.exists(v) else v for k, v in cfg.items() if k != "label"}
    return json.dumps(c, sort_keys=True, default=str)


def set_loading(side, label, seq=None):
    with STATE:
        if seq is None or seq == LOAD_SEQ[side]:
            STATUS[side] = dict(state="loading", label=label, error=None)


def stale(side, seq):
    with STATE:
        return seq is not None and seq != LOAD_SEQ[side]


def wait_for_generation(side, label, seq):
    """Models are swapped only while none is writing. A run that has started keeps its models until both sides have written (at most max_new_tokens
    each). No new run can start here, because this side is in state "loading". The simulations do not need the models and go on."""
    with STATE:
        if GENERATING:
            set_loading(side, f"{label} (waiting for the running generation)", seq)
            print(f"{side}: waiting for the running generation before swapping the model", flush=True)
        while GENERATING:
            STATE.wait(1)


def generation_over(run, side):
    with STATE:
        GENERATING.discard((run, side))
        STATE.notify_all()


def unload_side(side):
    """Drop this side's model. It is freed unless the other side shares it. Caller holds LOAD_LOCK and GEN_LOCK, no generation is running."""
    backend = BACKENDS[side]
    BACKENDS[side], MODEL_KEY[side], MODEL_FORMAT[side] = None, None, "objects"
    if backend is None:
        return
    label = getattr(backend, "label", "model")
    if backend is BACKENDS[other_side(side)]:
        print(f"{side}: let go of {label}, the other side still uses it", flush=True)
        return
    if hasattr(backend, "close"):
        backend.close()
    del backend
    llm_backends.release_memory()
    print(f"{side}: unloaded {label}, memory released", flush=True)


def mark_failed(side, label, error, seq=None):
    """A failed load leaves the side empty and says why. Runs after the except block, when the traceback no longer holds a half-loaded model."""
    with GEN_LOCK:
        unload_side(side)
        llm_backends.release_memory()
    with STATE:
        if seq is None or seq == LOAD_SEQ[side]:
            STATUS[side] = dict(state="error", label=label, error=error)
    print(f"{side} failed: {error}", flush=True)


def load_error(side, info, e):
    print(f"{side}: {type(e).__name__}: {str(e)[:400]}", flush=True)
    if llm_backends.is_oom(e):
        return llm_backends.memory_hint(info["label"], other_model(side))
    return models_scan.pairing_hint(info, e) or f"{type(e).__name__}: {e}"[:400]


def other_model(side):
    other = other_side(side)
    return STATUS[other]["label"] if BACKENDS[other] is not None and BACKENDS[other] is not BACKENDS[side] else None


def load_side(side, info, thinking=None, seq=None):
    """Load a model into one side of the duel (runs in a thread; one load at a time to keep memory and downloads sane).
    The old model goes first, so the peak is never old + new. A model the other side already holds is shared, not loaded again."""
    label, error = info["label"], None
    with LOAD_LOCK:
        if stale(side, seq):
            return
        set_loading(side, label, seq)
        wait_for_generation(side, label, seq)
        if stale(side, seq):
            return
        set_loading(side, label, seq)
        try:
            meta = models_scan.read_meta(info, HF_TOKEN["value"])
            if thinking is None:                               # the template menu wins; else enable_thinking or prompt_format of p2r_meta.json, else as before
                thinking = models_scan.thinking_of(meta, info["thinking"])
            cfg = models_scan.backend_cfg(info, thinking)
            key = model_key(cfg)
            if key != MODEL_KEY[side]:
                with GEN_LOCK:
                    unload_side(side)
                if key == MODEL_KEY[other_side(side)]:
                    BACKENDS[side] = BACKENDS[other_side(side)]
                    print(f"{side}: sharing {label} with the other side", flush=True)
                else:
                    print(f"loading {side}: {label} ...", flush=True)
                    BACKENDS[side] = llm_backends.make(cfg)
                MODEL_KEY[side] = key
            MODEL_FORMAT[side] = models_scan.prompt_format(meta)
            base = models_scan.base_info(info["base"]) if info.get("base") else None
            with STATE:
                if seq is None or seq == LOAD_SEQ[side]:           # meanwhile the side may have gone to Manual code, which then drops this model
                    STATUS[side] = dict(state="ready", label=label, error=None, base_info=base, max_cubes=models_scan.max_cubes(meta, TRAINED_CUBES))
            print(f"{side} ready", flush=True)
        except Exception as e:
            error = load_error(side, info, e)
        if error:
            mark_failed(side, label, error, seq)


def manual_side(side):
    """Manual code: this side needs no model. It is ready at once, a load in progress no longer counts. Its model goes as soon as no generation needs it."""
    with STATE:
        LOAD_SEQ[side] += 1
        seq = LOAD_SEQ[side]
        STATUS[side] = dict(state="ready", label="Manual code", error=None, manual=True)
    with LOAD_LOCK:
        with STATE:
            while GENERATING:
                STATE.wait(1)
        if not stale(side, seq):
            with GEN_LOCK:
                unload_side(side)


def on_base(info, base):
    """An MLX adapter on the base the user chose instead of its own. The adapter's repo or folder stays the same, the model is a different one (see model_key)."""
    if not base or info["kind"] != "adapter" or base == info["base"]:
        return info
    return dict(info, base=base, own_base=info["base"], label=f"{info['label']} · on {models_scan.base_info(base)['name']}")


def load_request(req):
    """Resolve what the user picked (list entry, folder or Hugging Face link; may download) and load it."""
    side, error = req["side"], None
    if req.get("manual"):
        return manual_side(side)
    with STATE:
        LOAD_SEQ[side] += 1
        seq = LOAD_SEQ[side]
        STATUS[side] = dict(state="loading", label="looking for the model ...", error=None)
    try:
        known = None if req.get("custom") else next((m for m in models_scan.list_models() if m["path"] == req["path"]), None)
        if known and known["kind"] in ("hf_repo", "hf_adapter"):      # an adapter: never straight to MLX, describe() picks the backend from the files
            set_loading(side, f"{known['label']} (downloading on first use)", seq)
            info = dict(models_scan.resolve(req["path"], HF_TOKEN["value"], log=lambda m: print(m, flush=True)), label=known["label"])
        else:
            info = known or models_scan.resolve(req["path"], HF_TOKEN["value"], log=lambda m: print(m, flush=True))
        if info["kind"] == "hf":                           # plain model id from the list, MLX fetches it itself
            set_loading(side, f"{info['label']} (downloading on first use)", seq)
        th = {"thinking": True, "nothinking": False}.get(req.get("template"))
        load_side(side, on_base(info, req.get("base")), th, seq)
    except Exception as e:
        error = str(e)[:400]
    if error:                                              # the model was not found: the side must not keep the old one under an error message
        label = str(req.get("path", ""))
        with LOAD_LOCK:
            wait_for_generation(side, label, seq)
            if not stale(side, seq):
                mark_failed(side, label, error)


def side_format(side, mode):
    """The prompt format a side uses: auto = what its model was trained with, else the one chosen on the page."""
    return MODEL_FORMAT[side] if mode == "auto" else models_scan.format_name(mode) or "objects"


def user_prompt(scene, instruction, scene_info=False):
    """The two user messages. Scene_Info is byte for byte what the models trained with it have seen; the others know only the Objects line.
    All formats from scene_v5 on use the Scene_Info message, only their system prompts (and thinking) differ (see PROMPTS)."""
    if scene_info:
        return instruction + "\n\nScene_Info:\n" + json.dumps({"scene": scene})
    objs = ", ".join(f"{i} ({o['color']})" for i, o in scene["objects"].items())
    return f"Objects: {objs}\n{instruction}"


class ThinkSplitter:
    """Splits a streamed reply into reasoning and the rest while it streams: feed(piece) -> [("think" | "token", text), ...].
    Only a <think> block at the very start counts (Qwen3's thinking format); the tags themselves never reach the page, even when split over pieces.
    thinking=True (the chat template opened the thinking mode): the reply is reasoning until </think> even if the model leaves out the opening <think>,
    as some chat templates write that tag into the prompt. tagged then says whether any tag came at all (see split_think)."""
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self, thinking=False):
        self.state, self.buf, self.lead, self.thinking = "start", "", {"think": True, "token": True}, thinking   # lead: no non-blank text of that kind sent yet
        self.tagged = False

    def _out(self, kind, text):
        if self.lead[kind]:
            text = text.lstrip()
            self.lead[kind] = not text
        return [(kind, text)] if text else []

    def _reply(self, text):
        return self._out("token", text)

    def feed(self, piece):
        self.buf += piece
        if self.state == "start":
            s = self.buf.lstrip()
            if not s or (self.OPEN.startswith(s) and s != self.OPEN):
                return []
            if s.startswith(self.OPEN):
                self.state, self.buf, self.tagged = "think", s[len(self.OPEN):], True
            elif self.thinking:
                self.state, self.buf = "think", s
            else:
                self.state, text, self.buf = "reply", self.buf, ""
                return self._reply(text)
        if self.state == "think":
            if self.CLOSE in self.buf:
                inner, rest = self.buf.split(self.CLOSE, 1)
                self.state, self.buf, self.tagged = "reply", "", True
                return self._out("think", inner.rstrip()) + self._reply(rest)
            keep = next((k for k in range(len(self.CLOSE) - 1, 0, -1) if self.buf.endswith(self.CLOSE[:k])), 0)     # maybe the start of </think>
            text = self.buf[:len(self.buf) - keep]
            text = text.rstrip()                                    # blanks wait for what follows: before </think> they are dropped
            self.buf = self.buf[len(text):]
            return self._out("think", text)
        text, self.buf = self.buf, ""
        return self._reply(text)

    def flush(self):
        """End of the stream: whatever is still held back."""
        text, self.buf = self.buf, ""
        return self._out("think", text) if self.state == "think" else self._reply(text)


def split_think(text, thinking=False):
    """A whole reply -> (reasoning, reply, closed). reasoning = the leading <think> block ("" if there is none), reply = what follows </think>;
    closed is False if the block never ended (the token limit came first), then reply is "". With thinking on, a </think> without the opening tag
    ends the reasoning too. A reply without any tag is all reply, also with thinking on (ThinkSplitter.tagged tells the page to move it)."""
    s = text.replace("<|im_end|>", "").lstrip()
    if s.startswith("<think>"):
        inner, tag, rest = s[len("<think>"):].partition("</think>")
        return inner.strip(), rest.strip(), bool(tag)
    if thinking and "</think>" in s:
        inner, _, rest = s.partition("</think>")
        return inner.strip(), rest.strip(), True
    return "", text, True


def extract_code(text):
    """Model text -> (code, notes). Tolerant on purpose: strips <think> blocks and markdown fences, and repairs the usual slips of
    small fine-tuned models (missing `def`, `def run(robot)` without colon, header written as `run(robot):`, body without header).
    `code` is "" if the text contains no program at all (a plain-text answer / refusal)."""
    notes = []
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).replace("<|im_end|>", "").strip()
    m = re.search(r"```(?:python|py)?[ \t]*\n?(.*?)(?:```|$)", text, re.S)
    if m and ("run" in m.group(1)):
        text = m.group(1).strip()
        notes.append("removed markdown fence")
    text = re.sub(r"^\s*(python|py)\s*\n", "", text)                    # a bare "python" line left by a fence
    robot_call = re.search(r"\brobot\.(get_object_position|get_gripper_position|move_to|open_gripper|close_gripper|rotate_gripper)\(", text)
    hdr = re.search(r"^[ \t]*(?:def[ \t]+)?run[ \t]*\([ \t]*robot[ \t]*\)[ \t]*:?[ \t]*$", text, re.M)
    if hdr:
        line = hdr.group(0)
        if not re.match(r"[ \t]*def[ \t]+run[ \t]*\([ \t]*robot[ \t]*\)[ \t]*:[ \t]*$", line):
            notes.append("repaired the function header (def run(robot):)")
        code = "def run(robot):" + text[hdr.end():]               # anything before the header (chatter) is dropped
    elif robot_call:                                                  # the program is there but the header is missing entirely
        lines = text.strip("\n").split("\n")
        code = "def run(robot):\n" + "\n".join(l if l.startswith((" ", "\t")) else "    " + l for l in lines)
        notes.append("added the missing function header (def run(robot):)")
    else:
        return "", notes
    return code.strip(), notes


def cube_count(value, default=3):
    """Whatever came from the page -> a whole number of at least 1. No upper limit here: random_scene places what fits."""
    try:
        return max(1, int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return default


def random_scene(n, seed):
    """Up to n cubes at least 10 cm apart. More than fit: as many as fit (len(BIG_GRID)), the page tells the user. Always ends after a bounded number of tries.
    Colours repeat above 12 cubes, the ids stay unique."""
    cells = GRID if n <= len(GRID) else BIG_GRID         # up to 24 cubes the scenes stay in the area they always had
    n = min(n, len(cells))
    rng = random.Random(seed)
    ids = sorted(rng.sample(range(1, max(n, 9) + 1), n))
    cols = (COLORS * (n // len(COLORS) + 1))[:n]
    rng.shuffle(cols)
    pos = []
    for _ in range(2000):
        if len(pos) == n:
            break
        p = (round(rng.uniform(0.30, 0.62), 2), round(rng.uniform(-0.28, 0.28), 2))
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > 0.10 ** 2 for q in pos):
            pos.append(p)
    if len(pos) < n:                                     # random placing got stuck (happens from about 14 cubes): a random part of the grid always fits
        pos = rng.sample(cells, n)
    return dict(objects={str(i): dict(color=c, position=[x, y, 0.02]) for i, c, (x, y) in zip(ids, cols, pos)},
                gripper=dict(position=[0.4, 0.0, 0.3], angle_deg=0, is_open=True))


class Run:
    def __init__(self, rid):
        self.id, self.events, self.cv, self.done, self.videos = rid, [], threading.Condition(), set(), {}     # videos: side -> (path, file name)
        self.meta = {}                                   # side -> what the export draws before the simulation: request, model, reasoning, answer

    def emit(self, side, type_, **kw):
        with self.cv:
            self.events.append(dict(side=side, type=type_, **kw))
            if type_ == "done":
                self.done.add(side)
            self.cv.notify_all()


def work(run, side, req):
    try:
        sample = find_example(req.get("example_id"), req.get("dataset"))
        answer_row = expects_answer(sample, None)                           # a question of the dataset: sim/check.py has no rule for it, see check_answer
        fmt, reasoning = None, ""
        text = (req.get("manual") or {}).get(side)                          # Manual code: the pasted text takes the place of the model's answer
        model = "manual code" if text is not None else STATUS[side]["label"]     # for the file name of the video
        if text is not None:
            generation_over(run, side)
            run.emit(side, "format", text="Manual code", manual=True)
            if not text.strip():
                run.emit(side, "result", kind="code", executed=False, error="nothing to run: paste a program (def run(robot): ...) into the code box")
                return
        else:
            run.emit(side, "status", text="waiting for the model")
            mode = (req.get("formats") or {}).get(side, "auto")
            fmt = side_format(side, mode)
            name, system, scene_info = PROMPTS[fmt]
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": user_prompt(req["scene"], req["instruction"], scene_info)}]
            if req.get("prompts"):                       # Manual model use: the edited texts replace the two messages; format and thinking stay
                msgs = [{"role": "system", "content": req["prompts"]["system"]}, {"role": "user", "content": req["prompts"]["user"]}]
            thinking = bool(getattr(BACKENDS[side], "thinking", False))
            run.emit(side, "format", text=name, auto=mode == "auto", thinking=thinking,
                     system=msgs[0]["content"], user=msgs[1]["content"], request=req["instruction"])      # shown as sent, so the prompt is no black box
            text, t0, n, split, coding = "", None, 0, ThinkSplitter(thinking), False
            limit = llm_backends.max_tokens(thinking or fmt == "posv3_think")      # reasoning + code: more room than for code alone
            with GEN_LOCK:                               # one model generates at a time (memory); the simulators run in parallel
                writing = "writing the reply" if fmt in ANSWERING else "writing code"
                run.emit(side, "status", text="thinking" if thinking else writing)
                if BACKENDS[side] is None:
                    raise RuntimeError("no model loaded on this side")
                for piece in BACKENDS[side].stream(msgs, max_new_tokens=limit):
                    t0 = t0 or time.time()
                    text, n = text + piece, n + 1
                    for kind, part in split.feed(piece):         # the reasoning goes to the Reasoning box, the rest to the code box
                        if kind == "token" and thinking and not coding:
                            coding = True
                            run.emit(side, "status", text=writing)
                        run.emit(side, kind, text=part)
                for kind, part in split.flush():
                    run.emit(side, kind, text=part)
            generation_over(run, side)
            if thinking and not split.tagged and text.strip():           # thinking on, but the model wrote no tag at all: it was the reply, not reasoning
                run.emit(side, "unthought", text=text.strip())
            reasoning, text, closed = split_think(text, thinking)
            if reasoning or not closed:
                run.emit(side, "thought", words=len(reasoning.split()), closed=closed, empty=closed and not text.strip())
            run.emit(side, "generated", seconds=round(time.time() - (t0 or time.time()), 1), pieces=n, answered=closed and bool(text.strip()) and not extract_code(text)[0] and expects_answer(sample, fmt),
                     stopped=not closed or bool(reasoning and not text.strip()))
            if not closed:                               # half a reasoning may hold half a program: nothing of it runs
                run.emit(side, "result", kind="code", executed=False, unfinished="reasoning",
                         error=f"the model stopped while still reasoning (no </think> within {limit} new tokens): no code, answer or refusal, so nothing was run")
                return
            if reasoning and not text.strip():
                run.emit(side, "result", kind="code", executed=False, unfinished="empty",
                         error="the model ended its reasoning but wrote no answer after </think>: no code, answer or refusal, so nothing was run")
                return
        code, notes = extract_code(text)
        if not code:
            kw = text_result(sample, fmt, text)
            if kw.get("answered"):                       # the arm did not move: the scene is the end state, so Chain prompt goes on from it
                kw.update(next_scene=req["scene"], dropped=[])
            run.emit(side, "result", kind="text", executed=False, **kw)
            return
        run.emit(side, "code", code=code, notes=notes)
        job = Path(tempfile.mkdtemp()) / "job.json"
        job.write_text(json.dumps(dict(scene=req["scene"], code=code, sample=None if answer_row else sample, speed=req.get("speed", 4.0),
                                         render=req.get("render") or "1280x720")))
        run.emit(side, "status", text="running in the simulator")
        p = subprocess.Popen([sys.executable, str(HERE / "sim_worker.py"), str(job)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        rec = recorder.Recorder(req.get("instruction"), model)               # the frames go to an MP4 on disk while they stream (Download video)
        result = None                                                        # the simulator's result event: the export's music follows it
        for line in p.stdout:
            e = json.loads(line)
            if e["t"] == "frame":                                             # the full-size frame goes to the recording, the small copy to the live view
                full = e.pop("jpg")
                rec.add(base64.b64decode(full))
                e["jpg"] = e.pop("live", None) or full
            if answer_row and e["t"] == "result" and e["executed"]:
                e.update(success=False, reasons=["this question needs one short sentence, not a program: the arm must not move"])
            if e["t"] == "result":
                result = {k: e.get(k) for k in ("executed", "error", "success", "strict_flags", "events", "frame_dt", "final", "sim_seconds", "calls")}
            run.emit(side, e.pop("t"), kind="code", **e)
        p.wait()
        video = rec.finish()
        if video:
            run.videos[side] = video
            run.meta[side] = dict(prompt=req.get("instruction") or "", model=model, reasoning=reasoning, answer=text, frames=min(rec.n, recorder.MAX_FRAMES) + recorder.HOLD,
                                  scene=req["scene"], result=result, code=code)
            run.emit(side, "video", url=f"/api/video/{run.id}/{side}", name=video[1], run=run.id)
    except Exception as e:
        error = llm_backends.memory_hint("this model", other_model(side)) if llm_backends.is_oom(e) else f"{type(e).__name__}: {e}"
        run.emit(side, "result", kind="code", executed=False, error=error)
    finally:
        generation_over(run, side)
        run.emit(side, "done")


def export(job, run, side, req):
    """Export video: renders the 1920x1080 MP4 of one side in exporter.py (its own niced process, so the live app stays smooth), with its music unless the switch is off."""
    st = EXPORTS[job]
    with EXPORT_LOCK:
        try:
            st["state"] = "rendering"
            meta, out = run.meta[side], EXPORT_DIR / f"{job}.mp4"
            spec = dict(ffmpeg=recorder.FFMPEG, video=str(run.videos[side][0]), frames=meta["frames"], out=str(out),
                        prompt=meta["prompt"] if req.get("prompt") else "", model=meta["model"] if req.get("model") else "",
                        reasoning=meta["reasoning"], answer=meta["answer"], pace=float(req.get("pace") or 25), code_pace=float(req.get("code_pace") or 0), speed=float(req.get("speed") or 2),
                        music=bool(req.get("music", True)), request=meta["prompt"], scene=meta.get("scene"), result=meta.get("result"), code=meta.get("code") or "")
            spec_file = Path(tempfile.mkdtemp()) / "export.json"
            spec_file.write_text(json.dumps(spec))
            p = subprocess.Popen([sys.executable, str(HERE / "exporter.py"), str(spec_file)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            for line in p.stdout:
                e = json.loads(line)
                st.update(progress=e.get("progress", st["progress"]), error=e.get("error"))
                if e.get("done"):
                    st["path"] = out
            p.wait()
            shutil.rmtree(spec_file.parent, ignore_errors=True)
            if not st["path"] and not st["error"]:
                st["error"] = "the export stopped without a video"
        except Exception as e:
            st["error"] = f"{type(e).__name__}: {e}"
        st["state"] = "done"
        for old in sorted(EXPORT_DIR.iterdir(), key=lambda q: q.stat().st_mtime)[:-EXPORT_KEEP]:
            old.unlink(missing_ok=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers()
        self.wfile.write(b)

    def _file(self, path, name):
        self.send_response(200); self.send_header("Content-Type", "video/mp4" if name.endswith(".mp4") else "image/gif")
        self.send_header("Content-Disposition", f'attachment; filename="{name}"'); self.send_header("Content-Length", str(path.stat().st_size)); self.end_headers()
        with path.open("rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            b = (HERE / "static" / "index.html").read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers()
            self.wfile.write(b)
        elif self.path == "/api/config":
            ex = [dict(id=e["id"], category=e["category"], subcategory=e["subcategory"], scene=e["scene"],
                       instruction=request_of(e)) for e in EXAMPLES]
            self._json(dict(examples=ex, datasets=DATASETS, status=STATUS, trained_cubes=TRAINED_CUBES, colors=len(COLORS),
                            color_names=COLORS, object_types=scene_gen.type_table()))
        elif self.path.startswith("/api/dataset/"):          # the rows of one dataset version for the ID overlay, without the answers
            name = self.path.rsplit("/", 1)[1]
            if name not in DATASETS:
                return self._json({"error": "unknown dataset"}, 404)
            self._json(dict(rows=[dict(id=e["id"], subcategory=e.get("subcategory", ""), split=e.get("split", ""), scene=e["scene"], dataset=name,
                                       instruction=e.get("request") or request_of(e), reasoning=bool(e.get("reasoning")), behavior=e.get("expected_behavior", ""),
                                       **{k: e[k] for k in PROVENANCE if e.get(k)}) for e in dataset_rows(name).values()]))
        elif self.path.startswith("/api/answer/"):           # the reference answer of one example; /api/config stays light without them
            path, _, dataset = self.path.partition("?dataset=")
            e = find_example(path.rsplit("/", 1)[1], dataset)
            if not e:
                return self._json({"error": "unknown example"}, 404)
            self._json(dict(id=e["id"], answer=e["assistant_answer"], program=bool(extract_code(e["assistant_answer"])[0]), expected=e.get("expected_behavior"),
                            reasoning=list(e.get("reasoning") or [])))      # data_posv3 train rows: the two stored reasonings of the answer
        elif self.path == "/api/status":
            self._json(STATUS)
        elif self.path.split("?")[0] == "/api/models":
            if "refresh=1" in self.path:                     # the ↻ button: ask Hugging Face again, the folders are rescanned anyway
                models_scan.refresh_account(HF_TOKEN["value"])
            ms = models_scan.list_models()
            self._json(dict(models=ms, bases=models_scan.mlx_bases(), status=STATUS, has_token=bool(HF_TOKEN["value"]), hf_models=len(models_scan.account_models()), note=models_scan.ACCOUNT["note"]))
        elif self.path.startswith("/api/video/"):            # Download video: the recorded run of one side as a file
            _, _, _, rid, side = (self.path.split("?")[0] + "//").split("/")[:5]
            video = RUNS[rid].videos.get(side) if rid in RUNS else None
            if not video or not video[0].exists():
                return self._json({"error": "no video for this run (it may be one of the older runs, only the last 20 are kept)"}, 404)
            self._file(*video)
        elif self.path.startswith("/api/export/"):           # Export video: the state of a job, or with /file its finished MP4
            job, _, rest = self.path.split("?")[0][len("/api/export/"):].partition("/")
            st = EXPORTS.get(job)
            if not st:
                return self._json({"error": "unknown export"}, 404)
            if rest != "file":
                return self._json(dict(state=st["state"], progress=st["progress"], error=st["error"], name=st["name"], ready=bool(st["path"])))
            if not st["path"] or not st["path"].exists():
                return self._json({"error": "this export is not ready or no longer kept"}, 404)
            self._file(st["path"], st["name"])
        elif self.path.startswith("/api/events/"):
            run = RUNS.get(self.path.rsplit("/", 1)[1])
            if not run:
                return self._json({"error": "unknown run"}, 404)
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.end_headers()
            i = 0
            try:
                while True:
                    with run.cv:
                        while i >= len(run.events) and len(run.done) < 2:
                            run.cv.wait(timeout=15)
                            if i >= len(run.events):
                                self.wfile.write(b": keepalive\n\n"); self.wfile.flush()
                        batch, i = run.events[i:], len(run.events)
                    for e in batch:
                        self.wfile.write(b"data: " + json.dumps(e).encode() + b"\n\n")
                    self.wfile.flush()
                    if len(run.done) >= 2 and i >= len(run.events):
                        break
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/api/scene":
            if not req.get("objects"):                          # no structures configured: exactly the scenes of before
                return self._json(dict(scene=random_scene(cube_count(req.get("n_cubes")), req.get("seed"))))
            try:
                self._json(scene_gen.generate(cube_count(req.get("n_cubes")), req.get("seed"), req["objects"], COLORS))
            except scene_gen.SceneError as e:
                self._json(dict(scene=None, error=str(e)))
        elif self.path == "/api/settings":
            tok = (req.get("hf_token") or "").strip().strip("\"'") or DEFAULT_TOKEN
            HF_TOKEN["value"] = tok or None
            if tok:
                os.environ["HF_TOKEN"] = tok                   # a token typed in Settings is kept in memory only, never written to disk
            else:
                os.environ.pop("HF_TOKEN", None)
            self._json(dict(ok=True, has_token=bool(tok), **models_scan.refresh_account(tok or None)))
        elif self.path == "/api/load":
            threading.Thread(target=load_request, args=(req,), daemon=True).start()
            self._json(dict(ok=True))
        elif self.path == "/api/prompt":                    # Manual model use: the two messages a side would send now, to edit before a run
            side = req.get("side")
            if side not in BACKENDS or not req.get("scene"):
                return self._json({"error": "unknown side or no scene"}, 400)
            name, system, scene_info = PROMPTS[side_format(side, req.get("format", "auto"))]
            self._json(dict(format=name, system=system, user=user_prompt(req["scene"], req.get("instruction") or "", scene_info)))
        elif self.path == "/api/export":
            run, side = RUNS.get(req.get("run")), req.get("side")
            video = run and run.videos.get(side)
            if not video or not video[0].exists() or side not in run.meta:
                return self._json({"error": "no video for this run (it may be one of the older runs, only the last 20 are kept)"}, 404)
            if not video[1].endswith(".mp4"):
                return self._json({"error": "exporting needs ffmpeg (pip install imageio-ffmpeg)"}, 409)
            job = uuid.uuid4().hex[:8]
            EXPORTS[job] = dict(state="waiting", progress=0.0, error=None, path=None, name=video[1])
            threading.Thread(target=export, args=(job, run, side, req), daemon=True).start()
            self._json(dict(job=job))
        elif self.path == "/api/run":
            with STATE:
                sides = [s for s in BACKENDS if s in req.get("sides", BACKENDS)]        # one side or both
                if not sides or any(STATUS[s]["state"] != "ready" for s in sides):
                    return self._json({"error": "models are not ready yet"}, 409)
                rid = uuid.uuid4().hex[:8]
                RUNS[rid] = run = Run(rid)
                GENERATING.update((run, s) for s in sides)
            for side in sides:
                threading.Thread(target=work, args=(run, side, req), daemon=True).start()
            self._json(dict(run=rid))
        else:
            self._json({"error": "not found"}, 404)


def autoload(mock):
    if mock:
        for side, cfg, fmt, cubes in (("base", {"backend": "mock", "label": "mock base (broken placement, forgets def)", "broken": True, "drop_def": True, "empty_think": True}, "objects", TRAINED_CUBES),
                                      ("finetuned", {"backend": "mock", "label": "mock fine-tuned (thinks, answers questions)", "answers": True, "enable_thinking": True}, "posv3_think", 27)):
            BACKENDS[side], MODEL_KEY[side], MODEL_FORMAT[side] = llm_backends.make(cfg), model_key(cfg), fmt
            STATUS[side] = dict(state="ready", label=cfg["label"], error=None, max_cubes=cubes)
        return
    ms = models_scan.list_models()
    local = sorted((m for m in ms if not m["kind"].startswith("hf")), key=lambda m: -m["mtime"])
    base = next(m for m in ms if m["path"] == models_scan.DEFAULT_BASE)
    ft = local[0] if local else dict(base, label="no fine-tuned model found: pick one above (base model again)")
    print("\nmodels found on this Mac:", flush=True)
    for m in local:
        print(f"   {m['label']}", flush=True)
    if not local:
        print("   (none yet - the right side runs the base model; choose or enter a path in the web page)", flush=True)
    for side, info in (("base", base), ("finetuned", ft)):
        threading.Thread(target=load_side, args=(side, info, None, LOAD_SEQ[side]), daemon=True).start()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("P2R_PORT", 8766)))
    ap.add_argument("--mock", action="store_true", help="fake models, to try the app without any model")
    ARGS = ap.parse_args()
    threading.Thread(target=models_scan.refresh_account, args=(HF_TOKEN["value"],), daemon=True).start()
    threading.Thread(target=lara_assets.ensure, daemon=True).start()
    autoload(ARGS.mock)
    print(f"ready: open http://localhost:{ARGS.port}  (models load in the background; progress is shown on the page)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler).serve_forever()
