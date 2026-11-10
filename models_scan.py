"""Finds the models available on this Mac and turns a chosen entry into a backend config.

Scans (depth-limited): <app>/models, <app>/../prompt2robo_run, ~/Desktop/prompt2robo_run
and every folder in the env var P2R_MODEL_DIRS (colon separated). Recognized folders:
  * MLX LoRA adapter   adapter_config.json + adapters.safetensors     (output of the Mac training script)
  * PEFT LoRA adapter  adapter_config.json + adapter_model.safetensors (output of the H100 training script, needs the transformers backend)
  * full model         config.json + *.safetensors                     (merged / converted MLX or HF model)
An optional p2r_meta.json next to the files may set {"name": "...", "prompt_format": "<one of PROMPT_FORMATS>", "enable_thinking": true|false, "scene_info": true|false,
"answers": true|false, "max_cubes": <largest training scene>, "base": "<MLX base of an adapter>"}.
"prompt_format" names the format the model was trained with (see PROMPT_FORMATS); without it the format is inferred from the older keys:
"answers": true -> answers_posv1, "scene_info": true -> scene_v5, neither -> objects. "enable_thinking" given explicitly always wins; otherwise
prompt_format decides it (posv3_think: on), and without either key the old defaults stay. max_cubes defaults to 6.

With a Hugging Face token the Prompt2Robo repos of the account HF_USER are listed as well (refresh_account); nothing about them is hard-coded,
so new uploads appear by themselves. Repo kinds by name: *-MLX-4bit and *-full go straight to MLX (kind "hf"); *-MLX-adapter is kind "hf_adapter" (small, runs
through MLX on its base); everything else, above all the PEFT *-adapter, is kind "hf_repo". Both must be downloaded with resolve() first, describe() then picks the
backend from the files. list_models() splits everything into GROUPS and says per entry whether it is on this Mac (is_cached: the Hugging Face cache, no network)."""
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = "mlx-community/Qwen3-8B-4bit"
HF_USER = "moritzzzz"
ACCOUNT = dict(models=[], note="")                      # the account's Prompt2Robo repos, filled by refresh_account (empty without a token)
ACCOUNT_LOCK = threading.Lock()
HF_PRESETS = [                                           # repo, name, download size in GB
    ("mlx-community/Qwen3-4B-4bit", "Qwen3-4B 4-bit", 2.3),
    ("mlx-community/Qwen3-8B-4bit", "Qwen3-8B 4-bit", 4.6),
    ("mlx-community/Qwen3-8B-8bit", "Qwen3-8B 8-bit (needs ~10 GB RAM)", 8.7),
    ("mlx-community/Qwen3-8B-bf16", "Qwen3-8B bf16 (needs ~18 GB RAM)", 16.4),
]
PEFT_BASES = {"Qwen/Qwen3-8B": ("Qwen3-8B bf16 (transformers)", 16.4), "Qwen/Qwen3-4B": ("Qwen3-4B bf16 (transformers)", 8.0)}
GROUPS = ("Base models", "Full models (Prompt2Robo fine-tunes)", "Adapters (small, run on a base model)")
SKIP = {".git", ".venv", "venv", "node_modules", "__pycache__", ".cache", "Library", "site-packages"}
# The prompt formats, name -> enable_thinking it implies. The system prompts and user messages are in server.PROMPTS.
#   objects        data_v0..v4: "Objects: 1 (red), ..." line before the request, system_prompt.md, no thinking
#   scene_v5       data_v5, data_posv0: request + "\n\nScene_Info:\n" + JSON of the scene, the Scene_Info system prompt, no thinking
#   answers_posv1  data_posv1, data_posv2: the Scene_Info message, system_prompt_answers.md (also answers scene questions), no thinking
#   posv3_think    data_posv3 with reasoning: the Scene_Info message, system_prompt_posv3.md, thinking on (<think>...</think> before the reply)
#   posv3_nothink  data_posv3 trained without reasoning: system_prompt_posv3.md minus its "Before the code, think briefly" line, no thinking
PROMPT_FORMATS = {"objects": False, "scene_v5": False, "answers_posv1": False, "posv3_think": True, "posv3_nothink": False}
FORMAT_ALIASES = {"scene": "scene_v5", "answers": "answers_posv1"}     # the names of before, still accepted in p2r_meta.json and from the page


def _roots():
    home = Path.home()
    roots = [HERE / "models", HERE.parent / "prompt2robo_run", home / "Desktop" / "prompt2robo_run"]
    roots += [Path(p).expanduser() for p in os.environ.get("P2R_MODEL_DIRS", "").split(":") if p]
    seen, out = set(), []
    for r in roots:
        if r.exists() and r.resolve() not in seen:
            seen.add(r.resolve()); out.append(r)
    return out


def adapter_base(meta, cfg):
    """The base of an MLX adapter: p2r_meta.json, then adapter_config.json, then DEFAULT_BASE."""
    if meta.get("base"):
        return meta["base"]
    base = cfg.get("model") or DEFAULT_BASE
    if not str(base).startswith("mlx-community/") and not Path(str(base)).expanduser().exists():
        base = DEFAULT_BASE                                  # base path from another machine -> fall back
    return base


def describe(folder):
    """Classify one folder -> dict or None."""
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        return None
    meta = {}
    if (folder / "p2r_meta.json").exists():
        try:
            meta = json.loads((folder / "p2r_meta.json").read_text())
        except Exception:
            pass
    cfgp = folder / "adapter_config.json"
    if cfgp.exists():
        try:
            cfg = json.loads(cfgp.read_text())
        except Exception:
            cfg = {}
        if (folder / "adapters.safetensors").exists():
            return dict(kind="adapter", path=str(folder), base=adapter_base(meta, cfg), thinking=thinking_of(meta, True),
                        name=meta.get("name") or "LoRA adapter (MLX)")
        if (folder / "adapter_model.safetensors").exists():
            base = Path(str(cfg.get("base_model_name_or_path") or "")).name    # often a path on the training machine: only the name counts
            return dict(kind="peft", path=str(folder), base=f"Qwen/{base}" if base.startswith("Qwen") else "Qwen/Qwen3-8B", thinking=thinking_of(meta, False),
                        name=meta.get("name") or "LoRA adapter (PEFT, slower)")
    if (folder / "config.json").exists() and list(folder.glob("*.safetensors")):
        return dict(kind="model", path=str(folder), base=None, thinking=thinking_of(meta, False),
                    name=meta.get("name") or "Full model")
    return None


def scan(max_depth=5, max_dirs=30000):
    found, seen, visited = [], set(), 0
    for root in _roots():
        base_depth = len(root.resolve().parts)
        for dp, dns, _ in os.walk(root):
            visited += 1
            if visited > max_dirs:
                break
            d = Path(dp)
            dns[:] = [x for x in dns if x not in SKIP and not x.startswith(".")]
            if len(d.resolve().parts) - base_depth >= max_depth:
                dns[:] = []
            info = describe(d)
            if info and info["path"] not in seen:
                seen.add(info["path"])
                try:
                    info["mtime"] = max(p.stat().st_mtime for p in d.iterdir() if p.is_file())
                except Exception:
                    info["mtime"] = 0
                try:
                    info["where"] = "~/" + str(d.relative_to(Path.home()))
                except ValueError:
                    info["where"] = str(d)
                info["label"] = f"{info['name']}  ·  {info['where']}"
                info["gb"] = sum(p.stat().st_size for p in d.glob("*.safetensors")) / 1e9
                found.append(info)
    found.sort(key=lambda m: -m["mtime"])
    return found


def hf_entries():
    return [dict(kind="hf", path=p, base=None, thinking=False, name=n, label=n, where="Hugging Face", gb=gb, group=GROUPS[0]) for p, n, gb in HF_PRESETS]


def repo_kind(repo):
    """Which path a repo takes: 'hf' = the id goes straight to MLX, 'hf_adapter' = MLX adapter, 'hf_repo' = anything else. The last two go through resolve()."""
    n = repo.lower()
    if n.endswith("-mlx-adapter"):
        return "hf_adapter"
    return "hf" if re.search(r"-mlx(-\d+bit)?$", n) or n.endswith("-full") else "hf_repo"


def _repo_parts(repo):
    """'moritzzzz/Prompt2Robo-Qwen3-8B-v3-qlora-adapter' -> ('Prompt2Robo v3 QLoRA, Qwen3-8B', 'adapter', 'Qwen3-8B')"""
    stem = re.sub(r"^Prompt2Robo-?", "", repo.split("/")[-1], flags=re.I)
    m = re.search(r"-(mlx-adapter|mlx(?:-\d+bit)?|full|adapter)$", stem, re.I)
    tail = m.group(1).lower() if m else ""
    stem = stem[:m.start()] if m else stem
    q = re.match(r"(Qwen[\d.]*-\d+B)-?(.*)", stem, re.I)
    model, rest = (q.group(1), q.group(2)) if q else ("", stem)
    rest = re.sub(r"(?i)\bqlora\b", "QLoRA", re.sub(r"(?i)\blora\b", "LoRA", rest))
    return " ".join(x for x in ("Prompt2Robo", rest) if x) + (f", {model}" if model else ""), tail, model


def peft_base(repo):
    """The bf16 base of a PEFT *-adapter repo, from its name."""
    return f"Qwen/{_repo_parts(repo)[2] or 'Qwen3-8B'}"


def repo_label(repo):
    """'moritzzzz/Prompt2Robo-Qwen3-8B-v5-qlora-MLX-adapter' -> 'Prompt2Robo v5 QLoRA, Qwen3-8B · MLX adapter'. No size or location: the label names the model
    in the side's heading and stays the same after the download."""
    head, tail, _ = _repo_parts(repo)
    base = peft_base(repo)
    gb = PEFT_BASES.get(base, ("", 16))[1]
    kind = {"full": "full bf16 weights, MLX", "mlx-adapter": "MLX adapter", "": "type is detected after the download",
            "adapter": f"PEFT adapter, slow (needs torch, peft and the {gb:.0f} GB {base})"}.get(tail, tail.replace("mlx", "MLX").replace("-", " ").replace("bit", "-bit"))
    return f"{head} · {kind}"


def repo_adapter_base(repo, token):
    """The base named by an MLX adapter repo, without downloading its weights. Small files only; anything unreadable ends at DEFAULT_BASE."""
    def read(name):
        try:
            from huggingface_hub import hf_hub_download
            return json.loads(Path(hf_hub_download(repo, name, token=token or None)).read_text())
        except Exception:
            return {}
    meta = read("p2r_meta.json")
    return adapter_base(meta, {} if meta.get("base") else read("adapter_config.json"))


def repo_entry(repo, size, base=None):
    """One account repo -> list entry. adapter = 'mlx' or 'peft' for the two kinds of adapter, base = what it runs on."""
    kind, tail = repo_kind(repo), _repo_parts(repo)[1]
    adapter = "mlx" if kind == "hf_adapter" else "peft" if tail == "adapter" else None
    label = repo_label(repo)
    return dict(kind=kind, path=repo, base=base or DEFAULT_BASE if adapter == "mlx" else peft_base(repo) if adapter else None, adapter=adapter, thinking=False,
                name=label, label=label, where="Hugging Face", gb=size / 1e9 if size else None, group=GROUPS[2 if adapter else 1])


def complete(snapshot):
    """Does this folder hold all weights of a model or adapter? Shards need their index, or the download is still running."""
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        return all((snapshot / f).exists() for f in set(json.loads(index.read_text())["weight_map"].values()))
    weights = list(snapshot.glob("*.safetensors"))
    return bool(weights) and not any("-of-" in w.name for w in weights)


def is_cached(repo):
    """Are the weights of this repo completely on this Mac? Looks at a few files in the Hugging Face cache or a local folder: no network, never raises.
    A cache entry alone says nothing: the small json files of a repo get there without its weights (repo_adapter_base, read_meta), and a download can stop halfway.
    Every snapshot counts, not just the newest: a small upstream commit moves refs/main, and the weights already here stay valid."""
    try:
        if Path(str(repo)).expanduser().is_dir():
            return True
        from huggingface_hub.constants import HF_HUB_CACHE
        snapshots = Path(HF_HUB_CACHE) / ("models--" + str(repo).replace("/", "--")) / "snapshots"
        return snapshots.is_dir() and any(complete(s) for s in snapshots.iterdir())
    except Exception:
        return False


def base_info(base):
    """What the page shows about a base model."""
    names = {p: (n, gb) for p, n, gb in HF_PRESETS} | PEFT_BASES
    name, gb = names.get(base, (str(base).rstrip("/").split("/")[-1], None))
    on_mac = is_cached(base)
    return dict(path=base, name=name, on_mac=on_mac,
                avail="✓ on this Mac" if on_mac else f"↓ {gb:.1f} GB, downloads on first use" if gb else "↓ downloads on first use",
                short="✓ on this Mac" if on_mac else f"↓ {gb:.1f} GB" if gb else "↓ download")


def mlx_bases():
    return [base_info(p) for p, _, _ in HF_PRESETS]


def _friendly(err, token):
    """An exception from the Hugging Face API -> a sentence for the page. Never contains the token."""
    msg = f"{type(err).__name__}: {err}".replace("\n", " ").replace(token or "\0", "***")
    if any(k in msg for k in ("401", "Invalid user token", "Unauthorized")):
        return "Hugging Face rejected this token. Check it and press Save again."
    if any(k in msg for k in ("403", "Forbidden")):
        return f"This token may not read the models of {HF_USER}. It needs read access."
    if any(k in msg for k in ("Connection", "Timeout", "timed out", "resolve", "offline", "Offline")):
        return "Could not reach Hugging Face (offline?). Press ↻ to try again."
    return f"Could not list the Hugging Face models ({msg[:160]}). Press ↻ to try again."


def refresh_account(token):
    """Ask Hugging Face for the Prompt2Robo repos of HF_USER. Never raises; a problem ends up as a sentence in ACCOUNT['note'].
    Without a token only the public repos are listed. The token is only passed on, never stored here."""
    with ACCOUNT_LOCK:
        models, note = [], ""
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token or False)
            repos = [m.id for m in api.list_models(author=HF_USER) if "prompt2robo" in m.id.lower()]
            if not repos:
                if token:
                    api.whoami()                           # list_models quietly returns nothing for a bad token; this one raises
                    note = f"The token works, but it sees no Prompt2Robo repos of {HF_USER}. It needs read access to them."

            def details(repo):
                try:
                    size = api.model_info(repo, expand=["usedStorage"]).used_storage or 0
                except Exception:
                    size = 0                               # a label without a size is fine
                return size, repo_adapter_base(repo, token) if repo_kind(repo) == "hf_adapter" else None
            with ThreadPoolExecutor(8) as ex:
                found = dict(zip(repos, ex.map(details, repos)))
            version = lambda r: int((re.search(r"-v(\d+)", r) or [0, 1])[1])  # noqa: E731
            rank = lambda r: (repo_kind(r) != "hf", r.lower().endswith("-full"), repo_kind(r) != "hf_adapter", -version(r))  # noqa: E731
            models = [repo_entry(r, *found[r]) for r in sorted(sorted(repos), key=rank)]      # MLX first, MLX adapters before PEFT, newest first
        except Exception as e:
            models, note = [], _friendly(e, token)
        ACCOUNT.update(models=models, note=note)
        return dict(hf_models=len(models), note=note)


def account_models():
    with ACCOUNT_LOCK:                                         # waits for a refresh that is still running
        return list(ACCOUNT["models"])


def local_group(info):
    return GROUPS[2 if info["kind"] in ("adapter", "peft") else 1]


def show(m):
    """Adds to a list entry what can change between two looks at the list: is it on this Mac, and its base."""
    local = not m["kind"].startswith("hf")
    on_mac = local or is_cached(m["path"])
    gb = f"{m['gb']:.1f} GB" if m.get("gb") else ""
    avail = " · ".join(x for x in (gb, "✓ on this Mac") if x) if on_mac else f"↓ {gb}" if gb else "↓ download"
    return dict(m, on_mac=on_mac, avail=avail, base_info=base_info(m["base"]) if m.get("base") else None)


def list_models():
    """Base models, full models, adapters: in that order, each as it comes (repos by rank, folders newest first)."""
    local = [dict(m, group=local_group(m), adapter={"adapter": "mlx", "peft": "peft"}.get(m["kind"])) for m in scan()]
    entries = hf_entries() + account_models() + local
    return [show(m) for m in sorted(entries, key=lambda m: GROUPS.index(m["group"]))]


def pairing_hint(info, err):
    """An adapter that does not fit the chosen base fails while loading with a long list of tensors -> one sentence for the panel, or None for any other error."""
    text = str(err)
    if info.get("kind") != "adapter" or not any(k in text for k in ("incompatible tensors", "does not match", "Expected shape", "not an adapter parameter")):
        return None
    own, base = info.get("own_base") or info["base"], info["base"]
    if own != base:
        return f"This adapter does not fit {base_info(base)['name']}: it was made for {base_info(own)['name']}, the layer sizes differ. Choose that base and it loads."
    return f"This adapter does not fit its base {base_info(base)['name']}: the layer sizes differ. The adapter files may name the wrong base."


def read_meta(info, token=None):
    """The model's p2r_meta.json. Missing or unreadable: {}. For an adapter info["path"] is the adapter folder, so its own file counts, not the base's."""
    try:
        if info["kind"] == "hf":
            from huggingface_hub import hf_hub_download
            meta = Path(hf_hub_download(info["path"], "p2r_meta.json", token=token or None))
        else:
            meta = Path(info["path"]) / "p2r_meta.json"
        data = json.loads(meta.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def format_name(name):
    """A format name from p2r_meta.json or the page -> a key of PROMPT_FORMATS, or None. The old names "scene" and "answers" still work."""
    name = FORMAT_ALIASES.get(name, name)
    return name if isinstance(name, str) and name in PROMPT_FORMATS else None


def prompt_format(meta):
    """What the model was trained with, a key of PROMPT_FORMATS: "prompt_format" if it names one, else inferred from the older keys as before."""
    fmt = format_name(meta.get("prompt_format"))
    if meta.get("prompt_format") is not None and fmt is None:
        print(f"p2r_meta.json: unknown prompt_format {meta['prompt_format']!r} (known: {', '.join(PROMPT_FORMATS)}), inferring it from the other keys", flush=True)
    return fmt or ("answers_posv1" if meta.get("answers") else "scene_v5" if meta.get("scene_info") else "objects")


def thinking_of(meta, default):
    """enable_thinking for the chat template: the explicit key, else what "prompt_format" implies, else default (the behaviour of before)."""
    if "enable_thinking" in meta:
        return meta["enable_thinking"]
    fmt = format_name(meta.get("prompt_format"))
    return PROMPT_FORMATS[fmt] if fmt else default


def max_cubes(meta, default):
    """The largest scene the model has seen, for the warning next to the cube field. Anything but a whole number from 1 up: default."""
    n = meta.get("max_cubes")
    return n if type(n) is int and n >= 1 else default


def backend_cfg(info, thinking=None):
    """info from describe()/list_models() -> kwargs for llm_backends.make()."""
    th = info["thinking"] if thinking is None else thinking
    kind = info["kind"]
    if kind == "hf":
        return dict(backend="mlx", path=info["path"], label=info["label"], enable_thinking=th)
    if kind == "model":
        return dict(backend="mlx", path=info["path"], label=info["label"], enable_thinking=th)
    if kind == "adapter":
        return dict(backend="mlx", path=info["base"], adapter=info["path"], label=info["label"], enable_thinking=th)
    if kind == "peft":
        return dict(backend="transformers", path=info["base"], adapter=info["path"], label=info["label"], enable_thinking=th)
    raise ValueError(f"unknown model kind {kind}")


def parse_hf(text):
    """'https://huggingface.co/org/repo[/tree/main/sub]' or 'org/repo[/sub...]' -> (repo_id, revision, subfolder) or None."""
    t = str(text).strip().strip("\"'").rstrip("/")
    for pre in ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/", "hf.co/"):
        if t.startswith(pre):
            t = t[len(pre):]
            break
    else:
        if t.startswith(("/", "~", ".")) or Path(t).expanduser().exists():
            return None
    parts = t.split("/")
    if len(parts) < 2 or not all(parts[:2]):
        return None
    repo, rev, sub = "/".join(parts[:2]), "main", ""
    rest = parts[2:]
    if rest[:1] in (["tree"], ["blob"]) and len(rest) >= 2:
        rev, sub = rest[1], "/".join(rest[2:])
    elif rest:
        sub = "/".join(rest)
    return repo, rev, sub


def resolve(path_or_id, token=None, log=print):
    """A user-typed folder or Hugging Face link/id -> info dict. Downloads from Hugging Face if needed (raises ValueError with a friendly message)."""
    s = str(path_or_id).strip().strip("\"'")
    hf = parse_hf(s)
    if hf:
        repo, rev, sub = hf
        from huggingface_hub import snapshot_download
        try:
            log(f"downloading {repo}{'/' + sub if sub else ''} from Hugging Face ...")
            local = Path(snapshot_download(repo, revision=rev, token=token or None, allow_patterns=[f"{sub}/*"] if sub else None))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}".replace("\n", " ").replace(token or "\0", "***")[:300]
            hint = " (private repo? paste a Hugging Face token in Settings)" if any(k in msg for k in ("401", "403", "Repository Not Found", "gated")) else ""
            raise ValueError(f"could not download '{repo}': {msg}{hint}")
        folder = local / sub if sub else local
        info = describe(folder) or next((describe(p.parent) for p in sorted(folder.rglob("adapter_config.json")) + sorted(folder.rglob("config.json")) if describe(p.parent)), None)
        if not info:
            raise ValueError(f"'{repo}' was downloaded but contains no model or LoRA adapter "
                             "(expected config.json + *.safetensors, or adapters.safetensors + adapter_config.json).")
        info["where"] = f"Hugging Face: {repo}" + (f"/{sub}" if sub else "")
        info["label"] = f"{info['name']}  ·  {info['where']}"
        return info
    p = Path(s).expanduser()
    if p.exists():
        info = describe(p)
        if info:
            info["where"], info["label"] = str(p), f"{info['name']}  ·  {p}"
            return info
        for sub in sorted(p.rglob("adapter_config.json"))[:1] + sorted(p.rglob("config.json"))[:1]:
            info = describe(sub.parent)
            if info:
                info["where"], info["label"] = str(sub.parent), f"{info['name']}  ·  {sub.parent}"
                return info
        raise ValueError(f"'{s}' is a folder, but I found no model or LoRA adapter in it "
                         "(expected adapters.safetensors + adapter_config.json, or config.json + *.safetensors).")
    raise ValueError(f"'{s}' is neither an existing folder nor a Hugging Face link (https://huggingface.co/<user>/<model>).")
