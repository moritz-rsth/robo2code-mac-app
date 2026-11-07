"""Verify Prompt2Robo samples in the MuJoCo simulator.

    check(sample) -> (ok, reasons)

CLI:
    python tools/check.py data/*.jsonl [--workers 32] [--write]   # --write sets `verified`
"""
from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import itertools
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

POS_TOL = 0.01    # 1 cm, the single global position tolerance
ANG_TOL = 1.0     # 1 degree
CUBE = 0.04
WS = np.array([[0.0, 0.8], [-0.4, 0.4], [0.0, 0.8]])

CATEGORIES = {"basic_single_explicit", "single_explicit", "abstract", "edge_cases"}
CONSTRAINT_TYPES = {"at", "on_top_of", "touching", "in_line", "left_of", "right_of", "in_front_of", "behind",
                    "within_distance", "same_height", "tower", "unchanged",
                    # extensions (documented in SCHEMA.md)
                    "gripper_at", "gripper_angle", "gripper_open", "held", "on_table", "stacked", "pyramid", "one_of"}


def default_workers():
    """Respect P2R_WORKERS, else the cgroup CPU quota (nproc can overstate available cores)."""
    if os.environ.get("P2R_WORKERS"):
        return max(1, int(os.environ["P2R_WORKERS"]))
    try:
        q, per = (int(open(f"/sys/fs/cgroup/cpu/cpu.cfs_{k}_us").read()) for k in ("quota", "period"))
        if q > 0:
            return max(1, q // per)
    except Exception:
        pass
    try:
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            return max(1, int(q) // int(per))
    except Exception:
        pass
    return os.cpu_count() or 1


# ---------------------------------------------------------------- helpers
def _p(state, oid):
    if oid == "gripper":
        return np.array(state["gripper"]["position"], float)
    return np.array(state["objects"][oid]["position"], float)


def _xy(a, b):
    return float(np.linalg.norm(a[:2] - b[:2]))


def _objs_in(c):
    out = set()
    if c.get("type") == "one_of":
        for opt in c.get("options", []):
            for cc in opt:
                out |= _objs_in(cc)
        return out
    for k in ("obj", "target"):
        if k in c and c[k] != "gripper":
            out.add(c[k])
    for k in ("targets", "objs"):
        for o in c.get(k, []):
            if o != "gripper":
                out.add(o)
    return out


def _on_top(state, o, t):
    po, pt = _p(state, o), _p(state, t)
    return abs(po[2] - pt[2] - CUBE) <= POS_TOL and _xy(po, pt) <= POS_TOL


def _touching_pair(state, a, b):
    pa, pb = _p(state, a), _p(state, b)
    return abs(pa[2] - pb[2]) <= POS_TOL and abs(_xy(pa, pb) - CUBE) <= POS_TOL


def _pyramid(state, objs):
    """Layers of n, n-1, ..., 1 cubes; bottom layer on the table, a touching row;
    every upper cube rests on two touching cubes of the layer below."""
    zs = {o: _p(state, o)[2] for o in objs}
    layers = {}
    for o, z in zs.items():
        k = int(round((z - CUBE / 2) / CUBE))
        layers.setdefault(k, []).append(o)
    n = len(layers)
    if n < 2 or sorted(layers) != list(range(n)):
        return False
    for k in range(n):
        if len(layers[k]) != n - k:
            return False
    base = layers[0]
    if len(base) > 1 and not eval_constraint({"type": "touching", "objs": base}, state, None, set())[0]:
        return False
    if len(base) > 2:
        ps = np.array([_p(state, o) for o in base])
        if min(np.ptp(ps[:, 0]), np.ptp(ps[:, 1])) > POS_TOL:
            return False
    for k in range(1, n):
        for o in layers[k]:
            if not any(_touching_pair(state, a, b) and
                       eval_constraint({"type": "on_top_of", "obj": o, "targets": [a, b]}, state, None, set())[0]
                       for a, b in itertools.combinations(layers[k - 1], 2)):
                return False
    return True


def eval_constraint(c, state, init, held):
    t = c["type"]
    if t not in CONSTRAINT_TYPES:
        return False, f"unknown constraint type {t}"
    try:
        if t == "at":
            d = float(np.linalg.norm(_p(state, c["obj"]) - np.array(c["position"], float)))
            return d <= POS_TOL, f"{c['obj']} is {d:.3f} m from {c['position']}"
        if t == "gripper_at":
            d = float(np.linalg.norm(_p(state, "gripper") - np.array(c["position"], float)))
            return d <= POS_TOL, f"gripper is {d:.3f} m from {c['position']}"
        if t == "gripper_angle":
            a = state["gripper"]["angle_deg"]
            return abs(a - c["angle_deg"]) <= ANG_TOL, f"gripper angle {a} != {c['angle_deg']}"
        if t == "gripper_open":
            return state["gripper"]["is_open"] == c["is_open"], f"gripper is_open != {c['is_open']}"
        if t == "held":
            return c["obj"] in held, f"{c['obj']} is not held by the gripper"
        if t == "on_table":
            z = _p(state, c["obj"])[2]
            return abs(z - CUBE / 2) <= POS_TOL, f"{c['obj']} not on the table (z={z:.3f})"
        if t == "on_top_of":
            o = c["obj"]
            if "targets" in c:
                ts = c["targets"]
                po = _p(state, o)
                mid = np.mean([_p(state, x) for x in ts], axis=0)
                top = max(_p(state, x)[2] for x in ts)
                ok = abs(po[2] - top - CUBE) <= POS_TOL and _xy(po, mid) <= POS_TOL
                return ok, f"{o} does not rest on {ts}"
            return _on_top(state, o, c["target"]), f"{o} is not on top of {c['target']}"
        if t == "touching":
            objs = c["objs"]
            # the "touching" graph over objs must be connected
            seen, todo = {objs[0]}, [objs[0]]
            while todo:
                a = todo.pop()
                for b in objs:
                    if b not in seen and _touching_pair(state, a, b):
                        seen.add(b)
                        todo.append(b)
            return len(seen) == len(objs), f"{objs} are not touching side by side"
        if t == "in_line":
            objs, ax = c["objs"], c["axis"]
            k = {"x": 1, "y": 0}[ax]  # a line along x has constant y
            ps = np.array([_p(state, o) for o in objs])
            ok = np.ptp(ps[:, k]) <= POS_TOL and np.ptp(ps[:, 2]) <= POS_TOL
            return bool(ok), f"{objs} are not in a line along {ax}"
        if t in ("left_of", "right_of", "in_front_of", "behind"):
            po, pt = _p(state, c["obj"]), _p(state, c["target"])
            d = {"left_of": po[1] - pt[1], "right_of": pt[1] - po[1],
                 "in_front_of": po[0] - pt[0], "behind": pt[0] - po[0]}[t]
            return d > POS_TOL, f"{c['obj']} is not {t} {c['target']}"
        if t == "within_distance":
            d = float(np.linalg.norm(_p(state, c["obj"]) - _p(state, c["target"])))
            return c["min"] <= d <= c["max"], f"distance {c['obj']}-{c['target']} = {d:.3f} not in [{c['min']}, {c['max']}]"
        if t == "same_height":
            zs = [_p(state, o)[2] for o in c["objs"]]
            return max(zs) - min(zs) <= POS_TOL, f"{c['objs']} not at the same height"
        if t == "tower":
            objs = c["objs"]
            ok = all(_on_top(state, objs[i + 1], objs[i]) for i in range(len(objs) - 1))
            return ok, f"tower {objs} not stacked in this order"
        if t == "stacked":
            objs = sorted(c["objs"], key=lambda o: _p(state, o)[2])
            ok = all(_on_top(state, objs[i + 1], objs[i]) for i in range(len(objs) - 1))
            return ok, f"{c['objs']} do not form a single tower"
        if t == "pyramid":
            return _pyramid(state, c["objs"]), f"{c['objs']} do not form a pyramid"
        if t == "one_of":
            for opt in c["options"]:
                if all(eval_constraint(cc, state, init, held)[0] for cc in opt):
                    return True, ""
            return False, "none of the one_of options holds"
        if t == "unchanged":
            bad = [o for o in c["objs"] if np.linalg.norm(_p(state, o) - _p(init, o)) > POS_TOL]
            return not bad, f"{bad} moved"
    except KeyError as e:
        return False, f"constraint {c} refers to unknown key {e}"
    return False, "unreachable"


def compare_exact(state, ref):
    reasons = []
    for oid, o in ref["objects"].items():
        d = float(np.linalg.norm(_p(state, oid) - np.array(o["position"], float)))
        if d > POS_TOL:
            reasons.append(f"object {oid} is {d:.3f} m away from the reference")
    g, rg = state["gripper"], ref["gripper"]
    d = float(np.linalg.norm(np.array(g["position"]) - np.array(rg["position"])))
    if d > POS_TOL:
        reasons.append(f"gripper is {d:.3f} m away from the reference")
    if abs(g["angle_deg"] - rg["angle_deg"]) > ANG_TOL:
        reasons.append(f"gripper angle {g['angle_deg']} != reference {rg['angle_deg']}")
    if g["is_open"] != rg["is_open"]:
        reasons.append("gripper open/closed differs from the reference")
    return reasons


def physical_checks(state, sim, held):
    """Workspace, floating, overlapping."""
    reasons = []
    for oid, o in state["objects"].items():
        p = np.array(o["position"])
        if np.any(p < WS[:, 0] - POS_TOL) or np.any(p > WS[:, 1] + POS_TOL):
            reasons.append(f"object {oid} ends outside the workspace")
    gp = np.array(state["gripper"]["position"])
    if np.any(gp < WS[:, 0] - POS_TOL) or np.any(gp > WS[:, 1] + POS_TOL):
        reasons.append("gripper ends outside the workspace")
    objs = state["objects"]
    for a, b in itertools.combinations(objs, 2):
        pa, pb = np.array(objs[a]["position"]), np.array(objs[b]["position"])
        if np.all(np.abs(pa - pb) < CUBE - 0.004):
            reasons.append(f"objects {a} and {b} overlap")
    # floating: resting on nothing (and not held) / tilted / not on a layer
    for oid in objs:
        if oid in held:
            continue
        p = np.array(objs[oid]["position"])
        layer = (p[2] - CUBE / 2) / CUBE
        if abs(layer - round(layer)) * CUBE > POS_TOL:
            reasons.append(f"object {oid} is not resting flat (z={p[2]:.3f})")
        if sim is not None:
            r = sim.data.xmat[sim.body[oid]].reshape(3, 3)
            if np.max(np.abs(r[2])) < 0.98:
                reasons.append(f"object {oid} is tilted")
    return reasons


# ---------------------------------------------------------------- refusals
_SUGGEST = re.compile(r"\?|\b(please|could you|can you|would you|you can|you could|let me know|tell me|specify|"
                      r"try|instead|suggest|send|ask|first|if you|should)\b", re.I)


def check_refusal(sample):
    reasons = []
    txt = sample["assistant_answer"].strip()
    if "def run" in txt or "robot." in txt or "```" in txt:
        reasons.append("refusal contains code")
    sents = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", txt) if s.strip()]
    if not 1 <= len(sents) <= 3:
        reasons.append(f"refusal has {len(sents)} sentences (1 to 3 required)")
    low = txt.lower()
    for oid, o in sample["scene"]["objects"].items():
        if not re.search(rf"(?<![\d.]){re.escape(oid)}(?![\d.])", txt):
            reasons.append(f"refusal does not mention object id {oid}")
        if o["color"].lower() not in low:
            reasons.append(f"refusal does not mention color {o['color']}")
    if not _SUGGEST.search(txt):
        reasons.append("refusal does not say what has to change (no question/suggestion)")
    ref = sample.get("destination", {}).get("reference")
    if ref is None or ref.get("objects") != sample["scene"]["objects"]:
        reasons.append("refusal reference must equal the initial scene")
    return reasons


# ---------------------------------------------------------------- format
def check_format(sample):
    reasons = []
    for k in ("id", "category", "subcategory", "num_cubes", "scene", "user_prompt", "expected_behavior",
              "destination", "assistant_answer", "verified"):
        if k not in sample:
            reasons.append(f"missing field {k}")
    if reasons:
        return reasons
    if sample["category"] not in CATEGORIES:
        reasons.append(f"unknown category {sample['category']}")
    objs = sample["scene"]["objects"]
    if sample["num_cubes"] != len(objs):
        reasons.append("num_cubes does not match the scene")
    head, _, req = sample["user_prompt"].partition("\n")
    exp = "Objects: " + ", ".join(f"{i} ({o['color']})" for i, o in objs.items())
    if head != exp:
        reasons.append(f"user_prompt header must be {exp!r}")
    if not req.strip():
        reasons.append("empty request")
    if sample["expected_behavior"] not in ("execute", "refuse"):
        reasons.append("expected_behavior must be execute or refuse")
    dest = sample["destination"]
    if dest.get("mode") not in ("exact", "constraints"):
        reasons.append("destination.mode must be exact or constraints")
    if "reference" not in dest:
        reasons.append("destination.reference missing")
    for c in dest.get("constraints", []):
        if c.get("type") not in CONSTRAINT_TYPES:
            reasons.append(f"unknown constraint {c.get('type')}")
        for o in _objs_in(c):
            if o not in objs:
                reasons.append(f"constraint refers to unknown object {o}")
    for p in [o["position"] for o in objs.values()]:
        if len(p) != 3:
            reasons.append("bad position")
    return reasons


# ---------------------------------------------------------------- main check
def check(sample):
    """Returns (ok, reasons)."""
    reasons = check_format(sample)
    if reasons:
        return False, reasons
    if sample["expected_behavior"] == "refuse":
        r = check_refusal(sample)
        return not r, r
    from robot import execute
    code = sample["assistant_answer"]
    if "def run" not in code:
        return False, ["execute sample without `def run`"]
    try:
        state, err, sim = execute(sample["scene"], code)
    except Exception as e:  # scene could not be built
        return False, [f"simulation setup failed: {e}"]
    if err:
        return False, [f"execution error: {err}"]
    init = {"objects": {i: {"position": p.tolist()} for i, p in sim.start_positions.items()},
            "gripper": sample["scene"]["gripper"]}
    held = sim.held_objects()
    dest = sample["destination"]
    reasons += physical_checks(state, sim, held)
    cons = dest.get("constraints", [])
    for c in cons:
        ok, why = eval_constraint(c, state, init, held)
        if not ok:
            reasons.append(f"constraint failed: {why}")
    if dest["mode"] == "exact":
        reasons += compare_exact(state, dest["reference"])
    else:
        if not cons:
            reasons.append("mode constraints without constraints")
        mentioned = set().union(*[_objs_in(c) for c in cons]) if cons else set()
        for oid in state["objects"]:
            if oid not in mentioned and np.linalg.norm(_p(state, oid) - _p(init, oid)) > POS_TOL:
                reasons.append(f"object {oid} is not part of the request but moved")
    return not reasons, reasons


def _work(line):
    s = json.loads(line)
    t = time.time()
    ok, reasons = check(s)
    return s, ok, reasons, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--write", action="store_true", help="rewrite files with verified set from the check")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    import multiprocessing as mp
    total_bad = 0
    with mp.Pool(a.workers) as pool:
        for f in a.files:
            lines = [l for l in open(f) if l.strip()]
            res = pool.map(_work, lines, chunksize=1)
            bad = 0
            for s, ok, reasons, dt in res:
                if not ok:
                    bad += 1
                    print(f"FAIL {f} {s.get('id')}: {'; '.join(reasons)}")
                elif not a.quiet:
                    print(f"ok   {f} {s.get('id')} ({dt:.2f}s)")
                s["verified"] = bool(ok)
            total_bad += bad
            times = [r[3] for r in res]
            if times:
                print(f"== {f}: {len(res) - bad}/{len(res)} ok, mean {np.mean(times):.2f}s max {np.max(times):.2f}s")
            if a.write:
                with open(f, "w") as fh:
                    for s, *_ in res:
                        fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    sys.exit(1 if total_bad else 0)


if __name__ == "__main__":
    main()
