"""Verify Prompt2Robo data_posv3 samples: tools/check_posv1.py (unchanged) plus ONE stricter grasp rule (data_posv3):
- strict grasp: when close_gripper() finishes, (a) at most one cube may be held between the fingers (lifting the second
  cube of a tower can catch the cube above it between the long fingers, which the simulator counts as "held"), and
  (b) no other cube may rest on the grasped cube (centre 4 cm higher, footprints overlapping by more than 5 mm on
  both axes). (b) catches pulling a cube out from under a cube that would stay balanced on its other support (a
  pyramid base cube), which the simulator does not flag. Both raise an execution error at the close_gripper() call.
  Nothing else changed; every check_posv1 rule applies as before.
- one new constraint type `outside_area` {"objs", "center": [x, y], "half": [hx, hy]}: no part of any of the cubes
  lies inside the rectangle (used by "clear the middle of the table").

data_posv1 checker (copy of tools/check_v5.py), additions only:
- expected_behavior "answer" (a question about the scene): destination {"mode": "answer", "question": {...},
  "accept": [...], "reject": [...]}; the checker derives the true answer from the scene (check_answer), requires the
  stored accept / reject lists to equal the derived ones, and checks the reply (one short sentence, no code).
- shape constraints grid, solid_cube, shape_cells, ring, pyramid_of (every cube of the shape within POS_TOL of its
  ideal cell, the cells anchored on one cube of the shape; cubes of the pool that are not part of the shape must stay unchanged). See data_posv1/SCHEMA.md.
- optional destination.steps (chained requests): a list of constraint lists that must hold, in this order, in the
  states recorded after each open_gripper() call (a step of "build X, then do Y" is verified where the final state
  cannot show it). destination.constraints still check the final state; `reverse_of_step` (final constraint) requires
  the final tower order to be the reverse of the order in the state where that step first held.
No physics, collision, refusal or existing constraint rule changed.

    check(sample) -> (ok, reasons)

CLI:
    python tools/check_posv3.py data_posv3/*.jsonl [--workers 32] [--write]   # --write sets `verified`; never on an existing data file
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
                    "gripper_at", "gripper_angle", "gripper_open", "held", "on_table", "stacked", "pyramid", "one_of",
                    # data_posv1 shape constraints
                    "grid", "solid_cube", "shape_cells", "ring", "pyramid_of", "reverse_of_step",
                    # data_posv3
                    "outside_area"}


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


# ---------------------------------------------------------------- data_posv1 shapes
# Every shape constraint names a pool `objs`; `count` cubes of the pool (default: all) must form the shape, the other
# pool cubes must stay where they started. Each cube of the shape must lie within POS_TOL of its ideal cell.
_TRANSFORMS = {  # (dx, dy) -> world offset; rotations are counter-clockwise seen from above, mirror flips left/right (y)
    "id": lambda dx, dy: (dx, dy), "rot90": lambda dx, dy: (-dy, dx), "rot180": lambda dx, dy: (-dx, -dy),
    "rot270": lambda dx, dy: (dy, -dx), "mirror": lambda dx, dy: (dx, -dy), "mirror_rot90": lambda dx, dy: (dy, dx),
    "mirror_rot180": lambda dx, dy: (-dx, dy), "mirror_rot270": lambda dx, dy: (-dy, -dx)}
MAX_SUBSETS = 5000


def _moved(state, init, o):
    return np.linalg.norm(_p(state, o) - _p(init, o)) > POS_TOL


def _subsets(state, init, pool, k):
    """Candidate shape cubes: every moved pool cube plus enough unmoved ones (the rest of the pool stays unchanged)."""
    moved = [o for o in pool if _moved(state, init, o)]
    rest = [o for o in pool if o not in moved]
    if len(moved) > k:
        return
    for n, extra in enumerate(itertools.combinations(rest, k - len(moved))):
        if n >= MAX_SUBSETS:
            return
        yield moved + list(extra)


def _match_cells(state, init, pool, offsets, count=None):
    """The ideal cells (offsets dx, dy, z with z absolute) are anchored on one cube of the shape (no averaging), and
    then every cell must hold exactly one pool cube within POS_TOL; pool cubes that are not on a cell are unchanged."""
    offs = np.array(offsets, float)
    k = len(offs)
    if count is not None and count != k:
        return False
    P = {o: _p(state, o) for o in pool}
    zmin = offs[:, 2].min()
    for o in pool:
        for base in offs[np.abs(offs[:, 2] - zmin) < 1e-9]:
            target = offs.copy()
            target[:, :2] += P[o][:2] - base[:2]
            used, match = set(), []
            for c in target:
                best = min(((np.linalg.norm(P[q] - c), q) for q in pool if q not in used), default=(9, None))
                if best[0] > POS_TOL:
                    break
                used.add(best[1])
                match.append(best[1])
            if len(match) == k and not any(_moved(state, init, q) for q in pool if q not in used):
                return True
    return False


def _cell_offsets(cells, pitch, transform):
    """cells: [[row, col], ...] of a glyph read from above with +x as its top: row 0 is the top row (largest x),
    columns run to the right of a person at the robot looking forward (-y). Returns world (dx, dy, 0.02) offsets."""
    f = _TRANSFORMS[transform]
    return [(*f(-r * pitch, -c * pitch), CUBE / 2) for r, c in cells]


def _pitches(state, pool, lo, hi):
    if hi - lo <= 1e-9:
        return [lo]
    ps = set()
    for a, b in itertools.combinations(pool, 2):
        pa, pb = _p(state, a), _p(state, b)
        for d in (abs(pa[0] - pb[0]), abs(pa[1] - pb[1])):
            if lo - POS_TOL <= d <= hi + POS_TOL:
                ps.add(round(d, 3))
    return sorted(ps)


def _lattice(state, objs, p):
    """All objs on the table on one axis-aligned square lattice of pitch p, one cube per point -> (rows, cols)."""
    P = np.array([_p(state, o) for o in objs])
    if np.any(np.abs(P[:, 2] - CUBE / 2) > POS_TOL):
        return None
    K = np.round((P[:, :2] - P[:, :2].min(axis=0)) / p)
    # the lattice is anchored on one of the cubes (no averaging), as in _match_cells
    if not any(np.all(np.linalg.norm(P[:, :2] - (P[j, :2] - K[j] * p) - K * p, axis=1) <= POS_TOL) for j in range(len(P))):
        return None
    if len({tuple(k) for k in K.astype(int)}) != len(objs):
        return None
    return int(K[:, 0].max()) + 1, int(K[:, 1].max()) + 1


def _grid(c, state, init):
    pool = c["objs"]
    k = c.get("count", len(pool))
    lo, hi = c["pitch"] if isinstance(c["pitch"], list) else (c["pitch"], c["pitch"])
    rows, cols = c.get("rows"), c.get("cols")
    if rows is None:  # "an even grid": all k cubes on one lattice, at most one incomplete row or column
        for sub in _subsets(state, init, pool, k):
            for p in _pitches(state, sub, lo, hi):
                rc = _lattice(state, sub, p)
                if rc and min(rc) >= 2 and rc[0] * rc[1] - k < min(rc):
                    return True
        return False
    shapes = [(rows, cols)] + ([(cols, rows)] if c.get("orient", "any") == "any" and rows != cols else [])
    for p in _pitches(state, pool, lo, hi):
        for r, cc in shapes:
            offs = [(-a * p, -b * p, CUBE / 2) for a in range(r) for b in range(cc)]
            if k == len(offs) and _match_cells(state, init, pool, offs):
                return True
    return False


def _solid_cube(c, state, init):
    n = c["n"]
    offs = [(a * CUBE, b * CUBE, CUBE / 2 + CUBE * z) for z in range(n) for a in range(n) for b in range(n)]
    return _match_cells(state, init, c["objs"], offs)


def _shape_cells(c, state, init):
    for tr in c.get("transforms", ["id"]):
        if _match_cells(state, init, c["objs"], _cell_offsets(c["cells"], c.get("pitch", CUBE), tr)):
            return True
    return False


def _ring(c, state, init):
    pool = c["objs"]
    k = c.get("count", len(pool))
    lo, hi = c.get("radius", [0.0, 1.0])
    for sub in _subsets(state, init, pool, k):
        P = np.array([_p(state, o) for o in sub])
        if np.any(np.abs(P[:, 2] - CUBE / 2) > POS_TOL):
            continue
        ctr = P[:, :2].mean(axis=0)
        d = P[:, :2] - ctr
        R = float(np.linalg.norm(d, axis=1).mean())
        if not lo - POS_TOL <= R <= hi + POS_TOL:
            continue
        th = np.arctan2(d[:, 1], d[:, 0])
        order = np.argsort(th)
        step = 2 * np.pi / k
        phase = np.angle(np.mean(np.exp(1j * (th[order] - step * np.arange(k)))))
        ideal = ctr + R * np.stack([np.cos(phase + step * np.arange(k)), np.sin(phase + step * np.arange(k))], 1)
        if np.all(np.linalg.norm(P[order, :2] - ideal, axis=1) <= POS_TOL):
            return True
    return False


def _pyramid_of(c, state, init):
    return any(_pyramid(state, sub) for sub in _subsets(state, init, c["objs"], c["count"]))


def _tower_order(state, objs):
    """bottom-to-top order if objs form one tower, else None"""
    o = sorted(objs, key=lambda i: _p(state, i)[2])
    return o if all(_on_top(state, o[k + 1], o[k]) for k in range(len(o) - 1)) else None


def _reverse_of_step(c, state, step_states):
    """Final state: objs form one tower whose order (bottom to top) is the reverse of their tower order in the state
    where step `step` (1-based) of destination.steps first held. A rebuild in the same order fails."""
    n = c.get("step", 1)
    if len(step_states) < n:
        return False, f"reverse_of_step: step {n} never held"
    before = _tower_order(step_states[n - 1], c["objs"])
    after = _tower_order(state, c["objs"])
    if before is None:
        return False, f"reverse_of_step: {c['objs']} were not one tower in step {n}"
    if after is None:
        return False, f"reverse_of_step: {c['objs']} are not one tower at the end"
    return after == before[::-1], f"tower {after} (bottom to top) is not the reverse of step {n}'s {before}"


SHAPES = {"grid": _grid, "solid_cube": _solid_cube, "shape_cells": _shape_cells, "ring": _ring, "pyramid_of": _pyramid_of}


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
        if t == "reverse_of_step":
            return False, "reverse_of_step is only valid in destination.constraints of a sample with steps"
        if t in SHAPES:
            return SHAPES[t](c, state, init), f"{t} of {c['objs']} does not hold"
        if t == "one_of":
            for opt in c["options"]:
                if all(eval_constraint(cc, state, init, held)[0] for cc in opt):
                    return True, ""
            return False, "none of the one_of options holds"
        if t == "outside_area":  # data_posv3: no part of any of objs lies inside the area (centre, half sizes)
            cx, cy = c["center"]
            hx, hy = c["half"]
            bad = [o for o in c["objs"] if abs(_p(state, o)[0] - cx) < hx + CUBE / 2 and abs(_p(state, o)[1] - cy) < hy + CUBE / 2]
            return not bad, f"{bad} still reach into the area around {c['center']}"
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


# ---------------------------------------------------------------- answers (data_posv1)
# A question about the scene is answered with one short sentence; the arm does not move. The checker derives the true
# answer from the scene and the question; `accept` items (alternatives separated by "|") must all appear in the reply
# as words, `reject` items must not. The stored lists must equal the derived ones.
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
             "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
             "twenty-one", "twenty-two", "twenty-three", "twenty-four", "twenty-five", "twenty-six", "twenty-seven"]
AT_TOL = 0.03     # a cube is "at" a queried table position if its centre is within 3 cm (xy)
REL_MARGIN = 0.02  # relations are only asked when the difference is >= 2 cm (clear), see SCHEMA.md
NOTHING = "no cube|nothing|empty|none|no object"


def _resolve(scene, ref):
    """-> (id or None, reason): a cube named by id or by colour."""
    objs = scene["objects"]
    if "id" in ref:
        return (ref["id"], "") if ref["id"] in objs else (None, "missing")
    hits = [i for i, o in objs.items() if o["color"] == ref["color"]]
    if len(hits) == 1:
        return hits[0], ""
    return None, "missing" if not hits else "ambiguous"


def _name(scene, ref):
    return ref["id"] if "id" in ref else ref["color"]


def derive_answer(scene, q):
    """-> (accept, reject) or raises ValueError if the question has no clear answer in this scene."""
    objs = scene["objects"]
    pos = {i: np.array(o["position"], float) for i, o in objs.items()}
    k = q["kind"]
    if k == "relation":
        a, wa = _resolve(scene, q["a"])
        b, wb = _resolve(scene, q["b"])
        for ref, w in ((q["a"], wa), (q["b"], wb)):
            if w == "missing":  # say it is not there and ask what they meant
                return [_name(scene, ref), "?"], ["yes"]
            if w == "ambiguous":
                raise ValueError("ambiguous reference")
        d = pos[a] - pos[b]
        val = {"left_of": d[1], "right_of": -d[1], "in_front_of": d[0], "behind": -d[0],
               "above": d[2], "below": -d[2]}[q["rel"]]
        if q["rel"] in ("above", "below"):
            same_col = _xy(pos[a], pos[b]) <= POS_TOL
            if abs(d[2]) > POS_TOL and not same_col:
                raise ValueError("above/below only for cubes in one column or on one layer")
            if abs(d[2]) <= POS_TOL:
                return ["no"], ["yes"]
        elif abs(val) < REL_MARGIN:
            raise ValueError("near tie")
        return (["yes"], ["no"]) if val > 0 else (["no"], ["yes"])
    if k == "count":
        n = sum(1 for o in objs.values() if "color" not in q or o["color"] == q["color"])
        return [f"{n}|{NUM_WORDS[n]}"], []
    if k == "color_of":
        a, w = _resolve(scene, q["a"])
        if a is None:
            return [_name(scene, q["a"]), "?"], []
        return [objs[a]["color"]], []
    if k == "at":
        xy = np.array(q["position"][:2], float)
        near = [i for i in objs if np.linalg.norm(pos[i][:2] - xy) <= AT_TOL]
        far_ok = all(np.linalg.norm(pos[i][:2] - xy) > 2 * AT_TOL for i in objs if i not in near)
        if not far_ok:
            raise ValueError("a cube near the queried position is neither clearly there nor clearly away")
        if not near:
            return [NOTHING], []
        top = max(near, key=lambda i: pos[i][2])
        if len(near) > 1:
            raise ValueError("a stack at the queried position")
        return [objs[top]["color"]], []
    if k == "top":
        zs = sorted(objs, key=lambda i: -pos[i][2])
        if pos[zs[0]][2] < CUBE / 2 + POS_TOL:
            return ["table"], []
        if len(zs) > 1 and pos[zs[1]][2] > pos[zs[0]][2] - POS_TOL:
            raise ValueError("two cubes share the top height")
        return [zs[0], objs[zs[0]]["color"]], []
    if k == "extreme":
        key = {"closest": lambda i: float(np.linalg.norm(pos[i][:2])), "furthest": lambda i: -float(np.linalg.norm(pos[i][:2])),
               "leftmost": lambda i: -pos[i][1], "rightmost": lambda i: pos[i][1],
               "frontmost": lambda i: -pos[i][0], "backmost": lambda i: pos[i][0]}[q["which"]]
        s = sorted(objs, key=key)
        if len(s) > 1 and abs(key(s[1]) - key(s[0])) < REL_MARGIN:
            raise ValueError("near tie")
        return [s[0], objs[s[0]]["color"]], []
    if k == "gripper_open":
        return (["yes"], ["no"]) if scene["gripper"]["is_open"] else (["no"], ["yes"])
    raise ValueError(f"unknown question kind {k}")


def _has(txt, item):
    for alt in item.split("|"):
        if alt == "?":
            if "?" in txt:
                return True
        elif re.search(rf"(?<![\w.-]){re.escape(alt)}(?![\w-]|\.\d)", txt, re.I):
            return True
    return False


def check_answer(sample):
    reasons = []
    txt = sample["assistant_answer"].strip()
    if "def run" in txt or re.search(r"robot\.\w+\(", txt) or "```" in txt:  # "... to the robot." is not code
        reasons.append("answer contains code")
    sents = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", txt) if s.strip()]
    if len(sents) != 1:
        reasons.append(f"answer has {len(sents)} sentences (exactly 1 required)")
    if len(txt.split()) > 25:
        reasons.append("answer longer than 25 words")
    dest = sample["destination"]
    try:
        acc, rej = derive_answer(sample["scene"], dest["question"])
    except (ValueError, KeyError) as e:
        return reasons + [f"question has no clear answer: {e}"]
    if dest.get("accept") != acc or dest.get("reject", []) != rej:
        reasons.append(f"stored accept/reject {dest.get('accept')}/{dest.get('reject')} != derived {acc}/{rej}")
    for a in acc:
        if not _has(txt, a):
            reasons.append(f"answer does not contain {a!r}")
    for r in rej:
        if _has(txt, r):
            reasons.append(f"answer contains {r!r}")
    ref = dest.get("reference")
    if ref is None or ref.get("objects") != sample["scene"]["objects"]:
        reasons.append("answer reference must equal the initial scene (the arm does not move)")
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
    # data_v5: the user_prompt is the request only (no "Objects:" header); the scene goes to the model as Scene_Info JSON
    if not isinstance(sample["user_prompt"], str) or not sample["user_prompt"].strip():
        reasons.append("user_prompt must be a non-empty request")
    if sample["expected_behavior"] not in ("execute", "refuse", "answer"):
        reasons.append("expected_behavior must be execute, refuse or answer")
    dest = sample["destination"]
    if (dest.get("mode") == "answer") != (sample["expected_behavior"] == "answer"):
        reasons.append("destination.mode answer goes with expected_behavior answer (and only with it)")
    if dest.get("mode") not in ("exact", "constraints", "answer"):
        reasons.append("destination.mode must be exact, constraints or answer")
    if "reference" not in dest:
        reasons.append("destination.reference missing")
    for c in dest.get("constraints", []) + [c for st in dest.get("steps", []) for c in st]:
        if c.get("type") not in CONSTRAINT_TYPES:
            reasons.append(f"unknown constraint {c.get('type')}")
        for o in _objs_in(c):
            if o not in objs:
                reasons.append(f"constraint refers to unknown object {o}")
    for p in [o["position"] for o in objs.values()]:
        if len(p) != 3:
            reasons.append("bad position")
    return reasons


# ---------------------------------------------------------------- data_posv3: strict grasp
REST_OVERLAP = 0.035  # centres closer than 3.5 cm on both axes = footprints overlap by more than 5 mm


def strict_grasp(sim):
    """Raised inside close_gripper(): two cubes held at once, or a cube resting on the grasped cube."""
    import robot as _robot
    held = sorted(sim.held_objects())
    if len(held) > 1:
        raise _robot.RobotError(f"strict grasp: cubes {', '.join(held)} are held at once (a cube is carried along); "
                                "take the cubes off one at a time from the top")
    for h in held:
        ph = sim.obj_pos(h)
        for j in sim.ids:
            if j == h:
                continue
            q = sim.obj_pos(j)
            if abs(q[2] - ph[2] - CUBE) < 0.01 and abs(q[0] - ph[0]) < REST_OVERLAP and abs(q[1] - ph[1]) < REST_OVERLAP:
                raise _robot.RobotError(f"strict grasp: cube {j} rests on the grasped cube {h}; take cube {j} off first")


# ---------------------------------------------------------------- main check
def check(sample):
    """Returns (ok, reasons)."""
    reasons = check_format(sample)
    if reasons:
        return False, reasons
    if sample["expected_behavior"] == "refuse":
        r = check_refusal(sample)
        return not r, r
    if sample["expected_behavior"] == "answer":
        r = check_answer(sample)
        return not r, r
    import robot as _robot
    code = sample["assistant_answer"]
    if "def run" not in code:
        return False, ["execute sample without `def run`"]
    snaps = []  # data_posv1: the state after every open_gripper() (released cube settled), for destination.steps
    orig_open = _robot.Sim.open_gripper
    orig_close = _robot.Sim.close_gripper

    def rec_open(self):
        orig_open(self)
        snaps.append(self.state())

    def strict_close(self):  # data_posv3: strict grasp rule (see the module docstring)
        orig_close(self)
        strict_grasp(self)
    _robot.Sim.open_gripper = rec_open
    _robot.Sim.close_gripper = strict_close
    try:
        state, err, sim = _robot.execute(sample["scene"], code)
    except Exception as e:  # scene could not be built
        return False, [f"simulation setup failed: {e}"]
    finally:
        _robot.Sim.open_gripper = orig_open
        _robot.Sim.close_gripper = orig_close
    if err:
        return False, [f"execution error: {err}"]
    init = {"objects": {i: {"position": p.tolist()} for i, p in sim.start_positions.items()},
            "gripper": sample["scene"]["gripper"]}
    held = sim.held_objects()
    dest = sample["destination"]
    reasons += physical_checks(state, sim, held)
    cons = dest.get("constraints", [])
    k, step_states = 0, []
    for n, step in enumerate(dest.get("steps", [])):
        while k < len(snaps) and not all(eval_constraint(c, snaps[k], init, set())[0] for c in step):
            k += 1
        if k == len(snaps):
            reasons.append(f"step {n + 1} of the request never held: "
                           + "; ".join(eval_constraint(c, state, init, set())[1] for c in step)[:200])
            break
        step_states.append(snaps[k])  # the first recorded state in which this step held
    for c in cons:
        if c.get("type") == "reverse_of_step":
            ok, why = _reverse_of_step(c, state, step_states)
        else:
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


# ---------------------------------------------------------------- result cache (data/check_cache_posv3.json)
# A check result depends only on the sample (minus `verified`) and on the checker / simulator sources, so unchanged rows
# reuse their result. The key hashes both; editing any of these files or the row invalidates the entry.
CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "check_cache_posv3.json")
_SRC = ("check_posv3.py", "robot.py", "scene.py", "make_reference.py")


def _code_hash():
    import hashlib
    h = hashlib.sha256()
    d = os.path.dirname(os.path.abspath(__file__))
    for f in _SRC:
        p = os.path.join(d, f)
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()


def _key(line, code):
    import hashlib
    s = json.loads(line)
    s.pop("verified", None)
    return hashlib.sha256((code + json.dumps(s, sort_keys=True)).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--write", action="store_true", help="rewrite files with verified set from the check")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="re-simulate every row (the cache is still updated)")
    a = ap.parse_args()
    import multiprocessing as mp
    total_bad = 0
    with mp.Pool(a.workers) as pool:
        for f in a.files:
            lines = [l for l in open(f) if l.strip()]
            code = _code_hash()
            try:
                cache = json.load(open(CACHE_FILE))
            except Exception:
                cache = {}
            keys = [_key(l, code) for l in lines]
            todo = [k for k, key in enumerate(keys) if a.no_cache or key not in cache]
            fresh = dict(zip(todo, pool.map(_work, [lines[k] for k in todo], chunksize=1)))
            res = []
            for k, (l, key) in enumerate(zip(lines, keys)):
                if k in fresh:
                    r = fresh[k]
                    cache[key] = [bool(r[1]), list(r[2])]
                    res.append(r)
                else:
                    ok, reasons = cache[key]
                    res.append((json.loads(l), ok, reasons, 0.0))
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            tmp = CACHE_FILE + ".tmp"
            json.dump(cache, open(tmp, "w"))
            os.replace(tmp, CACHE_FILE)
            print(f"== {f}: {len(lines) - len(todo)} cached, {len(todo)} simulated")
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
