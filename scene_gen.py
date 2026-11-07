"""The scene generator of the edit button next to "New random scene": structures (towers, pyramids, flat squares and shapes) at random free spots,
then spare single cubes. Without structures server.random_scene runs as before, so this module is only used when the page sends objects.

A structure is a list of cubes in grid units (i, j, k): centre = origin + 0.04 * (i, j) and z = 0.02 + 0.04 * k, so neighbours touch exactly
(sim/scene.py makes the collision boxes 0.5 mm smaller in x/y for that). Footprints of different structures and spare cubes stay at least
GAP apart, the same 6 cm gap the spare cubes always had (centres 10 cm apart), so the open gripper fits between them."""
import math
import random

CUBE = 0.04
GAP = 0.06                                       # free floor between two structures, or a structure and a spare cube
SPARE_DIST = 0.10                                # centre distance of two spare cubes, as in server.random_scene
NORMAL = (0.30, 0.62, -0.28, 0.28)               # x0, x1, y0, y1 of the cube centres: the area random_scene uses up to 24 cubes
BIG = (0.20, 0.72, -0.364, 0.364)                # the area of server.BIG_GRID: all reachable and graspable; used above 24 cubes or when NORMAL is too small
GRIPPER_KEEP_OUT = (0.39, 0.41, -0.05, 0.05)     # the open fingers of the start pose (0.4, 0, 0.3) reach down to z = 0.288
TALL = 0.27                                      # a structure whose top face is higher keeps GAP / 2 away from those fingers
MAX_OBJECTS = 30

# type -> (label, smallest size, largest size, default size, what the size means). Sizes outside are clamped.
TYPES = {
    "cube": ("single cube", 1, 1, 1, ""),
    "tower": ("tower", 2, 10, 3, "height"),
    "pyramid": ("pyramid", 2, 5, 3, "layers"),
    "square": ("flat square", 2, 4, 2, "side"),
    "line": ("line", 2, 6, 3, "length"),
    "L": ("L shape", 1, 3, 2, "arm"),
    "T": ("T shape", 1, 3, 1, "arm"),
    "cross": ("cross", 1, 2, 1, "arm"),
}


def type_table():
    """What the page's editor offers: per type its label, size range, default, what the size means and the cubes of every size."""
    return {k: dict(label=label, min=lo, max=hi, default=d, unit=unit, cubes={s: len(cells(k, s)) for s in range(lo, hi + 1)})
            for k, (label, lo, hi, d, unit) in TYPES.items()}


class SceneError(ValueError):
    """The configuration cannot be built; the message is shown on the page as it is."""


def cells(kind, size):
    """The cubes of one structure in grid units (i, j, k), before it is turned."""
    if kind == "cube":
        return [(0, 0, 0)]
    if kind == "tower":
        return [(0, 0, k) for k in range(size)]
    if kind == "pyramid":                        # flat, as in the dataset: each layer a touching row, every upper cube on two below
        return [(k / 2 + m, 0, k) for k in range(size) for m in range(size - k)]
    if kind == "square":
        return [(i, j, 0) for i in range(size) for j in range(size)]
    if kind == "line":
        return [(i, 0, 0) for i in range(size)]
    if kind == "L":
        return [(0, 0, 0)] + [(i, 0, 0) for i in range(1, size + 1)] + [(0, j, 0) for j in range(1, size + 1)]
    if kind == "T":
        return [(i, 0, 0) for i in range(-size, size + 1)] + [(0, -j, 0) for j in range(1, size + 1)]
    if kind == "cross":
        return [(0, 0, 0)] + [c for d in range(1, size + 1) for c in ((d, 0, 0), (-d, 0, 0), (0, d, 0), (0, -d, 0))]
    raise SceneError(f"unknown object type {kind!r}")


TURNS = {"line": 2, "pyramid": 2, "L": 4, "T": 4}   # how many quarter turns give different layouts; the others look the same every way


def turned(cs, quarter):
    for _ in range(quarter):
        cs = [(-j, i, k) for i, j, k in cs]
    i0, j0 = min(c[0] for c in cs), min(c[1] for c in cs)
    return [(i - i0, j - j0, k) for i, j, k in cs]


def normalize(objects):
    """The page's list -> [(kind, size, colour or None)]. Raises SceneError for anything it cannot build."""
    if not isinstance(objects, list):
        raise SceneError("objects must be a list")
    if len(objects) > MAX_OBJECTS:
        raise SceneError(f"at most {MAX_OBJECTS} objects")
    out = []
    for o in objects:
        if not isinstance(o, dict) or o.get("type") not in TYPES:
            raise SceneError(f"unknown object type {o.get('type') if isinstance(o, dict) else o!r}")
        _, lo, hi, default, _ = TYPES[o["type"]]
        try:
            size = int(o.get("size", default))
        except (TypeError, ValueError):
            size = default
        out.append((o["type"], min(hi, max(lo, size)), o.get("color") if o.get("unicolor") else None))
    return out


def needed(objects):
    """Cubes the configured structures take together: the smallest scene they make."""
    return sum(len(cells(k, s)) for k, s, _ in normalize(objects))


def describe(kind, size):
    label, _, _, _, unit = TYPES[kind]
    if kind == "square":
        return f"{size}x{size} {label}"
    if kind == "cube":
        return label
    return f"{label} ({unit} {size})"


def _gap(a, b):
    dx = max(a[0] - b[1], b[0] - a[1], 0.0)
    dy = max(a[2] - b[3], b[2] - a[3], 0.0)
    return math.hypot(dx, dy)


def _rect(centres):
    h = CUBE / 2
    return (min(p[0] for p in centres) - h, max(p[0] for p in centres) + h, min(p[1] for p in centres) - h, max(p[1] for p in centres) + h)


def _inside(p, area):
    return area[0] - 1e-9 <= p[0] <= area[1] + 1e-9 and area[2] - 1e-9 <= p[1] <= area[3] + 1e-9


def _try_place(shapes, area, rng):
    """One go at putting every structure (biggest first) at a random free spot. Returns [(index, centres, rect)] or None."""
    placed = []
    order = sorted(range(len(shapes)), key=lambda n: -len(shapes[n][2]))
    for n in order:
        kind, size, cs0 = shapes[n]
        tall = CUBE * (max(c[2] for c in cs0) + 1) > TALL
        for _ in range(300):
            cs = turned(cs0, rng.randrange(TURNS.get(kind, 1)))
            span_i, span_j = max(c[0] for c in cs) * CUBE, max(c[1] for c in cs) * CUBE
            if span_i > area[1] - area[0] + 1e-9 or span_j > area[3] - area[2] + 1e-9:
                return None
            ox, oy = round(rng.uniform(area[0], area[1] - span_i), 2), round(rng.uniform(area[2], area[3] - span_j), 2)
            centres = [(round(ox + CUBE * i, 3), round(oy + CUBE * j, 3), round(CUBE / 2 + CUBE * k, 3)) for i, j, k in cs]
            if not all(_inside(p, area) for p in centres):
                continue                         # the rounding pushed it over the edge
            r = _rect(centres)
            if any(_gap(r, q[2]) < GAP - 1e-9 for q in placed):
                continue
            if tall and _gap(r, GRIPPER_KEEP_OUT) < GAP / 2:
                continue
            placed.append((n, centres, r))
            break
        else:
            return None
    return sorted(placed)


def _spares(n, area, rects, rng):
    """Up to n single cubes, random spots first as in random_scene. If that gets stuck, two fallbacks fill up and the better one wins:
    the random spots topped up from a fine lattice, or the 10.4 cm grid of server.BIG_GRID (packs the most) topped up the same way."""
    h = CUBE / 2

    def free(p, pos):
        r = (p[0] - h, p[0] + h, p[1] - h, p[1] + h)
        return (_inside(p, area) and all(_gap(r, q) >= GAP - 1e-9 for q in rects)
                and all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > SPARE_DIST ** 2 for q in pos))

    def fill(pos, spots):
        for p in spots:
            if len(pos) == n:
                break
            if free(p, pos):
                pos.append(p)
        return pos
    pos = []
    for _ in range(2000):
        if len(pos) == n:
            return pos
        p = (round(rng.uniform(area[0], area[1]), 2), round(rng.uniform(area[2], area[3]), 2))
        if free(p, pos):
            pos.append(p)
    fine = [(round(area[0] + 0.02 * a, 3), round(area[2] + 0.02 * b, 3))
            for a in range(int((area[1] - area[0]) / 0.02 + 1e-6) + 1) for b in range(int((area[3] - area[2]) / 0.02 + 1e-6) + 1)]
    rng.shuffle(fine)
    grid = [(round(area[0] + 0.104 * a, 3), round(area[2] + 0.104 * b, 3))
            for a in range(int((area[1] - area[0]) / 0.104 + 1e-6) + 1) for b in range(int((area[3] - area[2]) / 0.104 + 1e-6) + 1)]
    rng.shuffle(grid)
    one, two = fill(pos, fine), fill(fill([], grid), fine)
    return one if len(one) >= len(two) else two


def generate(n, seed, objects, colors):
    """A scene with the configured structures and n cubes in all (at least the structures' own). Spare cubes that do not fit are left out,
    as random_scene does, and the page tells the user; a structure is never left out: SceneError instead.
    Returns dict(scene, groups=[{type, size, label, color, ids}], spare=[ids], needed)."""
    objs = normalize(objects)
    unknown = [c for _, _, c in objs if c not in colors and c is not None]
    if unknown:
        raise SceneError(f"unknown colour {unknown[0]!r}")
    shapes = [(k, s, cells(k, s)) for k, s, _ in objs]
    need = sum(len(c) for _, _, c in shapes)
    n = max(n, need)
    rng = random.Random(seed)
    best = None                                  # (placed, spare cubes) of the first area where everything fits, else of the one that holds the most
    for area in ([NORMAL, BIG] if n <= 24 else [BIG]):
        for _ in range(40):
            placed = _try_place(shapes, area, rng)
            if placed:
                break
        if placed:
            spare = _spares(n - need, area, [r for _, _, r in placed], rng)
            if not best or len(spare) > len(best[1]):
                best = placed, spare
            if len(spare) == n - need:
                break
    if not best:
        count = {}
        for k, s, _ in objs:
            count[describe(k, s)] = count.get(describe(k, s), 0) + 1
        names = ", ".join(f"{c} x {d}" if c > 1 else d for d, c in count.items())
        raise SceneError(f"These objects do not fit on the table together ({names}). Remove one or make one smaller.")
    placed, spare = best
    total = need + len(spare)

    # colours: the mixed structures and then the spare cubes take the colours nobody chose, in random order, round after round
    # (from 12 cubes on colours repeat, the ids stay unique). With fewer than 4 colours left over, all 12 take part, so mixed stays mixed.
    chosen = {c for _, _, c in objs if c}
    free_colors = [c for c in colors if c not in chosen]
    free_colors = free_colors if len(free_colors) >= 4 else list(colors)
    pool = []

    def next_color():
        if not pool:
            pool.extend(rng.sample(free_colors, len(free_colors)))
        return pool.pop(0)
    ids = rng.sample(range(1, max(total, 9) + 1), total)     # random order, so an id says nothing about the structure it belongs to
    objects_out, groups, at = {}, [], 0
    for idx, centres, _ in placed:
        kind, size, color = objs[idx]
        gid = ids[at:at + len(centres)]
        at += len(centres)
        for i, p in zip(gid, centres):
            objects_out[str(i)] = dict(color=color or next_color(), position=list(p))
        groups.append(dict(type=kind, size=size, label=describe(kind, size), color=color or "mixed", ids=[str(i) for i in gid]))
    spare_ids = [str(i) for i in ids[at:]]
    for i, (x, y) in zip(spare_ids, spare):
        objects_out[i] = dict(color=next_color(), position=[x, y, 0.02])
    scene = dict(objects=dict(sorted(objects_out.items(), key=lambda kv: int(kv[0]))),
                 gripper=dict(position=[0.4, 0.0, 0.3], angle_deg=0, is_open=True))
    return dict(scene=scene, groups=groups, spare=spare_ids, needed=need)
