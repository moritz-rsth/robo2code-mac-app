"""Runs ONE program in the MuJoCo simulation and streams JSON lines on stdout: frames (base64 JPEG) while it runs, the events of the picture
(grasps, releases, cubes landing: see Tracker), then a result.
usage: python sim_worker.py <job.json>     job = {"scene": {...}, "code": "...", "sample": {...} or null, "speed": 4.0}

The program runs through the exact same API/sandbox the dataset was verified with (sim/robot.py), so what you see here is what
the checker counts. If `sample` (a dataset row with its destination) is given, the result also says whether the task was solved.
The frames are drawn from a second, visual-only model (floor_view.py: endless grid floor); the physics model is the untouched one of sim/scene.py."""
import ast
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "sim")]
import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

import importlib  # noqa: E402
import floor_view  # noqa: E402
import lara_view  # noqa: E402
import numpy as np  # noqa: E402
import robot as R  # noqa: E402
from scene import CUBE, TIMESTEP  # noqa: E402

OUT = os.fdopen(os.dup(1), "w")          # protocol channel; anything else printed on fd 1 goes to stderr
os.dup2(2, 1)
# Frame size, 16:9: what the recording and the 1920x1080 export are made from. The job's "render" ("1920x1080", "1280x720", "960x540" or "640x360",
# chosen in the app) wins over P2R_RENDER; P2R_JPEG sets the JPEG quality. The live view gets a small copy of every frame (LIVE), so a high
# resolution costs render time but not the browser's bandwidth.
RENDER_SIZES = {"1920x1080", "1280x720", "960x540", "640x360"}
W, H = 1280, 720
JPEG_QUALITY = int(os.environ.get("P2R_JPEG", "88"))
LIVE, LIVE_JPEG = (640, 360), 75


def set_size(spec):
    global W, H
    if spec and spec.lower() in RENDER_SIZES:
        W, H = (int(v) for v in spec.lower().split("x"))
FRAME_EVERY = max(1, round(1 / (30 * TIMESTEP)))     # ~30 frames per simulated second


def emit(**kw):
    OUT.write(json.dumps(kw) + "\n")
    OUT.flush()


# Lenient mode (default in the app): the physics runs to the end even if a finger or the arm touches a cube/the table. What the dataset checker
# would have flagged (and aborted on) is collected and shown as a warning. Set P2R_STRICT=1 to abort on contact like the checker.
LENIENT = os.environ.get("P2R_STRICT", "0") != "1"
FLAGS, SEEN = [], set()


def _soften(orig):
    def wrapped(self, *a, **k):
        try:
            return orig(self, *a, **k)
        except R.RobotError as e:
            msg = str(e)
            if msg not in SEEN and len(FLAGS) < 5:
                SEEN.add(msg)
                f = dict(sim_s=round(self.steps * TIMESTEP, 1), msg=msg)
                FLAGS.append(f)
                emit(t="flag", **f)
    return wrapped


if LENIENT:
    R.Sim._check_collisions = _soften(R.Sim._check_collisions)
    R.Sim._check_clearance = _soften(R.Sim._check_clearance)


def count_calls(code):
    try:
        return sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                   and n.func.value.id == "robot" for n in ast.walk(ast.parse(code)))
    except SyntaxError:
        return 0


# What happens in the picture, with its moment on the frames' clock (sim_s = steps * TIMESTEP): the export's music plays these in sync with the video.
# Every event is read off the simulation state at frame cadence (the same steps the frames are drawn at, so an event is never off by more than a
# frame) and only read: nothing here steps, settles or touches the physics.
#   grasp    the gripper closed (cube = what it holds now, None if it closed on nothing)
#   release  the gripper opened (cube = what it let go, None if it was empty)
#   land     a cube came to rest after it was let go or knocked: onto = "table" or the cube under it, height = the stack it tops (1 on the table),
#            fall = how far it fell (m); "drop" instead of "land" when it fell more than a cube's edge
#   Each is a line {t: "event", sim_s, what, ...} as it happens and all of them are the result's "events".
EVENTS = []
MOVING, STILL, FALL_DROP = 0.05, 0.01, CUBE                            # m/s: a cube that is on its way / that has come to rest; a drop


class Tracker:
    def __init__(self):
        self.gripper, self.pending, self.fingers = None, {}, None         # gripper: the call in progress, pending: cubes on their way

    def sim_s(self, sim):
        return round(sim.steps * TIMESTEP, 3)

    def event(self, sim, **kw):
        e = dict(sim_s=self.sim_s(sim), **kw)
        EVENTS.append(e)
        emit(t="event", **e)

    def gripper_call(self, sim, opening):
        """close_gripper / open_gripper started: the frames of this call tell when the fingers meet the cube or let go of it."""
        self.gripper = dict(opening=opening, before=set(sim.held_objects()), done=False, frames=0)
        self.fingers = sim.data.qpos[sim.finger_qadr].copy()

    def gripper_done(self, sim):
        """The call returned: whatever the frames did not catch (a close on nothing that stopped between two frames) is recorded now."""
        g, self.gripper = self.gripper, None
        held = set(sim.held_objects())
        if g and not g["done"]:
            cubes = sorted(g["before"] - held if g["opening"] else held - g["before"])
            self.event(sim, what="release" if g["opening"] else "grasp", cube=cubes[0] if cubes else None)
            for i in cubes if g["opening"] else ():
                self.pending[i] = dict(z0=float(sim.obj_pos(i)[2]), moved=False, frames=0)

    def observe(self, sim):
        """Once per frame."""
        held = set(sim.held_objects())
        g = self.gripper
        if g and not g["done"]:
            g["frames"] += 1
            fingers = sim.data.qpos[sim.finger_qadr].copy()
            stopped = g["frames"] > 2 and np.max(np.abs(fingers - self.fingers)) < 1e-4
            self.fingers = fingers
            if g["opening"] and (g["before"] - held or (stopped and not g["before"])):
                let_go = sorted(g["before"] - held)
                g["done"] = True
                self.event(sim, what="release", cube=let_go[0] if let_go else None)
                for i in let_go:
                    self.pending[i] = dict(z0=float(sim.obj_pos(i)[2]), moved=False, frames=0)
            elif not g["opening"] and (held - g["before"] or stopped):
                got = sorted(held - g["before"])
                g["done"] = True
                self.event(sim, what="grasp", cube=got[0] if got else None)
        for i in sim.ids:                                                   # cubes on their way: let go, pushed, knocked off a stack
            if i in held:
                self.pending.pop(i, None)
                continue
            da = sim.model.jnt_dofadr[sim.model.body_jntadr[sim.body[i]]]
            v = float(np.linalg.norm(sim.data.qvel[da:da + 3]))
            p = self.pending.get(i)
            if p is None:
                if v > MOVING:
                    self.pending[i] = dict(z0=float(sim.obj_pos(i)[2]), moved=True, frames=0)
                continue
            p["frames"] += 1
            p["moved"] = p["moved"] or v > MOVING
            if v < STILL and (p["moved"] or p["frames"] >= 2):
                del self.pending[i]
                onto, height = self.support(sim, i)
                fall = max(0.0, p["z0"] - float(sim.obj_pos(i)[2]))
                self.event(sim, what="drop" if fall > FALL_DROP else "land", cube=i, onto=onto, height=height, fall=round(fall, 3))

    def support(self, sim, i):
        """What cube i rests on (the nearest cube one edge below it, else the table) and the height of the stack it tops."""
        pos = {j: sim.obj_pos(j) for j in sim.ids}
        height, seen = 1, {i}
        onto = None
        while True:
            p = pos[i]
            below = [(abs(p[2] - pos[j][2] - CUBE) + np.hypot(*(p[:2] - pos[j][:2])), j) for j in sim.ids
                     if j not in seen and abs(p[2] - pos[j][2] - CUBE) < CUBE / 3 and np.hypot(*(p[:2] - pos[j][:2])) < CUBE * 0.9]
            if not below:
                return onto or "table", height
            i = min(below)[1]
            onto = onto or i
            height += 1
            seen.add(i)


def install_frame_hook(speed):
    """Wrap Sim._step so that a rendered frame is streamed every FRAME_EVERY physics steps, paced to `speed` x real time; the gripper calls are
    wrapped so that the Tracker above sees them. Returns a function that removes the hooks again."""
    st = dict(renderer=None, cam=None, last=time.time())
    orig_step, orig_open, orig_close = R.Sim._step, R.Sim.open_gripper, R.Sim.close_gripper
    track = Tracker()

    def render(sim):
        if st["renderer"] is None:
            st["model"] = floor_view.render_model(sim.scene, W, H)     # a copy for drawing only, so nothing here can reach the physics
            st["model"].vis.global_.offwidth, st["model"].vis.global_.offheight = max(W, st["model"].vis.global_.offwidth), max(H, st["model"].vis.global_.offheight)
            st["data"] = mujoco.MjData(st["model"])
            st["follow"] = lara_view.Follow(st["model"], sim) if lara_view.enabled() else None
            st["model"].vis.headlight.ambient[:] = 0.60                # the two lights of floor_view.py do most of the work, the headlight only lifts the shade
            st["model"].vis.headlight.diffuse[:] = 0.12
            st["model"].vis.headlight.specular[:] = 0.0
            st["renderer"] = mujoco.Renderer(st["model"], H, W)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0.40, 0.0, 0.12]
            cam.distance, cam.azimuth, cam.elevation = 1.45, 135.0, -30.0
            st["cam"] = cam
        if st["follow"]:
            st["follow"].sync(st["data"], sim)                   # the LARA 5 follows the physics gripper
        else:
            st["data"].qpos[:] = sim.data.qpos
        mujoco.mj_kinematics(st["model"], st["data"])
        mujoco.mj_camlight(st["model"], st["data"])               # places the scene's lights (kinematics alone leaves them at the origin)
        st["renderer"].update_scene(st["data"], camera=st["cam"])
        img, buf, live = Image.fromarray(st["renderer"].render()), io.BytesIO(), io.BytesIO()
        img.save(buf, "JPEG", quality=JPEG_QUALITY, subsampling=0)                     # 4:4:4, so coloured edges stay crisp (the recording)
        (img.resize(LIVE, Image.BILINEAR) if img.width > LIVE[0] else img).save(live, "JPEG", quality=LIVE_JPEG)   # the live view
        emit(t="frame", jpg=base64.b64encode(buf.getvalue()).decode(), live=base64.b64encode(live.getvalue()).decode(),
             sim_s=round(sim.steps * TIMESTEP, 2))
        wait = FRAME_EVERY * TIMESTEP / speed - (time.time() - st["last"])
        if wait > 0:
            time.sleep(wait)
        st["last"] = time.time()

    def step(self, *a, **k):
        out = orig_step(self, *a, **k)
        self._n = getattr(self, "_n", 0) + 1
        if self._n % FRAME_EVERY == 0:
            track.observe(self)
            render(self)
        return out

    def gripper(orig, opening):
        def wrapped(self):
            track.gripper_call(self, opening)
            try:
                return orig(self)
            finally:
                track.gripper_done(self)
        return wrapped

    R.Sim._step, R.Sim.open_gripper, R.Sim.close_gripper = step, gripper(orig_open, True), gripper(orig_close, False)

    def remove():
        R.Sim._step, R.Sim.open_gripper, R.Sim.close_gripper = orig_step, orig_open, orig_close
    return remove


def next_scene(final, sim):
    """The start scene of a chained prompt: the end state, brought to rest. A scene cannot hold a cube in the gripper, so a held cube is let go and falls."""
    held = sorted(sim.held_objects())
    rest = R.Sim(dict(final, gripper=dict(final["gripper"], is_open=final["gripper"]["is_open"] or bool(held))))
    rest._settle(max_steps=3000, min_steps=200)
    return rest.state(), held


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    code, scene, sample = job["code"], job["scene"], job.get("sample")
    C = importlib.import_module((sample or {}).get("_check", "check"))   # check_posv1 for data_posv1 / data_posv2 rows
    set_size(job.get("render") or os.environ.get("P2R_RENDER"))
    remove_hook = install_frame_hook(max(float(job.get("speed", 4.0)), 0.1))
    t0 = time.time()
    final, err, sim = R.execute(scene, code)
    remove_hook()                                        # C.check below runs the program again: no second film in the stream
    res = dict(t="result", executed=err is None, error=err, final=final, sim_seconds=round(sim.steps * TIMESTEP, 1),
               calls=count_calls(code), wall=round(time.time() - t0, 1))
    res["strict_flags"] = list(FLAGS)
    res["events"], res["frame_dt"] = list(EVENTS), FRAME_EVERY * TIMESTEP        # frame_dt: simulated seconds between two recorded frames
    if sample and err is None:
        try:
            ok, reasons = C.check(dict(sample, assistant_answer=code))
            res["success"], res["reasons"] = bool(ok), list(reasons)[:4]
        except Exception as e:
            res["success"], res["reasons"] = False, [f"checker error: {e}"]
    try:
        res["next_scene"], res["dropped"] = next_scene(final, sim)
    except Exception as e:
        res["chain_error"] = str(e)
    emit(**res)


if __name__ == "__main__":
    main()
