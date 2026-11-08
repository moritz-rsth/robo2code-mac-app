"""Runs ONE program in the MuJoCo simulation and streams JSON lines on stdout: frames (base64 JPEG) while it runs, then a result.
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
import robot as R  # noqa: E402
from scene import TIMESTEP  # noqa: E402

OUT = os.fdopen(os.dup(1), "w")          # protocol channel; anything else printed on fd 1 goes to stderr
os.dup2(2, 1)
W, H = 640, 360
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


def install_frame_hook(speed):
    """Wrap Sim._step so that a rendered frame is streamed every FRAME_EVERY physics steps, paced to `speed` x real time.
    Returns a function that removes the hook again."""
    st = dict(renderer=None, cam=None, last=time.time())
    orig_step = R.Sim._step

    def render(sim):
        if st["renderer"] is None:
            st["model"] = floor_view.render_model(sim.scene)     # a copy for drawing only, so nothing here can reach the physics
            st["data"] = mujoco.MjData(st["model"])
            st["follow"] = lara_view.Follow(st["model"], sim) if lara_view.enabled() else None
            st["model"].vis.headlight.ambient[:] = 0.55
            st["model"].vis.headlight.diffuse[:] = 0.75
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
        st["renderer"].update_scene(st["data"], camera=st["cam"])
        buf = io.BytesIO()
        Image.fromarray(st["renderer"].render()).save(buf, "JPEG", quality=75)
        emit(t="frame", jpg=base64.b64encode(buf.getvalue()).decode(), sim_s=round(sim.steps * TIMESTEP, 2))
        wait = FRAME_EVERY * TIMESTEP / speed - (time.time() - st["last"])
        if wait > 0:
            time.sleep(wait)
        st["last"] = time.time()

    def step(self, *a, **k):
        out = orig_step(self, *a, **k)
        self._n = getattr(self, "_n", 0) + 1
        if self._n % FRAME_EVERY == 0:
            render(self)
        return out

    R.Sim._step = step
    return lambda: setattr(R.Sim, "_step", orig_step)


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
    remove_hook = install_frame_hook(max(float(job.get("speed", 4.0)), 0.1))
    t0 = time.time()
    final, err, sim = R.execute(scene, code)
    remove_hook()                                        # C.check below runs the program again: no second film in the stream
    res = dict(t="result", executed=err is None, error=err, final=final, sim_seconds=round(sim.steps * TIMESTEP, 1),
               calls=count_calls(code), wall=round(time.time() - t0, 1))
    res["strict_flags"] = list(FLAGS)
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
