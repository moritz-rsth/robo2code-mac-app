"""MuJoCo-backed robot API (exactly the methods documented in system_prompt.md)
plus a restricted executor for generated `run(robot)` code.

Arm control: the six arm joints are position controlled (stiff, industrial-arm
style): every physics step the joint targets from damped-least-squares IK on
the MuJoCo Jacobian are written to the arm and the physics is stepped, so
cubes, fingers and contacts are fully simulated. Grasping works through finger
contact + friction only (no welds, no teleporting).
"""
from __future__ import annotations

import ast
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene import (ARM_JOINTS, CUBE, FINGER_JOINTS, FINGER_OPEN, TIMESTEP,  # noqa: E402
                   build_xml, target_rotation)

WORKSPACE = np.array([[0.0, 0.8], [-0.4, 0.4], [0.0, 0.8]])
WS_EPS = 1e-6
SPEED = 0.30          # m/s Cartesian TCP speed
ROT_SPEED = 180.0     # deg/s gripper rotation speed
REACH_TOL = 0.001     # 1 mm
CLEARANCE = 0.001     # arm/fingers must stay >= 1 mm away from the table and non-held cubes
DISPLACE_TOL = 0.003  # a non-grasped cube pushed by more than 3 mm = collision
MAX_SIM_STEPS = 200_000  # hard cap per program (~400 s simulated)


class RobotError(Exception):
    pass


class Sim:
    def __init__(self, scene: dict):
        self.scene = scene
        self.model = mujoco.MjModel.from_xml_string(build_xml(scene))
        self.data = mujoco.MjData(self.model)
        self.kin = mujoco.MjData(self.model)  # scratch data for IK
        m = self.model
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        self.arm_dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
        self.arm_range = np.array([m.jnt_range[m.joint(j).id] for j in ARM_JOINTS])
        self.finger_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in FINGER_JOINTS])
        self.tcp = m.site("tcp").id
        self.ids = list(scene["objects"].keys())
        self.body = {i: m.body(f"obj_{i}").id for i in self.ids}
        self.geom = {i: m.geom(f"obj_{i}_geom").id for i in self.ids}
        self.cube_geoms = {g: i for i, g in self.geom.items()}
        self.table = m.geom("table").id
        robot_bodies = set()
        for b in range(m.nbody):
            name = m.body(b).name
            if name in ("base", "link1", "link2", "link3", "link4", "link5", "hand", "finger_l", "finger_r"):
                robot_bodies.add(b)
        self.robot_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in robot_bodies}
        self.finger_geoms = {m.geom("finger_l_geom").id, m.geom("finger_r_geom").id}
        self.steps = 0

        g = scene["gripper"]
        self.angle = float(g.get("angle_deg", 0.0))
        self.is_open = bool(g.get("is_open", True))
        q0 = np.array([0.0, 0.3, 1.6, 0.0, 1.24, 0.0])
        q = self._ik(np.array(g["position"], float), self.angle, q0, iters=300)
        if q is None:
            raise RobotError("gripper start pose is unreachable")
        self.q = q
        self.data.qpos[self.arm_qadr] = q
        fo = FINGER_OPEN if self.is_open else 0.0
        self.data.qpos[self.finger_qadr] = fo
        self.data.ctrl[:] = fo
        mujoco.mj_forward(self.model, self.data)
        self.start_positions = {i: self.obj_pos(i) for i in self.ids}
        # let the scene settle briefly (cubes are placed exactly touching)
        self._settle(max_steps=150, min_steps=50)

    # ------------------------------------------------------------------ state
    def obj_pos(self, i):
        return self.data.xpos[self.body[i]].copy()

    def tcp_pos(self):
        return self.data.site_xpos[self.tcp].copy()

    def measured_angle(self):
        r = self.data.site_xmat[self.tcp].reshape(3, 3)
        f = r[:, 1]  # finger axis
        meas = math.degrees(math.atan2(-f[0], f[1]))
        d = (meas - self.angle + 180.0) % 360.0 - 180.0
        return self.angle + d

    def contacts(self):
        d = self.data
        for k in range(d.ncon):
            c = d.contact[k]
            yield c.geom1, c.geom2, c.dist

    def held_objects(self):
        touching = {}
        for g1, g2, dist in self.contacts():
            for a, b in ((g1, g2), (g2, g1)):
                if a in self.finger_geoms and b in self.cube_geoms and dist < 0.001:
                    touching.setdefault(self.cube_geoms[b], set()).add(a)
        return {i for i, f in touching.items() if len(f) == 2}

    def state(self):
        objs = {}
        for i in self.ids:
            p = self.obj_pos(i)
            objs[i] = {"color": self.scene["objects"][i]["color"], "position": [round(float(v), 4) for v in p]}
        return {
            "objects": objs,
            "gripper": {
                "position": [round(float(v), 4) for v in self.tcp_pos()],
                "angle_deg": round(float(self.measured_angle()), 2),
                "is_open": bool(self.is_open),
            },
        }

    # ------------------------------------------------------------------ IK
    def _ik(self, pos, angle, q_init, iters=60, tol=1e-5):
        m, d = self.model, self.kin
        r_t = target_rotation(angle)
        q = q_init.copy()
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        lam = 1e-3
        for _ in range(iters):
            d.qpos[self.arm_qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self.tcp]
            r = d.site_xmat[self.tcp].reshape(3, 3)
            ep = pos - p
            er = 0.5 * (np.cross(r[:, 0], r_t[:, 0]) + np.cross(r[:, 1], r_t[:, 1]) + np.cross(r[:, 2], r_t[:, 2]))
            e = np.concatenate([ep, er])
            if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 1e-4:
                return q
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp)
            J = np.vstack([jacp[:, self.arm_dadr], jacr[:, self.arm_dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.arm_range[:, 0], self.arm_range[:, 1])
        d.qpos[self.arm_qadr] = q
        mujoco.mj_kinematics(m, d)
        p = d.site_xpos[self.tcp]
        r = d.site_xmat[self.tcp].reshape(3, 3)
        er = 0.5 * (np.cross(r[:, 0], r_t[:, 0]) + np.cross(r[:, 1], r_t[:, 1]) + np.cross(r[:, 2], r_t[:, 2]))
        if np.linalg.norm(pos - p) < REACH_TOL * 0.2 and np.linalg.norm(er) < 2e-3:
            return q
        return None

    # ------------------------------------------------------------------ stepping
    def _step(self, q_target=None):
        self.steps += 1
        if self.steps > MAX_SIM_STEPS:
            raise RobotError("program takes too long")
        if q_target is not None:
            self.data.qvel[self.arm_dadr] = (q_target - self.q) / TIMESTEP
            self.q = q_target
        else:
            self.data.qvel[self.arm_dadr] = 0.0
        self.data.qpos[self.arm_qadr] = self.q
        mujoco.mj_step(self.model, self.data)
        # stiff arm: hold the commanded joint positions exactly
        self.data.qpos[self.arm_qadr] = self.q
        self.data.qvel[self.arm_dadr] = 0.0

    def _cube_speed(self):
        v = 0.0
        for i in self.ids:
            b = self.body[i]
            jid = self.model.body_jntadr[b]
            da = self.model.jnt_dofadr[jid]
            v = max(v, float(np.linalg.norm(self.data.qvel[da:da + 6])))
        return v

    def _settle(self, max_steps=1500, min_steps=20):
        for k in range(max_steps):
            self._step()
            if k >= min_steps and self._cube_speed() < 2e-3:
                break

    def _check_collisions(self, held, ref_pos):
        # A closed, empty gripper may push cubes on purpose (fingertips only).
        pushing = (not self.is_open) and not held
        for g1, g2, dist in self.contacts():
            if dist > -2e-4:
                continue
            for a, b in ((g1, g2), (g2, g1)):
                if a in self.robot_geoms:
                    if b == self.table:
                        raise RobotError("collision: the gripper/arm hit the table")
                    if b in self.cube_geoms and self.cube_geoms[b] not in held:
                        if pushing and a in self.finger_geoms:
                            continue
                        raise RobotError(f"collision: the gripper/arm hit cube {self.cube_geoms[b]}")
        if pushing:
            return
        self._check_clearance(held)
        for i in self.ids:
            if i in held:
                continue
            if np.linalg.norm(self.obj_pos(i) - ref_pos[i]) > DISPLACE_TOL:
                raise RobotError(f"collision: cube {i} was pushed")

    def _check_clearance(self, exclude):
        """Fingers must keep >= CLEARANCE from every cube that is not held/excluded
        (a finger grazing a neighbour counts as a collision)."""
        for i, g in self.geom.items():
            if i in exclude:
                continue
            for f in self.finger_geoms:
                d = mujoco.mj_geomDistance(self.model, self.data, f, g, CLEARANCE * 5, None)
                if d < CLEARANCE:
                    raise RobotError(f"collision: a gripper finger hit cube {i} "
                                     "(rotate the gripper so the fingers point to free sides)")

    # ------------------------------------------------------------------ API
    def move_to(self, x, y, z):
        tgt = np.array([float(x), float(y), float(z)])
        if not np.all(np.isfinite(tgt)):
            raise RobotError("invalid target")
        for k in range(3):
            if tgt[k] < WORKSPACE[k, 0] - WS_EPS or tgt[k] > WORKSPACE[k, 1] + WS_EPS:
                raise RobotError(f"target {tgt.round(4).tolist()} is outside the workspace")
        self._cartesian(tgt, self.angle)

    def rotate_gripper(self, angle_deg):
        a = float(angle_deg)
        if not (0.0 - 1e-9 <= a <= 180.0 + 1e-9):
            raise RobotError("rotate_gripper angle must be between 0 and 180 degrees")
        self._cartesian(self.tcp_pos(), a)

    def _cartesian(self, tgt, angle):
        held = self.held_objects()
        ref_pos = {i: self.obj_pos(i) for i in self.ids}
        p0 = self.tcp_pos()
        a0 = self.angle
        dist = float(np.linalg.norm(tgt - p0))
        T = max(dist / SPEED, abs(angle - a0) / ROT_SPEED, 0.05)
        n = max(int(math.ceil(T / TIMESTEP)), 1)
        q = self.q.copy()
        for k in range(1, n + 1):
            s = k / n
            s = 3 * s * s - 2 * s * s * s  # smooth start/stop
            p = p0 + s * (tgt - p0)
            a = a0 + s * (angle - a0)
            qn = self._ik(p, a, q)
            if qn is None:
                raise RobotError(f"target {tgt.round(4).tolist()} is not reachable")
            q = qn
            self._step(q)
            self._check_collisions(held, ref_pos)
        self.angle = angle
        for _ in range(25):  # let the held object follow / damp
            self._step()
        self._check_collisions(held, ref_pos)
        if np.linalg.norm(self.tcp_pos() - tgt) > REACH_TOL:
            raise RobotError("target not reached")

    def _drive_fingers(self, target):
        self.data.ctrl[:] = target
        prev = self.data.qpos[self.finger_qadr].copy()
        for k in range(600):
            self._step()
            cur = self.data.qpos[self.finger_qadr].copy()
            if k > 30 and np.max(np.abs(cur - prev)) < 2e-6:
                break
            prev = cur

    def open_gripper(self):
        released = self.held_objects()
        self._drive_fingers(FINGER_OPEN)
        self._check_clearance(released)
        self.is_open = True
        self._settle()

    def close_gripper(self):
        self._drive_fingers(0.0)
        self.is_open = False
        self._settle(max_steps=300)

    def finish(self):
        self._settle()


class Robot:
    """The object handed to generated code: exactly the documented API."""
    __slots__ = ("_s",)

    def __init__(self, sim: Sim):
        object.__setattr__(self, "_s", sim)

    def get_object_position(self, id):
        s = object.__getattribute__(self, "_s")
        key = str(id)
        if key not in s.body:
            raise RobotError(f"unknown object id {id!r}")
        return s.obj_pos(key)

    def get_gripper_position(self):
        return object.__getattribute__(self, "_s").tcp_pos()

    def move_to(self, x, y, z):
        object.__getattribute__(self, "_s").move_to(x, y, z)

    def open_gripper(self):
        object.__getattribute__(self, "_s").open_gripper()

    def close_gripper(self):
        object.__getattribute__(self, "_s").close_gripper()

    def rotate_gripper(self, angle_deg):
        object.__getattribute__(self, "_s").rotate_gripper(angle_deg)


# ---------------------------------------------------------------------- sandbox
_SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in ["abs", "min", "max", "range", "len", "round", "float", "int", "list", "tuple", "dict", "set",
              "enumerate", "zip", "sorted", "sum", "reversed", "any", "all", "map", "filter", "str", "bool",
              "True", "False", "None", "ValueError", "Exception", "isinstance"]
}


class CodeError(Exception):
    pass


def validate_code(code: str) -> ast.Module:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeError(f"syntax error: {e}")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != "run":
        raise CodeError("answer must be exactly one function `run(robot)`")
    args = tree.body[0].args
    if [a.arg for a in args.args] != ["robot"]:
        raise CodeError("run must take exactly one argument `robot`")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            raise CodeError("imports/global statements are not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise CodeError("access to private attributes is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise CodeError("dunder names are not allowed")
        if isinstance(node, ast.Name) and node.id in ("open", "exec", "eval", "compile", "getattr", "setattr",
                                                     "globals", "locals", "vars", "input", "breakpoint"):
            raise CodeError(f"`{node.id}` is not allowed")
    return tree


def execute(scene: dict, code: str):
    """Run `code` against a fresh simulation. Returns (final_state, error_or_None, sim)."""
    sim = Sim(scene)
    try:
        validate_code(code)
        ns = {"__builtins__": _SAFE_BUILTINS, "np": np, "math": math}
        exec(compile(code, "<assistant_answer>", "exec"), ns)
        ns["run"](Robot(sim))
        sim.finish()
        return sim.state(), None, sim
    except (RobotError, CodeError) as e:
        return sim.state(), str(e), sim
    except Exception as e:  # bug in the generated code
        return sim.state(), f"{type(e).__name__}: {e}", sim
