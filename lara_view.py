"""Draws the NEURA LARA 5 of lara5/ instead of the simulator's generic arm. Drawing only: the physics, the robot API and the checker stay the generic ones
of sim/, which the dataset was verified with. Every frame the LARA 5 is solved (inverse kinematics) to the gripper pose of the physics model, so it follows
the same motion. Gripper as in Stack Lab: a generic parallel gripper on the flange. Very far targets are outside the real arm's reach; there the
drawn gripper falls short of the true one (see README)."""
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import lara_assets

HERE = Path(__file__).resolve().parent / "lara5"
JOINTS = [f"joint{i}" for i in range(1, 7)]
PAD = 0.005                                      # inner face of a closed finger, from the TCP axis
SEEDS = [(0.0, 0.5, 1.0, 0.0, 1.6, 0.0), (0.0, -0.3, 1.8, 0.0, 1.6, 0.0), (0.0, 0.5, 1.0, 0.0, -1.6, 0.0), (0.0, -0.3, 1.8, 0.0, -1.6, 0.0)]


def enabled():
    return os.environ.get("P2R_ARM", "lara5") != "generic" and lara_assets.ready()


def swap_arm(xml):
    """Physics scene xml -> the same scene with the generic arm replaced by the LARA 5 (for drawing only)."""
    root, lara = ET.fromstring(xml), ET.parse(HERE / "lara5.xml").getroot()
    world = root.find("worldbody")
    world.remove(world.find("body[@name='base']"))
    root.remove(root.find("actuator"))
    comp = root.find("compiler")
    if comp is None:
        comp = ET.Element("compiler")
        root.insert(0, comp)
    comp.set("angle", "radian"); comp.set("autolimits", "true")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    for m in lara.findall("asset/mesh"):
        ET.SubElement(asset, "mesh", name=m.get("name"), file=str(HERE / "assets" / m.get("file")), smoothnormal="true")     # smooth shading instead of visible facets
    ET.SubElement(asset, "material", name="lara_shell", specular="0.5", shininess="0.6", reflectance="0.04")      # white housings: soft highlights
    ET.SubElement(asset, "material", name="lara_joint", specular="0.35", shininess="0.4")                          # grey joint rings and gripper
    arm = lara.find("worldbody/body[@name='base']")
    for parent in arm.iter("body"):
        for g in parent.findall("geom"):
            if g.get("name", "").startswith("collision"):
                parent.remove(g)
    for g in arm.iter("geom"):
        if g.get("type") == "mesh":
            g.set("material", "lara_shell" if g.get("rgba", "").startswith("0.85") else "lara_joint")
    g = ET.SubElement(arm.find(".//body[@name='link6']"), "body", name="lara_gripper")
    box = dict(type="box", contype="0", conaffinity="0", group="2", mass="0.1", material="lara_joint")
    ET.SubElement(g, "geom", name="lara_palm", size=".035 .055 .018", pos="0 0 .02", rgba=".17 .20 .24 1", **box)
    ET.SubElement(g, "site", name="lara_tcp", pos="0 0 .09", size=".003", rgba="0 0 0 0")
    for sign, name in ((1, "lara_fl"), (-1, "lara_fr")):
        b = ET.SubElement(g, "body", name=name, pos=f"0 {sign * .009} 0")
        ET.SubElement(b, "joint", name=name, type="slide", axis=f"0 {sign} 0", range="0 .04")
        ET.SubElement(b, "geom", name=name + "_pad", size=".018 .004 .03", pos="0 0 .077", rgba=".24 .27 .30 1", **box)
    world.append(arm)
    return ET.tostring(root, encoding="unicode")


class Follow:
    """Copies the cubes and solves the LARA 5 to the physics gripper pose, in the drawing model."""

    def __init__(self, model, sim):
        self.m, m = model, model
        self.adr = np.array([m.jnt_qposadr[m.joint(j).id] for j in JOINTS])
        self.dof = np.array([m.jnt_dofadr[m.joint(j).id] for j in JOINTS])
        self.lim = np.array([m.jnt_range[m.joint(j).id] for j in JOINTS])
        self.fingers = [m.jnt_qposadr[m.joint(j).id] for j in ("lara_fl", "lara_fr")]
        self.site = m.site("lara_tcp").id
        self.cubes = [(m.jnt_qposadr[m.joint(f"obj_{i}_free").id], sim.model.jnt_qposadr[sim.model.joint(f"obj_{i}_free").id]) for i in sim.ids]
        self.q = None
        self.scratch = mujoco.MjData(m)

    def _solve(self, q, pos, rot, iters):
        m, d = self.m, self.scratch
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        err = 1e9
        for _ in range(iters):
            d.qpos[self.adr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p, r = d.site_xpos[self.site], d.site_xmat[self.site].reshape(3, 3)
            e = np.concatenate([pos - p, 0.5 * sum(np.cross(r[:, k], rot[:, k]) for k in range(3))])
            err = float(np.linalg.norm(e[:3]) + 0.1 * np.linalg.norm(e[3:]))
            if err < 2e-4:
                break
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            J = np.vstack([jp[:, self.dof], jr[:, self.dof]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            n = np.linalg.norm(dq)
            q = np.clip(q + (dq * 0.4 / n if n > 0.4 else dq), self.lim[:, 0], self.lim[:, 1])
        return q, err

    def sync(self, data, sim):
        for dst, src in self.cubes:
            data.qpos[dst:dst + 7] = sim.data.qpos[src:src + 7]
        pos, rot = sim.tcp_pos(), sim.data.site_xmat[sim.tcp].reshape(3, 3).copy()
        if self.q is None:
            yaw = math.atan2(pos[1], pos[0])
            tries = [self._solve(np.array((yaw,) + s[1:]), pos, rot, 200) for s in SEEDS]
            self.q = min(tries, key=lambda t: t[1])[0]
        self.q, self.err = self._solve(self.q, pos, rot, 25)
        data.qpos[self.adr] = self.q
        opening = float(np.mean(np.abs(sim.data.qpos[sim.finger_qadr])))
        for a in self.fingers:
            data.qpos[a] = min(max(opening - PAD, 0.0), 0.04)
