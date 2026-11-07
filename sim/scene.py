"""MuJoCo scene builder for the Prompt2Robo tabletop.

World frame (as in system_prompt.md): origin at the robot base, +x front,
+y left, +z up, table top is the plane z = 0.

The arm is a generic 6-DOF arm (no LARA 5 model is available in MuJoCo
Menagerie): base yaw, shoulder pitch, elbow pitch, forearm roll, wrist pitch,
tool roll, with link lengths chosen so the TCP covers x 0..0.8,
y -0.4..0.4, z 0..0.8 with the tool pointing down (LARA 5-like reach).
It carries a two-finger parallel gripper (max opening 0.08 m). The TCP site
sits between the fingertip pads.
"""
from __future__ import annotations

import numpy as np

CUBE = 0.04          # cube edge length [m]
CUBE_MASS = 0.05     # [kg], wooden block
TIMESTEP = 0.002
# Collision boxes are 0.5 mm smaller per side in x/y (rounded edges / manufacturing
# tolerance), so cubes placed exactly 0.04 m apart just meet instead of overlapping.
XY_SLACK = 0.0005

ARM_JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
FINGER_JOINTS = ["fl", "fr"]
FINGER_OPEN = 0.04   # inner face of each finger 0.04 m from the TCP axis -> 0.08 m opening

# collision bits: robot geoms collide only with the environment, not with themselves
ROBOT_COL = 'contype="2" conaffinity="1"'
ENV_COL = 'contype="1" conaffinity="3"'


def _cube_xml(oid: str, color: str, pos) -> str:
    rgba = COLORS.get(color, "0.6 0.6 0.6 1")
    x, y, z = pos
    return (
        f'<body name="obj_{oid}" pos="{x} {y} {z}">'
        f'<freejoint name="obj_{oid}_free"/>'
        f'<geom name="obj_{oid}_geom" type="box" size="{CUBE/2 - XY_SLACK} {CUBE/2 - XY_SLACK} {CUBE/2}" '
        f'mass="{CUBE_MASS}" rgba="{rgba}" friction="1.0 0.01 0.001" condim="4" {ENV_COL}/>'
        f"</body>"
    )


COLORS = {
    "red": "0.85 0.1 0.1 1", "blue": "0.1 0.2 0.85 1", "green": "0.1 0.7 0.2 1",
    "yellow": "0.95 0.85 0.1 1", "orange": "1 0.55 0 1", "purple": "0.55 0.2 0.7 1",
    "white": "0.95 0.95 0.95 1", "black": "0.1 0.1 0.1 1", "pink": "1 0.5 0.7 1",
    "brown": "0.5 0.3 0.1 1", "gray": "0.5 0.5 0.5 1", "grey": "0.5 0.5 0.5 1",
    "cyan": "0.1 0.8 0.85 1",
}


def build_xml(scene: dict) -> str:
    cubes = "".join(_cube_xml(oid, o["color"], o["position"]) for oid, o in scene["objects"].items())
    return f"""
<mujoco model="prompt2robo">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{TIMESTEP}" integrator="implicitfast" cone="elliptic" impratio="10" gravity="0 0 -9.81"/>
  <default>
    <geom solref="0.004 1" solimp="0.95 0.99 0.001"/>
  </default>
  <worldbody>
    <geom name="table" type="plane" size="2 2 0.1" rgba="0.8 0.75 0.65 1" friction="0.8 0.01 0.001" {ENV_COL}/>
    <body name="base" pos="0 0 0">
      <geom name="base_geom" type="cylinder" fromto="0 0 0 0 0 0.12" size="0.07" rgba="0.3 0.3 0.3 1" {ROBOT_COL}/>
      <body name="link1" pos="0 0 0.12" gravcomp="1">
        <joint name="j1" axis="0 0 1" range="-3.1416 3.1416"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.23" size="0.05" mass="3" {ROBOT_COL}/>
        <body name="link2" pos="0 0 0.23" gravcomp="1">
          <joint name="j2" axis="0 1 0" range="-2.0 2.0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.55" size="0.04" mass="2" {ROBOT_COL}/>
          <body name="link3" pos="0 0 0.55" gravcomp="1">
            <joint name="j3" axis="0 1 0" range="-2.9 2.9"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.30" size="0.035" mass="1.5" {ROBOT_COL}/>
            <body name="link4" pos="0 0 0.30" gravcomp="1">
              <joint name="j4" axis="0 0 1" range="-3.1416 3.1416"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.03" mass="1" {ROBOT_COL}/>
              <body name="link5" pos="0 0 0.25" gravcomp="1">
                <joint name="j5" axis="0 1 0" range="-2.4 2.4"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.06" size="0.03" mass="0.5" {ROBOT_COL}/>
                <body name="hand" pos="0 0 0.06" gravcomp="1">
                  <joint name="j6" axis="0 0 1" range="-6.3 6.3"/>
                  <geom name="palm" type="box" pos="0 0 0.022" size="0.02 0.06 0.008" mass="0.3" rgba="0.2 0.2 0.2 1" {ROBOT_COL}/>
                  <geom type="cylinder" fromto="0 0 0 0 0 0.014" size="0.025" mass="0.2" {ROBOT_COL}/>
                  <site name="tcp" pos="0 0 0.10" size="0.004" rgba="1 0 0 1"/>
                  <body name="finger_l" pos="0 0 0" gravcomp="1">
                    <joint name="fl" type="slide" axis="0 1 0" range="0 {FINGER_OPEN}" damping="20"/>
                    <geom name="finger_l_geom" type="box" pos="0 0.005 0.071" size="0.01 0.005 0.041" mass="0.05"
                          friction="1.2 0.02 0.001" condim="4" rgba="0.4 0.4 0.4 1" {ROBOT_COL}/>
                  </body>
                  <body name="finger_r" pos="0 0 0" gravcomp="1">
                    <joint name="fr" type="slide" axis="0 -1 0" range="0 {FINGER_OPEN}" damping="20"/>
                    <geom name="finger_r_geom" type="box" pos="0 -0.005 0.071" size="0.01 0.005 0.041" mass="0.05"
                          friction="1.2 0.02 0.001" condim="4" rgba="0.4 0.4 0.4 1" {ROBOT_COL}/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    {cubes}
  </worldbody>
  <actuator>
    <position name="act_fl" joint="fl" kp="600" forcerange="-15 15" ctrlrange="0 {FINGER_OPEN}"/>
    <position name="act_fr" joint="fr" kp="600" forcerange="-15 15" ctrlrange="0 {FINGER_OPEN}"/>
  </actuator>
</mujoco>
"""


def target_rotation(angle_deg: float) -> np.ndarray:
    """Rotation matrix of the TCP frame: z points down, finger axis = local y.

    angle 0 -> fingers along world y, angle 90 -> fingers along world x.
    """
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    r0 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
    return rz @ r0
