"""Visual-only floor for the rendered frames: an endless plane with a grid, lines aligned with the world origin.

The physics never sees this. sim_worker.py steps the model of sim/scene.py exactly as the dataset does; render_model() builds a second model
that is only ever drawn (its joint positions are copied from the physics model, it is never stepped). sim/ stays byte-identical to the dataset's tools/.
If you change GRID_CELL, also change the caption in static/index.html."""
import re

import mujoco
import numpy as np
from scene import build_xml

import lara_view

GRID_CELL = 0.10                                 # [m] one grid cell, chosen over 4 cm (see README)
TEX = 2000                                       # one texture tile = 1 m x 1 m (MuJoCo: texrepeat 1 = 2 m on a plane, hence "2 2"), so 1 pixel = 0.5 mm
FLOOR, MINOR, MAJOR = (222, 216, 204), (190, 183, 170), (150, 142, 128)   # a light warm floor, so the coloured cubes stand out; lines every cell / every metre
# What the frames look like. All of it is visual: it only exists in the drawing model.
# offsamples = multisampling of the offscreen buffer (smooth edges), shadowsize = shadow map, offwidth/offheight = the largest frame sim_worker.py may ask for.
VISUAL = """<visual>
  <global offwidth="{w}" offheight="{h}"/>
  <quality offsamples="4" shadowsize="4096"/>
  <map shadowclip="1.6" shadowscale="0.6" znear="0.02"/>
  <rgba haze="0.88 0.90 0.93 1"/>
</visual>
"""
ASSET = f"""<asset>
  <texture name="p2r_grid" type="2d" builtin="flat" rgb1="0.87 0.85 0.8" width="{TEX}" height="{TEX}"/>
  <texture name="p2r_sky" type="skybox" builtin="gradient" rgb1="0.93 0.95 0.98" rgb2="0.76 0.80 0.87" width="512" height="512"/>
  <material name="p2r_grid" texture="p2r_grid" texuniform="true" texrepeat="2 2" specular="0.05" shininess="0.1" reflectance="0"/>
  <material name="p2r_cube" specular="0.45" shininess="0.5" reflectance="0"/>
</asset>
"""
# A warm key light from the front left that casts the shadows, a cool fill from the other side without shadows (it lifts the shaded cube faces)
LIGHTS = """<light name="p2r_key" pos="-0.6 1.2 2.4" dir="0.30 -0.40 -1" directional="true" castshadow="true" diffuse="0.36 0.34 0.32" specular="0.30 0.30 0.28"/>
    <light name="p2r_fill" pos="1.6 -1.2 1.6" dir="-0.5 0.4 -0.6" directional="true" castshadow="false" diffuse="0.12 0.13 0.15" specular="0.05 0.05 0.06"/>
"""


def _grid():
    """Lines sit on the tile's edges and at every GRID_CELL, so on the floor they fall on multiples of GRID_CELL from the world origin."""
    t = np.empty((TEX, TEX, 3), np.uint8)
    t[:] = FLOOR
    step = round(GRID_CELL * TEX)
    for k, colour, half in [(k, MINOR, 1) for k in range(step, TEX, step)] + [(0, MAJOR, 3)]:
        rows = np.arange(k - half, k + half + 1) % TEX
        t[rows, :] = colour
        t[:, rows] = colour
    return t


def render_model(scene, width=1280, height=720):
    """The scene's model with the table's look replaced by the grid plane (size 0 0 = infinite). For drawing only, never step it."""
    xml, n = re.subn(r'(<geom name="table" type="plane") size="[^"]*" rgba="[^"]*"', r'\1 size="0 0 1" material="p2r_grid"', build_xml(scene))
    if n != 1:
        return mujoco.MjModel.from_xml_string(build_xml(scene))        # scene.py changed: plain floor rather than no film
    xml = re.sub(r'(<geom name="obj_[^"]*_geom" type="box")', r'\1 material="p2r_cube"', xml)     # cube colours stay the scene's own rgba; the material only adds a gloss
    xml = xml.replace("<worldbody>", VISUAL.format(w=width, h=height) + ASSET + "<worldbody>\n    " + LIGHTS, 1)
    if lara_view.enabled():
        xml = lara_view.swap_arm(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    model.tex_data[model.tex_adr[0]:model.tex_adr[0] + TEX * TEX * 3] = _grid().ravel()
    return model
