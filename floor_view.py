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
TEX = 1000                                       # one texture tile = 1 m x 1 m (MuJoCo: texrepeat 1 = 2 m on a plane, hence "2 2"), so 1 pixel = 1 mm
FLOOR, MINOR, MAJOR = (204, 191, 166), (132, 122, 102), (92, 84, 70)      # floor colour as in scene.py; lines every cell / every metre
ASSET = f"""<asset>
  <texture name="p2r_grid" type="2d" builtin="flat" rgb1="0.8 0.75 0.65" width="{TEX}" height="{TEX}"/>
  <material name="p2r_grid" texture="p2r_grid" texuniform="true" texrepeat="2 2" reflectance="0"/>
</asset>
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


def render_model(scene):
    """The scene's model with the table's look replaced by the grid plane (size 0 0 = infinite). For drawing only, never step it."""
    xml, n = re.subn(r'(<geom name="table" type="plane") size="[^"]*" rgba="[^"]*"', r'\1 size="0 0 1" material="p2r_grid"', build_xml(scene))
    if n != 1:
        return mujoco.MjModel.from_xml_string(build_xml(scene))        # scene.py changed: plain floor rather than no film
    xml = xml.replace("<worldbody>", ASSET + "<worldbody>", 1)
    if lara_view.enabled():
        xml = lara_view.swap_arm(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    model.tex_data[model.tex_adr[0]:model.tex_adr[0] + TEX * TEX * 3] = _grid().ravel()
    return model
