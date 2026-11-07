"""Builds the LARA 5 meshes on this computer on first use, the way Stack Lab does (https://github.com/carlo2004/Simulation-Neura-LARA-5, MIT, see lara5/LICENSE).

The arm's geometry comes from the public RoboDK file below. Its licence does not cover redistribution, so the meshes are not shipped with the app:
they are downloaded and checked here once, then kept in lara5/assets. Without them the app simply draws the generic arm. See lara5/NOTICE.md."""
import hashlib
import struct
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent / "lara5"
URL = "https://cdn.robodk.com/downloads-library/library-robots/Neura-LARA-5.robot"
SHA = "4dc8df5ff0e8396c86a2388f9f0a582ba3d8c34c9a310fb8f98d290a25dee3c3"


def ready():
    try:
        return all((HERE / "assets" / m.get("file")).is_file() for m in ET.parse(HERE / "lara5.xml").findall("asset/mesh"))
    except (OSError, ET.ParseError):
        return False


def _links(raw):
    """Surfaces (rgba, vertices) per link of the .robot file; layout as decoded in Stack Lab's stacklab/sim/convert.py."""
    offset, links = 32008, []
    for _ in range(7):
        count = struct.unpack_from("<Q", raw, offset)[0]
        offset += 8
        surfaces = []
        for _ in range(count):
            nt, *rgba = struct.unpack_from("<Q4f", raw, offset)
            offset += 24
            ar = np.ndarray(nt * 3, dtype=[("p", "<f4", 3), ("n", "<u2")], buffer=raw, offset=offset)
            offset += nt * 42
            verts = ar["p"].copy().reshape(nt, 3, 3)
            packed = ar["n"].astype(float)                     # packed spherical normals orient the triangles
            az, polar = np.floor(packed / 256) * 2 * np.pi / 255, np.mod(packed, 256) * np.pi / 255
            normals = np.array([np.cos(az) * np.sin(polar), np.sin(az) * np.sin(polar), np.cos(polar)]).T.reshape(nt, 3, 3).mean(1)
            flip = np.sum(normals * np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0]), axis=1) < 0
            verts[flip] = verts[flip][:, [0, 2, 1]]
            surfaces.append((rgba, verts.reshape(-1, 3)))
        offset += 16
        links.append(surfaces)
    if offset != len(raw):
        raise ValueError("unexpected file layout")
    return links


def build(data):
    if hashlib.sha256(data).hexdigest() != SHA:
        raise ValueError("the downloaded robot file has an unexpected checksum")
    raw = zlib.decompress(data[4:])
    out = HERE / "assets"
    out.mkdir(exist_ok=True)
    for i, surfaces in enumerate(_links(raw)):
        for j, colour in enumerate(sorted({tuple(np.round(c, 5)) for c, _ in surfaces})):
            tris = np.concatenate([v for c, v in surfaces if tuple(np.round(c, 5)) == colour]) / 1000
            vertices, ids = np.unique(tris, axis=0, return_inverse=True)
            with (out / f"link{i}_{j}.obj").open("w") as f:
                f.writelines("v " + " ".join(f"{x:.8f}" for x in v) + "\n" for v in vertices)
                f.writelines("f " + " ".join(map(str, t)) + "\n" for t in ids.reshape(-1, 3) + 1)


def ensure(log=print):
    """True when the meshes are there. Never raises: a failure only means the generic arm is drawn."""
    if ready():
        return True
    try:
        log("downloading the LARA 5 model (once) ...")
        req = urllib.request.Request(URL, headers={"User-Agent": "Prompt2Robo-app"})
        with urllib.request.urlopen(req, timeout=60) as r:
            build(r.read(25 * 1024 * 1024 + 1))
    except Exception as e:
        log(f"LARA 5 model not available, drawing the generic arm: {type(e).__name__}: {e}")
    return ready()


if __name__ == "__main__":
    sys.exit(0 if ensure() else 1)
