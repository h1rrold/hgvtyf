#!/usr/bin/env python3
"""Builds the 3D mini Ender Dragon: two item models (wings up / wings down), one painted atlas texture.

The whole dragon fits inside ONE block (16 x 16 x 16 model units): head tip to tail tip = 16, wingspan = 16.
The head points toward +Z (south) so it follows the display entity's facing direction.

Run:  python3 tools/gen_dragon.py            (writes the pack files)
      python3 tools/gen_dragon.py --preview  (also writes preview PNGs to /tmp)
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resourcepack", "assets", "weapons")
TEX = os.path.join(ROOT, "textures", "item")
MOD = os.path.join(ROOT, "models", "item")
ITM = os.path.join(ROOT, "items")

PPU = 8          # texture pixels per model unit
ATLAS = 256      # atlas size (px)
rng = np.random.default_rng(42)

FACES = ("north", "south", "east", "west", "up", "down")


class Box:
    def __init__(self, x1, y1, z1, x2, y2, z2, mat, decals=None, name=""):
        self.f = (x1, y1, z1)
        self.t = (x2, y2, z2)
        self.mat = mat
        self.decals = decals or {}
        self.name = name

    @property
    def size(self):
        return tuple(self.t[i] - self.f[i] for i in range(3))

    def face_size_units(self, face):
        sx, sy, sz = self.size
        return {"north": (sx, sy), "south": (sx, sy), "east": (sz, sy), "west": (sz, sy),
                "up": (sx, sz), "down": (sx, sz)}[face]

    def face_px(self, face):
        w, h = self.face_size_units(face)
        return max(2, int(round(w * PPU))), max(2, int(round(h * PPU)))

    def point(self, face, uf, vf):
        """World position of face-local fraction (uf, vf). Matches Minecraft's default UV orientation."""
        (x1, y1, z1), (x2, y2, z2) = self.f, self.t
        sx, sy, sz = self.size
        if face == "north":
            return (x2 - uf * sx, y2 - vf * sy, z1)
        if face == "south":
            return (x1 + uf * sx, y2 - vf * sy, z2)
        if face == "east":
            return (x2, y2 - vf * sy, z2 - uf * sz)
        if face == "west":
            return (x1, y2 - vf * sy, z1 + uf * sz)
        if face == "up":
            return (x1 + uf * sx, y2, z1 + vf * sz)
        return (x1 + uf * sx, y1, z2 - vf * sz)  # down


def mirror(b):
    """Mirror a box across the dragon's centre plane (x -> 16 - x)."""
    nb = Box(16 - b.t[0], b.f[1], b.f[2], 16 - b.f[0], b.t[1], b.t[2], b.mat, name=b.name + "_r")
    swap = {"east": "west", "west": "east"}
    nb.decals = {swap.get(k, k): v for k, v in b.decals.items()}
    return nb


# ------------------------------------------------------------------------------------- the dragon

def build(L):
    """L = wing lift: +1 wings up, -1 wings down."""
    boxes = []
    add = boxes.append

    def wy(x):  # wing height at x (left wing, x from 6 at the shoulder to 0 at the tip)
        return 10.3 + L * 0.34 * (6.0 - x)

    # ---- head (points to +Z)
    add(Box(7.0, 8.95, 13.6, 9.0, 10.05, 16.0, "scale2", name="snout"))
    add(Box(7.15, 8.1, 13.4, 8.85, 8.95, 15.8, "scale", name="lowerjaw"))
    add(Box(7.2, 8.95, 13.7, 8.8, 9.05, 15.7, "mouth", name="mouth"))
    for tx in (7.1, 8.5):
        for tz in (14.2, 15.2):
            add(Box(tx, 8.7, tz, tx + 0.4, 9.15, tz + 0.35, "tooth", name="tooth"))
    add(Box(7.15, 10.05, 15.0, 7.7, 10.5, 15.6, "scale2", name="nostril"))
    add(Box(8.3, 10.05, 15.0, 8.85, 10.5, 15.6, "scale2", name="nostril"))
    eye_l = {"west": [(0.52, 0.28, 0.90, 0.52, "eye")]}
    eye_r = {"east": [(0.10, 0.28, 0.48, 0.52, "eye")]}
    skull = Box(6.6, 8.3, 11.9, 9.4, 10.4, 13.9, "scale2", name="skull")
    skull.decals = {"west": [(0.52, 0.30, 0.90, 0.55, "eye")], "east": [(0.10, 0.30, 0.48, 0.55, "eye")]}
    add(skull)
    add(Box(6.9, 10.4, 12.2, 9.1, 10.8, 13.4, "scale", name="brow"))
    # horns (swept back)
    for side in (0, 1):
        x0 = 6.45 if side == 0 else 9.05
        add(Box(x0, 10.2, 12.2, x0 + 0.5, 11.1, 12.8, "horn", name="horn"))
        add(Box(x0, 10.9, 11.5, x0 + 0.5, 11.7, 12.2, "horn", name="horn"))
        add(Box(x0 + 0.05, 11.5, 11.0, x0 + 0.45, 12.1, 11.6, "horn", name="horn"))
    # cheek spikes
    add(Box(6.3, 8.4, 12.2, 6.65, 9.0, 12.9, "horn", name="cheek"))
    add(Box(9.35, 8.4, 12.2, 9.7, 9.0, 12.9, "horn", name="cheek"))

    # ---- neck (curving back and up toward the body)
    add(Box(6.9, 8.5, 10.4, 9.1, 10.5, 12.1, "scale", name="neck1"))
    add(Box(6.8, 8.2, 8.9, 9.2, 10.3, 10.5, "scale", name="neck2"))
    add(Box(6.6, 7.9, 7.4, 9.4, 10.1, 9.0, "scale", name="neck3"))
    for z, h in ((11.0, 0.95), (9.6, 0.9), (8.2, 0.9)):
        add(Box(7.75, 10.3, z, 8.25, 10.3 + h, z + 0.55, "spike", name="spine"))

    # ---- body
    add(Box(5.9, 6.7, 4.4, 10.1, 10.0, 8.2, "scale", name="body"))
    add(Box(6.3, 6.5, 4.8, 9.7, 6.75, 8.0, "belly", name="belly"))
    for z in (7.4, 6.2, 5.0):
        add(Box(7.8, 10.0, z, 8.2, 10.95, z + 0.5, "spike", name="spine"))

    # ---- legs (tucked)
    for x in (6.2, 8.8):
        add(Box(x, 5.7, 6.4, x + 1.0, 6.8, 7.7, "scale", name="foreleg"))
        add(Box(x + 0.1, 5.0, 7.0, x + 0.9, 5.8, 8.1, "scale2", name="forefoot"))
        add(Box(x + 0.2, 4.95, 7.9, x + 0.8, 5.3, 8.5, "claw", name="claw"))
        add(Box(x, 5.8, 4.4, x + 1.1, 6.9, 5.7, "scale", name="hindleg"))
        add(Box(x + 0.15, 5.1, 4.7, x + 0.95, 5.9, 5.9, "scale2", name="hindfoot"))

    # ---- tail (tapers, droops)
    add(Box(6.5, 7.2, 2.7, 9.5, 9.5, 4.6, "scale", name="tail1"))
    add(Box(6.85, 7.1, 1.5, 9.15, 9.1, 2.9, "scale", name="tail2"))
    add(Box(7.15, 6.9, 0.5, 8.85, 8.6, 1.7, "scale", name="tail3"))
    add(Box(7.45, 6.7, 0.0, 8.55, 8.0, 0.7, "scale2", name="tail4"))
    for z, y in ((3.4, 9.5), (2.0, 9.1), (0.9, 8.6)):
        add(Box(7.8, y, z, 8.2, y + 0.8, z + 0.45, "spike", name="spine"))

    # ---- left wing (x < 8). Staircase of arm bones + fingers + membrane
    wing = []
    a = wing.append
    a(Box(3.9, wy(3.9) - 0.1, 6.0, 6.1, wy(3.9) + 1.0, 7.5, "wingbone", name="humerus"))
    a(Box(1.5, wy(1.5) - 0.1, 6.0, 4.1, wy(1.5) + 0.9, 7.3, "wingbone", name="forearm"))
    a(Box(0.0, wy(0.0) - 0.05, 6.2, 1.7, wy(0.0) + 0.8, 7.1, "wingbone", name="hand"))
    # fingers (swept back toward the tail, outer finger longest)
    for x, zlen in ((0.1, 0.4), (1.9, 1.9), (3.7, 3.4), (5.35, 4.6)):
        a(Box(x, wy(x + 0.2), zlen, x + 0.55, wy(x + 0.2) + 0.55, 6.3, "wingbone", name="finger"))
    # membrane panels between fingers (thin, slightly overlapping the bones) -> scalloped trailing edge
    for x0, x1, zlen in ((0.5, 2.05, 0.8), (2.4, 3.85, 2.2), (4.2, 5.4, 3.7), (5.7, 6.0, 4.8)):
        xm = (x0 + x1) / 2
        a(Box(x0, wy(xm) + 0.05, zlen, x1, wy(xm) + 0.4, 6.2, "membrane", name="membrane"))
    boxes.extend(wing)
    for b in wing:
        boxes.append(mirror(b))
    return boxes


# ------------------------------------------------------------------------------------- painting

BASE = {
    "scale": (30, 20, 42),
    "scale2": (46, 31, 62),
    "belly": (84, 62, 100),
    "spike": (70, 52, 90),
    "horn": (176, 158, 190),
    "wingbone": (52, 36, 70),
    "membrane": (26, 14, 38),
    "mouth": (118, 26, 76),
    "tooth": (238, 232, 242),
    "claw": (208, 196, 214),
}


def clamp(c):
    return tuple(int(max(0, min(255, v))) for v in c)


def paint_face(box, face, w, h):
    """Returns an (h, w, 4) uint8 array for one face."""
    base = np.array(BASE[box.mat], dtype=float)
    img = np.zeros((h, w, 4), dtype=float)
    yy, xx = np.mgrid[0:h, 0:w]
    img[..., :3] = base
    img[..., 3] = 255
    mat = box.mat

    if mat in ("scale", "scale2", "wingbone"):
        # staggered scale pattern: 3x2 px scales, lighter rim on the upper edge of each scale
        row = yy // 2
        off = (row % 2) * 1
        sx = (xx + off) % 3
        sy = yy % 2
        shade = np.where(sy == 0, 1.22, 0.9) * np.where(sx == 0, 0.9, 1.0)
        img[..., :3] *= shade[..., None]
        # sparse violet sheen on the lit rows
        sheen = (rng.random((h, w)) < 0.10) & (sy == 0)
        img[..., 0] += sheen * 26
        img[..., 2] += sheen * 38
    elif mat == "belly":
        band = np.where(yy % 2 == 0, 1.08, 0.88)
        img[..., :3] *= band[..., None]
    elif mat == "spike":
        t = (yy / max(1, h - 1))  # 0 at top of face (tip) .. 1 at base
        light = 1.0 + (1.0 - t) * 1.1
        img[..., :3] *= light[..., None]
    elif mat == "horn":
        t = (yy / max(1, h - 1))
        img[..., :3] *= (0.85 + 0.3 * (1 - t))[..., None]
        img[..., :3] *= np.where(xx % 3 == 0, 0.9, 1.0)[..., None]
    elif mat == "membrane":
        # dark leathery membrane with purple veins fanning toward the back
        vein = ((xx * 5 + yy * 2) % 11 == 0) | ((xx * 5 - yy * 2) % 13 == 0)
        img[..., :3] = np.where(vein[..., None], np.array([112, 52, 150]), base)
        edge = (xx == 0) | (yy == 0) | (xx == w - 1) | (yy == h - 1)
        img[..., :3] = np.where(edge[..., None], np.array([64, 32, 96]), img[..., :3])
    elif mat in ("mouth", "tooth", "claw"):
        pass

    # soft edge darkening (cheap ambient occlusion)
    ed = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
    ao = np.where(ed == 0, 0.82, 1.0)
    img[..., :3] *= ao[..., None]
    # noise
    img[..., :3] *= (1.0 + (rng.random((h, w)) - 0.5) * 0.10)[..., None]

    # decals (eyes)
    for (u0, v0, u1, v1, kind) in box.decals.get(face, []):
        x0, x1 = int(round(u0 * w)), max(int(round(u1 * w)), int(round(u0 * w)) + 2)
        y0, y1 = int(round(v0 * h)), max(int(round(v1 * h)), int(round(v0 * h)) + 2)
        if kind == "eye":
            img[y0:y1, x0:x1, :3] = (200, 60, 255)
            # bright core
            cy0, cy1 = y0 + max(0, (y1 - y0) // 4), y1 - (y1 - y0) // 4
            cx0, cx1 = x0 + max(0, (x1 - x0) // 4), x1 - (x1 - x0) // 4
            img[cy0:cy1, cx0:cx1, :3] = (255, 200, 255)
            # glow halo
            hy0, hy1 = max(0, y0 - 1), min(h, y1 + 1)
            hx0, hx1 = max(0, x0 - 1), min(w, x1 + 1)
            halo = img[hy0:hy1, hx0:hx1, :3]
            m = np.ones(halo.shape[:2], dtype=bool)
            m[1:-1, 1:-1] = False
            halo[m] = halo[m] * 0.4 + np.array([150, 30, 200]) * 0.6
    return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------------- atlas + model

def pack(boxes):
    """Shelf-pack every face rectangle into the atlas. Returns {(box_index, face): (x, y, w, h)}."""
    rects = []
    for i, b in enumerate(boxes):
        for f in FACES:
            w, h = b.face_px(f)
            rects.append((h, w, i, f))
    rects.sort(reverse=True)
    placed = {}
    x = y = shelf = 0
    for h, w, i, f in rects:
        if x + w > ATLAS:
            x = 0
            y += shelf + 1
            shelf = 0
        if y + h > ATLAS:
            raise RuntimeError("atlas too small")
        placed[(i, f)] = (x, y, w, h)
        x += w + 1
        shelf = max(shelf, h)
    return placed


def make_model(boxes, placed):
    elements = []
    for i, b in enumerate(boxes):
        faces = {}
        for f in FACES:
            x, y, w, h = placed[(i, f)]
            k = 16.0 / ATLAS
            faces[f] = {"uv": [round(x * k, 4), round(y * k, 4), round((x + w) * k, 4), round((y + h) * k, 4)],
                        "texture": "#0"}
        elements.append({"name": b.name, "from": [round(v, 3) for v in b.f], "to": [round(v, 3) for v in b.t],
                         "faces": faces})
    return {
        "texture_size": [ATLAS, ATLAS],
        "textures": {"0": "weapons:item/mini_dragon", "particle": "weapons:item/mini_dragon"},
        "elements": elements,
        "gui_light": "side",
    }


def make_atlas(boxes, placed):
    atlas = np.zeros((ATLAS, ATLAS, 4), dtype=np.uint8)
    for (i, f), (x, y, w, h) in placed.items():
        atlas[y:y + h, x:x + w] = paint_face(boxes[i], f, w, h)
    return atlas


# ------------------------------------------------------------------------------------- preview renderer

NORMAL = {"north": (0, 0, -1), "south": (0, 0, 1), "east": (1, 0, 0), "west": (-1, 0, 0), "up": (0, 1, 0), "down": (0, -1, 0)}
SHADE = {"up": 1.0, "down": 0.5, "north": 0.8, "south": 0.8, "east": 0.6, "west": 0.6}


def render(boxes, placed, atlas, yaw, pitch, size=420, scale=22.0):
    ca, sa = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    # camera looks toward -z' after rotating the world by yaw (around Y) then pitch (around X)
    view = np.array([0.0, 0.0, 1.0])
    pts, cols, deps = [], [], []
    for i, b in enumerate(boxes):
        for f in FACES:
            n = np.array(NORMAL[f], dtype=float)
            # rotate normal for culling
            nx = n[0] * ca + n[2] * sa
            nz = -n[0] * sa + n[2] * ca
            ny = n[1]
            ny2 = ny * cp - nz * sp
            nz2 = ny * sp + nz * cp
            if nz2 < -0.001:
                continue  # facing away
            x, y, w, h = placed[(i, f)]
            tex = atlas[y:y + h, x:x + w]
            us = (np.arange(w) + 0.5) / w
            vs = (np.arange(h) + 0.5) / h
            for jv, v in enumerate(vs):
                for ju, u in enumerate(us):
                    px, py, pz = b.point(f, u, v)
                    px -= 8
                    py -= 8
                    pz -= 8
                    rx = px * ca + pz * sa
                    rz = -px * sa + pz * ca
                    ry = py * cp - rz * sp
                    rz2 = py * sp + rz * cp
                    c = tex[jv, ju]
                    if c[3] == 0:
                        continue
                    sh = SHADE[f] * 0.85 + 0.25
                    pts.append((rx, ry))
                    cols.append(np.clip(c[:3].astype(float) * sh, 0, 255))
                    deps.append(rz2)
    pts = np.array(pts)
    cols = np.array(cols)
    deps = np.array(deps)
    order = np.argsort(-deps)  # far first
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (58, 66, 96)
    sx = (pts[order, 0] * scale + size / 2).astype(int)
    sy = (size / 2 - pts[order, 1] * scale).astype(int)
    c = cols[order].astype(np.uint8)
    sz = int(math.ceil(scale / PPU)) + 1
    for dx in range(sz):
        for dy in range(sz):
            xx = sx + dx - sz // 2
            yy = sy + dy - sz // 2
            ok = (xx >= 0) & (xx < size) & (yy >= 0) & (yy < size)
            img[yy[ok], xx[ok]] = c[ok]
    return Image.fromarray(img)


def main():
    os.makedirs(MOD, exist_ok=True)
    up = build(+1)
    down = build(-1)
    placed = pack(up)
    atlas = make_atlas(up, placed)
    Image.fromarray(atlas).save(os.path.join(TEX, "mini_dragon.png"))
    for name, bx in (("mini_dragon", up), ("mini_dragon_down", down)):
        with open(os.path.join(MOD, name + ".json"), "w") as f:
            json.dump(make_model(bx, placed), f)
        with open(os.path.join(ITM, name + ".json"), "w") as f:
            json.dump({"model": {"type": "minecraft:model", "model": f"weapons:item/{name}"}}, f, indent=2)
    # bounding box sanity: must stay inside one block
    for nm, bx in (("up", up), ("down", down)):
        mn = [min(b.f[i] for b in bx) for i in range(3)]
        mx = [max(b.t[i] for b in bx) for i in range(3)]
        print(nm, "bounds", [round(v, 2) for v in mn], [round(v, 2) for v in mx], "elements", len(bx))
    if "--preview" in sys.argv:
        views = [("side", 90, 15), ("front34", 35, 25), ("top", 0, 78), ("back", 200, 20)]
        sheet = Image.new("RGB", (420 * 2, 420 * 2))
        for k, (nm, yaw, pitch) in enumerate(views):
            im = render(up, placed, atlas, yaw, pitch)
            sheet.paste(im, ((k % 2) * 420, (k // 2) * 420))
        sheet.save("/tmp/dragon_up.png")
        sheet2 = Image.new("RGB", (420 * 2, 420))
        sheet2.paste(render(down, placed, atlas, 35, 25), (0, 0))
        sheet2.paste(render(down, placed, atlas, 0, 78), (420, 0))
        sheet2.save("/tmp/dragon_down.png")
        Image.fromarray(atlas).resize((512, 512), Image.NEAREST).save("/tmp/dragon_atlas.png")


if __name__ == "__main__":
    main()
