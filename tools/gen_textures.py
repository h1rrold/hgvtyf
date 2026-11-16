#!/usr/bin/env python3
"""High-detail 64x64 weapon textures for the SMP Weapons resource pack.

Every weapon is built from shaded parts (palette ramps + bevel lighting + ordered dithering), then gets a
selective dark outline and, at Tier II, an energy halo.  The 3D mini dragon lives in gen_dragon.py.

Run:  python3 tools/gen_textures.py
"""
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resourcepack")
TEX = os.path.join(ROOT, "assets", "weapons", "textures", "item")
MOD = os.path.join(ROOT, "assets", "weapons", "models", "item")
ITM = os.path.join(ROOT, "assets", "weapons", "items")
for d in (TEX, MOD, ITM):
    os.makedirs(d, exist_ok=True)

S = 64
A0, A1 = 7.0, 57.0                        # pixel position of s=0 (bottom-left)
D0, D1 = math.sqrt(0.5), -math.sqrt(0.5)  # unit vector along the weapon (to the top right)
N0, N1 = math.sqrt(0.5), math.sqrt(0.5)   # unit vector across the weapon (+v = down-right)

yy, xx = np.mgrid[0:S, 0:S]
X = xx + 0.5
Y = yy + 0.5
SS = (X - A0) * D0 + (Y - A1) * D1        # s of every pixel
VV = (X - A0) * N0 + (Y - A1) * N1        # v of every pixel

BAY = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=float) / 16.0
BAY = np.tile(BAY, (S // 4, S // 4))


def P(s, v):
    return (A0 + s * D0 + v * N0, A1 + s * D1 + v * N1)


def hexrgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def R(*hexes):
    return np.array([hexrgb(h) for h in hexes], dtype=float)


def clip01(x):
    return np.clip(x, 0.0, 1.0)


# ------------------------------------------------------------------ masks

def poly(pts):
    im = Image.new("L", (S, S), 0)
    ImageDraw.Draw(im).polygon([(P(s, v)[0] - 0.5, P(s, v)[1] - 0.5) for s, v in pts], fill=255)
    return np.array(im) > 0


def ell(s0, v0, rs, rv):
    return ((SS - s0) / rs) ** 2 + ((VV - v0) / rv) ** 2 <= 1.0


def circle_px(cx, cy, r):
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r * r


def dist_seg(p0, p1):
    (s0, v0), (s1, v1) = p0, p1
    ds, dv = s1 - s0, v1 - v0
    L2 = ds * ds + dv * dv
    t = clip01(((SS - s0) * ds + (VV - v0) * dv) / (L2 if L2 else 1))
    return np.hypot(SS - (s0 + t * ds), VV - (v0 + t * dv))


def stroke(pts, w):
    d = np.full((S, S), 1e9)
    for a, b in zip(pts, pts[1:]):
        d = np.minimum(d, dist_seg(a, b))
    return d <= w / 2.0


def bev(v, c, hw, hl=-0.4):
    """Cylinder-ish bevel light across v: highlight on the up-left side."""
    t = np.clip((v - c) / hw, -1, 1)
    return clip01(0.86 - 0.62 * np.abs(t - hl) ** 1.25)


def radial(s0, v0, rs, rv, hl=(0.25, -0.35)):
    d = np.sqrt(((SS - s0 - hl[0] * rs) / rs) ** 2 + ((VV - v0 - hl[1] * rv) / rv) ** 2)
    return clip01(1.05 - 0.85 * d)


def interp_env(xs, ys):
    return np.interp(SS, xs, ys)


def shift(arr, dy, dx):
    """Shift an array without wrap-around (new cells are zero/False)."""
    out = np.zeros_like(arr)
    h, w = arr.shape[:2]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs_, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = arr[ys, xs_]
    return out


def noise_field(seed, cells=9):
    r = np.random.default_rng(seed).random((cells, cells))
    im = Image.fromarray((r * 255).astype(np.uint8)).resize((S, S), Image.BICUBIC)
    return np.array(im) / 255.0


NZ = noise_field(5)
NZ2 = noise_field(17, 16)


def rim_ul(mask):
    """Pixels on the up-left edge of a mask (for rim lighting)."""
    return mask & ~shift(mask, 1, 1)


# ------------------------------------------------------------------ canvas

class Art:
    def __init__(self, seed=1):
        self.rgb = np.zeros((S, S, 3))
        self.a = np.zeros((S, S), dtype=bool)
        self.rng = np.random.default_rng(seed)

    def paint(self, mask, ramp, light, dither=0.11):
        n = len(ramp)
        l = clip01(light + (BAY - 0.5) * dither)
        idx = np.clip((l * n).astype(int), 0, n - 1)
        col = ramp[idx]
        self.rgb[mask] = col[mask]
        self.a |= mask

    def px(self, s, v, color):
        x, y = P(s, v)
        x, y = int(math.floor(x)), int(math.floor(y))
        if 0 <= x < S and 0 <= y < S:
            self.rgb[y, x] = hexrgb(color) if isinstance(color, str) else color
            self.a[y, x] = True

    def tube(self, pts, r, ramp, hl=(0.0, -0.7)):
        """Stamp shaded circles along a polyline (bamboo / vines / curved limbs)."""
        for (s0, v0) in pts:
            m = ((SS - s0) ** 2 + (VV - v0) ** 2) <= r * r
            d = np.sqrt((SS - s0 - hl[0] * r) ** 2 + (VV - v0 - hl[1] * r) ** 2) / (r * 1.9)
            self.paint(m, ramp, 1.0 - d)

    def sparkles(self, n, region, color, seed=99):
        r = np.random.default_rng(seed)
        ys, xs = np.nonzero(region)
        if len(ys) == 0:
            return
        col = hexrgb(color) if isinstance(color, str) else color
        for _ in range(n):
            k = r.integers(len(ys))
            self.rgb[ys[k], xs[k]] = col
            self.a[ys[k], xs[k]] = True

    def finish(self, glow=None, glow_r=3, glow_a=0.55, outline_dark=0.26):
        a = self.a.copy()
        rgb = self.rgb.copy()
        out = np.zeros((S, S, 4), dtype=np.uint8)
        acc = np.zeros((S, S, 3))
        cnt = np.zeros((S, S))
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh_a = shift(a, dy, dx)
            sh_c = shift(rgb, dy, dx)
            acc += sh_c * sh_a[..., None]
            cnt += sh_a
        edge = (~a) & (cnt > 0)
        ocol = acc / np.maximum(cnt, 1)[..., None] * outline_dark + np.array([4, 2, 10]) * (1 - outline_dark)
        solid = a | edge
        rgb[edge] = ocol[edge]
        out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        out[..., 3] = np.where(solid, 255, 0)
        if glow is not None:
            gcol = np.array(hexrgb(glow), dtype=np.uint8)
            cur = solid.copy()
            for k in range(1, glow_r + 1):
                nxt = cur.copy()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt |= shift(cur, dy, dx)
                new = nxt & ~cur
                alpha = glow_a * (1 - (k - 1) / glow_r) ** 1.6
                out[new, :3] = gcol
                out[new, 3] = int(255 * alpha)
                cur = nxt
        return out

    def save(self, name, **kw):
        Image.fromarray(self.finish(**kw)).save(os.path.join(TEX, name + ".png"))


def wrap_handle(art, s0, s1, hw, ramp, gap=2.4):
    m = poly([(s0, -hw), (s1, -hw), (s1, hw), (s0, hw)])
    l = bev(VV, 0, hw) * 0.8 + 0.2 * (np.sin((SS + VV * 0.6) * (2 * math.pi / gap)) * 0.5 + 0.5)
    art.paint(m, ramp, l)


def facets_blade(art, xs, ws, ramp, c=0.0, top=0.66, bot=0.30):
    """Two-facet blade with a ridge: lit upper half, shaded lower half."""
    pts = [(x, c - w) for x, w in zip(xs, ws)] + [(x, c + w) for x, w in zip(xs[::-1], ws[::-1])]
    m = poly(pts)
    w_s = np.maximum(interp_env(xs, ws), 0.5)
    rel = (VV - c) / w_s
    l = np.where(rel < 0, top + 0.28 * (-rel), bot + 0.20 * (1 - rel))
    art.paint(m, ramp, l)
    return m, w_s


# ------------------------------------------------------------------ 1. Beamer Sword

def beamer(t):
    a = Art(11 + t)
    obs = R("#08050f", "#120a22", "#1f1240", "#321e66", "#4f2f9a", "#7a52d0")
    cry = R("#240a4a", "#5a1aa0", "#a040f0", "#e080ff", "#ffd0ff", "#ffffff")
    cyan = R("#083a5a", "#1488c0", "#40d0ff", "#9af4ff", "#e8ffff", "#ffffff")
    leather = R("#0c0914", "#181024", "#271a3a", "#3a2858", "#523a78")
    wrap_handle(a, 8, 22, 2.4, leather)
    for s in (11, 15, 19):  # silver rings
        a.paint(poly([(s, -2.6), (s + 1, -2.6), (s + 1, 2.6), (s, 2.6)]), R("#2a2440", "#5a5280", "#9a92c0", "#d0c8f0"), bev(VV, 0, 2.6))
    a.paint(ell(5.2, 0, 4.6, 4.2), cry, radial(5.2, 0, 4.6, 4.2))          # pommel crystal
    # end-crystal style guard
    g = 1.0 if t == 1 else 1.25
    gm = poly([(22, -4), (24, -10 * g), (28, -14 * g), (32, -12 * g), (30.5, -6), (30.5, 6), (32, 12 * g), (28, 14 * g), (24, 10 * g), (22, 4)])
    a.paint(gm, obs, clip01(0.75 - 0.35 * np.abs(SS - 26.5) / 5.5 + 0.18 * (VV < 0)))
    a.paint(stroke([(23.5, -9 * g), (27, -12.5 * g), (31, -11 * g)], 1.0), R("#6a48b8", "#a07ce8", "#d8b8ff"), 0.7)
    a.paint(stroke([(23.5, 9 * g), (27, 12.5 * g), (31, 11 * g)], 1.0), R("#2a1a58", "#46308c"), 0.4)
    a.paint(ell(26.5, 0, 3.0, 3.0), cry, radial(26.5, 0, 3.0, 3.0))
    # crystalline blade
    wmax = 6.4 if t == 1 else 8.2
    xs = [30.5, 34, 56, 62, 66, 70]
    ws = [wmax * 0.8, wmax, wmax, wmax * 0.66, wmax * 0.32, 0.25]
    m, w_s = facets_blade(a, xs, ws, obs, top=0.62, bot=0.26)
    a.paint(m & (VV < -w_s + 1.3), R("#8a68d8", "#b898f8", "#e0ccff"), 0.7)
    a.paint(m & (VV > w_s - 1.1), R("#1a0e38", "#2c1a5c"), 0.4)
    # glowing core
    cw = 1.1 if t == 1 else 1.7
    core = m & (np.abs(VV) <= cw) & (SS > 31) & (SS < 69)
    a.paint(core, cyan, clip01(1.0 - np.abs(VV) / (cw + 0.4) * 0.55 + 0.25 * (SS - 31) / 38))
    # fuller dashes
    for s in range(36, 62, 5):
        a.paint(stroke([(s, -3.6 * wmax / 6.4), (s + 2.6, -3.6 * wmax / 6.4)], 0.9) & m, R("#5a32aa", "#9a68f0"), 0.7)
        a.paint(stroke([(s, 3.6 * wmax / 6.4), (s + 2.6, 3.6 * wmax / 6.4)], 0.9) & m, R("#5a32aa", "#9a68f0"), 0.4)
    if t == 2:
        for s0 in (36, 44, 52, 59):                      # crystal fins along both edges
            wl = float(np.interp(s0 + 3, xs, ws))
            a.paint(poly([(s0, -wl), (s0 + 3.4, -wl - 6.2), (s0 + 6.4, -wl)]), obs, clip01(0.55 + 0.4 * ((SS - s0) / 6.4)))
            a.paint(poly([(s0, wl), (s0 + 3.4, wl + 5.0), (s0 + 6.4, wl)]), obs, clip01(0.25 + 0.3 * ((SS - s0) / 6.4)))
            a.px(s0 + 3.4, -wl - 5.4, "#e8ffff")
        for s0 in (38, 47, 56):                          # glyphs
            a.px(s0, -1.9, "#ffffff")
            a.px(s0 + 1, 1.9, "#ffffff")
        for (s0, v0) in ((20, -12), (45, 15), (62, -14), (14, 12), (58, 14)):  # floating shards
            a.paint(poly([(s0, v0 - 1.8), (s0 + 1.3, v0), (s0, v0 + 1.8), (s0 - 1.3, v0)]), cyan, 0.85)
        a.save("beamer_sword_t2", glow="#a040ff", glow_r=3, glow_a=0.5)
    else:
        for (s0, v0) in ((20, -11), (46, 12), (62, -11)):
            a.paint(poly([(s0, v0 - 1.4), (s0 + 1, v0), (s0, v0 + 1.4), (s0 - 1, v0)]), cyan, 0.8)
        a.save("beamer_sword_t1")


# ------------------------------------------------------------------ 2. Tropical Bow Staff

def tropical(t):
    a = Art(21 + t)
    wood = R("#24140a", "#3e2612", "#5e3a1c", "#80542a", "#a87a40", "#cfa266")
    bam = R("#0a3016", "#14522a", "#238a40", "#3cc060", "#7eea8a", "#d0ffc0")
    vine = R("#0b3012", "#14541c", "#25842c", "#44b844", "#86e468")
    leaf = R("#0e4018", "#1c7a2a", "#34b840", "#6ce060", "#b8ff90")
    crys = R("#0a4a1a", "#1c8a30", "#40d060", "#98ff88", "#e8ffe0")
    m = poly([(2, -1.8), (68, -1.8), (68, 1.8), (2, 1.8)])
    a.paint(m, wood, bev(VV, 0, 1.8) * 0.92 + 0.08 * (np.sin(SS * 1.3) * 0.5 + 0.5))
    a.paint(poly([(17, -2.2), (30, -2.2), (30, 2.2), (17, 2.2)]), R("#0e3a16", "#16602a", "#26923a", "#46c458"),
            bev(VV, 0, 2.2) * 0.8 + 0.2 * (np.sin((SS + VV) * 2.6) * 0.5 + 0.5))
    # the bow: C-shape opening toward the grip, centred on the staff
    amp = 18 if t == 1 else 21
    arc = []
    for k in range(-24, 25):
        tt = k / 24.0
        arc.append((37 + 12 * (1 - tt * tt), amp * tt))
    a.tube(arc, 2.3 if t == 1 else 2.7, bam)
    for sg in (-1, 1):
        a.tube([(36.2, sg * (amp + 1.6)), (34.5, sg * (amp + 3.0))], 1.3, bam)
    a.paint(stroke([(36.2, -amp - 1.6), (36.2, amp + 1.6)], 0.9), R("#9ad880", "#d8ffc0", "#ffffff"), 0.75)
    # vines spiralling up the shaft
    sp = [(4 + 0.45 * k, 2.3 * math.sin(0.62 * k)) for k in range(0, 40)]
    a.tube(sp, 0.95, vine)
    for k in range(3, 40, 6):
        s0, v0 = sp[k]
        sg = 1 if (k // 6) % 2 == 0 else -1
        a.paint(poly([(s0, v0), (s0 + 2.0, v0 + sg * 3.8), (s0 + 3.4, v0 + sg * 1.2)]), leaf, clip01(0.35 + 0.5 * (SS - s0) / 3.4))
    a.paint(poly([(60, 0), (64, -3.4), (70, 0), (64, 3.4)]), crys, radial(64.5, 0, 5, 3.4))
    # kiwi charm
    a.paint(circle_px(*P(10, 4.2), 3.4), R("#4a2c14", "#6a4222", "#8a5a30"), 0.5)
    a.paint(circle_px(*P(10, 4.2), 2.7), R("#2a8a2a", "#58c038", "#98e868", "#d8ff98"), radial(10, 4.2, 2.7, 2.7))
    a.px(10, 4.2, "#f4ffe0")
    if t == 2:
        for (s0, v0) in ((29, 2.6), (47, -23.5), (47, 23.5)):             # hibiscus flowers
            cx, cy = P(s0, v0)
            for k in range(5):
                ang = k * 2 * math.pi / 5
                a.paint(circle_px(cx + 2.6 * math.cos(ang), cy + 2.6 * math.sin(ang), 2.1),
                        R("#8a1a4a", "#d03a80", "#ff6aaa", "#ffb0d8"), 0.55 + 0.25 * math.cos(ang - 2.2))
            a.paint(circle_px(cx, cy, 1.5), R("#c07a10", "#ffcc30", "#fff090"), 0.7)
        for (s0, v0, sg) in ((40, -amp * 0.85, -1), (40, amp * 0.85, 1), (44, -amp * 0.6, -1), (44, amp * 0.6, 1)):
            a.paint(poly([(s0, v0), (s0 + 5.4, v0 + sg * 2.6), (s0 + 3, v0 + sg * 6.4)]), leaf, clip01(0.3 + 0.6 * (SS - s0) / 5.4))
        a.paint(ell(46, 0, 4.4, 4.4), R("#2a6a10", "#58b81c", "#9aee3a", "#d4ff80", "#ffffff"), radial(46, 0, 4.4, 4.4))
        a.sparkles(18, a.a, "#e8ffc8", 7)
        a.save("tropical_bow_staff_t2", glow="#60ff60", glow_r=3, glow_a=0.45)
    else:
        a.save("tropical_bow_staff_t1")


# ------------------------------------------------------------------ 3. Ice Axe (frost scythe)

def ice_axe(t):
    a = Art(31 + t)
    steel = R("#0a1220", "#162438", "#27405e", "#40668e", "#6a9cc8", "#aedcf4")
    ice = R("#0e3a6a", "#1f6aac", "#3a9ee0", "#78d4fa", "#c4f4ff", "#ffffff")
    wrap_handle(a, 2, 44, 1.8, steel, gap=3.2)
    for s_ in (9, 21, 33):
        a.paint(poly([(s_, -2.4), (s_ + 1.4, -2.4), (s_ + 1.4, 2.4), (s_, 2.4)]), R("#26486c", "#4a7aa8", "#9ad0f0", "#e0faff"), bev(VV, 0, 2.4))
    # spear tip continuing past the blade
    a.paint(poly([(46, -2.0), (58, 0), (46, 2.0)]), ice, clip01(0.45 + 0.5 * (VV < 0) - 0.1 * (SS - 46) / 12))
    a.paint(poly([(37, -2.8), (46, -2.8), (46, 2.8), (37, 2.8)]), R("#1a3050", "#34608c", "#6aa0cc", "#c4ecff"), bev(VV, 0, 2.8))
    # crescent scythe blade, hooked toward the tip
    k = 1.0 if t == 1 else 1.1
    outer = [(38, -1), (38.5, -8 * k), (41, -15 * k), (46, -20.5 * k), (52, -22 * k), (57, -19.5 * k), (60.5, -15.5 * k)]
    inner = [(58.5, -14.5), (54, -16.5 * k), (48.5, -14 * k), (44, -9.5), (42, -4.5), (42, 0)]
    bm = poly(outer + inner)
    a.paint(bm, ice, clip01(0.28 + 0.6 * clip01(-VV / (22.0 * k)) + 0.28 * (NZ2 - 0.5)))
    a.paint(rim_ul(bm), R("#c4f4ff", "#ffffff"), 0.8)
    a.paint(stroke(inner, 1.4) & bm, R("#c4f4ff", "#e8ffff", "#ffffff"), 0.85)
    a.paint(stroke([(39, -3), (40, -9 * k), (43, -15.5 * k), (48, -19 * k), (53, -19.5 * k)], 1.0) & bm, R("#0e3a6a", "#1f6aac"), 0.3)
    # frost shards growing off the spine of the blade
    for (s0, v0, ln) in ((38.5, -8 * k, 6.0), (41, -15 * k, 6.5), (46, -20.5 * k, 5.5)):
        a.paint(poly([(s0 - 2.2, v0 + 0.6), (s0 - 0.2, v0 - ln), (s0 + 2.4, v0 + 0.4)]), ice, clip01(0.5 + 0.45 * (VV < v0 - 0.3)))
    if t == 2:
        for (s0, v0, ln) in ((43, -8.5, 6.5), (47, -15, 7.0), (53, -17.5, 6.5), (57, -15.5, 5.5)):  # icicles off the cutting edge
            cx, cy = P(s0, v0)
            for j in range(int(ln)):
                w = max(0.5, 1.5 * (1 - j / ln))
                a.paint(circle_px(cx, cy + 1 + j, w), ice, 0.88 - 0.1 * j / ln)
        for (s0, v0) in ((22, -6), (30, 6), (12, 6), (50, 7)):
            a.paint(poly([(s0, v0 - 2), (s0 + 1.2, v0), (s0, v0 + 2), (s0 - 1.2, v0)]), ice, 0.95)
        a.paint(poly([(57, -2.6), (66, 0), (57, 2.6)]), ice, clip01(0.6 + 0.4 * (VV < 0)))
        a.sparkles(26, a.a, "#ffffff", 5)
        a.save("ice_axe_t2", glow="#6ae0ff", glow_r=3, glow_a=0.55)
    else:
        a.sparkles(14, a.a, "#ffffff", 5)
        a.save("ice_axe_t1")


# ------------------------------------------------------------------ 4. Dynasty Sword

def dynasty(t):
    a = Art(41 + t)
    gold = R("#3a2204", "#6e4608", "#b07a10", "#e8b42a", "#ffd95a", "#fff6b8")
    ruby = R("#4a0010", "#a0082a", "#e02848", "#ff7a90", "#ffd0d8")
    sapph = R("#06185a", "#1038a8", "#2c70e8", "#78b4ff", "#d0ecff")
    emer = R("#05381a", "#0c7a38", "#22c060", "#80f0a0", "#d8ffe8")
    amet = R("#2a0a58", "#5a20a8", "#9a50e8", "#d09cff", "#f4e0ff")
    cloth = R("#3a0008", "#780014", "#b8102a", "#e83a4c")
    for k, dx in enumerate((-2, 0, 2)):                                   # tassel hanging down the screen
        a.paint(poly([(3, -2 + dx * 0.4), (-3.5, dx * 1.4 - 1.2), (-3.5, dx * 1.4 + 1.2), (3, 2 + dx * 0.4)]) & (Y > 52), cloth, 0.6 + 0.1 * k)
    m = poly([(4, -2.3), (66, -2.3), (66, 2.3), (4, 2.3)])
    a.paint(m, gold, bev(VV, 0, 2.3) * 0.9 + 0.1 * (np.sin(SS * 1.1) * 0.5 + 0.5))
    for s in range(10, 64, 12):                                           # ornate rings
        a.paint(poly([(s, -3.2), (s + 2.2, -3.2), (s + 2.2, 3.2), (s, 3.2)]), gold, bev(VV, 0, 3.2) * 0.95 + 0.05)
        a.paint(stroke([(s + 1.1, -3.2), (s + 1.1, 3.2)], 0.7), R("#3a2204", "#6e4608"), 0.3)
    ln = 6.4 if t == 1 else 8.8
    for i, s0 in enumerate((15, 24, 33, 42, 51, 60)):                    # spikes
        sg = -1 if i % 2 == 0 else 1
        a.paint(poly([(s0 - 1.9, sg * 2.2), (s0 + 0.6, sg * (2.2 + ln)), (s0 + 2.4, sg * 2.2)]), gold,
                clip01(0.45 - 0.5 * (np.abs(VV) - 2.2) / ln + 0.3 * (SS < s0 + 0.6)))
        if t == 2:
            a.paint(poly([(s0 - 1.5, -sg * 2.2), (s0 + 0.4, -sg * (2.2 + ln * 0.6)), (s0 + 2.0, -sg * 2.2)]), gold,
                    clip01(0.55 - 0.4 * (np.abs(VV) - 2.2) / ln + 0.2 * (SS < s0 + 0.4)))
    for s0, ramp in ((8, ruby), (20, sapph), (29, emer), (38, amet), (47, ruby), (56, sapph)):
        a.paint(ell(s0 + 1.1, 0, 2.2, 1.9), ramp, radial(s0 + 1.1, 0, 2.2, 1.9))
        a.px(s0 + 0.5, -0.8, "#ffffff")
    for (s0, v0) in ((26, 3.9), (44, -3.9), (13, -3.9), (53, 3.9)):       # coins
        a.paint(ell(s0, v0, 2.4, 2.4), gold, radial(s0, v0, 2.4, 2.4))
        a.paint(ell(s0, v0, 1.2, 1.2), R("#8a5a08", "#c08a18"), 0.5)
    a.paint(poly([(61, 0), (64, -6.2), (70, 0), (64, 6.2)]), gold, clip01(0.4 + 0.5 * (VV < 0) + 0.1 * (SS > 64)))
    a.paint(poly([(62, 0), (64.4, -3.4), (68, 0), (64.4, 3.4)]), ruby, radial(65, 0, 3.4, 3.4))
    for (s0, v0) in ((60.5, -4.4), (60.5, 4.4), (68.5, -2.6), (68.5, 2.6)):
        a.paint(ell(s0, v0, 1.5, 1.5), gold, radial(s0, v0, 1.5, 1.5))
    a.px(64.5, -0.8, "#ffffff")
    if t == 2:
        for (s0, v0, ramp) in ((17, -4.2, emer), (35, 4.2, ruby), (49, -4.2, sapph), (58, 4.4, amet)):
            a.paint(ell(s0, v0, 1.8, 1.8), ramp, radial(s0, v0, 1.8, 1.8))
        a.sparkles(22, a.a, "#fff6c8", 3)
        a.save("dynasty_sword_t2", glow="#ffd040", glow_r=3, glow_a=0.5)
    else:
        a.save("dynasty_sword_t1")


# ------------------------------------------------------------------ 5. Inferno Sword (double sided)

def inferno(t):
    a = Art(51 + t)
    iron = R("#08060a", "#16111a", "#241c28", "#382c3c", "#54425a")
    flame = R("#3a0400", "#7a0e00", "#c42400", "#ff5a00", "#ff9a14", "#ffd83a", "#fffbc0")
    cloth = R("#2a0004", "#620010", "#a0101c", "#d8322c")
    wmax = 7.2 if t == 1 else 9.6
    wrap_handle(a, 29, 43, 2.7, iron, gap=2.0)
    a.paint(poly([(31, -2.9), (33, -2.9), (33, 2.9), (31, 2.9)]), cloth, bev(VV, 0, 2.9))
    a.paint(poly([(39, -2.9), (41, -2.9), (41, 2.9), (39, 2.9)]), cloth, bev(VV, 0, 2.9))
    for s0, sg in ((43.5, 1), (28.5, -1)):                                 # guards
        a.paint(poly([(s0, -7.4), (s0 + sg * 2.4, -6), (s0 + sg * 2.4, 6), (s0, 7.4), (s0 - sg * 1.0, 0)]),
                R("#3a0e04", "#7a2008", "#c4501a", "#ffa040"), clip01(0.75 - 0.35 * np.abs(VV) / 7.4 + 0.15 * (VV < 0)))
    a.paint(ell(36, 0, 3.0, 3.0), R("#7a1000", "#e03a00", "#ff9a14", "#fff0a0"), radial(36, 0, 3.0, 3.0))

    def blade(base, tip_s):
        sgn = 1 if tip_s > base else -1
        xs = np.linspace(base, tip_s, 14)
        Lb = abs(tip_s - base)
        prog = np.abs(xs - base) / Lb
        ws = wmax * (1 - prog ** 1.25) + 0.3
        ws = ws + 0.9 * np.sin(prog * 9 + (0 if sgn > 0 else 1.7))
        ws = np.maximum(ws, 0.3)
        pts = [(x, -w) for x, w in zip(xs, ws)] + [(x, w) for x, w in zip(xs[::-1], ws[::-1])]
        m = poly(pts)
        if sgn > 0:
            w_s = np.interp(SS, xs, ws)
        else:
            w_s = np.interp(SS, xs[::-1], ws[::-1])
        w_s = np.maximum(w_s, 0.4)
        heat = 1.0 - np.abs(VV) / w_s
        along = np.abs(SS - base) / Lb
        l = clip01(0.12 + 0.78 * np.clip(heat, 0, 1) ** 0.75 + 0.42 * (NZ2 - 0.5) - 0.16 * along)
        a.paint(m, flame, l)

    blade(43.5, 70)
    blade(28.5, 0)
    for (s0, v0, sg) in ((50, -wmax, -1), (58, wmax * 0.7, 1), (20, -wmax * 0.8, -1), (11, wmax * 0.6, 1)):
        a.paint(poly([(s0, v0), (s0 + 2.4, v0 + sg * 5.5), (s0 + 4.4, v0 + sg * 0.6)]), flame, 0.62)
    if t == 2:
        for (s0, v0, sg) in ((46, wmax * 0.8, 1), (63, -wmax * 0.5, -1), (24, wmax * 0.8, 1), (6, -wmax * 0.4, -1), (54, wmax, 1), (16, -wmax, -1)):
            a.paint(poly([(s0, v0), (s0 + 3.0, v0 + sg * 7.0), (s0 + 5.2, v0 + sg * 0.8)]), flame, 0.78)
        a.sparkles(24, (~a.a) & (np.hypot(X - 32, Y - 32) < 30), "#ffe070", 6)
        a.save("inferno_sword_t2", glow="#ff7a10", glow_r=3, glow_a=0.55)
    else:
        a.save("inferno_sword_t1")


# ------------------------------------------------------------------ 6. Hellsfire Axe

def hellsfire(t):
    a = Art(61 + t)
    char = R("#0a0606", "#160c0c", "#241414", "#382222", "#523434")
    obs = R("#06030a", "#110818", "#1e0c26", "#341436", "#5a1e3a", "#8c3040")
    lava = R("#6a0e00", "#c42400", "#ff5a00", "#ff9a14", "#ffd83a", "#fff6a0")
    iron = R("#14121a", "#2a2630", "#46404c", "#706878", "#a098a8")
    k = 1.0 if t == 1 else 1.12
    m = poly([(2, -2.0), (58, -2.0), (58, 2.0), (2, 2.0)])
    a.paint(m, char, bev(VV, 0, 2.0) * 0.9 + 0.1 * (np.sin(SS * 1.7) * 0.5 + 0.5))
    for s in (14, 28):                                                    # iron bands
        a.paint(poly([(s, -2.7), (s + 2.2, -2.7), (s + 2.2, 2.7), (s, 2.7)]), iron, bev(VV, 0, 2.7))
    for s in (6, 20, 34):                                                 # lava seams in the haft
        a.paint(stroke([(s, -1.4), (s + 3, 0.4), (s + 5, -0.6)], 0.8), lava, 0.62)
    head = [(42, -1.5), (40, -6), (36, -12 * k), (30.5, -21 * k), (37, -19 * k), (45, -18.5 * k), (53, -19.5 * k),
            (56.5, -21.5 * k), (54, -12), (51, -5), (51, 1.5)]
    hm = poly(head)
    a.paint(hm, obs, clip01(0.26 + 0.55 * clip01((-VV - 2) / (20 * k)) + 0.3 * (NZ - 0.5)))
    a.paint(rim_ul(hm), R("#5a1e3a", "#8c3040", "#c05050"), 0.8)
    spike = poly([(51, 1.5), (49.5, 6.5), (54, 11), (56.5, 4.5), (55, 1)])
    a.paint(spike, obs, clip01(0.55 - 0.3 * (VV - 1) / 14 + 0.2 * (SS > 55)))
    a.paint(rim_ul(spike), R("#5a1e3a", "#8c3040"), 0.7)
    edge_pts = [(30.5, -21 * k), (37, -19 * k), (45, -18.5 * k), (53, -19.5 * k), (56.5, -21.5 * k)]
    a.paint(stroke(edge_pts, 2.0) & hm, lava, 0.72 + 0.25 * (NZ2 - 0.5))                 # molten cutting edge
    a.paint(stroke(edge_pts, 0.8) & hm, R("#ffd83a", "#fff6a0"), 0.75)
    cracks = [[(41, -5), (39, -10), (35, -15 * k), (34, -19 * k)], [(46, -4), (45, -10), (46, -15 * k)],
              [(50, -4), (52, -9), (54, -15 * k)], [(40, -9), (44, -11), (48, -10)]]
    if t == 2:
        cracks += [[(44, -14), (41, -17 * k)], [(50, -12), (49, -17 * k)], [(37, -13), (36, -17 * k)]]
    for c in cracks:
        a.paint(stroke(c, 1.2) & hm, lava, 0.78)
        a.paint(stroke(c, 0.5) & hm, R("#ffd83a", "#fff6a0"), 0.75)
    a.paint(stroke([(53, 2), (52, 7), (55, 11)], 1.0) & spike, lava, 0.65)
    # a few thick lava drips fall straight down the screen from the underside of the head
    cols = (17, 21, 26, 31) if t == 1 else (15, 18, 22, 26, 30, 34, 38)
    for cx in cols:
        ys = np.nonzero(hm[:, cx])[0]
        if len(ys) == 0:
            continue
        y0 = ys.max()
        ln = int(a.rng.integers(3, 6 if t == 1 else 8))
        for j in range(1, ln + 1):
            yq = y0 + j
            if yq < S - 1:
                for dxx in ((0, 1) if j < ln - 1 else (0,)):
                    if cx + dxx < S:
                        a.rgb[yq, cx + dxx] = lava[min(5, 3 - j // 3 + 1)]
                        a.a[yq, cx + dxx] = True
        if y0 + ln + 1 < S - 1:
            a.rgb[y0 + ln + 1, cx] = lava[4]
            a.a[y0 + ln + 1, cx] = True
    if t == 2:
        a.sparkles(18, hm, "#ffd83a", 8)
        a.save("hellsfire_axe_t2", glow="#ff5a10", glow_r=3, glow_a=0.55)
    else:
        a.save("hellsfire_axe_t1")


# ------------------------------------------------------------------ 7. Dragon's Fang (dagger)

def dragon(t):
    a = Art(71 + t)
    skin = R("#10081e", "#1c0e34", "#2e1a52", "#46287a", "#6a40a8", "#9a6ae0")
    bone = R("#4a4034", "#7e7058", "#b8aa8a", "#e0d4b4", "#f8f0dc", "#ffffff")
    mag = R("#40065a", "#8a1ab4", "#d440ff", "#f090ff", "#ffe0ff")
    m = poly([(8, -2.3), (26, -2.3), (26, 2.3), (8, 2.3)])
    scale = ((np.floor(SS / 1.9) + np.floor((VV + 3) / 1.9) * 0.5) % 2) * 0.16
    a.paint(m, skin, bev(VV, 0, 2.3) * 0.84 + scale)
    for s in (11, 16, 21):
        a.paint(stroke([(s, -2.3), (s + 0.4, 2.3)], 0.7), R("#06020e", "#100622"), 0.3)
    a.paint(ell(6.2, 0, 3.8, 3.5), mag, radial(6.2, 0, 3.8, 3.5))
    gv = 6.8 if t == 1 else 8.6
    for sg in (-1, 1):                                                     # curved bone claws
        claw = [(24.8, sg * 2.0), (25.4, sg * 4.2), (27.0, sg * (gv - 1.8)), (29.4, sg * (gv - 0.2)), (32.0, sg * (gv + 0.2))]
        a.tube(claw, 1.55, bone, hl=(0.0, -0.7 * sg))
        a.paint(poly([(31, sg * (gv - 0.9)), (34.2, sg * (gv + 0.4)), (31, sg * (gv + 1.3))]), bone, 0.9)
    a.paint(poly([(24.5, -2.4), (28.5, -2.4), (28.5, 2.4), (24.5, 2.4)]), bone, bev(VV, 0, 2.4))
    xs = np.linspace(28.5, 66, 20)
    prog = (xs - 28.5) / (66 - 28.5)
    wmax = 5.4 if t == 1 else 6.8
    cen = 10.5 * prog ** 1.7 * (1 if t == 1 else 1.1)
    ws = wmax * (1 - prog ** 1.5) + 0.3
    pts = [(x, c - w) for x, c, w in zip(xs, cen, ws)] + [(x, c + w) for x, c, w in zip(xs[::-1], cen[::-1], ws[::-1])]
    bm = poly(pts)
    cs = np.interp(SS, xs, cen)
    wl = np.maximum(np.interp(SS, xs, ws), 0.4)
    rel = (VV - cs) / wl
    a.paint(bm & (rel < 0), bone, 0.62 + 0.3 * (-rel))
    a.paint(bm & (rel >= 0), R("#1a0e32", "#34205a", "#5e4090", "#9a80c8"), 0.30 + 0.45 * (1 - rel))
    a.paint(bm & (VV < cs - wl + 1.3), bone, 0.95)
    a.paint(bm & (VV > cs + wl - 1.2), bone, 0.12)
    vein = list(zip(xs[1:-3], cen[1:-3]))
    a.paint(stroke(vein, 1.2 if t == 1 else 1.8) & bm, mag, 0.7)
    if t == 2:
        for s0 in (38, 46, 54):                                            # bone spurs on the spine
            cc = float(np.interp(s0, xs, cen))
            ww = float(np.interp(s0, xs, ws))
            a.paint(poly([(s0 - 2.4, cc - ww), (s0 + 1.6, cc - ww - 5.2), (s0 + 2.8, cc - ww + 0.4)]), bone, clip01(0.5 + 0.5 * (VV < cc - ww)))
        for s0 in (36, 42, 48, 54):
            a.px(s0, float(np.interp(s0, xs, cen)) + 2.6, "#f090ff")
        a.sparkles(10, bm, "#ffe0ff", 4)
        a.save("dragons_fang_t2", glow="#c040ff", glow_r=3, glow_a=0.5)
    else:
        a.save("dragons_fang_t1")


# ------------------------------------------------------------------ 8. Heavy Hitter (mace / warhammer)

def heavy(t):
    a = Art(81 + t)
    steel = R("#12151e", "#262b3a", "#424b62", "#6a7690", "#9ca8c4", "#d8e0f4", "#ffffff")
    gold = R("#3a2204", "#6e4608", "#b07a10", "#e8b42a", "#ffd95a", "#fff6b8")
    wood = R("#1c1008", "#32200e", "#4e3016", "#6e4622", "#8e6232")
    glowc = R("#7a2a00", "#e86a10", "#ffb030", "#fff090")
    m = poly([(3, -2.7), (40, -2.7), (40, 2.7), (3, 2.7)])
    a.paint(m, wood, bev(VV, 0, 2.7) * 0.82 + 0.18 * (np.sin((SS + VV * 0.8) * 2.0) * 0.5 + 0.5))
    for s in (7, 17, 27):
        a.paint(poly([(s, -3.3), (s + 1.8, -3.3), (s + 1.8, 3.3), (s, 3.3)]), gold, bev(VV, 0, 3.3))
    a.paint(ell(3.2, 0, 3.4, 3.4), gold, radial(3.2, 0, 3.4, 3.4))
    hv = 11.0 if t == 1 else 11.6
    hm = poly([(39, -hv + 2), (41, -hv), (62, -hv), (64.5, -hv + 2.5), (64.5, hv - 2.5), (62, hv), (41, hv), (39, hv - 2)])
    a.paint(hm, steel, bev(VV, 0, hv, hl=-0.35) * 0.86 + 0.14 * (SS > 56) - 0.1 * (SS < 43))
    for v in (-6.2, 0.0, 6.2):                                             # engraved flanges
        a.paint(stroke([(42, v), (62, v)], 1.0) & hm, steel, 0.12 if v >= 0 else 0.55)
    for s in (40.0, 62.0):                                                 # gold bands
        a.paint(poly([(s - 1.1, -hv - 0.4), (s + 1.1, -hv - 0.4), (s + 1.1, hv + 0.4), (s - 1.1, hv + 0.4)]) & hm, gold, bev(VV, 0, hv))
    for (s0, v0) in ((45, -8.6), (45, 8.6), (57, -8.6), (57, 8.6), (51, -3.1), (51, 3.1)):   # studs
        a.paint(ell(s0, v0, 1.5, 1.5), steel, radial(s0, v0, 1.5, 1.5))
    a.paint(poly([(64.5, -4.2), (71, 0), (64.5, 4.2)]), steel, clip01(0.35 + 0.55 * (VV < 0) + 0.1 * (SS > 66)))
    if t == 2:
        for c in ([(44, -7), (49, -4), (52, 0), (56, 3), (60, 7)], [(47, 8), (52, 5), (55, 7)], [(48, -9), (53, -6), (58, -8)]):
            a.paint(stroke(c, 1.1) & hm, glowc, 0.7)
            a.paint(stroke(c, 0.5) & hm, R("#fff090", "#ffffff"), 0.7)
        a.paint(poly([(51, -hv), (55.5, -hv - 6.5), (60, -hv)]), steel, clip01(0.55 + 0.4 * (SS > 55)))
        a.paint(poly([(46, hv), (50.5, hv + 5.0), (55, hv)]), steel, clip01(0.25 + 0.3 * (SS > 50)))
        a.sparkles(16, hm, "#fff090", 5)
        a.save("heavy_hitter_t2", glow="#ffb030", glow_r=3, glow_a=0.5)
    else:
        a.save("heavy_hitter_t1")


# ------------------------------------------------------------------ extras

def kiwi():
    a = Art(91)
    cx, cy = 32, 33
    rim = R("#2e1a0a", "#4a2c14", "#6e4624", "#8e6234")
    flesh = R("#1e6a1e", "#3a9a2a", "#66c63a", "#9ae85e", "#d0ff9a")
    a.paint(circle_px(cx, cy, 25), rim, 1.0 - np.hypot(X - cx + 7, Y - cy + 8) / 40)
    for _ in range(90):                                                    # fuzzy skin
        ang = a.rng.uniform(0, 2 * math.pi)
        r_ = a.rng.uniform(23, 26)
        px_, py_ = int(cx + r_ * math.cos(ang)), int(cy + r_ * math.sin(ang))
        if 0 <= px_ < S and 0 <= py_ < S:
            a.rgb[py_, px_] = hexrgb("#a88050")
            a.a[py_, px_] = True
    d = np.hypot(X - cx, Y - cy)
    a.paint(circle_px(cx, cy, 21), flesh, clip01(0.35 + 0.65 * np.clip(1 - d / 21, 0, 1) ** 0.7 + 0.12 * np.sin(np.arctan2(Y - cy, X - cx) * 14) * (d > 9)))
    a.paint(circle_px(cx, cy, 8.2), R("#e8ffd0", "#f4ffe8", "#ffffff"), 0.8 - d / 30)
    for k in range(18):                                                    # seeds
        ang = k * 2 * math.pi / 18
        a.paint(circle_px(cx + 12.6 * math.cos(ang), cy + 12.6 * math.sin(ang), 1.2), R("#06100a", "#102014"), 0.4)
    a.paint(circle_px(cx + 2, cy - 27, 1.5), R("#4a3016", "#6e4a22"), 0.5)
    a.paint(circle_px(cx - 9, cy - 11, 2.5), R("#a0ff70", "#e8ffc0"), 0.9)
    a.save("kiwi_fruit")


def coin():
    a = Art(92)
    gold = R("#4a2c04", "#8a5a0c", "#c8901c", "#f0c038", "#ffe270", "#fff8c0")
    cx, cy = 32, 32
    d = np.hypot(X - cx, Y - cy)
    a.paint(circle_px(cx, cy, 27), R("#5a3606", "#8a5a0c", "#b8800e"), 0.6 - 0.3 * (X + Y - 64) / 60)
    a.paint(circle_px(cx, cy, 24), gold, clip01(0.9 - 0.6 * np.hypot(X - cx + 9, Y - cy + 10) / 40))
    a.paint(((d > 19.5) & (d < 21.5)), R("#8a5a0c", "#c8901c"), 0.4)
    star = [(cx + (14 if k % 2 == 0 else 6) * math.sin(k * math.pi / 5), cy - (14 if k % 2 == 0 else 6) * math.cos(k * math.pi / 5)) for k in range(10)]
    im = Image.new("L", (S, S), 0)
    ImageDraw.Draw(im).polygon([(x - 0.5, y - 0.5) for x, y in star], fill=255)
    a.paint(np.array(im) > 0, gold, clip01(0.95 - 0.55 * np.hypot(X - cx + 4, Y - cy + 5) / 22))
    a.paint(circle_px(cx, cy, 3.2), R("#b8800e", "#ffe270"), 0.8)
    a.save("coin")


def reroller():
    a = Art(93)
    cx, cy = 32, 32
    d = np.hypot(X - cx, Y - cy)
    a.paint(circle_px(cx, cy, 25), R("#10041e", "#240a3e", "#3e1468", "#5e2096", "#8a3ad0"), clip01(0.9 - 0.7 * np.hypot(X - cx + 6, Y - cy + 8) / 40))
    ang = np.arctan2(Y - cy, X - cx)
    a.paint((np.sin(ang * 3 + d * 0.35) > 0.55) & (d < 24) & (d > 8), R("#8a3ad0", "#c070ff", "#e8b0ff"), 0.75)
    a.paint(circle_px(cx, cy, 11), R("#7a1c00", "#d04a00", "#ff8a10", "#ffd040", "#fff6a0"), clip01(1.0 - d / 12.0))
    a.paint(circle_px(cx, cy, 5), R("#fff6a0", "#ffffff"), 0.8)
    for (px_, py_) in ((11, 13), (53, 51)):
        a.paint(circle_px(px_, py_, 4.0), R("#c070ff", "#f0c8ff"), 0.75)
    a.sparkles(18, a.a, "#ffffff", 9)
    a.save("reroller", glow="#c070ff", glow_r=3, glow_a=0.5)


# ------------------------------------------------------------------ models / pack meta

def write_models():
    names = []
    for w in ("beamer_sword", "tropical_bow_staff", "ice_axe", "dynasty_sword", "inferno_sword",
              "hellsfire_axe", "dragons_fang", "heavy_hitter"):
        for t in (1, 2):
            names.append((f"{w}_t{t}", "minecraft:item/handheld"))
    for n in ("kiwi_fruit", "coin", "reroller"):
        names.append((n, "minecraft:item/generated"))
    for name, parent in names:
        with open(os.path.join(MOD, name + ".json"), "w") as f:
            json.dump({"parent": parent, "textures": {"layer0": f"weapons:item/{name}"}}, f, indent=2)
        with open(os.path.join(ITM, name + ".json"), "w") as f:
            json.dump({"model": {"type": "minecraft:model", "model": f"weapons:item/{name}"}}, f, indent=2)


def write_meta():
    meta = {"pack": {"description": "SMP Weapons - custom weapon models and sounds", "pack_format": 46,
                     "min_format": 46, "max_format": 2000,
                     "supported_formats": {"min_inclusive": 46, "max_inclusive": 2000}}}
    with open(os.path.join(ROOT, "pack.mcmeta"), "w") as f:
        json.dump(meta, f, indent=2)
    bg = Image.new("RGBA", (S, S), (28, 18, 46, 255))
    ic = Image.open(os.path.join(TEX, "heavy_hitter_t2.png")).convert("RGBA")
    bg.alpha_composite(ic)
    bg.resize((128, 128), Image.NEAREST).save(os.path.join(ROOT, "pack.png"))


if __name__ == "__main__":
    for t in (1, 2):
        beamer(t)
        tropical(t)
        ice_axe(t)
        dynasty(t)
        inferno(t)
        hellsfire(t)
        dragon(t)
        heavy(t)
    kiwi()
    coin()
    reroller()
    write_models()
    write_meta()
    print("done")
