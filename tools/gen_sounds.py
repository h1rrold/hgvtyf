#!/usr/bin/env python3
"""Synthesises every custom sound for the SMP Weapons resource pack (numpy -> wav -> ffmpeg/libvorbis ogg).

Run:  python3 tools/gen_sounds.py      (needs numpy + ffmpeg)
The sounds are original synthesised effects, so there are no licensing worries. Swap any .ogg in
resourcepack/assets/weapons/sounds/ with your own audio if you want - the names stay the same.
"""
import json
import os
import subprocess
import tempfile
import wave

import numpy as np

SR = 44100
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resourcepack", "assets", "weapons")
OUT = os.path.join(ROOT, "sounds")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1234)


def t_(dur):
    return np.linspace(0, dur, int(SR * dur), endpoint=False)


def sine(f, dur, phase=0.0):
    t = t_(dur)
    if np.isscalar(f):
        return np.sin(2 * np.pi * f * t + phase)
    return np.sin(2 * np.pi * np.cumsum(f) / SR + phase)


def saw(f, dur):
    t = t_(dur)
    if np.isscalar(f):
        ph = (f * t) % 1.0
    else:
        ph = (np.cumsum(f) / SR) % 1.0
    return 2 * ph - 1


def square(f, dur):
    return np.sign(sine(f, dur))


def noise(dur):
    return rng.uniform(-1, 1, int(SR * dur))


def sweep(f0, f1, dur, expo=True):
    t = t_(dur)
    if expo:
        f = f0 * (f1 / f0) ** (t / dur)
    else:
        f = f0 + (f1 - f0) * (t / dur)
    return f


def env(n, a=0.01, d=None, curve=3.0):
    """attack/decay envelope over n samples"""
    t = np.linspace(0, 1, n)
    atk = np.minimum(1, t / max(a, 1e-4))
    dec = np.exp(-curve * t) if d is None else np.clip(1 - (t - a) / max(d, 1e-4), 0, 1)
    return atk * dec


def fade(x, a=0.005, r=0.03):
    n = len(x)
    ai, ri = int(SR * a), int(SR * r)
    x = x.copy()
    if ai > 0:
        x[:ai] *= np.linspace(0, 1, ai)
    if ri > 0:
        x[-ri:] *= np.linspace(1, 0, ri)
    return x


def fft_filter(x, lo=None, hi=None):
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), 1 / SR)
    m = np.ones_like(fr)
    if lo:
        m *= 1 / (1 + (lo / np.maximum(fr, 1e-3)) ** 4)
    if hi:
        m *= 1 / (1 + (fr / hi) ** 4)
    return np.fft.irfft(X * m, len(x))


def mix(*parts):
    n = max(len(p) for p, _ in parts)
    o = np.zeros(n)
    for p, g in parts:
        o[: len(p)] += p * g
    return o


def at(x, offset, total):
    o = np.zeros(int(SR * total))
    i = int(SR * offset)
    o[i:i + len(x)] += x[: max(0, len(o) - i)]
    return o


def tremolo(x, rate, depth=0.5):
    t = np.arange(len(x)) / SR
    return x * (1 - depth + depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t)))


def reverb(x, delays=(0.031, 0.067, 0.113), gains=(0.35, 0.25, 0.15)):
    o = np.concatenate([x, np.zeros(int(SR * (max(delays) + 0.3)))])
    for d, g in zip(delays, gains):
        i = int(SR * d)
        o[i:i + len(x)] += x * g
    return o


def crackle(dur, density=60, hi=3500):
    n = int(SR * dur)
    o = np.zeros(n)
    for _ in range(int(density * dur)):
        i = rng.integers(0, n - 200)
        l = rng.integers(40, 200)
        o[i:i + l] += rng.uniform(-1, 1, l) * np.exp(-np.linspace(0, 6, l)) * rng.uniform(0.3, 1)
    return fft_filter(o, lo=800, hi=hi)


def ping(f, dur, g=1.0, harm=(1.0, 2.76, 5.4)):
    out = np.zeros(int(SR * dur))
    for i, h in enumerate(harm):
        out += sine(f * h, dur) * env(len(out), 0.002, curve=5 + 2 * i) * (0.6 ** i)
    return out * g


def whoosh(dur, f0, f1, g=1.0):
    n = noise(dur)
    band = fft_filter(n, lo=200)
    t = np.linspace(0, 1, len(n))
    cut = f0 * (f1 / f0) ** t
    # crude time-varying lowpass: blend of progressively filtered copies
    outs = [fft_filter(band, hi=f) for f in np.geomspace(f0, f1, 8)]
    o = np.zeros(len(n))
    for k, seg in enumerate(np.array_split(np.arange(len(n)), 8)):
        o[seg] = outs[k][seg]
    return o * np.sin(np.pi * t) ** 1.5 * g


def boom(dur=1.2, f0=140, f1=28, g=1.0):
    body = sine(sweep(f0, f1, dur), dur) * env(int(SR * dur), 0.002, curve=3.2)
    rumble = fft_filter(noise(dur), hi=300) * env(int(SR * dur), 0.001, curve=2.4)
    crack = fft_filter(noise(0.18), lo=1500) * env(int(SR * 0.18), 0.0005, curve=9)
    return mix((body, 0.9), (rumble, 0.8), (crack, 0.5)) * g


# ----------------------------------------------------------------------------- sounds
S = {}


def snd(name, data, vol=0.9, vol_scale=True):
    data = np.asarray(data, dtype=np.float64)
    data = fade(data, 0.003, 0.04)
    pk = np.max(np.abs(data)) + 1e-9
    S[name] = data / pk * vol


# Beamer sword
snd("beam_charge", mix((sine(sweep(180, 1400, 1.2), 1.2) * env(int(SR * 1.2), 0.2, d=0.8), 0.6),
                       (saw(sweep(90, 700, 1.2), 1.2) * 0.25, 1),
                       (fft_filter(noise(1.2), lo=2500) * np.linspace(0, 1, int(SR * 1.2)) * 0.3, 1)))
beam = mix((saw(110, 1.0) * 0.5, 1), (sine(440, 1.0) * 0.3, 1), (sine(880 + 40 * np.sin(np.arange(SR) / SR * 40), 1.0) * 0.2, 1),
           (fft_filter(noise(1.0), lo=3000) * 0.18, 1))
snd("beam_loop", tremolo(beam, 28, 0.4), 0.7)
snd("dash", mix((whoosh(0.5, 400, 7000), 1.0), (boom(0.4, 90, 40) * 0.6, 1)), 0.9)

# Tropical bow staff
vines = crackle(0.9, 140, 5000) * env(int(SR * 0.9), 0.05, curve=1.6)
snd("vines", mix((vines, 0.9), (whoosh(0.9, 300, 2500) * 0.5, 1), (sine(sweep(120, 70, 0.9), 0.9) * 0.4, 1)), 0.85)
bub = np.zeros(int(SR * 0.9))
for i, off in enumerate((0.0, 0.18, 0.4, 0.62)):
    f0 = 300 + 90 * i
    b = sine(sweep(f0, f0 * 2.2, 0.16), 0.16) * env(int(SR * 0.16), 0.01, curve=4)
    bub += at(b, off, 0.9)
snd("bubble", mix((bub, 1.0), (ping(1800, 0.9, 0.12), 1)), 0.8)
snd("bubble_drain", tremolo(mix((sine(sweep(900, 150, 0.9), 0.9), 0.7), (saw(sweep(450, 75, 0.9), 0.9), 0.2)), 14, 0.5) * env(int(SR * 0.9), 0.01, curve=1.8), 0.8)
snd("kiwi_eat", mix((crackle(0.45, 160, 6000) * env(int(SR * 0.45), 0.003, curve=4), 1.0),
                    (sine(sweep(500, 160, 0.2), 0.2) * env(int(SR * 0.2), 0.003, curve=5) * 0.5, 1)), 0.8)

# Ice axe
shimmer = mix(*[(at(ping(f, 0.7, 0.5, (1.0, 2.0, 3.0)), o, 1.1), 1) for f, o in ((2200, 0.0), (2900, 0.08), (3600, 0.16), (4400, 0.24))])
snd("freeze", mix((shimmer, 1.0), (crackle(1.1, 200, 8000) * env(int(SR * 1.1), 0.02, curve=2.5), 0.7),
                  (whoosh(1.1, 800, 6000) * 0.4, 1)), 0.85)
snd("ice_crit", mix((fft_filter(noise(0.35), lo=2000) * env(int(SR * 0.35), 0.0005, curve=9), 1.0),
                    (at(ping(1600, 0.4, 0.8), 0.0, 0.5), 0.8), (at(ping(2400, 0.4, 0.6), 0.05, 0.5), 0.6)), 0.9)

# Dynasty sword
snd("coin_throw", mix((ping(1500, 0.35, 0.8, (1.0, 2.4, 4.1)), 0.9), (whoosh(0.3, 1500, 6000) * 0.4, 1)), 0.75)
snd("coin_hit", mix((ping(950, 0.45, 1.0, (1.0, 2.5, 3.9)), 0.9), (boom(0.2, 220, 80) * 0.5, 1)), 0.8)
snd("spin", tremolo(whoosh(1.0, 500, 5000), 12, 0.6) + 0.2 * saw(sweep(200, 420, 1.0), 1.0) * env(int(SR), 0.1, curve=2), 0.8)
snd("double_jump", mix((sine(sweep(260, 700, 0.22), 0.22) * env(int(SR * 0.22), 0.005, curve=5), 0.8),
                       (whoosh(0.3, 800, 5000) * 0.6, 1)), 0.7)

# Inferno sword
snd("lava_wall", mix((fft_filter(noise(1.6), hi=350) * env(int(SR * 1.6), 0.15, curve=1.0) * 1.2, 1.0),
                     (crackle(1.6, 90, 4500), 0.8), (sine(sweep(80, 45, 1.6), 1.6) * 0.5, 1)), 0.9)
roar = mix((saw(sweep(95, 60, 1.2) * (1 + 0.04 * np.sin(np.arange(int(SR * 1.2)) / SR * 40)), 1.2), 0.6),
           (fft_filter(noise(1.2), lo=300, hi=2500), 0.5), (crackle(1.2, 120, 5000), 0.7))
snd("blaze_summon", roar * env(int(SR * 1.2), 0.03, curve=1.8), 0.85)
arp = mix(*[(at(ping(f, 0.6, 0.8, (1.0, 2.0)), o, 1.2), 1) for f, o in ((523, 0.0), (659, 0.12), (784, 0.24), (1047, 0.36))])
snd("heal", reverb(arp), 0.8)

# Hellsfire axe
snd("inferno_slam", mix((boom(1.5, 150, 25), 1.0), (crackle(1.5, 140, 5000) * env(int(SR * 1.5), 0.01, curve=1.5), 0.6)), 0.95)
snd("cleave", mix((whoosh(0.9, 300, 4500), 0.9), (crackle(0.9, 150, 5500), 0.7), (boom(0.5, 110, 40) * 0.5, 1)), 0.85)

# Dragon's fang
snd("dragon_breath", mix((fft_filter(noise(1.6), lo=500, hi=4200) * env(int(SR * 1.6), 0.25, d=1.2), 0.9),
                         (saw(sweep(70, 55, 1.6), 1.6) * 0.3 * env(int(SR * 1.6), 0.2, curve=1.2), 1),
                         (crackle(1.6, 60, 3500) * 0.3, 1)), 0.85)
dr = mix((saw(sweep(180, 70, 1.3) * (1 + 0.06 * np.sin(np.arange(int(SR * 1.3)) / SR * 30)), 1.3), 0.6),
         (square(sweep(90, 45, 1.3), 1.3), 0.3), (fft_filter(noise(1.3), lo=400, hi=3000) * 0.5, 1))
dr = dr * env(int(SR * 1.3), 0.02, curve=1.7)
snd("dragon_slam", mix((dr, 0.8), (at(boom(0.9, 130, 30), 0.4, 1.3), 0.9)), 0.95)

# Heavy hitter mace
snd("cataclysm_charge", mix((sine(sweep(40, 220, 1.0), 1.0) * np.linspace(0.2, 1, int(SR)), 0.8),
                            (fft_filter(noise(1.0), hi=900) * np.linspace(0, 1, int(SR)) ** 2, 0.7),
                            (saw(sweep(60, 300, 1.0), 1.0) * 0.2, 1)), 0.85)
snd("cataclysm_blast", reverb(mix((boom(1.8, 170, 20), 1.0), (whoosh(1.2, 600, 9000) * 0.6, 1),
                                  (crackle(1.0, 80, 3000) * 0.3, 1)), (0.05, 0.11, 0.21), (0.4, 0.3, 0.2)), 1.0)
tc = mix((sine(sweep(50, 400, 2.0), 2.0) * np.linspace(0.1, 1, int(SR * 2.0)), 0.7),
         (saw(sweep(55, 220, 2.0), 2.0) * np.linspace(0.1, 1, int(SR * 2.0)) * 0.35, 1),
         (fft_filter(noise(2.0), lo=1200) * np.linspace(0, 1, int(SR * 2.0)) ** 3 * 0.4, 1))
snd("titan_charge", tremolo(tc, 9, 0.35), 0.85)
snd("titan_slam", reverb(mix((boom(1.6, 120, 22), 1.0), (at(fft_filter(noise(0.8), lo=700, hi=7000) * env(int(SR * 0.8), 0.001, curve=4), 0.02, 1.6), 0.7))), 1.0)
snd("colossus_hit", mix((boom(0.45, 110, 45) * 1.0, 1.0), (fft_filter(noise(0.1), lo=1800) * env(int(SR * 0.1), 0.0005, curve=9) * 0.4, 1)), 0.85)
snd("colossus_shockwave", mix((boom(0.9, 130, 35), 1.0), (whoosh(0.7, 500, 5000) * 0.5, 1)), 0.9)
snd("double_hit", mix((at(fft_filter(noise(0.1), lo=1800) * env(int(SR * 0.1), 0.0005, curve=9), 0.0, 0.4), 1.0),
                      (at(fft_filter(noise(0.1), lo=1800) * env(int(SR * 0.1), 0.0005, curve=9), 0.11, 0.4), 1.0),
                      (at(ping(700, 0.2, 0.6), 0.11, 0.4), 0.6)), 0.85)

# Global
fan = mix(*[(at(ping(f, 0.8, 0.8, (1.0, 2.0, 3.0)), o, 1.8), 1) for f, o in ((392, 0.0), (523, 0.14), (659, 0.28), (784, 0.42), (1047, 0.56))])
snd("upgrade", reverb(mix((fan, 1.0), (sine(sweep(200, 800, 0.5), 0.5) * env(int(SR * 0.5), 0.1, curve=3) * 0.2, 1))), 0.9)
rr = mix(*[(at(ping(f, 0.5, 0.7, (1.0, 2.0)), o, 1.4), 1) for f, o in
           ((1568, 0.0), (1175, 0.06), (1397, 0.12), (988, 0.18), (1319, 0.24), (1760, 0.3))])
snd("reroll", reverb(mix((rr, 1.0), (at(whoosh(1.0, 500, 6000), 0.0, 1.4), 0.4))), 0.8)
bell = mix((ping(110, 2.0, 1.0, (1.0, 2.0, 2.4, 3.0)), 1.0), (sine(55, 2.0) * env(int(SR * 2.0), 0.02, curve=1.5), 0.8),
           (saw(sweep(98, 98.0, 2.0), 2.0) * env(int(SR * 2.0), 0.3, curve=1.4) * 0.25, 1))
snd("hh_claim", reverb(bell, (0.09, 0.19, 0.31), (0.4, 0.3, 0.2)), 0.95)
snd("passive_chime", reverb(ping(1319, 0.6, 0.6, (1.0, 2.0, 3.0))), 0.6)
snd("error", mix((square(150, 0.28) * env(int(SR * 0.28), 0.005, curve=4), 0.5), (saw(sweep(200, 90, 0.28), 0.28) * env(int(SR * 0.28), 0.005, curve=3), 0.5)), 0.6)


def write_ogg(name, data):
    data = np.clip(data, -1, 1)
    pcm = (data * 32767).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    out = os.path.join(OUT, name + ".ogg")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path, "-c:a", "libvorbis", "-q:a", "4", "-ac", "1", out], check=True)
    os.remove(wav_path)


if __name__ == "__main__":
    events = {}
    for name, data in S.items():
        write_ogg(name, data)
        events[name] = {"sounds": [{"name": f"weapons:{name}", "stream": False}], "subtitle": f"weapons.subtitle.{name}"}
    # sounds.json: events keyed by name, files at assets/weapons/sounds/<name>.ogg
    with open(os.path.join(ROOT, "sounds.json"), "w") as f:
        json.dump(events, f, indent=2)
    lang = {f"weapons.subtitle.{n}": n.replace("_", " ").title() for n in S}
    os.makedirs(os.path.join(ROOT, "lang"), exist_ok=True)
    with open(os.path.join(ROOT, "lang", "en_us.json"), "w") as f:
        json.dump(lang, f, indent=2)
    print("rendered", len(S), "sounds:", ", ".join(sorted(S)))
