"""Composes and synthesises the soundtrack of an exported video (exporter.py) from the video's own data. Pure numpy: no samples, no files,
no network; the same plan always gives the same track.     compose(plan) -> float32 array (samples, 2) at SR; write_wav(path, audio)

plan = {"duration" (seconds of the video), "request" (the instruction), "model", "speed", "scene" ({objects: {id: {color, position}}}),
        "result" (the simulator's result event or None: executed, error, success, strict_flags, final), "code" (the program or ""),
        "sections": {name: (start, end) in seconds; any of scene, prompt, handoff, think, code, sim, hold},
        "events": {"prompt_words", "reason_words", "code_lines", "flags": [seconds], "code_text": [the lines], "pulse": seconds or None, "program_end": seconds or None}}

What the video decides:  the request text seeds everything (key, mode, tempo, the progressions, the drum feel, the phrasing), so the same
request always sounds the same and a new request sounds new.  The cubes are the motif: every cube is a note (its colour picks the scale step,
its place on the table pans it), heard once at the very start as the scene is shown, then as the theme of the simulation; cubes that moved during
the run are the accented notes.  The model's name picks the lead timbre and the colour of the pad, so the two sides of a duel sound alike in
their harmony (same request) but different in their voice.  The intro stays calm: a pad on two chords, a soft pluck for every word as the request is typed in, a riser that
lands on the hand-off pulse.  The reasoning gets a steady electric-piano arpeggio with sparkles on its words; every line of code strikes a note of a
rising arpeggio as it appears (a close_gripper line is a low grab, an open_gripper line a high release, a comment or a def is silent), and the
drums come in underneath.  The simulation is the full groove: kick, snare and hats (half time below 1x,
driving above 8x: the speed sets the feel, the number of robot calls per shown second the density), a bass line, the cube theme in phrases; every
collision flag of the simulator is a dissonant stab at its moment in the picture; the pad breathes with the kick, and every eighth bar the
drums step out for a bar and come back with a crash.  The program's end gets its cadence: a bright major chord with
bells if the task was solved, a plain tonic if nothing was checked, a dark minor chord if the run failed; it rings through the final hold.
Everything is rendered into three buses (dry, delay send, reverb send), then a ping-pong delay, a convolution reverb with a synthetic tail, a
soft clipper, a look-ahead peak limiter (ceiling -1 dBFS) and a loudness fit around -15 dBFS RMS, which social platforms play back unchanged."""
import hashlib
import math
import re
import struct

import numpy as np

SR = 48000
F32 = np.float32
MODES = {"ionian": [0, 2, 4, 5, 7, 9, 11], "lydian": [0, 2, 4, 6, 7, 9, 11], "mixolydian": [0, 2, 4, 5, 7, 9, 10], "dorian": [0, 2, 3, 5, 7, 9, 10]}
MINOR = {"dorian"}
COLOR_STEP = {"red": 0, "orange": 1, "yellow": 2, "green": 3, "cyan": 4, "blue": 5, "purple": 6, "pink": 7, "white": 8, "gray": 9, "brown": 10, "black": 11}
INTRO_PROGS = [[0, 3], [0, 5], [0, 1], [3, 0]]                      # scale degrees, two bars each, while the request is typed
THINK_PROGS = [[0, 5, 3, 4], [0, 3, 5, 4], [5, 3, 0, 4], [0, 2, 3, 4]]    # one bar each, under the reasoning and the code
SIM_PROGS = [[0, 4, 5, 3], [5, 3, 0, 4], [0, 3, 0, 4], [0, 5, 1, 4], [3, 4, 5, 0]]
RHYTHMS = [[0, 0.5, 1, 1.5, 2, 3], [0, 0.75, 1.5, 2, 2.5, 3.5], [0, 1, 1.5, 2, 2.5, 3], [0.5, 1, 1.5, 2.5, 3, 3.5]]     # phrase onsets in beats
CEILING, TARGET_RMS = 10 ** (-1 / 20), 10 ** (-15 / 20)


# ---- small helpers
def _t(n):
    return np.arange(n, dtype=F32) / SR


def _n(seconds):
    return max(1, int(round(seconds * SR)))


def _env(n, a, d, s, r, gate):
    """Attack (linear), exponential decay to the sustain level, release from the gate (in samples) on."""
    t = _t(n)
    e = np.minimum(t / max(a, 1e-4), 1) * (s + (1 - s) * np.exp(-t / max(d, 1e-4)))
    if gate < n:
        e[gate:] *= np.exp(-(t[gate:] - t[gate]) / max(r, 1e-3))
    return e.astype(F32)


def _midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def _table(harmonics):
    """One period of a waveform with the given harmonic amplitudes."""
    x = np.arange(2048) / 2048
    return sum(a * np.sin(2 * np.pi * (k + 1) * x) for k, a in enumerate(harmonics)).astype(F32)


DARK, BRIGHT = _table([1, 0.5, 0.25, 0.1]), _table([1, 0.7, 0.5, 0.4, 0.3, 0.22, 0.16, 0.12, 0.08, 0.05])
BASS_T = _table([1, 0.45, 0.2, 0.08])
_noise = np.random.RandomState(7).standard_normal(SR * 3).astype(F32)       # one fixed noise, read from random offsets (deterministic)


def _wave(table, freq, n, detune_cents=0.0, vibrato=0.0):
    """Wavetable oscillator; freq may be a number or an array (one value per sample)."""
    f = np.broadcast_to(np.asarray(freq, dtype=np.float64) * 2 ** (detune_cents / 1200), (n,))
    if vibrato:
        f = f * (1 + vibrato * np.sin(2 * np.pi * 5.2 * _t(n)))
    ph = np.cumsum(f) / SR
    return table[(ph * 2048).astype(np.int64) & 2047]


def _noise_slice(n, rng):
    i = rng.randint(0, len(_noise) - n) if n < len(_noise) else 0
    return np.resize(_noise[i:], n) if n >= len(_noise) else _noise[i:i + n]


def _hp_noise(n, rng, order=1):
    y = _noise_slice(n, rng)
    for _ in range(order):                                      # differencing: a crude but cheap high-pass
        y = np.diff(y, prepend=F32(0))
    return y / (2 ** order)


# ---- instruments: each returns a mono float32 signal starting at the note's onset
def pad(f, dur, bright=0.35, duck=None):
    """Two detuned wavetable voices, one per channel (stereo), a slow brightness swell between a dark and a bright table.
    duck = beat length: the pad dips on every beat and swells back, breathing with the kick."""
    n = _n(dur + 1.2)
    t = _t(n)
    b = np.clip(bright + 0.18 * np.sin(2 * np.pi * 0.11 * t + 1.0), 0, 1).astype(F32)
    e = _env(n, 0.7, 2.0, 0.85, 1.0, _n(dur)) * 0.5
    if duck:
        e *= 1 - 0.4 * np.exp(-((t % duck) / duck) * 6).astype(F32)
    voices = [((1 - b) * _wave(DARK, f, n, cents, 0.0015) + b * _wave(BRIGHT, f, n, cents, 0.0015)) * e for cents in (-7, 7)]
    return np.stack(voices, axis=1)


def pluck(f, vel=0.8, dur=0.3, ratio=1.0, bright=2.5):
    n = _n(min(dur + 0.8, 2.2))
    t = _t(n)
    idx = bright * vel * np.exp(-t * 9)
    mod = np.sin(2 * np.pi * f * ratio * t)
    y = np.sin(2 * np.pi * f * t + idx * mod) + 0.4 * np.sin(2 * np.pi * f * 1.004 * t + idx * mod)
    return (y * np.exp(-t * 3.2) * (1 - np.exp(-t * 900)) * _env(n, 0.001, 9, 1, 0.25, _n(dur)) * 0.5 * vel).astype(F32)


def epiano(f, vel=0.7, dur=0.4):
    n = _n(dur + 0.9)
    t = _t(n)
    y = np.sin(2 * np.pi * f * t + 1.6 * vel * np.exp(-t * 2.5) * np.sin(2 * np.pi * f * t))
    y += 0.12 * np.sin(2 * np.pi * f * 7 * t) * np.exp(-t * 40)      # the tine
    return (y * np.exp(-t * 1.8) * _env(n, 0.002, 9, 1, 0.3, _n(dur)) * 0.5 * vel).astype(F32)


def bell(f, vel=0.5, dur=1.4):
    n = _n(dur)
    t = _t(n)
    y = np.zeros(n, F32)
    for ratio, amp, decay in ((1, 1, 1.4), (2.76, 0.45, 2.6), (5.4, 0.2, 4.5), (8.93, 0.07, 7)):
        if f * ratio < SR / 2.2:
            y += amp * np.sin(2 * np.pi * f * ratio * t) * np.exp(-t * decay)
    return (y * (1 - np.exp(-t * 1500)) * 0.35 * vel).astype(F32)


def bass(f, dur, vel=0.8):
    n = _n(dur + 0.15)
    y = _wave(BASS_T, f, n) * _env(n, 0.006, 0.35, 0.6, 0.06, _n(dur))
    return (np.tanh(1.6 * y) * 0.6 * vel).astype(F32)


def sub(f, dur):
    n = _n(dur)
    return (np.sin(2 * np.pi * f * _t(n)) * _env(n, 0.01, 1.5, 0.3, 0.3, _n(dur * 0.8)) * 0.6).astype(F32)


def kick(vel=1.0, rng=None):
    n = _n(0.42)
    t = _t(n)
    ph = np.cumsum(48 + 120 * np.exp(-t * 26)) / SR
    y = np.sin(2 * np.pi * ph) * np.exp(-t * 7) + 0.25 * _hp_noise(n, rng, 2) * np.exp(-t * 350)
    return (np.tanh(1.5 * y) * 0.9 * vel).astype(F32)


def snare(vel=0.8, rng=None):
    n = _n(0.32)
    t = _t(n)
    y = _hp_noise(n, rng) * np.exp(-t * 17) * 0.9 + 0.5 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 28)
    return (y * 0.75 * vel).astype(F32)


def hat(vel=0.5, open_=False, rng=None):
    n = _n(0.45 if open_ else 0.09)
    t = _t(n)
    return (_hp_noise(n, rng, 3) * np.exp(-t * (8 if open_ else 65)) * 0.5 * vel).astype(F32)


def crash(vel=0.5, rng=None):
    n = _n(2.6)
    t = _t(n)
    return (_hp_noise(n, rng, 2) * np.exp(-t * 1.5) * (1 - np.exp(-t * 200)) * 0.35 * vel).astype(F32)


def riser(f, dur, rng):
    """Detuned sines sweeping up an octave plus a noise swell: everything rises to the hand-off pulse."""
    n = _n(dur)
    t = _t(n)
    k = (t / dur) ** 2
    y = np.zeros(n, F32)
    for cents in (-12, 0, 12):
        y += _wave(DARK, f * 2 ** (k + cents / 1200), n)
    y = y * k * 0.25 + _hp_noise(n, rng) * k ** 2.5 * 0.5
    return (y * np.minimum(1, (dur - t) / 0.03)).astype(F32)


def stab(f, rng):
    """A collision: a short, dissonant cluster with a thud."""
    y = pluck(f, 1.0, 0.12, 1.41, 5) + pluck(f * 2 ** (1 / 12), 1.0, 0.12, 1.41, 5) + pluck(f * 2 ** (6 / 12), 0.7, 0.12, 1, 4)
    k = kick(0.8, rng)
    y[:len(k)] += k
    return y * 0.7


# ---- the mix: three stereo buses, notes are added at a time in seconds
class Mix:
    def __init__(self, seconds):
        self.n = _n(seconds)
        self.dry, self.verb, self.dly = (np.zeros((self.n, 2), F32) for _ in range(3))

    def add(self, y, t, pan=0.0, gain=1.0, verb=0.25, delay=0.0):
        i = int(round(t * SR))
        if i >= self.n or i + len(y) <= 0:
            return
        if i < 0:
            y, i = y[-i:], 0
        y = y[:self.n - i]
        a = (math.pi / 4) * (pan + 1)                               # equal-power panning; a stereo signal keeps its own image, pan shifts it
        if y.ndim == 2:
            st = y * gain * np.array([1 - max(pan, 0) * 0.7, 1 + min(pan, 0) * 0.7], F32)
        else:
            st = np.stack((y * (gain * math.cos(a)), y * (gain * math.sin(a))), axis=1)
        self.dry[i:i + len(y)] += st
        if verb:
            self.verb[i:i + len(y)] += st * verb
        if delay:
            self.dly[i:i + len(y)] += st * delay

    def render(self, beat, rng):
        """The buses in place (a 20-minute export is a 460 MB bus; nothing here keeps a second copy of one)."""
        out = self.dry
        d = _n(beat * 0.75)                                         # ping-pong at a dotted eighth, three repeats, channels swapped
        for k, g in enumerate((0.42, 0.22, 0.1)):
            s = d * (k + 1)
            if s < self.n:
                tap = self.dly[:-s] * g
                out[s:] += tap[:, ::-1] if k % 2 == 0 else tap
                self.verb[s:] += tap * 0.5
        self.dly = None
        _reverb(self.verb, out, rng)
        self.verb = None
        return out


def _reverb(x, out, rng, seconds=2.1):
    """Adds the reverb of x to out: convolution with a synthetic tail (decaying noise, a different one per channel, darker as it fades, 18 ms of pre-delay)."""
    L = _n(seconds)
    t = _t(L)
    ir = np.zeros((L, 2), F32)
    for c in range(2):
        nz = rng.standard_normal(L).astype(F32)
        nz = np.convolve(nz, np.ones(6, F32) / 6, mode="same")       # a little low-pass, so the tail is not hissy
        ir[:, c] = nz * np.exp(-t * (6.9 / seconds)) * (1 - np.exp(-t * 60))
    ir[:_n(0.018)] = 0
    ir *= 0.08 * 0.9 / np.sqrt(np.mean(ir ** 2) * L)
    _fft_convolve(x, ir, out)


def _fft_convolve(x, ir, out):
    """Overlap-add block convolution of a stereo signal with a stereo impulse response, added to out (memory stays flat for long videos)."""
    n, L = len(x), len(ir)
    B = 1 << 17
    nfft = 1 << int(math.ceil(math.log2(B + L - 1)))
    IR = np.fft.rfft(ir, nfft, axis=0)
    for i in range(0, n, B):
        blk = np.fft.irfft(np.fft.rfft(x[i:i + B], nfft, axis=0) * IR, nfft, axis=0)
        m = min(n - i, B + L - 1)
        out[i:i + m] += blk[:m].astype(F32)


def _limit(x, lookahead=0.004, release=0.12):
    """Look-ahead peak limiter: the gain needed per 128-sample block, held over the look-ahead, released exponentially, interpolated."""
    blk = 128
    nb = (len(x) + blk - 1) // blk
    peaks = np.abs(np.pad(x, ((0, nb * blk - len(x)), (0, 0)))).reshape(nb, blk, 2).max(axis=(1, 2))
    g = np.minimum(1.0, CEILING / np.maximum(peaks, 1e-9))
    la = max(1, int(lookahead * SR / blk))
    held = np.minimum.reduce([np.pad(g, (k, 0), constant_values=1)[:nb] for k in range(la + 1)])
    rel = math.exp(-blk / (release * SR))
    cur = 1.0
    for i in range(nb):                                             # release: the gain climbs back slowly, drops instantly
        cur = held[i] if held[i] < cur else cur + (1 - rel) * (held[i] - cur)
        held[i] = cur
    gain = np.interp(np.arange(len(x)), np.arange(nb) * blk + blk / 2, held).astype(F32)
    x *= gain[:, None]
    return x


def _master(x, mix):
    x *= 1.1
    np.tanh(x, out=x)                                               # a soft clipper catches the odd transient before the loudness fit
    x /= math.tanh(1.1)
    win = _n(0.4)
    blocks = x[:len(x) // win * win].reshape(-1, win, 2)
    rms = np.sqrt(np.mean(blocks ** 2, axis=(1, 2))) if len(blocks) else np.array([np.sqrt(np.mean(x ** 2))])
    loud = np.sort(rms)[len(rms) // 3:]                             # the quietest third of the blocks (pauses, the fade) is left out
    x *= TARGET_RMS / max(float(np.mean(loud)), 1e-6)
    x = _limit(x)
    f_in, f_out = _n(0.05), _n(min(0.6, mix.n / SR / 4))
    x[:f_in] *= np.linspace(0, 1, f_in, dtype=F32)[:, None]
    x[-f_out:] *= np.linspace(1, 0, f_out, dtype=F32)[:, None] ** 1.5
    return np.clip(x, -CEILING, CEILING).astype(F32)


# ---- harmony
class Key:
    """Root and mode; scale steps (0 = root) to MIDI; diatonic chords with voice leading (every chord moves as little as possible from the last)."""
    def __init__(self, root, mode):
        self.root, self.steps, self.minor = root, MODES[mode], mode in MINOR
        self.prev = None

    def midi(self, step, octave=4):
        """Scale step (0 = root, 7 = the root an octave up) in the given octave (4: from middle C's octave)."""
        return self.root % 12 + 12 * (octave + 1 + step // 7) + self.steps[step % 7]

    def chord(self, degree, add9=False):
        tones = [degree, degree + 2, degree + 4] + ([degree + 8] if add9 else [])
        classes = [self.midi(s) % 12 for s in tones]
        best, best_cost = None, 1e9
        for combo in np.ndindex(*(3,) * len(classes)):             # each tone in one of three octaves around middle C
            v = sorted(c + 12 * (4 + o) for c, o in zip(classes, combo))
            if v[-1] - v[0] > 19 or v[0] < 55 or v[-1] > 81:
                continue
            cost = sum(abs(a - b) for a, b in zip(v, self.prev)) if self.prev and len(self.prev) == len(v) else abs(v[0] - 62)
            if cost < best_cost:
                best, best_cost = v, cost
        self.prev = best
        return best

    def root_of(self, degree):
        return self.midi(degree, 2)                                 # bass register


def _seed(text):
    return int.from_bytes(hashlib.sha256((text or "").strip().lower().encode()).digest()[:4], "big")


def _thin(times, max_rate):
    """Keeps events at least 1 / max_rate seconds apart (the first of each cluster)."""
    out, last = [], -1e9
    for t in times:
        if t - last >= 1 / max_rate:
            out.append(t); last = t
    return out


def motif_of(scene, result):
    """The cubes as notes: (scale step, pan, moved) per cube, sorted by id; moved = more than 2 cm away in the simulator's final state."""
    objs = (scene or {}).get("objects") or {}
    final = ((result or {}).get("final") or {}).get("objects") or {}
    notes = []
    for i in sorted(objs, key=lambda k: (len(k), k)):
        o = objs[i]
        pos = o.get("position") or [0.45, 0, 0]
        fin = (final.get(i) or {}).get("position")
        moved = bool(fin) and math.dist(pos[:2], fin[:2]) > 0.02
        notes.append((COLOR_STEP.get(str(o.get("color", "")).lower(), 0), float(np.clip(-pos[1] / 0.3, -1, 1)), moved))
    return notes[:16] or [(0, 0.0, False), (4, 0.0, False), (2, 0.0, False)]


# ---- the composition
def compose(plan):
    end = float(plan["duration"])
    seed = _seed(plan.get("request") or plan.get("code") or "run")
    rng = np.random.RandomState(seed)
    key = Key(seed % 12, ["ionian", "ionian", "lydian", "mixolydian", "dorian"][(seed >> 8) % 5])
    bpm = 92 + ((seed >> 12) % 5) * 4
    beat, bar = 60 / bpm, 240 / bpm
    sec, ev = plan.get("sections") or {}, plan.get("events") or {}
    result = plan.get("result") or {}
    motif = motif_of(plan.get("scene"), result)
    speed = float(plan.get("speed") or 1)
    voice = _seed(plan.get("model") or "") % 3                      # the model's voice: 0 electric piano, 1 plucks, 2 bells (and the pad's colour)
    lead = (epiano, lambda f, v, d: pluck(f, v, d, 1.0, 2.0), lambda f, v, d: bell(f, v * 0.8, d + 0.8))[voice]
    tint = (0.45, 0.6, 0.3)[voice]
    mix = Mix(end)
    sim = sec.get("sim")
    pulse = ev.get("pulse")
    t_sim = sim[0] if sim else end
    t_end = ev.get("program_end") or t_sim
    hold = (sec.get("hold") or (t_end, end))[0]

    # -- chords: the phases (intro / reasoning and code / simulation) each start their own bar grid, the pad bridges the seams
    chords = []                                                     # (start, end, degree, voicing)

    def schedule(t0, t1, prog, span):
        k = 0
        while t0 < t1 - 0.05:
            d = prog[k % len(prog)]
            chords.append([t0, min(t0 + span, t1), d, key.chord(d, add9=span > bar)])
            t0, k = t0 + span, k + 1
    intro_prog = INTRO_PROGS[(seed >> 16) % len(INTRO_PROGS)]
    think_prog = THINK_PROGS[(seed >> 20) % len(THINK_PROGS)]
    sim_prog = SIM_PROGS[(seed >> 24) % len(SIM_PROGS)]
    grid_b = pulse if pulse is not None else t_sim
    schedule(0, grid_b, intro_prog, 2 * bar)
    schedule(grid_b, t_sim, think_prog, bar)
    schedule(t_sim, t_end, sim_prog, bar)
    success, executed = result.get("success"), result.get("executed")
    failed = executed is False or success is False or bool(result.get("error"))
    final_deg = (5 if not key.minor else 0) if failed else 0
    chords.append([t_end, end, final_deg, key.chord(final_deg, add9=not failed)])

    def chord_at(t):
        for c in chords:
            if c[0] <= t < c[1]:
                return c
        return chords[-1]

    for t0, t1, d, voicing in chords:
        phase_gain = 0.55 if t1 <= grid_b else 0.7 if t1 <= t_sim else 0.85
        for m in voicing:
            mix.add(pad(_midi(m), t1 - t0, 0.25 if t1 <= grid_b else tint, beat if t_sim <= t0 < t_end else None), t0, pan=(m % 5 - 2) * 0.35, gain=0.16 * phase_gain, verb=0.5)

    # -- the scene: the cubes chime once, left to right in pitch order of the ids, as the sharp first frame is shown
    scene_t = (sec.get("scene") or (0, 0))[0]
    for k, (step, pan, moved) in enumerate(motif[:10]):
        mix.add(bell(_midi(key.midi(step, 5)), 0.45), scene_t + 0.25 + k * 0.16, pan=pan, gain=0.4, verb=0.7, delay=0.3)

    # -- the request: a pluck for every typed word, walking the motif; the last word lands on the root
    words = _thin(ev.get("prompt_words") or [], 7)
    for k, t in enumerate(words):
        step = motif[k % len(motif)][0] if k < len(words) - 1 else 7
        mix.add(pluck(_midi(key.midi(step, 4)), 0.6, 0.25), t, pan=-0.3 + 0.6 * (k % 3) / 2, gain=0.42, verb=0.35, delay=0.3)

    # -- the hand-off: a riser into the pulse, a soft hit on it
    if pulse is not None:
        h0 = (sec.get("handoff") or (pulse - 1.0, pulse))[0]
        mix.add(riser(_midi(key.midi(0, 3)), max(pulse - h0, 0.3), rng), h0, gain=0.5, verb=0.6)
        mix.add(kick(0.7, rng), pulse, gain=0.7, verb=0.1)
        mix.add(bass(_midi(key.root_of(0)), 1.2, 0.8), pulse, gain=0.7, verb=0.0)
        mix.add(bell(_midi(key.midi(4, 5)), 0.5), pulse, pan=0.2, gain=0.4, verb=0.7, delay=0.4)
        mix.add(crash(0.35, rng), pulse, gain=0.4, verb=0.3)

    # -- the reasoning: an eighth-note electric-piano arpeggio on the chords, a sparkle on every n-th word, bass on each chord root
    think, code = sec.get("think"), sec.get("code")
    texture = (think or code or (None, None))[0]
    if texture is not None:
        t0, t1 = texture, t_sim
        pat = [0, 1, 2, 1, 3, 2, 1, 2] if rng.rand() < 0.5 else [0, 2, 1, 3, 2, 1, 0, 1]
        k, t = 0, grid_b
        while t < t1 - 0.05:
            c = chord_at(t)
            v = c[3] + [c[3][0] + 12]
            if t >= t0:
                in_code = code and t >= code[0]
                mix.add(epiano(_midi(v[pat[k % 8] % len(v)]), 0.55 if k % 2 == 0 else 0.4, beat * 0.45), t,
                        pan=0.55 * math.sin(k * 0.9), gain=0.5 if not in_code else 0.4, verb=0.3, delay=0.25)
                if k % 8 == 0:
                    mix.add(bass(_midi(key.root_of(c[2])), bar * 0.9, 0.7), t, gain=0.6, verb=0.0)
                if in_code and k % 4 == 0:                        # the drums enter softly under the code
                    mix.add(kick(0.55, rng), t, gain=0.6, verb=0.05)
                if in_code and k % 2 == 1:
                    mix.add(hat(0.35, False, rng), t, pan=0.3, gain=0.5)
            t, k = t + beat / 2, k + 1
        for k, t in enumerate(_thin(ev.get("reason_words") or [], 2.5)):
            step = motif[(k * 3) % len(motif)][0]
            mix.add(bell(_midi(key.midi(step, 5)), 0.3, 1.0), t, pan=motif[(k * 3) % len(motif)][1], gain=0.3, verb=0.8, delay=0.3)
        # every line of code strikes the next note of a rising arpeggio of the current chord (two octaves, then it starts over); the program's
        # vocabulary colours it: close_gripper is a low grab, open_gripper a high release, a comment or a def says nothing
        lines, texts = ev.get("code_lines") or [], ev.get("code_text") or []
        kinds = [("grab" if "close_gripper" in x else "release" if "open_gripper" in x else "quiet" if x.startswith(("#", "def ", "import ", "from ")) else "note")
                 for x in texts] + ["note"] * (len(lines) - len(texts))
        last, k = -1e9, 0
        for t, kind in zip(lines, kinds):
            if kind == "quiet" or t - last < 1 / 9:
                continue
            last, c = t, chord_at(t)
            v = c[3] + [m + 12 for m in c[3]]
            if kind == "grab":
                mix.add(bass(_midi(key.root_of(c[2])), beat * 0.5, 0.9), t, gain=0.6)
                mix.add(pluck(_midi(v[0] - 12), 0.8, 0.2, 1.0, 3.0), t, gain=0.45, verb=0.3)
            elif kind == "release":
                mix.add(bell(_midi(v[-1] + 12), 0.5, 1.2), t, pan=0.4, gain=0.4, verb=0.7, delay=0.4)
            else:
                mix.add(pluck(_midi(v[k % len(v)]), 0.55, 0.15, 2.0, 2.0), t, pan=-0.5 + (k % len(v)) / len(v), gain=0.42, verb=0.3, delay=0.35)
                k += 1
        if sim and t1 - t0 > 2 * beat:                              # a snare roll into the simulation
            for k in range(8):
                mix.add(snare(0.25 + 0.08 * k, rng), t_sim - beat + k * beat / 8, gain=0.5, verb=0.2)

    # -- the simulation: the groove, the bass line, the cube theme in phrases, stabs on the collision flags
    if sim and t_end - t_sim > 0.3:
        calls = result.get("calls") or len(re.findall(r"robot\.\w+\(", plan.get("code") or "")) or 4
        density = calls / max(t_end - t_sim, 1)                     # robot calls per shown second
        feel = "half" if speed <= 1 and density < 1.5 else "drive" if speed >= 10 or density > 3 else "normal"
        kicks = {"half": [0, 2.5], "normal": [0, 1.5, 2.5], "drive": [0, 1, 2, 3, 3.5]}[feel]
        snares = {"half": [2], "normal": [1, 3], "drive": [1, 3]}[feel]
        hats = {"half": [i / 2 for i in range(8)], "normal": [i / 2 for i in range(8)], "drive": [i / 4 for i in range(16)]}[feel]
        mix.add(crash(0.5, rng), t_sim, gain=0.5, verb=0.4)
        mix.add(kick(1.0, rng), t_sim, gain=0.9, verb=0.05)
        b, t = 0, t_sim
        n_bars = int((t_end - t_sim) / bar)
        while t < t_end - 0.05:                                     # bar by bar
            c = chord_at(t)
            root = key.root_of(c[2])
            if b % 8 == 7 and n_bars >= 12 and b < n_bars - 2:      # a breakdown: the drums step out for a bar, the bass holds, a crash brings them back
                mix.add(bass(_midi(root), bar * 0.9, 0.7), t, gain=0.6)
                mix.add(crash(0.5, rng), t + bar, gain=0.5, verb=0.4)
                for k in range(6):
                    mix.add(hat(0.2 + 0.06 * k, False, rng), t + bar - beat + k * beat / 6, pan=0.3, gain=0.4)
                t, b = t + bar, b + 1
                continue
            for q in kicks:
                if t + q * beat < t_end:
                    mix.add(kick(0.9 if q == 0 else 0.75, rng), t + q * beat, gain=0.85, verb=0.05)
            for q in snares:
                if t + q * beat < t_end:
                    mix.add(snare(0.8, rng), t + q * beat, gain=0.55, verb=0.25)
            for i, q in enumerate(hats):
                if t + q * beat < t_end:
                    opn = feel != "half" and i % 4 == 2 and b % 2 == 1
                    mix.add(hat(0.5 if i % 2 == 0 else 0.3, opn, rng), t + q * beat, pan=0.45 if i % 2 == 0 else -0.25, gain=0.45 if not opn else 0.35)
            line = [0, 0, 7, 0, 0, 12, 7, 5] if feel != "half" else [0, None, None, 0, 7, None, 0, None]
            for i, iv in enumerate(line):                           # the bass walks root, fifth and octave in eighths
                if iv is not None and t + i * beat / 2 < t_end:
                    mix.add(bass(_midi(root + iv), beat * 0.42, 0.85 if i == 0 else 0.7), t + i * beat / 2, gain=0.65, verb=0.0)
            if b % 2 == 0:                                          # the cube theme, a phrase every two bars; moved cubes play louder
                rhythm = RHYTHMS[((seed >> 28) + b // 2) % len(RHYTHMS)]
                shift = c[2] if b % 4 == 2 else 0
                for i, q in enumerate(rhythm):
                    step, pan, moved = motif[(i + b // 2) % len(motif)]
                    tn = t + q * beat
                    if tn < t_end - 0.1:
                        mix.add(lead(_midi(key.midi((step + shift) % 14, 4)), 0.9 if moved else 0.65, beat * 0.6), tn, pan=pan * 0.6, gain=0.55, verb=0.35, delay=0.3)
                if b % 4 == 0 and b:                                # answered by a high bell on the chord's fifth
                    mix.add(bell(_midi(c[3][-1] + 12), 0.4), t + 3.5 * beat, pan=-0.3, gain=0.3, verb=0.7)
            t, b = t + bar, b + 1
        for t in _thin(sorted(ev.get("flags") or []), 2):           # the simulator flagged a collision here
            if t_sim <= t < t_end:
                mix.add(stab(_midi(key.midi(0, 3)), rng), t, gain=0.6, verb=0.3)

    # -- the end of the program: the cadence rings through the hold
    if failed:
        mix.add(sub(_midi(key.root_of(final_deg)), min(3.0, end - t_end)), t_end, gain=0.9, verb=0.1)
        mix.add(kick(0.9, rng), t_end, gain=0.8)
        for k, iv in enumerate((0, -2, -5)):                        # a falling bass figure
            mix.add(bass(_midi(key.root_of(final_deg) + iv), 0.5, 0.7), t_end + k * beat / 2, gain=0.6)
    else:
        mix.add(kick(0.8, rng), t_end, gain=0.7)
        mix.add(bass(_midi(key.root_of(0)), min(2.5, end - t_end), 0.9), t_end, gain=0.7)
        mix.add(crash(0.45, rng), t_end, gain=0.45, verb=0.4)
        if success:                                                 # the task was solved: bright bells up the chord
            v = key.chord(0, add9=True)
            for k, m in enumerate(v + [v[0] + 12]):
                mix.add(bell(_midi(m + 12), 0.5, 2.5), t_end + 0.08 * k, pan=-0.4 + 0.2 * k, gain=0.45, verb=0.8, delay=0.4)
        else:
            mix.add(bell(_midi(key.midi(0, 5)), 0.45, 2.0), t_end, gain=0.4, verb=0.8)
    return _master(mix.render(beat, rng), mix)


def write_wav(path, audio):
    """32-bit float WAV, stereo, SR."""
    data = np.ascontiguousarray(audio, dtype="<f4").tobytes()
    hdr = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16, 3, 2, SR, SR * 8, 8, 32, b"data", len(data))
    with open(path, "wb") as f:
        f.write(hdr + data)
