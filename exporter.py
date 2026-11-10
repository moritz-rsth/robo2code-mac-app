"""Turns the recorded run of one side into a shareable 1920x1080 MP4 for the Export video overlay. Runs as its own process (niced), so an export never
slows the live app; progress goes to stdout as JSON lines.     usage: python exporter.py <job.json>
job = {"ffmpeg", "video" (the 640x360 recording of recorder.py), "frames" (its frame count), "out", "prompt" (request or ""), "model" (label or ""),
       "reasoning", "answer", "pace" (reasoning words per second), "speed" (simulation speed)}

The sequence, each part only if its text is there: prompt card (the request typed into a bubble over the blurred first frame of the scene),
hand-off (the bubble flies into a model card, which moves up to become the header), thinking (cycling words, the reasoning streams in),
answer (highlighted Python, or plain text), a pause to read, then the simulation at the chosen speed with its end-state hold.
Every frame is drawn with Pillow and piped into one ffmpeg (H.264, yuv420p, TV range); the recording is decoded and scaled by a second ffmpeg,
one frame at a time: memory stays flat however long the run or the reasoning is."""
import bisect
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pygments.lexers import PythonLexer
from pygments.token import Token

HERE = Path(__file__).resolve().parent
W, H, FPS = 1920, 1080, 30
FRAME = W * H * 3
FONT_DIR = HERE / "fonts"                                           # Geist and Geist Mono, SIL Open Font License (fonts/OFL.txt)
BG, PANEL, EDGE, INK, MUTED, ACCENT = (14, 15, 18), (24, 25, 30, 236), (52, 54, 63, 255), (238, 238, 235), (142, 146, 155), (47, 111, 228)
WORDS = ["Baking", "Marinating", "Simmering", "Kneading", "Caramelizing", "Zesting"]
CODE_COLORS = [(Token.Comment, (128, 134, 145)), (Token.Keyword, (199, 146, 234)), (Token.Name.Function, (130, 170, 255)), (Token.Name.Builtin, (255, 203, 107)),
               (Token.Literal.String, (195, 232, 141)), (Token.Literal.Number, (247, 140, 108)), (Token.Operator, (137, 221, 255))]
METHOD = (130, 170, 255)                                            # robot.move_to: a name after a dot
HEAD_Y, PANEL_TOP, PANEL_BOTTOM = 40, 252, 1036


def font(name, size):
    return ImageFont.truetype(str(FONT_DIR / name), size)


def emit(**kw):
    print(json.dumps(kw), flush=True)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def code_runs(code):
    """Python source -> [(text, colour)] with the colours of CODE_COLORS."""
    runs, prev = [], ""
    for tt, val in PythonLexer().get_tokens(code):
        col = METHOD if tt in Token.Name and prev == "." else next((c for t, c in CODE_COLORS if tt in t), INK)
        runs.append((val, col))
        prev = val.strip() or prev
    return runs


def is_code(text):
    return bool(re.search(r"^\s*(def |run\(robot\)|robot\.)", text, re.M))


class Typed:
    """A text laid out once (wrapped to `width`) onto a tall transparent image; draw() shows its first n characters inside a view that scrolls
    to keep the newest line in sight."""
    def __init__(self, runs, fnt, width, lh, code=False):
        self.font, self.lh, self.width, self.off = fnt, lh, width, 0.0
        lines, cur, x, indent = [], [], 0.0, ""

        def wrap(more):                                                     # more: the same source line goes on (code: under its indent + 4)
            nonlocal cur, x
            lines.append(cur)
            cur = [(indent + "    ", INK)] if more and code else []
            x = fnt.getlength(cur[0][0]) if cur else 0.0
        for text, col in runs:
            for piece in re.findall(r"\n| +|[^ \n]+", text.replace("\t", "    ")):
                if piece == "\n":
                    indent = ""
                    wrap(False)
                    continue
                if not cur and piece.isspace():
                    indent = piece[:24]
                w = fnt.getlength(piece)
                if x + w > width and not piece.isspace() and "".join(t for t, _ in cur).strip():
                    wrap(True)                                              # wrap before the word (or code token)
                if piece.isspace() and x + w > width:
                    continue                                                # a blank at a wrap is dropped
                while fnt.getlength(piece) > width - x:                     # longer than a whole line: break it
                    k = max(1, next((k for k in range(len(piece) - 1, 0, -1) if fnt.getlength(piece[:k]) <= width - x), 0))
                    cur.append((piece[:k], col))
                    piece = piece[k:]
                    wrap(True)
                if piece:
                    cur.append((piece, col)); x += fnt.getlength(piece)
        lines.append(cur)
        while len(lines) > 1 and not "".join(t for t, _ in lines[-1]).strip():
            lines.pop()
        self.text = ["".join(t for t, _ in l) for l in lines]
        self.cum = [0]
        for t in self.text:
            self.cum.append(self.cum[-1] + len(t) + 1)
        self.total = self.cum[-1]
        self.img = Image.new("RGBA", (int(width) + 8, len(lines) * lh + lh), (0, 0, 0, 0))
        d = ImageDraw.Draw(self.img)
        for i, l in enumerate(lines):
            s = ""
            for t, col in l:
                d.text((fnt.getlength(s), i * lh + lh * 0.12), t, font=fnt, fill=col)
                s += t

    def draw(self, frame, x0, y0, view_h, n, caret=False):
        n = min(max(int(n), 0), self.total)
        L = min(bisect.bisect_right(self.cum, n) - 1, len(self.text) - 1)
        j = min(n - self.cum[L], len(self.text[L]))
        target = max(0.0, (L + 1) * self.lh - view_h)
        self.off += (target - self.off) * 0.25 if abs(target - self.off) > 1 else target - self.off
        off, top = int(self.off), L * self.lh
        if top > off:                                                       # the lines before the current one, as far as they are in view
            frame.alpha_composite(self.img.crop((0, off, self.img.width, min(top, off + view_h))), (x0, y0))
        xw = int(math.ceil(self.font.getlength(self.text[L][:j])))
        y1, y2 = max(top, off), min(top + self.lh, off + view_h)
        if xw and y2 > y1:
            frame.alpha_composite(self.img.crop((0, y1, xw, y2)), (x0, y0 + y1 - off))
        if off > 0:                                                         # lines scrolled out of view fade at the top edge
            fade = Image.new("RGBA", (self.img.width, 40), PANEL[:3] + (0,))
            fade.putalpha(Image.linear_gradient("L").transpose(Image.FLIP_TOP_BOTTOM).resize((self.img.width, 40)))
            frame.alpha_composite(fade, (x0, y0))
        if caret and top + self.lh <= off + view_h:
            ImageDraw.Draw(frame).rectangle((x0 + xw + 3, y0 + top - off + self.lh * 0.18, x0 + xw + 6, y0 + top - off + self.lh * 0.88), fill=ACCENT)


def rounded(size, radius, fill, outline=None, width=2):
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius, fill=fill, outline=outline, width=width)
    return im


def shadow(size, radius, blur=28, alpha=120):
    pad = blur * 2
    im = Image.new("RGBA", (size[0] + 2 * pad, size[1] + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle((pad, pad + 10, pad + size[0], pad + size[1] + 10), radius, fill=(0, 0, 0, alpha))
    return im.filter(ImageFilter.GaussianBlur(blur)), pad


def paste(frame, sprite, cx, cy, scale=1.0, alpha=1.0):
    """sprite centred at (cx, cy), scaled and faded."""
    if alpha <= 0.01 or scale <= 0.01:
        return
    if scale != 1.0:
        sprite = sprite.resize((max(1, int(sprite.width * scale)), max(1, int(sprite.height * scale))), Image.BILINEAR)
    if alpha < 1.0:
        sprite = sprite.copy()
        sprite.putalpha(sprite.getchannel("A").point(lambda a: int(a * alpha)))
    frame.alpha_composite(sprite, (int(cx - sprite.width / 2), int(cy - sprite.height / 2)))


class Intro:
    def __init__(self, job, first):
        self.prompt, self.model = (job.get("prompt") or "").strip(), (job.get("model") or "").strip()
        self.reasoning = (job.get("reasoning") or "").strip() if self.model else ""
        answer = re.sub(r"^\s*```\w*\s*$", "", (job.get("answer") or "").replace("<|im_end|>", ""), flags=re.M).strip("\n")
        self.answer = answer if self.model else ""
        pace = min(max(float(job.get("pace") or 25), 5), 200)
        self.cps_reason, self.cps_code, self.cps_prompt = pace * 6.0, pace * 9.0, max(pace * 2.2, 30)
        bg = first.resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(18)).convert("RGBA")
        self.bg = Image.alpha_composite(bg, Image.new("RGBA", (W, H), BG + (165,)))
        self.f_label, self.f_bubble = font("GeistMono-Regular.ttf", 22), font("Geist-Regular.ttf", 50)
        self.f_name, self.f_small, self.f_head = font("Geist-SemiBold.ttf", 44), font("Geist-Regular.ttf", 28), font("Geist-SemiBold.ttf", 30)
        self.f_reason, self.f_code = font("Geist-Regular.ttf", 33), font("GeistMono-Regular.ttf", 26)
        self.segments = []
        if self.prompt:
            self._bubble()
        if self.model:
            self._card()
            self._stage()
        self._plan()

    # ---- sprites and the static stage
    def _bubble(self):
        self.p_typed = Typed([(self.prompt, (255, 255, 255))], self.f_bubble, 1180, 66)
        tw = max(self.f_bubble.getlength(t) for t in self.p_typed.text)
        bw, bh = int(tw) + 88, len(self.p_typed.text) * 66 + 58
        self.b_size = (bw, bh)
        self.b_pos = ((W - bw) // 2, (H - bh) // 2 - 40)
        self.bubble = rounded(self.b_size, 40, ACCENT + (255,))
        full = self.bubble.copy()
        self.p_typed.draw(full, 44, 24, bh, self.p_typed.total)
        self.bubble_full = full

    def _card(self):
        name = self.model
        while self.f_name.getlength(name) > 1500 and len(name) > 4:
            name = name[:-2].rstrip() + "…" if name.endswith("…") else name[:-1]
        cw, ch = max(int(self.f_name.getlength(name)) + 120, 560), 132
        card = rounded((cw, ch), 30, PANEL[:3] + (250,), outline=EDGE)
        d = ImageDraw.Draw(card)
        d.ellipse((48, 34, 62, 48), fill=ACCENT)
        d.text((76, 28), "MODEL", font=self.f_label, fill=MUTED)
        d.text((48, 60), name, font=self.f_name, fill=INK)
        self.card = card
        glow = Image.new("RGBA", (cw + 120, ch + 120), (0, 0, 0, 0))
        ImageDraw.Draw(glow).rounded_rectangle((60, 60, 60 + cw, 60 + ch), 30, outline=ACCENT + (255,), width=10)
        self.glow = glow.filter(ImageFilter.GaussianBlur(16))

    def _stage(self):
        """Background, header card, request line and the empty panels: everything that does not move while the texts stream."""
        st = self.bg.copy()
        paste(st, self.card, W / 2, HEAD_Y + self.card.height / 2)
        if self.prompt:
            line = f"“{self.prompt}”"
            while self.f_small.getlength(line) > 1600:
                line = line[:-3] + "…”" if line.endswith("…”") else line[:-2] + "…”"
            ImageDraw.Draw(st).text((W / 2, HEAD_Y + self.card.height + 20), line, font=self.f_small, fill=(205, 207, 212), anchor="ma")
        if self.reasoning:
            self.panels = {"reason": (80, 870), "answer": (910, 1840)}
        else:
            self.panels = {"answer": (260, 1660)}
        for x1, x2 in self.panels.values():
            sh, pad = shadow((x2 - x1, PANEL_BOTTOM - PANEL_TOP), 28)
            st.alpha_composite(sh, (x1 - pad, PANEL_TOP - pad))
            st.alpha_composite(rounded((x2 - x1, PANEL_BOTTOM - PANEL_TOP), 28, PANEL, outline=EDGE), (x1, PANEL_TOP))
        self.stage = st
        inner = lambda k: self.panels[k][1] - self.panels[k][0] - 88
        if self.reasoning:
            self.r_typed = Typed([(self.reasoning, (222, 223, 226))], self.f_reason, inner("reason"), 50)
        runs = code_runs(self.answer) if is_code(self.answer) else [(self.answer, INK)]
        f = self.f_code if is_code(self.answer) else self.f_reason
        self.a_typed = Typed(runs, f, inner("answer"), 42 if f is self.f_code else 50, code=f is self.f_code)

    # ---- timeline: a list of (frames, draw(i) -> RGBA frame)
    def _plan(self):
        sec = lambda s: max(1, int(round(s * FPS)))
        if self.prompt:
            n_type = sec(self.p_typed.total / self.cps_prompt)
            self.segments.append((sec(0.35), lambda i, n=sec(0.35): self.prompt_frame(0, ease(i / n))))
            self.segments.append((n_type, lambda i, n=n_type: self.prompt_frame(self.p_typed.total * (i + 1) / n, 1)))
            self.segments.append((sec(0.8 if self.model else 1.6), lambda i: self.prompt_frame(self.p_typed.total, 1, blink=i)))
        if not self.model:
            return
        n = sec(1.9 if self.prompt else 1.4)
        self.segments.append((n, lambda i, n=n: self.handoff_frame(i / FPS)))
        if self.reasoning:
            lead, n_r = sec(0.7), sec(self.r_typed.total / self.cps_reason)
            self.segments.append((lead, lambda i: self.text_frame(think_t=i / FPS, r=0, a=None)))
            self.segments.append((n_r, lambda i, n=n_r, lead=lead: self.text_frame(think_t=(i + lead) / FPS, r=self.r_typed.total * (i + 1) / n, a=None)))
            self.segments.append((sec(0.35), lambda i: self.text_frame(r=self.r_typed.total, a=0)))
        n_a = sec(self.a_typed.total / self.cps_code)
        self.segments.append((n_a, lambda i, n=n_a: self.text_frame(r=self.reasoning and self.r_typed.total, a=self.a_typed.total * (i + 1) / n)))
        words = len(self.reasoning.split()) + len(self.answer.split())
        self.segments.append((sec(min(2.5 + words / 120, 7)), lambda i: self.text_frame(r=self.reasoning and self.r_typed.total, a=self.a_typed.total, hold=True)))

    def frames(self):
        return sum(n for n, _ in self.segments)

    def __iter__(self):
        for n, draw in self.segments:
            for i in range(n):
                yield draw(i)

    # ---- the frames of each part
    def prompt_frame(self, n, appear, blink=None):
        fr = self.bg.copy()
        bx, by = self.b_pos
        lab_a = int(255 * appear)
        ImageDraw.Draw(fr).text((bx + 8, by - 44), "REQUEST", font=self.f_label, fill=(200, 202, 208, lab_a))
        b = self.bubble.copy()
        self.p_typed.draw(b, 44, 24, b.height, n, caret=blink is None or (blink // 15) % 2 == 0)
        paste(fr, b, bx + b.width / 2, by + b.height / 2 + 30 * (1 - appear), alpha=appear)
        return fr

    def handoff_frame(self, t):
        fr = self.bg.copy()
        head_cy = HEAD_Y + self.card.height / 2
        if self.prompt:                                                     # the bubble flies into the card, which then moves up
            bx, by = self.b_pos
            bcx, bcy = bx + self.b_size[0] / 2, by + self.b_size[1] / 2
            card_cy0 = 780
            k_in, k_fly, k_up, k_pulse = ease(t / 0.45), ease((t - 0.3) / 0.6), ease((t - 1.2) / 0.5), (t - 0.85) / 0.4
            cy = card_cy0 + 260 * (1 - k_in) + (head_cy - card_cy0) * k_up
            if 0 <= k_pulse <= 1:
                paste(fr, self.glow, W / 2, cy, alpha=math.sin(math.pi * k_pulse))
            paste(fr, self.card, W / 2, cy, scale=1 + 0.04 * math.sin(math.pi * min(max(k_pulse, 0), 1)), alpha=k_in)
            if k_fly < 1:
                paste(fr, self.bubble_full, bcx, bcy + (card_cy0 - bcy) * k_fly, scale=1 - 0.85 * k_fly, alpha=1 - k_fly ** 3)
            fade = ease((t - 1.45) / 0.45)
        else:
            k_in, k_pulse, k_up = ease(t / 0.35), (t - 0.3) / 0.4, ease((t - 0.7) / 0.45)
            cy = 520 + (head_cy - 520) * k_up
            if 0 <= k_pulse <= 1:
                paste(fr, self.glow, W / 2, cy, alpha=math.sin(math.pi * k_pulse))
            paste(fr, self.card, W / 2, cy, scale=0.94 + 0.06 * k_in, alpha=k_in)
            fade = ease((t - 0.95) / 0.45)
        if fade > 0:                                                        # the stage (panels, header) fades in
            fr = Image.blend(fr, self.text_frame(r=0, a=None, static=True), fade)
        return fr

    def _panel_head(self, fr, key, title, live_t=None, note=""):
        x1 = self.panels[key][0] + 44
        d = ImageDraw.Draw(fr)
        if live_t is not None:                                              # three pulsing dots and the cycling word with a shimmer
            for k in range(3):
                a = 0.5 + 0.5 * math.sin(2 * math.pi * (live_t * 1.4 - k * 0.18))
                r = 6 + 3 * a
                cx, cy = x1 + 8 + k * 26, PANEL_TOP + 50
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=ACCENT + (int(110 + 145 * a),))
            word = WORDS[int(live_t / 1.6) % len(WORDS)] + "…"
            self._shimmer(fr, word, x1 + 92, PANEL_TOP + 30, live_t)
        else:
            d.text((x1, PANEL_TOP + 30), title, font=self.f_head, fill=INK)
            if note:
                d.text((x1 + self.f_head.getlength(title) + 16, PANEL_TOP + 34), note, font=self.f_small, fill=MUTED)
        d.line((self.panels[key][0] + 1, PANEL_TOP + 92, self.panels[key][1] - 2, PANEL_TOP + 92), fill=EDGE, width=2)

    def _shimmer(self, fr, word, x, y, t):
        w, h = int(self.f_head.getlength(word)) + 4, 44
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).text((0, 2), word, font=self.f_head, fill=255)
        pos = ((t % 1.6) / 1.6) * (w + 240) - 120
        band = Image.linear_gradient("L").rotate(90, expand=True).resize((w, 1))      # 0..255 left to right
        band = band.point(lambda v: max(0, 255 - int(abs(v / 255 * w - pos) * 255 / 110))).resize((w, h))
        hi = Image.new("RGBA", (w, h), (255, 255, 255, 0))
        hi.putalpha(Image.composite(band, Image.new("L", (w, h), 0), mask))
        base = Image.new("RGBA", (w, h), MUTED + (0,))
        base.putalpha(mask)
        fr.alpha_composite(base, (x, y))
        fr.alpha_composite(hi, (x, y))

    def text_frame(self, r=0, a=None, think_t=None, hold=False, static=False):
        fr = self.stage.copy()
        view_top, view_h = PANEL_TOP + 118, PANEL_BOTTOM - PANEL_TOP - 150
        if self.reasoning:
            done = r >= self.r_typed.total
            if static:
                self._panel_head(fr, "reason", "Thinking")
            elif done:
                self._panel_head(fr, "reason", "Thought", note=f"{len(self.reasoning.split())} words")
            else:
                self._panel_head(fr, "reason", "", live_t=think_t or 0)
            if r and not static:
                self.r_typed.draw(fr, self.panels["reason"][0] + 44, view_top, view_h, r, caret=not done)
        typing = a is not None and a < self.a_typed.total
        self._panel_head(fr, "answer", "Answer", note="Python" if is_code(self.answer) and not static else "")
        if typing and not static:
            t = a / self.cps_code
            d = ImageDraw.Draw(fr)
            k = 0.5 + 0.5 * math.sin(2 * math.pi * t * 1.4)
            cx = self.panels["answer"][1] - 60
            d.ellipse((cx - 7 - 2 * k, PANEL_TOP + 46 - 7 - 2 * k, cx + 7 + 2 * k, PANEL_TOP + 46 + 7 + 2 * k), fill=ACCENT + (int(120 + 135 * k),))
        if a is None and self.reasoning and not static:
            ImageDraw.Draw(fr).text((self.panels["answer"][0] + 44, view_top), "comes after the reasoning", font=self.f_small, fill=(98, 101, 110))
        elif a:
            self.a_typed.draw(fr, self.panels["answer"][0] + 44, view_top, view_h, a, caret=typing)
        return fr


def first_frame(job):
    out = subprocess.run([job["ffmpeg"], "-loglevel", "error", "-i", job["video"], "-frames:v", "1", "-vf", "scale=640:360", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return Image.frombytes("RGB", (640, 360), out[:640 * 360 * 3])


def main():
    try:
        os.nice(10)                                       # the live app and its simulations come first
    except (AttributeError, OSError):
        pass
    job = json.loads(Path(sys.argv[1]).read_text())
    speed = min(max(float(job.get("speed") or 1), 0.25), 8)
    intro = Intro(job, first_frame(job)) if (job.get("prompt") or job.get("model")) else None
    n_intro, n_sim = intro.frames() if intro else 0, int(job.get("frames", 0) / speed) + FPS
    total, done = max(1, n_intro + n_sim), 0
    enc = subprocess.Popen([job["ffmpeg"], "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", str(FPS), "-i", "-",
                            "-vf", "scale=out_range=tv:out_color_matrix=bt709,format=yuv420p,setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                            "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                            "-movflags", "+faststart", job["out"]], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def put(rgb_bytes):
        nonlocal done
        enc.stdin.write(rgb_bytes)
        done += 1
        if done % 15 == 0:
            emit(progress=round(min(done / total, 0.99), 3))

    last = None
    try:
        if intro:
            for fr in intro:
                last = fr.convert("RGB")
                put(last.tobytes())
        dec = subprocess.Popen([job["ffmpeg"], "-loglevel", "error", "-i", job["video"], "-vf", f"scale={W}:{H}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        src, k, cur, n_fade = -1, 0, None, FPS // 2 if last else 0
        while True:                                       # output frame k shows recorded frame k * speed
            want = int(k * speed)
            while src < want:
                b = dec.stdout.read(FRAME)
                if len(b) < FRAME:
                    break
                cur, src = b, src + 1
            if src < want or cur is None:
                break
            if k < n_fade:                                # the last intro frame fades into the simulation
                put(Image.blend(last, Image.frombytes("RGB", (W, H), cur), ease((k + 1) / n_fade)).tobytes())
            else:
                put(cur)
            k += 1
        dec.stdout.close(); dec.wait()
        if cur is None:
            raise RuntimeError("the recording of this run could not be read")
        for _ in range(FPS):                              # the end state stays on screen for one more second
            put(cur)
        enc.stdin.close()
        if enc.wait(timeout=300):
            raise RuntimeError("ffmpeg could not encode the video: " + enc.stderr.read().decode(errors="replace")[-300:])
        emit(progress=1.0, done=True)
    except Exception as e:
        if enc.poll() is None:
            enc.kill()
        Path(job["out"]).unlink(missing_ok=True)
        emit(error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
