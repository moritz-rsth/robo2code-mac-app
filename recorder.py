"""Records the frames of one simulated run into a video file for the "Download video" button.
The JPEG frames of sim_worker.py (1280x720, 4:4:4) are piped straight into ffmpeg (bundled with imageio-ffmpeg), which writes an H.264 MP4 to disk while the run
goes on: memory stays flat however long the run is. The video plays at simulated real time (30 frames per simulated second), whatever the speed slider says.
Without ffmpeg it falls back to a small GIF (every 3rd frame at half size, so 10 fps)."""
import re
import shutil
import subprocess
import tempfile
import time
from io import BytesIO
from pathlib import Path

from PIL import Image

FPS = 30                                     # sim_worker.py renders about 30 frames per simulated second
MAX_FRAMES = FPS * 600                       # 10 simulated minutes; a longer run keeps its first 10 minutes and its end state
HOLD = FPS                                   # the end state stays on screen for one more second
GIF_STEP, GIF_MAX = 3, 600                   # fallback: 10 fps, at most a minute (about 35 MB in memory)
KEEP = 20                                    # videos of older runs are deleted
DIR = Path(tempfile.mkdtemp(prefix="p2r_videos_"))


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


FFMPEG = ffmpeg_exe()


def slug(text, n):
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:n].strip("-")


def file_name(instruction, model, ext):
    """prompt slug + model + timestamp, e.g. put-the-red-cube-on-the-blue-cube_qwen3-1-7b-ft_20261108-143012.mp4"""
    return f"{slug(instruction, 48) or 'run'}_{slug(model, 32) or 'model'}_{time.strftime('%Y%m%d-%H%M%S')}.{ext}"


def prune():
    old = sorted(DIR.iterdir(), key=lambda p: p.stat().st_mtime)[:-KEEP]
    for p in old:
        p.unlink(missing_ok=True)


class Recorder:
    def __init__(self, instruction, model):
        self.name = file_name(instruction, model, "mp4" if FFMPEG else "gif")
        self.path, self.n, self.last, self.proc, self.gif, self.error = DIR / f"{time.time_ns()}_{self.name}", 0, None, None, [], None

    def add(self, jpg):
        """jpg: the bytes of one frame. Never raises: a broken recording must not break the run."""
        self.n += 1
        self.last = jpg
        if self.error or self.n > MAX_FRAMES:
            return
        try:
            if FFMPEG:
                self._pipe(jpg)
            elif self.n % GIF_STEP == 1 and len(self.gif) < GIF_MAX:
                im = Image.open(BytesIO(jpg))
                self.gif.append(im.resize((im.width // 2, im.height // 2)).convert("P", palette=Image.ADAPTIVE))
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def _pipe(self, jpg):
        if self.proc is None:                # yuv420p + faststart: plays in QuickTime, Safari, iMessage and WhatsApp
            self.proc = subprocess.Popen([FFMPEG, "-loglevel", "error", "-y", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(FPS), "-i", "-",
                                          "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:in_range=pc:out_range=tv,format=yuv420p",    # JPEG is full range, players want TV range
                                          "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-movflags", "+faststart", str(self.path)],
                                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.proc.stdin.write(jpg)

    def finish(self):
        """Closes the file. Returns (path, name) of the finished video, or None if nothing was recorded."""
        if not self.n:
            return None
        try:
            if self.error:
                pass
            elif FFMPEG:
                for _ in range(HOLD):
                    self._pipe(self.last)
                self.proc.stdin.close()
                if self.proc.wait(timeout=120):
                    raise RuntimeError("ffmpeg could not encode the video")
            else:
                last = Image.open(BytesIO(self.last))
                frames = self.gif + [last.resize((last.width // 2, last.height // 2)).convert("P", palette=Image.ADAPTIVE)]
                frames[0].save(self.path, save_all=True, append_images=frames[1:], duration=[1000 * GIF_STEP // FPS] * (len(frames) - 1) + [1000], loop=0)
                self.gif = []
        except Exception as e:
            self.error = self.error or f"{type(e).__name__}: {e}"
        finally:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
        if self.error or not self.path.exists():
            self.path.unlink(missing_ok=True)
            return None
        prune()
        return self.path, self.name
