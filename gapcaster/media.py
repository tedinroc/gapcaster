"""ffmpeg helpers. Every function shells out to the ffmpeg/ffprobe on PATH."""
import json
import shutil
import subprocess
from pathlib import Path

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


def run(args):
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def duration(path) -> float:
    out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return float(json.loads(out)["format"]["duration"])


def cut(src, dst, start: float, end: float):
    """Re-encode a time window so keyframes and timestamps start at zero."""
    run([FFMPEG, "-v", "error", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "192k", str(dst)])


def extract_audio(src, dst, sr: int = 16000):
    run([FFMPEG, "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(sr), str(dst)])


def frame(src, t: float, dst, width: int = 768):
    run([FFMPEG, "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1",
         "-vf", f"scale={width}:-2", "-q:v", "3", str(dst)])


def mux_mix(video, ad_wav, dst, duck_db: float = -10.0):
    """Mix the AD track over the programme audio, ducking the programme while AD speaks."""
    filt = (
        "[1:a]aresample=48000,asplit=2[sc][adv];"
        "[0:a]aresample=48000[pgm];"
        "[pgm][sc]sidechaincompress=threshold=0.015:ratio=8:attack=15:release=350:makeup=1[duck];"
        "[adv]volume=1.6[adl];"
        "[duck][adl]amix=inputs=2:duration=first:normalize=0[out]"
    )
    run([FFMPEG, "-v", "error", "-y", "-i", str(video), "-i", str(ad_wav), "-filter_complex", filt,
         "-map", "0:v", "-map", "[out]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(dst)])


def ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
