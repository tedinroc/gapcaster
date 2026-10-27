"""Re-mix a run from the describer's decisions (approved.json downloaded from the review page).

approved -> keep the line's audio; edited -> speak the new text and re-check that it fits its
silence (speed ladder, no rewrite: the describer's words are final); rejected or pending -> drop.
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from . import export, media
from .config import Settings
from .tts import TTS


def apply(run_dir, approved_path, settings: Settings | None = None):
    s = settings or Settings()
    run = Path(run_dir)
    doc = json.loads((run / "script.json").read_text())
    decisions = {d["id"]: d for d in json.loads(Path(approved_path).read_text())["lines"]}
    tts, kept, report = None, [], []
    for r in doc["lines"]:
        d = decisions.get(r["id"], {"status": "pending"})
        if d["status"] in ("rejected", "pending") or not r.get("audio"):
            report.append({"id": r["id"], "status": d["status"], "kept": False})
            continue
        line = dict(r, review=d["status"])
        if d["status"] == "edited" and d.get("text") and d["text"] != r["text"]:
            tts = tts or TTS(s.tts_model, s.tts_voice, s.lang, backend=s.tts_backend)
            fit = None
            for sp in s.speeds:
                a, dur = tts.synth(d["text"], sp)
                if dur <= r["avail"]:
                    fit = (a, dur, sp)
                    break
            if not fit:
                report.append({"id": r["id"], "status": "edited", "kept": False, "reason": f"edited line is {dur:.1f}s, silence is {r['avail']:.1f}s"})
                continue
            p = run / "lines" / f"{r['id']}_edited.wav"
            sf.write(p, fit[0], tts.sr)
            line.update(text=d["text"], duration=round(fit[1], 2), speed=fit[2], audio=str(p.relative_to(run)))
        kept.append(line)
        report.append({"id": r["id"], "status": d["status"], "kept": True})
    total = doc["metrics"]["clip_seconds"]
    sr = 24000
    track = np.zeros(int(total * sr) + sr, dtype=np.float32)
    for r in kept:
        a, _ = sf.read(run / r["audio"], dtype="float32")
        i = int(r["start"] * sr)
        track[i:i + len(a)] += a[: len(track) - i]
    out = run / "approved"
    out.mkdir(exist_ok=True)
    sf.write(out / "ad_track.wav", track, sr)
    media.mux_mix(run / "clip.mp4", out / "ad_track.wav", out / "clip_with_ad.mp4")
    export.write_all({"lines": kept}, out)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return {"kept": len(kept), "dropped": len(report) - len(kept), "out": str(out)}
