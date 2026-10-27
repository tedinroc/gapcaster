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


def apply(run_dir, approved_path, settings: Settings | None = None, tts_backend: str | None = None):
    run = Path(run_dir)
    doc = json.loads((run / "script.json").read_text())
    rec = doc.get("settings", {})
    # re-time edits with the same voice and runtime the run used, unless told otherwise
    s = settings or Settings(lang=rec.get("lang", "en"), tts_voice=rec.get("tts_voice", Settings.tts_voice),
                             tts_model=rec.get("tts_model", Settings.tts_model),
                             tts_backend=tts_backend or rec.get("tts_backend", "mlx"),
                             speeds=tuple(rec.get("speeds", Settings.speeds)))
    decisions = {d["id"]: d for d in json.loads(Path(approved_path).read_text())["lines"]}
    tts, kept, report = None, [], []
    for r in doc["lines"]:
        d = decisions.get(r["id"], {"status": "pending"})
        status, new_text = d.get("status", "pending"), (d.get("text") or "").strip()
        if status in ("rejected", "pending"):
            report.append({"id": r["id"], "status": status, "kept": False})
            continue
        changed = bool(new_text) and new_text != r["text"]
        if status == "approved" and changed:        # approved after typing a change: treat it as an edit
            status = "edited"
        needs_voice = status == "edited" and (changed or not r.get("audio"))
        if not r.get("audio") and not needs_voice:
            report.append({"id": r["id"], "status": status, "kept": False,
                           "reason": "line was not placed by the pipeline; edit its text to place it"})
            continue
        line = dict(r, review=status)
        if needs_voice:
            tts = tts or TTS(s.tts_model, s.tts_voice, s.lang, backend=s.tts_backend)
            text, fit, dur = new_text or r["text"], None, 0.0
            for sp in s.speeds:
                a, dur = tts.synth(text, sp)
                if dur <= r["avail"]:
                    fit = (a, dur, sp)
                    break
            if not fit:
                report.append({"id": r["id"], "status": "edited", "kept": False,
                               "reason": f"edited line is {dur:.1f}s, silence is {r['avail']:.1f}s"})
                continue
            p = run / "lines" / f"{r['id']}_edited.wav"
            sf.write(p, fit[0], tts.sr)
            line.update(text=text, duration=round(fit[1], 2), speed=fit[2], audio=str(p.relative_to(run)))
        kept.append(line)
        report.append({"id": r["id"], "status": status, "kept": True})
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
