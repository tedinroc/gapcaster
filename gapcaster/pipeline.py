"""End-to-end run: shots -> dialogue gaps -> shot facts -> write -> fit loop -> verify -> mix."""
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from . import gapmap, llm, media, prompts, scenes, styleguard
from .config import Settings
from .tts import TTS


CREDITS = re.compile(r"supported by|presents|film fund|foundation|copyright|©|www\.|\.org|\.com|produced by|directed by", re.I)


class Timer:
    def __init__(self):
        self.stages = {}

    def __call__(self, name):
        timer = self

        class _T:
            def __enter__(self):
                self.t = time.time()

            def __exit__(self, *a):
                timer.stages[name] = round(timer.stages.get(name, 0) + time.time() - self.t, 2)
        return _T()


def cast_text(cast, t: float | None = None):
    """Cast sheet as prompt text. Before an entry's `named_from` time the film has not
    introduced the name yet, so the writer is told to use the look instead (AD practice)."""
    rows = []
    for c in cast:
        nf = c.get("named_from")
        if t is not None and nf is not None and t < float(nf):
            rows.append(f"{c['look']} (not named yet: describe by look, do not say '{c['name']}')")
        else:
            rows.append(f"{c['name']}: {c['look']}")
    return "; ".join(rows) or "none provided"


def facts_text(shots_by_id, ids):
    rows = []
    for i in ids:
        f = shots_by_id[i].get("facts") or {}
        sh = shots_by_id[i]
        rows.append(f"- {i} ({sh['start']:.1f}-{sh['end']:.1f}s): who={f.get('who','?')}; where={f.get('where','?')}; "
                    f"action={f.get('action','?')}; text={f.get('text','')}; key={f.get('key','')}")
    return "\n".join(rows)


def dialogue_near(spans, t0, t1, before=True):
    if before:
        prev = [s for s in spans if s["end"] <= t0 + 0.05]
        return prev[-1]["text"][-160:] if prev else ""
    nxt = [s for s in spans if s["start"] >= t1 - 0.05]
    return nxt[0]["text"][:160] if nxt else ""


def build_slots(gaps, shots, s: Settings):
    slots = []
    for g in gaps:
        cuts = [g["start"]]
        if g["end"] - g["start"] > s.max_slot:
            for sh in shots:
                if g["start"] + 2.0 < sh["start"] < g["end"] - 2.0 and sh["start"] - cuts[-1] >= 2.5:
                    cuts.append(sh["start"])
        cuts.append(g["end"])
        for a, b in zip(cuts, cuts[1:]):
            now = [sh["id"] for sh in shots if gapmap.overlap(a, b, sh["start"], sh["end"]) > 0.2]
            # the shot that cut away just before the silence: only its unread on-screen text may still be read
            prev = [sh["id"] for sh in shots if sh["id"] not in now and a - 2.5 < sh["end"] <= a + 0.2]
            slots.append({"start": round(a, 3), "end": round(b, 3), "shots": now, "prev": prev})
    return slots


def run(video, out_dir, start=0.0, end=None, cast_path=None, ref_srt=None, settings: Settings | None = None,
        inject: int = 0, log=print):
    s = settings or Settings()
    out = media.ensure_dir(out_dir)
    frames_dir = media.ensure_dir(out / "frames")
    lines_dir = media.ensure_dir(out / "lines")
    T = Timer()
    cast = yaml.safe_load(open(cast_path)) if cast_path else []
    t_all = time.time()

    with T("prepare"):
        end = end or media.duration(video)
        clip = out / "clip.mp4"
        media.cut(video, clip, start, end)
        total = media.duration(clip)
        wav16 = out / "clip16k.wav"
        media.extract_audio(clip, wav16)
    log(f"clip {total:.1f}s")

    with T("shots"):
        shot_list = scenes.shots(clip)
        for sh in shot_list:
            sh["frames"] = []
            for k, t in enumerate(scenes.keyframe_times(sh)):
                p = frames_dir / f"{sh['id']}_{k}.jpg"
                media.frame(clip, t, p)
                sh["frames"].append(str(p))
    log(f"{len(shot_list)} shots")

    with T("dialogue_map"):
        if s.asr_backend == "openvino":
            words, transcript = gapmap.transcribe_openvino(wav16, s.asr_ov_model, s.asr_device, s.lang)
        else:
            words, transcript = gapmap.transcribe(wav16, s.whisper_model, s.lang if s.lang != "zh" else None)
        asr_spans = gapmap.speech_spans(words, s.merge_pause)
        vad_spans = gapmap.vad_speech(wav16)
        spans = gapmap.merge(asr_spans, vad_spans, s.merge_pause)
        gaps = gapmap.gaps(spans, total, s.lead, s.tail, s.min_slot)
    log(f"{len(spans)} dialogue spans (ASR {len(asr_spans)}, VAD {len(vad_spans)}), {len(gaps)} gaps, {sum(g['end'] - g['start'] for g in gaps):.1f}s usable")

    with T("load_models"):
        model = llm.make(s)
        tts = TTS(s.tts_model, s.tts_voice, s.lang, backend=s.tts_backend)

    by_id = {sh["id"]: sh for sh in shot_list}
    with T("shot_facts"):
        def facts_for(sh):
            ctx = " ".join(x["text"] for x in spans if gapmap.overlap(sh["start"] - 3, sh["end"] + 3, x["start"], x["end"]) > 0)[:300]
            reply = model.chat(prompts.SHOT.format(cast=cast_text(cast, sh["start"]), ctx=ctx or "(none)", dur=sh["end"] - sh["start"]),
                               images=sh["frames"], max_tokens=220, temperature=0.2)
            return sh, reply
        # shots are independent: an OpenAI-compatible server (vLLM) batches them; in-process MLX runs them one by one
        for sh, reply in llm.pmap(model, facts_for, shot_list, s.concurrency):
            sh["facts"] = llm.parse_json(reply) or {"key": reply[:200], "confidence": 0.3, "parse_error": True}
    log(f"shot facts done ({model.stats.calls} VLM calls)")

    slots = build_slots(gaps, shot_list, s)
    lines, recent, prev_end = [], [], -1.0
    read_text, skipped_repeats = set(), 0
    with T("write_fit_verify"):
        for slot in slots:
            begin = max(slot["start"], prev_end + s.breath)
            avail = slot["end"] - begin
            if avail < s.min_slot or not slot["shots"]:
                continue
            budget = max(3, min(18, math.floor(avail * s.words_per_sec)))
            facts = facts_text(by_id, slot["shots"])
            unread = []
            for i in slot.get("prev", []) + slot["shots"]:
                t = str((by_id[i].get("facts") or {}).get("text", "") or "").strip()
                if len(t) >= 3 and t.lower() not in read_text and not CREDITS.search(t):
                    unread.append(t[:90])
                    read_text.add(t.lower())
            text_note = ("On-screen text not read yet (read the story-relevant part, e.g. a headline): "
                         + " | ".join(f'"{u}"' for u in unread)) if unread else ""
            text = model.chat(prompts.WRITE.format(text_note=text_note,
                cast=cast_text(cast, slot["start"]), recent=" | ".join(recent[-3:]) or "(none)",
                before=dialogue_near(spans, slot["start"], slot["end"], True),
                after=dialogue_near(spans, slot["start"], slot["end"], False),
                facts=facts, avail=avail, budget=budget), max_tokens=80).strip().strip('"')
            if text.upper().startswith("SKIP") or not text:
                continue
            if styleguard.is_repeat(text, recent):
                skipped_repeats += 1
                continue
            attempts, status, audio, dur, speed = [], "overflow", None, 0.0, 1.0
            # text gate: story text on screen must be read (deterministic check, model rewrite)
            story = unread
            if story and not any(styleguard.content_words(u) & styleguard.content_words(text) for u in story):
                attempts.append({"text": text, "seconds": None, "fit": False, "reason": "text gate: on-screen text not read"})
                text = model.chat(prompts.READ_TEXT.format(text=" | ".join(story), budget=budget, line=text),
                                  max_tokens=80).strip().strip('"')
            for k in range(s.max_rewrites + 1):
                hard, soft = styleguard.check(text, budget)
                if CREDITS.search(text):
                    hard.append("credits")
                if hard:
                    attempts.append({"text": text, "seconds": None, "fit": False, "reason": "style: " + ", ".join(hard)})
                    if k == s.max_rewrites:
                        status = "style"
                        break
                    text = model.chat(prompts.RESTYLE.format(problems=", ".join(hard), budget=budget, line=text),
                                      max_tokens=80).strip().strip('"')
                    continue
                fit = False
                for sp in s.speeds:
                    a, d = tts.synth(text, sp)
                    if sp == s.speeds[0]:
                        d1 = d
                    if d <= avail:
                        audio, dur, speed, fit = a, d, sp, True
                        break
                attempts.append({"text": text, "seconds": round(d1, 2), "fit": fit, "speed": speed if fit else None,
                                 "reason": "fits" if fit else f"overflow {d1:.1f}s > {avail:.1f}s"})
                if fit:
                    status = "fit"
                    break
                if k == s.max_rewrites:
                    break
                target = max(2, math.floor(len(text.split()) * avail / d1 * 0.9))
                text = model.chat(prompts.SHORTEN.format(dur=d1, avail=avail, target=target, line=text),
                                  max_tokens=60).strip().strip('"')
            hard, soft = styleguard.check(text, budget)
            conf = min([float((by_id[i].get("facts") or {}).get("confidence", 0.5) or 0.5) for i in slot["shots"]])
            flags = list(soft)
            if conf < 0.6:
                flags.append(f"low vision confidence {conf:.2f}")
            if status != "fit":
                flags.append("needs extended AD (pause) or a cut")
            n = len(lines) + 1
            rec = {"id": f"AD{n:03d}", "start": round(begin, 2), "window_end": slot["end"], "avail": round(avail, 2),
                   "text": text, "duration": round(dur, 2) if audio is not None else None, "speed": speed,
                   "budget_words": budget, "attempts": attempts, "status": status, "shots": slot["shots"],
                   "text_read": unread, "verify": {}, "flags": flags, "review": "pending"}
            if audio is not None:
                p = lines_dir / f"{rec['id']}.wav"
                sf.write(p, audio, tts.sr)
                rec["audio"] = str(p.relative_to(out))
                prev_end = begin + dur
                recent.append(text)
            lines.append(rec)
            log(f"{rec['id']} {begin:6.1f}s [{status}] {text}")

    with T("verify"):
        # every line is checked only against the shots on screen while it plays (plus text it was asked to read)
        def check(rec):
            t1 = rec["start"] + (rec["duration"] or rec["avail"])
            ids = [sh["id"] for sh in shot_list if gapmap.overlap(rec["start"], t1, sh["start"], sh["end"]) > 0.05] or rec["shots"]
            f = facts_text(by_id, ids) + ("\nOn-screen text shown just before: " + " | ".join(rec["text_read"]) if rec["text_read"] else "")
            v = llm.parse_json(model.chat(prompts.VERIFY.format(facts=f, line=rec["text"]), max_tokens=120, temperature=0.0)) or {}
            return rec, ids, v
        for rec, ids, v in llm.pmap(model, check, lines, s.concurrency):
            rec["verify"], rec["verify_shots"] = v, ids
            if v.get("supported") is False:
                rec["flags"].append("unsupported: " + "; ".join(v.get("unsupported", []))[:160])

    with T("mix"):
        track = np.zeros(int(total * tts.sr) + tts.sr, dtype=np.float32)
        for rec in lines:
            if rec.get("audio"):
                a, _ = sf.read(out / rec["audio"], dtype="float32")
                i = int(rec["start"] * tts.sr)
                track[i:i + len(a)] += a[: len(track) - i]
        sf.write(out / "ad_track.wav", track, tts.sr)
        media.mux_mix(clip, out / "ad_track.wav", out / "clip_with_ad.mp4")

    elapsed = time.time() - t_all
    placed = [r for r in lines if r.get("audio")]
    overlap_asr = sum(gapmap.overlap(r["start"], r["start"] + r["duration"], sp["start"], sp["end"]) for r in placed for sp in spans)
    metrics = {
        "clip_seconds": round(total, 1), "shots": len(shot_list), "dialogue_spans": len(spans), "gaps": len(gaps),
        "speech_seconds_asr": round(sum(x["end"] - x["start"] for x in asr_spans), 1),
        "speech_seconds_vad": round(sum(x["end"] - x["start"] for x in vad_spans), 1),
        "speech_seconds_merged": round(sum(x["end"] - x["start"] for x in spans), 1),
        "usable_gap_seconds": round(sum(g["end"] - g["start"] for g in gaps), 1), "slots": len(slots),
        "lines": len(lines), "lines_placed": len(placed), "skipped_repeats": skipped_repeats,
        "first_try_fit": sum(1 for r in lines if r["attempts"] and r["attempts"][0]["fit"] and r["attempts"][0].get("speed") == 1.0),
        "fit_after_loop": sum(1 for r in lines if r["status"] == "fit"),
        "mean_attempts": round(sum(len(r["attempts"]) for r in lines) / max(1, len(lines)), 2),
        "rewrites": sum(max(0, len(r["attempts"]) - 1) for r in lines),
        "ad_seconds": round(sum(r["duration"] for r in placed), 1),
        "ad_overlap_with_detected_dialogue_s": round(overlap_asr, 2),
        "flagged_lines": sum(1 for r in lines if r["flags"]),
        "unsupported_flags": sum(1 for r in lines if any(f.startswith("unsupported") for f in r["flags"])),
        "rescued_by_speed": sum(1 for r in lines if r["status"] == "fit" and r["speed"] != 1.0),
        "rescued_by_rewrite": sum(1 for r in lines if r["status"] == "fit" and any(str(a.get("reason", "")).startswith("overflow") for a in r["attempts"])),
        "style_or_text_gate_rewrites": sum(1 for r in lines for a in r["attempts"] if str(a.get("reason", "")).startswith(("style", "text gate"))),
        "shots_covered": len({i for r in placed for i in r["shots"]}),
        "processing_seconds": round(elapsed, 1),
        "seconds_per_content_minute": round(elapsed / (total / 60), 1),
        "stages": T.stages, "llm": model.stats.as_dict(), "tts": {"calls": tts.calls, "audio_seconds": round(tts.audio_seconds, 1)},
    }
    try:
        import mlx.core as mx
        metrics["mlx_peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 1)
    except Exception:
        pass
    if ref_srt:
        ref = [x for x in gapmap.load_srt(ref_srt) if x["end"] > start and x["start"] < start + total]
        ref = [{"start": x["start"] - start, "end": x["end"] - start} for x in ref]
        metrics["reference_dialogue_cues"] = len(ref)
        metrics["ad_overlap_with_reference_subtitles_s"] = round(sum(
            gapmap.overlap(r["start"], r["start"] + r["duration"], x["start"], x["end"]) for r in placed for x in ref), 2)

    if inject:
        from .inject import injected_error_test
        with T("injected_error_test"):
            metrics["injected_error_test"] = injected_error_test(model, lines, by_id, inject, s.concurrency)

    doc = {"source": {"file": Path(video).name, "start": start, "end": start + total}, "settings": vars(s) | {"speeds": list(s.speeds)},
           "cast": cast, "shots": [{k: v for k, v in sh.items()} for sh in shot_list], "dialogue": spans,
           "dialogue_asr": asr_spans, "speech_vad": vad_spans, "gaps": gaps,
           "slots": slots, "lines": lines, "metrics": metrics}
    for sh in doc["shots"]:
        sh["frames"] = [str(Path(f).relative_to(out)) for f in sh["frames"]]
    (out / "script.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    from . import export
    export.write_all(doc, out)
    return doc
