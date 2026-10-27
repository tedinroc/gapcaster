"""Parity check for the Stage II speech path, run on a laptop CPU.

Compares the dialogue map from OpenVINO GenAI Whisper (CPU) with the MLX Whisper map,
and Kokoro durations from the PyTorch runtime with the MLX runtime, on one run's lines.
Usage: python eval/parity.py runs/<run>  ->  prints JSON and writes <run>/parity.json
"""
import json
import sys
import time
from pathlib import Path

from gapcaster import gapmap
from gapcaster.config import Settings
from gapcaster.tts import TTS


def span_set_iou(a, b, total, step=0.05):
    def mask(sp):
        m = [0] * int(total / step + 1)
        for x in sp:
            for i in range(int(x["start"] / step), min(len(m), int(x["end"] / step) + 1)):
                m[i] = 1
        return m
    ma, mb = mask(a), mask(b)
    inter = sum(1 for x, y in zip(ma, mb) if x and y)
    union = sum(1 for x, y in zip(ma, mb) if x or y)
    return inter / max(1, union)


def main(run):
    run = Path(run)
    doc = json.loads((run / "script.json").read_text())
    s, total = Settings(), doc["metrics"]["clip_seconds"]
    wav = run / "clip16k.wav"
    out = {}
    t = time.time()
    w_mlx, _ = gapmap.transcribe(wav, s.whisper_model)
    out["mlx_asr_seconds"] = round(time.time() - t, 1)
    t = time.time()
    w_ov, _ = gapmap.transcribe_openvino(wav, s.asr_ov_model, "CPU")
    out["openvino_cpu_asr_seconds"] = round(time.time() - t, 1)
    a, b = gapmap.speech_spans(w_mlx, s.merge_pause), gapmap.speech_spans(w_ov, s.merge_pause)
    out["asr_only_iou_mlx_vs_openvino"] = round(span_set_iou(a, b, total), 3)
    vad = gapmap.vad_speech(wav)          # the pipeline's dialogue map is ASR ∪ VAD
    a, b = gapmap.merge(a, vad, s.merge_pause), gapmap.merge(b, vad, s.merge_pause)
    out["speech_iou_mlx_vs_openvino"] = round(span_set_iou(a, b, total), 3)
    ga = gapmap.gaps(a, total, s.lead, s.tail, s.min_slot)
    gb = gapmap.gaps(b, total, s.lead, s.tail, s.min_slot)
    out["gaps_mlx"], out["gaps_openvino"] = len(ga), len(gb)
    out["usable_gap_seconds_mlx"] = round(sum(g["end"] - g["start"] for g in ga), 1)
    out["usable_gap_seconds_openvino"] = round(sum(g["end"] - g["start"] for g in gb), 1)
    lines = [r for r in doc["lines"] if r.get("duration")]
    k = TTS(s.tts_model, s.tts_voice, backend="kokoro", device="cpu")
    diffs = []
    t = time.time()
    for r in lines:
        _, d = k.synth(r["text"], r["speed"])
        diffs.append(abs(d - r["duration"]))
    out["kokoro_pytorch_cpu_seconds"] = round(time.time() - t, 1)
    out["tts_lines"] = len(lines)
    out["tts_duration_abs_diff_mean_s"] = round(sum(diffs) / max(1, len(diffs)), 3)
    out["tts_duration_abs_diff_max_s"] = round(max(diffs or [0]), 3)
    (run / "parity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
