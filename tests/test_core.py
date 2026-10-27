from gapcaster import gapmap, styleguard
from gapcaster.config import Settings
from gapcaster.export import ts
from gapcaster.pipeline import build_slots


def test_speech_spans_merge_short_pauses():
    words = [{"w": "a", "start": 0.0, "end": 0.4}, {"w": "b", "start": 0.7, "end": 1.0},
             {"w": "c", "start": 3.0, "end": 3.5}]
    spans = gapmap.speech_spans(words, merge_pause=0.6)
    assert [(s["start"], s["end"]) for s in spans] == [(0.0, 1.0), (3.0, 3.5)]


def test_gaps_respect_lead_and_tail():
    spans = [{"start": 1.0, "end": 2.0}, {"start": 6.0, "end": 7.0}]
    g = gapmap.gaps(spans, total=10.0, lead=0.25, tail=0.30, min_slot=1.1)
    assert g[0] == {"start": 2.25, "end": 5.7}
    assert g[-1]["end"] == 9.8          # 0.2 s kept free at the end of the clip
    assert all(b["end"] - b["start"] >= 1.1 for b in g)


def test_short_silence_is_not_a_gap():
    spans = [{"start": 0.0, "end": 2.0}, {"start": 3.0, "end": 4.0}]
    assert gapmap.gaps(spans, total=4.0, lead=0.25, tail=0.3, min_slot=1.1) == []


def test_long_silence_splits_at_shot_cuts():
    shots = [{"id": "S1", "start": 0, "end": 5}, {"id": "S2", "start": 5, "end": 11}, {"id": "S3", "start": 11, "end": 20}]
    slots = build_slots([{"start": 1.0, "end": 19.0}], shots, Settings())
    assert [s["start"] for s in slots] == [1.0, 5, 11]
    assert slots[-1]["end"] == 19.0


def test_style_gate_blocks_camera_language_and_speculation():
    hard, _ = styleguard.check("We see Celia, who seems upset, on the bridge.")
    assert "camera-language" in hard and "speculation" in hard


def test_style_gate_soft_flags_inner_state_only():
    hard, soft = styleguard.check("Celia frowns and turns away, angry.")
    assert hard == [] and "inner-state" in soft


def test_style_gate_passes_plain_description():
    assert styleguard.check("Celia lifts her robotic hand toward Thom.") == ([], [])


def test_overlap():
    assert gapmap.overlap(0, 2, 1, 3) == 1
    assert gapmap.overlap(0, 1, 2, 3) == 0


def test_timestamp_format():
    assert ts(3723.5) == "01:02:03,500"
    assert ts(1.25, ".") == "00:00:01.250"


def test_merge_takes_union_of_detectors():
    a = [{"start": 1.0, "end": 2.0, "text": "hi"}]
    b = [{"start": 2.3, "end": 3.0}, {"start": 8.0, "end": 8.5}]
    m = gapmap.merge(a, b, merge_pause=0.6)
    assert [(x["start"], x["end"]) for x in m] == [(1.0, 3.0), (8.0, 8.5)]


def test_first_person_is_case_insensitive_for_we():
    hard, _ = styleguard.check("We follow Celia across the bridge.")
    assert "first-person" in hard


def test_roman_numeral_and_quoted_text_pass():
    assert styleguard.check("A caption: Act I.") == ([], [])
    assert styleguard.check('A sign reads "We are open".') == ([], [])


def test_slots_separate_on_screen_and_cut_away_shots():
    shots = [{"id": "S1", "start": 0, "end": 4}, {"id": "S2", "start": 4, "end": 9}]
    slots = build_slots([{"start": 5.0, "end": 8.0}], shots, Settings())
    assert slots[0]["shots"] == ["S2"] and slots[0]["prev"] == ["S1"]


def test_cast_sheet_hides_names_before_named_from():
    from gapcaster.pipeline import cast_text
    cast = [{"name": "Celia", "look": "young woman, red hair", "named_from": 23}]
    assert "Celia:" not in cast_text(cast, 10.0) and "do not say 'Celia'" in cast_text(cast, 10.0)
    assert cast_text(cast, 30.0).startswith("Celia:")


def test_possessives_do_not_hide_rules():
    assert styleguard.check("Celia's hand probably reaches for Thom's arm.")[0] == ["speculation"]


def test_on_screen_text_read_verbatim_is_exempt():
    line = "A caption reads: Look, Celia, we have to follow our passions."
    assert "first-person" in styleguard.check(line)[0]
    assert styleguard.check(line, exempt=["Look, Celia, we have to follow our passions"])[0] == []


def test_timestamps_round_into_the_next_second():
    assert ts(59.9996) == "00:01:00,000"
    assert ts(3599.9999, ".") == "01:00:00.000"


def test_model_replies_are_cleaned():
    from gapcaster.pipeline import clean
    assert clean("SKIP") == "" and clean("  ") == "" and clean("<think>long reasoning") == ""
    assert clean("reasoning...</think>\n\n\"The sniper aims.\"") == "The sniper aims."


def test_credits_pattern_spares_ordinary_description():
    from gapcaster.pipeline import CREDITS
    assert CREDITS.search("Blender Foundation presents")
    assert CREDITS.search("Credits: render farm JustaCluster")
    assert not CREDITS.search("She presents a gift to Thom.")
    assert not CREDITS.search("Cracks run through the concrete foundation.")


def test_cut_free_clip_is_one_shot(tmp_path):
    import shutil
    import subprocess
    import pytest
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    clip = tmp_path / "still.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=320x240:d=3", "-r", "24", str(clip)], check=True)
    from gapcaster import scenes
    assert len(scenes.shots(clip)) == 1


def test_export_keeps_approved_drops_rejected_and_reports_unplaced(tmp_path):
    import json
    import numpy as np
    import soundfile as sf
    from gapcaster import approve
    run = tmp_path
    (run / "lines").mkdir()
    sf.write(run / "lines" / "AD001.wav", np.zeros(2400, dtype=np.float32), 24000)
    sf.write(run / "lines" / "AD002.wav", np.zeros(2400, dtype=np.float32), 24000)
    lines = [
        {"id": "AD001", "start": 1.0, "duration": 0.1, "avail": 2.0, "text": "A.", "audio": "lines/AD001.wav", "speed": 1.0},
        {"id": "AD002", "start": 3.0, "duration": 0.1, "avail": 2.0, "text": "B.", "audio": "lines/AD002.wav", "speed": 1.0},
        {"id": "AD003", "start": 5.0, "duration": None, "avail": 2.0, "text": "C.", "speed": 1.0},
    ]
    (run / "script.json").write_text(json.dumps({"lines": lines, "metrics": {"clip_seconds": 8.0}, "settings": {}}))
    subprocess_free = {"lines": [{"id": "AD001", "status": "approved", "text": "A."},
                                 {"id": "AD002", "status": "rejected", "text": "B."},
                                 {"id": "AD003", "status": "approved", "text": "C."}]}
    (run / "approved.json").write_text(json.dumps(subprocess_free))
    approve.media.mux_mix = lambda *a, **k: None          # no video in this unit test
    out = approve.apply(run, run / "approved.json")
    report = {r["id"]: r for r in json.loads((run / "approved" / "report.json").read_text())}
    assert out["kept"] == 1 and report["AD001"]["kept"] and not report["AD002"]["kept"]
    assert not report["AD003"]["kept"] and "not placed" in report["AD003"]["reason"]
