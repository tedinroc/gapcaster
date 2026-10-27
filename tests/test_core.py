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
