"""Shot boundaries with PySceneDetect (BSD-3)."""
from scenedetect import AdaptiveDetector, detect


def shots(video_path, min_len: float = 0.6):
    scenes = detect(str(video_path), AdaptiveDetector(), start_in_scene=True)  # a clip with no cuts is one shot
    out = []
    for i, (a, b) in enumerate(scenes):
        s, e = (a.seconds, b.seconds) if hasattr(a, "seconds") else (a.get_seconds(), b.get_seconds())
        if out and e - s < min_len:          # fold flicker-length shots into the previous one
            out[-1]["end"] = e
            continue
        out.append({"id": f"S{len(out) + 1:03d}", "start": round(s, 3), "end": round(e, 3)})
    return out


def keyframe_times(shot, max_frames: int = 2):
    s, e = shot["start"], shot["end"]
    if e - s > 5.0 and max_frames > 1:
        return [s + (e - s) / 3, s + 2 * (e - s) / 3]
    return [(s + e) / 2]
