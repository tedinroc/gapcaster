"""Script exports: SRT, WebVTT (kind="descriptions" text track), plain-text script."""


def ts(t: float, sep=","):
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_all(doc, out):
    lines = [r for r in doc["lines"] if r.get("audio")]
    srt, vtt, txt = [], ["WEBVTT", "Kind: descriptions", ""], []
    for k, r in enumerate(lines, 1):
        a, b = r["start"], r["start"] + r["duration"]
        srt += [str(k), f"{ts(a)} --> {ts(b)}", r["text"], ""]
        vtt += [r["id"], f"{ts(a, '.')} --> {ts(b, '.')}", r["text"], ""]
        txt.append(f"[{ts(a, '.')}] {r['text']}")
    (out / "ad.srt").write_text("\n".join(srt))
    (out / "ad.vtt").write_text("\n".join(vtt))
    (out / "ad_script.txt").write_text("\n".join(txt) + "\n")
