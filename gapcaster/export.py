"""Script exports: SRT, WebVTT (kind="descriptions" text track), plain-text script."""


def ts(t: float, sep=","):
    h, rem = divmod(max(0.0, t), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int(round((s - int(s)) * 1000)):03d}"


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
