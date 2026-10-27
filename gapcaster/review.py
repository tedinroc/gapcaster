"""Static review page for a run: timeline, every AD line with its evidence, attempts and flags."""
import base64
import html
import json
from pathlib import Path

CSS = """
:root{--bg:#f4f5f2;--card:#fff;--ink:#1b1f1d;--muted:#5f6862;--rule:#d9ddd6;--dlg:#3d4a44;--gap:#e6ece4;--ad:#c2410c;--ok:#2f7a4b;--warn:#a16207}
@media (prefers-color-scheme:dark){:root{--bg:#121513;--card:#1a1e1c;--ink:#e8ece9;--muted:#9aa49e;--rule:#2c332f;--dlg:#8fa197;--gap:#25302a;--ad:#fb923c;--ok:#6cc392;--warn:#e2b44c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,"Inter",system-ui,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 60px;display:flex;flex-direction:column;gap:22px}
h1{margin:0;font-size:26px;letter-spacing:-.01em}h1 span{color:var(--ad)}
.sub{color:var(--muted)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:24px;font-variant-numeric:tabular-nums}.kpi small{color:var(--muted)}
.tl{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:14px}
.tl svg{width:100%;height:auto;display:block}
.legend{display:flex;gap:16px;font-size:13px;color:var(--muted);margin-top:6px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
video{width:100%;border-radius:10px;background:#000}
.line{display:grid;grid-template-columns:180px 1fr 230px;gap:16px;background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:14px}
.line img{width:100%;border-radius:6px;display:block}
.t{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;color:var(--muted)}
.txt{font-size:17px;font-weight:600;margin:4px 0 8px}
.att{font-size:13px;color:var(--muted);border-left:2px solid var(--rule);padding-left:10px;display:flex;flex-direction:column;gap:3px}
.att .x{text-decoration:line-through}
.fit{font-family:ui-monospace,Menlo,monospace;font-size:13px}
.bar{height:8px;border-radius:4px;background:var(--gap);position:relative;margin:4px 0 10px}
.bar i{position:absolute;inset:0 auto 0 0;border-radius:4px;background:var(--ok)}
.chip{display:inline-block;font-size:12px;font-weight:600;padding:2px 8px;border-radius:999px;margin:2px 4px 2px 0;background:var(--gap)}
.chip.ok{color:var(--ok)}.chip.warn{color:var(--warn)}
.facts{font-size:12.5px;color:var(--muted)}
audio{width:100%;height:32px}
@media (max-width:820px){.line{grid-template-columns:1fr}}
"""


def _img64(p, width=None):
    try:
        return "data:image/jpeg;base64," + base64.b64encode(Path(p).read_bytes()).decode()
    except OSError:
        return ""


def timeline_svg(doc):
    total = doc["metrics"]["clip_seconds"]
    W, H = 1000, 74
    x = lambda t: 10 + (W - 20) * t / total
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="timeline">']
    parts.append(f'<rect x="10" y="10" width="{W-20}" height="22" rx="4" fill="var(--gap)"/>')
    for s in doc["dialogue"]:
        parts.append(f'<rect x="{x(s["start"]):.1f}" y="10" width="{max(1,x(s["end"])-x(s["start"])):.1f}" height="22" fill="var(--dlg)"/>')
    for r in doc["lines"]:
        if r.get("duration"):
            parts.append(f'<rect x="{x(r["start"]):.1f}" y="38" width="{max(2,x(r["start"]+r["duration"])-x(r["start"])):.1f}" height="14" rx="3" fill="var(--ad)"><title>{html.escape(r["text"])}</title></rect>')
    for t in range(0, int(total) + 1, 10):
        parts.append(f'<text x="{x(t):.1f}" y="68" font-size="10" fill="var(--muted)" text-anchor="middle">{t}s</text>')
    parts.append("</svg>")
    return "".join(parts)


def build(run_dir):
    run = Path(run_dir)
    doc = json.loads((run / "script.json").read_text())
    m = doc["metrics"]
    shots = {s["id"]: s for s in doc["shots"]}
    n = max(1, m["lines"])
    inj = m.get("injected_error_test")
    kpis = [
        (f'{m["fit_after_loop"]}/{m["lines"]}', "lines fit their silence"),
        (f'{m["first_try_fit"]}/{m["lines"]}', "fit on the first draft"),
        (f'{m["ad_overlap_with_detected_dialogue_s"]:.1f} s', "AD over dialogue"),
        (f'{m["seconds_per_content_minute"]:.0f} s', "processing per content minute"),
        (f'{m["shots"]}', "shots described"),
    ]
    if "ad_overlap_with_reference_subtitles_s" in m:
        kpis.insert(3, (f'{m["ad_overlap_with_reference_subtitles_s"]:.1f} s', "AD over official subtitle cues"))
    if inj:
        kpis.append((f'{inj["caught"]}/{inj["planted"]}', "planted false details flagged"))
    out = [f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>GapCaster review</title><style>{CSS}</style><div class=wrap>"]
    out.append(f'<div><h1>Gap<span>Caster</span> review</h1><div class=sub>{html.escape(doc["source"]["file"])} · {doc["source"]["start"]:.0f}–{doc["source"]["end"]:.0f} s · {m["lines"]} draft lines awaiting describer approval</div></div>')
    out.append('<div class=kpis>' + "".join(f'<div class=kpi><b>{a}</b><small>{b}</small></div>' for a, b in kpis) + '</div>')
    out.append(f'<div class=tl>{timeline_svg(doc)}<div class=legend><span><i style="background:var(--dlg)"></i>dialogue</span><span><i style="background:var(--gap)"></i>silence</span><span><i style="background:var(--ad)"></i>audio description</span></div></div>')
    if (run / "clip_with_ad.mp4").exists():
        poster = run / "poster.jpg"
        if not poster.exists():
            from . import media
            media.frame(run / "clip_with_ad.mp4", m["clip_seconds"] * 0.57, poster, width=1280)
        out.append('<video controls preload=none poster="poster.jpg" src="clip_with_ad.mp4"></video>')
    for r in doc["lines"]:
        sh = shots.get(r["shots"][0]) if r["shots"] else None
        img = _img64(run / sh["frames"][0]) if sh and sh.get("frames") else ""
        fit_pct = min(100, 100 * (r["duration"] or 0) / max(0.01, r["avail"]))
        atts = "".join(
            f'<div class="{"" if a["fit"] else "x"}">{html.escape(a["text"])} <span class=t>— {html.escape(a["reason"])}</span></div>'
            for a in r["attempts"][:-1]) if len(r["attempts"]) > 1 else ""
        chips = '<span class="chip ok">fits</span>' if r["status"] == "fit" else '<span class="chip warn">does not fit</span>'
        chips += "".join(f'<span class="chip warn">{html.escape(f)}</span>' for f in r["flags"])
        facts = "<br>".join(f'{i}: {html.escape(str((shots[i].get("facts") or {}).get("key", "")))}' for i in r["shots"] if i in shots)
        audio = f'<audio controls preload=none src="{html.escape(r["audio"])}"></audio>' if r.get("audio") else ""
        out.append(f'''<div class=line><div>{f'<img alt="" src="{img}">' if img else ''}<div class=t>{r["id"]} · {r["start"]:.1f}s</div></div>
<div><div class=txt>{html.escape(r["text"])}</div>{f'<div class=att><b>Earlier drafts</b>{atts}</div>' if atts else ''}<div class=facts>Evidence · {facts}</div></div>
<div><div class=fit>{(r["duration"] or 0):.1f} s of {r["avail"]:.1f} s silence{f" · {r['speed']}×" if r.get("speed") and r["speed"] != 1.0 else ""}</div><div class=bar><i style="width:{fit_pct:.0f}%"></i></div>{chips}{audio}</div></div>''')
    out.append("</div>")
    p = run / "review.html"
    p.write_text("\n".join(out))
    return str(p)
