"""Injected-error test for the verifier.

easy: absurd details appended to a finished line (a red umbrella, a white horse, ...).
hard: plausible in-film errors built from the film's own shot facts:
      a noun or verb borrowed from another shot, a swapped character, or the line
      written for a neighbouring silence (a timing error).
Each corrupted line is checked against the facts of the shots on screen while the
original line plays; a catch is a verdict of "not supported".
"""
import re

from . import llm, prompts

ADDS = [", holding a red umbrella", " as rain falls", " in a crowded train station", ". A sign reads EXIT", " beside a white horse"]
SWAPS = [("Thom", "Celia"), ("Celia", "Thom"), ("the sniper", "the old man"), ("the old man", "the sniper"),
         ("The sniper", "The old man"), ("The old man", "The sniper"), (" man", " woman"), (" woman", " man")]
_nlp = None


def _spacy():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def easy(text: str, i: int) -> str:
    if i % 3 == 1:
        for a, b in SWAPS:
            if a in text:
                return text.replace(a, b, 1)
    return text.rstrip(".") + ADDS[i % len(ADDS)] + "."


ABSTRACT = {"expression", "right", "left", "capital", "scene", "shot", "frame", "view", "image", "background", "foreground",
            "side", "center", "centre", "camera", "moment", "part", "area", "sense", "kind", "type", "show", "shows", "showing",
            "appears", "features", "featuring", "caption", "headline", "text", "title"}


def _pool(shots_by_id, exclude_ids, pos):
    """Words of a part of speech that appear in OTHER shots' facts."""
    nlp, out = _spacy(), []
    for sid, sh in shots_by_id.items():
        if sid in exclude_ids:
            continue
        f = sh.get("facts") or {}
        for tok in nlp(" ".join(str(f.get(k, "")) for k in ("action", "key"))):
            if tok.pos_ == pos and tok.is_alpha and len(tok.text) > 3 and tok.text.lower() not in ABSTRACT:
                out.append((tok.text, tok.tag_))
    return out


def hard(rec, i, lines, shots_by_id, facts_words):
    """Return (kind, corrupted line) or None."""
    text, nlp = rec["text"], _spacy()
    kinds = ["noun", "verb", "character", "neighbour"]
    for kind in kinds[i % 4:] + kinds[: i % 4]:
        if kind == "character":
            for a, b in SWAPS[:6]:
                if a in text:
                    return kind, text.replace(a, b, 1)
        elif kind == "neighbour":
            placed = [r for r in lines if r.get("audio")]
            k = placed.index(rec)
            for j in (k + 2, k - 2, k + 3, k - 3):
                if 0 <= j < len(placed) and placed[j]["text"] != text:
                    return kind, placed[j]["text"]
        else:
            pos = "NOUN" if kind == "noun" else "VERB"
            doc = nlp(text)
            targets = [t for t in doc if t.pos_ == pos and t.is_alpha and len(t.text) > 3 and t.text.lower() not in ABSTRACT]
            pool = [(w, tag) for w, tag in _pool(shots_by_id, set(rec.get("verify_shots") or rec["shots"]), pos)
                    if w.lower() not in facts_words]
            for t in targets:
                same = [w for w, tag in pool if tag == t.tag_ and w.lower() != t.text.lower()]
                if same:
                    w = same[(i * 7) % len(same)]
                    return kind, text[: t.idx] + w + text[t.idx + len(t.text):]
    return None


def injected_error_test(model, lines, shots_by_id, n: int, workers: int = 1):
    from .pipeline import facts_text
    pool = [r for r in lines if r.get("audio")][:n]

    def facts_for(r):
        f = facts_text(shots_by_id, r.get("verify_shots") or r["shots"])
        if r.get("text_read"):
            f += "\nOn-screen text shown just before: " + " | ".join(r["text_read"])
        return f

    jobs = []
    for i, r in enumerate(pool):
        f = facts_for(r)
        jobs.append(("easy", "append/swap", r, f, easy(r["text"], i)))
        h = hard(r, i, lines, shots_by_id, set(re.findall(r"[a-z]+", f.lower())))
        if h:
            jobs.append(("hard", h[0], r, f, h[1]))

    def run(job):
        level, kind, r, f, bad = job
        v = llm.parse_json(model.chat(prompts.VERIFY.format(facts=f, line=bad), max_tokens=120, temperature=0.0)) or {}
        return {"set": level, "kind": kind, "line": r["id"], "original": r["text"], "corrupted": bad,
                "caught": v.get("supported") is False, "unsupported": v.get("unsupported", [])}

    cases = list(llm.pmap(model, run, jobs, workers))
    out = {"clean_lines": len(pool),
           "false_alarms_on_clean_lines": sum(1 for r in pool if (r.get("verify") or {}).get("supported") is False),
           "cases": cases}
    for level in ("easy", "hard"):
        cs = [c for c in cases if c["set"] == level]
        out[level] = {"planted": len(cs), "caught": sum(c["caught"] for c in cs),
                      "catch_rate": round(sum(c["caught"] for c in cs) / max(1, len(cs)), 2)}
        if level == "hard":
            out[level]["by_kind"] = {k: f'{sum(c["caught"] for c in cs if c["kind"] == k)}/{sum(1 for c in cs if c["kind"] == k)}'
                                     for k in ("noun", "verb", "character", "neighbour")}
    # kept for older readers of metrics.json
    out["planted"], out["caught"], out["catch_rate"] = out["easy"]["planted"], out["easy"]["caught"], out["easy"]["catch_rate"]
    return out
