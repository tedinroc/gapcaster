"""Injected-error test: plant false visual claims in finished lines and count how many the verifier flags."""
from . import llm, prompts

SWAPS = [("Thom", "Celia"), ("Celia", "Thom"), (" man", " woman"), (" woman", " man")]
ADDS = [", holding a red umbrella", " as rain falls", " in a crowded train station", ". A sign reads EXIT", " beside a white horse"]


def corrupt(text: str, i: int) -> str:
    if i % 3 == 1:
        for a, b in SWAPS:
            if a in text:
                return text.replace(a, b, 1)
    base = text.rstrip(".")
    return base + ADDS[i % len(ADDS)] + "."


def injected_error_test(model, lines, shots_by_id, n: int):
    from .pipeline import facts_text
    pool = [r for r in lines if r.get("audio")][:n]
    caught, false_alarm, cases = 0, 0, []
    for i, r in enumerate(pool):
        facts = facts_text(shots_by_id, r["shots"])
        if (r.get("verify") or {}).get("supported") is False:
            false_alarm += 1
        bad = corrupt(r["text"], i)
        v = llm.parse_json(model.chat(prompts.VERIFY.format(facts=facts, line=bad), max_tokens=120, temperature=0.0)) or {}
        hit = v.get("supported") is False
        caught += hit
        cases.append({"line": r["id"], "corrupted": bad, "caught": hit, "unsupported": v.get("unsupported", [])})
    return {"planted": len(pool), "caught": caught, "catch_rate": round(caught / max(1, len(pool)), 2),
            "false_alarms_on_clean_lines": false_alarm, "clean_lines": len(pool), "cases": cases}
