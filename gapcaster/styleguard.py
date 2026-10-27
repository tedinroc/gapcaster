"""Deterministic style gate for audio description (no model involved).

Hard rules block a line and force a rewrite; soft rules only flag it for the human describer.
Based on common AD practice: describe what is visible, in the present tense, without
interpretation, camera language or guesses.
"""
import re

HARD = {
    "camera-language": r"\b(we see|we hear|we watch|the camera|camera (?:pans|zooms|cuts)|shot of|the scene shows|on screen we)\b",
    "speculation": r"\b(seems?|seemingly|appears? to|apparently|probably|perhaps|maybe|possibly|it looks like|presumably|likely)\b",
    "first-person": r"\bI\b(?=\s+(?:am|was|have|had|see|saw|think|hear|notice|feel|can|will|would|do|did|know)\b)|\bI['’](?:m|ve|d|ll)\b|\b(?i:we|us|our)\b",
}
SOFT = {
    "inner-state": r"\b(sad|happy|angry|upset|worried|nervous|scared|afraid|excited|confused|annoyed|frustrated|guilty|jealous|in love|exhausted|exhaustion|tired|weary|relieved|determined|hopeful|desperate|bored)\b",
    "evaluative": r"\b(beautiful|ugly|stunning|amazing|terrible|creepy|disgusting|gorgeous)\b",
    "simile": r"\bas if\b",
}


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", t.lower())


def check(line: str, max_words: int | None = None, exempt=()):
    """Return (hard, soft) rule names. `exempt` holds on-screen texts the line may read verbatim:
    they are removed before the rules run, so a caption containing "we" is not first person."""
    hard, soft = [], []
    # quoted text is read verbatim, not judged; a single quote inside a word (Celia's) is an apostrophe
    line = re.sub(r"[\"“][^\"”]*[\"”]|(?<![A-Za-z])['‘][^'’]{3,}['’](?![A-Za-z])", "", line)
    for t in exempt or ():
        words = [re.escape(w) for w in _norm(t).split() if w]
        if len(words) >= 2:
            line = re.sub(r"\W+".join(words), " ", line, flags=re.I)
    low = line.lower()
    for k, pat in HARD.items():
        flags = 0 if k == "first-person" else re.I
        if re.search(pat, line if k == "first-person" else low, flags):
            hard.append(k)
    for k, pat in SOFT.items():
        if re.search(pat, low):
            soft.append(k)
    if max_words and len(line.split()) > max_words * 1.6:
        soft.append("long")
    return hard, soft


STOP = set("a an the of and or to in on at with from by for his her their its is are as into onto over under up down".split())


def content_words(line: str):
    return {w for w in re.findall(r"[a-z']+", line.lower()) if w not in STOP}


def is_repeat(line: str, recent, threshold: float = 0.5) -> bool:
    """True when a draft mostly re-says one of the last lines (Jaccard on content words)."""
    a = content_words(line)
    for prev in recent[-2:]:
        b = content_words(prev)
        if a and b and len(a & b) / len(a | b) >= threshold:
            return True
    return False
