SHOT = """You are helping draft audio description (AD) for blind and low-vision viewers.
Cast sheet (use a name only if the person clearly matches its description):
{cast}
Dialogue heard around this shot: "{ctx}"

The image(s) come from one film shot lasting {dur:.1f} s. Report only what is visible.
Reply with JSON only:
{{"who": "names, or brief visual descriptions of the people visible, or none",
  "where": "the setting",
  "action": "what is happening, present tense",
  "text": "legible on-screen text, or empty",
  "key": "the single most important visual fact for a blind viewer",
  "confidence": 0.0 to 1.0}}"""

WRITE = """You write ONE audio description line for a film, for blind and low-vision viewers.
Rules: present tense. Describe only what is visible. If the facts show story text on screen (a title card such as "40 YEARS LATER", a sign, a headline) that has not been read yet, read it first, e.g. "A caption: Forty years later." Ignore sponsor credits and logos. Name feelings only through a visible expression or gesture. No camera language ("we see", "shot of"). Do not repeat what the dialogue already says. Use cast names or cast labels (such as \"the sniper\") when the person matches. Plain, concrete words.
Cast: {cast}
Recent AD lines (do not repeat them): {recent}
Dialogue just before: "{before}"
Dialogue just after: "{after}"
Visible during this window:
{facts}
{text_note}
The window is {avail:.1f} s long. Hard limit: {budget} words.
If nothing new is visible, reply SKIP. Otherwise reply with the line only."""

SHORTEN = """Shorten this audio description line. Spoken, it takes {dur:.1f} s, but the silence is only {avail:.1f} s.
Rewrite it in at most {target} words. Keep the most important visual fact (who and what they do comes first).
Present tense, no interpretation, no camera language. Reply with the line only.
Line: {line}"""

RESTYLE = """Rewrite this audio description line to fix these problems: {problems}.
Describe only what is visible, present tense, no guessing words (seems, appears, maybe), no camera language, no "I/we".
If the problem is "credits": describe the visible image instead of reading production credits, sponsors or logos.
At most {budget} words. Reply with the line only.
Line: {line}"""

VERIFY = """Check an audio description line against the facts a vision model extracted from the same shots.
Facts:
{facts}
Line: "{line}"
Is every concrete claim in the line (people, names, objects, colours, actions, places, text) supported by the facts?
Reply with JSON only: {{"supported": true or false, "unsupported": ["claim", ...]}}"""

READ_TEXT = """Rewrite this audio description line so it first reads the story text visible on screen, for example "A headline: Roboticist creates killer robots." or "A caption: Forty years later."
On-screen text: {text}
Then add the most important visible action if words remain. At most {budget} words. Present tense, no interpretation. Reply with the line only.
Line: {line}"""
