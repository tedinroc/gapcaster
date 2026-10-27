"""Text-to-speech with real duration measurement (Kokoro-82M, Apache-2.0, via mlx-audio)."""
import numpy as np


class TTS:
    def __init__(self, model: str, voice: str, lang: str = "en"):
        from mlx_audio.tts.utils import load_model
        self.model = load_model(model)
        self.voice = voice
        self.lang_code = {"en": "a", "zh": "z"}.get(lang, "a")
        self.sr = self.model.sample_rate
        self.calls = 0
        self.audio_seconds = 0.0
        self.synth("Ready.")  # warm the phonemizer so timings below are steady-state

    def synth(self, text: str, speed: float = 1.0):
        chunks = [np.array(r.audio, dtype=np.float32) for r in
                  self.model.generate(text=text, voice=self.voice, speed=speed, lang_code=self.lang_code)]
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        audio = trim(audio, self.sr)
        self.calls += 1
        self.audio_seconds += len(audio) / self.sr
        return audio, len(audio) / self.sr


def trim(a, sr, thresh: float = 0.01):
    """Cut leading/trailing near-silence so the measured duration is the spoken duration."""
    idx = np.where(np.abs(a) > thresh)[0]
    if len(idx) == 0:
        return a
    pad = int(0.03 * sr)
    return a[max(0, idx[0] - pad): min(len(a), idx[-1] + pad)]
