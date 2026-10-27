import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    lang: str = "en"
    # Gap map
    merge_pause: float = 0.6        # pauses shorter than this stay inside a dialogue span
    lead: float = 0.25              # silence kept after dialogue ends
    tail: float = 0.30              # silence kept before dialogue resumes
    min_slot: float = 1.1           # shortest window worth narrating
    breath: float = 0.8             # minimum quiet between two AD lines
    max_slot: float = 7.0           # long silences are split at shot boundaries
    # Writing budget
    words_per_sec: float = 2.6      # first-guess speaking rate for the budget
    # Fit loop
    speeds: tuple = (1.0, 1.08, 1.15)
    max_rewrites: int = 3
    # Models
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"
    # Hugging Face repo id or a local folder (set GAPCASTER_VLM to use a copy already on disk)
    vlm_model: str = os.environ.get("GAPCASTER_VLM", "lmstudio-community/Qwen3.8-27B-MLX-8bit")
    tts_model: str = "mlx-community/Kokoro-82M-bf16"
    tts_voice: str = "af_heart"
    llm_backend: str = "mlx"        # "mlx" (in-process, Mac) or "openai" (any OpenAI-compatible server, e.g. vLLM on Arc Pro B70)
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_model_name: str = "qwen"
    extra: dict = field(default_factory=dict)
