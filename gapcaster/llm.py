"""One interface, two backends.

- MLXBackend runs the VLM in-process on Apple Silicon (Stage I laptop prototype).
- OpenAIBackend talks to any OpenAI-compatible server, e.g. vLLM / llm-scaler on an
  Intel Arc Pro B70 (Stage II target). Thinking is switched off through
  chat_template_kwargs, which vLLM forwards to the Qwen chat template.
"""
import base64
import json
import os
import re
import time


class Stats:
    def __init__(self):
        self.calls = 0
        self.seconds = 0.0
        self.prompt_tokens = 0
        self.gen_tokens = 0

    def as_dict(self):
        return {"calls": self.calls, "seconds": round(self.seconds, 1),
                "prompt_tokens": self.prompt_tokens, "gen_tokens": self.gen_tokens}


class MLXBackend:
    def __init__(self, model_path: str):
        from mlx_vlm import load
        self.path = os.path.expanduser(model_path)
        self.model, self.processor = load(self.path)
        self.stats = Stats()

    def chat(self, prompt: str, images=None, max_tokens: int = 200, temperature: float = 0.4) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        images = [str(i) for i in (images or [])]
        msgs = [{"role": "user", "content": prompt}]
        templ = apply_chat_template(self.processor, self.model.config, msgs, num_images=len(images),
                                    enable_thinking=False)
        t = time.time()
        out = generate(self.model, self.processor, templ, image=images or None, max_tokens=max_tokens,
                       temperature=temperature, top_p=0.8, verbose=False)
        self.stats.calls += 1
        self.stats.seconds += time.time() - t
        self.stats.prompt_tokens += int(getattr(out, "prompt_tokens", 0) or 0)
        self.stats.gen_tokens += int(getattr(out, "generation_tokens", 0) or 0)
        return (out.text if hasattr(out, "text") else str(out)).strip()


class OpenAIBackend:
    def __init__(self, base_url: str, model: str):
        import requests
        self.requests = requests
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.stats = Stats()

    def chat(self, prompt: str, images=None, max_tokens: int = 200, temperature: float = 0.4) -> str:
        content = [{"type": "text", "text": prompt}]
        for p in images or []:
            b64 = base64.b64encode(open(p, "rb").read()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.8,
                "chat_template_kwargs": {"enable_thinking": False}}
        t = time.time()
        r = self.requests.post(self.url, json=body, timeout=600)
        r.raise_for_status()
        data = r.json()
        self.stats.calls += 1
        self.stats.seconds += time.time() - t
        u = data.get("usage") or {}
        self.stats.prompt_tokens += u.get("prompt_tokens", 0)
        self.stats.gen_tokens += u.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"].strip()


def make(settings):
    if settings.llm_backend == "openai":
        return OpenAIBackend(settings.llm_base_url, settings.llm_model_name)
    return MLXBackend(settings.vlm_model)


def parse_json(text: str):
    """Pull the first JSON object out of a model reply."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
