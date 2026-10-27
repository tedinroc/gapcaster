"""Minimal OpenAI-compatible /v1/chat/completions test server backed by the in-process MLX model.

Used only to exercise GapCaster's `--backend openai` client path (images as data URLs,
chat_template_kwargs, concurrency) on a laptop before Stage II moves to vLLM on a B70.
Requests are served one at a time; this checks the code path, not throughput.
Usage: python eval/local_server.py [port]
"""
import base64
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gapcaster.config import Settings
from gapcaster.llm import MLXBackend

MODEL = MLXBackend(Settings().vlm_model)
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt, images = "", []
        for part in body["messages"][-1]["content"]:
            if part["type"] == "text":
                prompt += part["text"]
            elif part["type"] == "image_url":
                b64 = part["image_url"]["url"].split(",", 1)[1]
                f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                f.write(base64.b64decode(b64))
                f.close()
                images.append(f.name)
        assert body.get("chat_template_kwargs", {}).get("enable_thinking") is False
        with LOCK:
            text = MODEL.chat(prompt, images=images, max_tokens=body.get("max_tokens", 200), temperature=body.get("temperature", 0.4))
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}],
                          "usage": {"prompt_tokens": 0, "completion_tokens": len(text.split())}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"serving on 127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
