import argparse
import json

from .config import Settings


def main():
    ap = argparse.ArgumentParser(prog="gapcaster", description="Draft an audio-description track that fits the silences between dialogue.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process a video")
    r.add_argument("video")
    r.add_argument("--out", required=True)
    r.add_argument("--start", type=float, default=0.0)
    r.add_argument("--end", type=float, default=None)
    r.add_argument("--cast", help="YAML cast sheet: [{name, look}]")
    r.add_argument("--ref-srt", help="reference subtitles, used only for evaluation")
    r.add_argument("--lang", default="en", choices=["en"], help="narration language (English; Traditional Chinese is planned)")
    r.add_argument("--backend", default="mlx", choices=["mlx", "openai"])
    r.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    r.add_argument("--model", default="qwen", help="model name on the OpenAI-compatible server (vLLM --served-model-name)")
    r.add_argument("--concurrency", type=int, default=8, help="parallel requests to the server (shot facts, verifier)")
    r.add_argument("--asr", default="mlx", choices=["mlx", "openvino"], help="speech recognition runtime")
    r.add_argument("--asr-device", default="CPU", help="OpenVINO device for ASR: CPU or GPU")
    r.add_argument("--tts", default="mlx", choices=["mlx", "kokoro"], help="TTS runtime: mlx-audio or PyTorch Kokoro (CPU / XPU)")
    r.add_argument("--inject", type=int, default=0, help="run the injected-error test on N lines")
    v = sub.add_parser("review", help="build the review page for a run")
    v.add_argument("run_dir")
    x = sub.add_parser("export", help="re-mix a run from the describer's approved.json")
    x.add_argument("run_dir")
    x.add_argument("--approved", required=True)
    x.add_argument("--tts", default=None, choices=["mlx", "kokoro"], help="TTS runtime for edited lines (default: the run's own)")
    a = ap.parse_args()
    if a.cmd == "run":
        from .pipeline import run
        s = Settings(lang=a.lang, llm_backend=a.backend, llm_base_url=a.base_url, llm_model_name=a.model,
                     concurrency=a.concurrency, asr_backend=a.asr, asr_device=a.asr_device, tts_backend=a.tts)
        doc = run(a.video, a.out, a.start, a.end, a.cast, a.ref_srt, s, inject=a.inject)
        from .review import build
        build(a.out)
        print(json.dumps({k: v for k, v in doc["metrics"].items() if k != "injected_error_test"}, indent=1))
    elif a.cmd == "export":
        from .approve import apply
        print(json.dumps(apply(a.run_dir, a.approved, tts_backend=a.tts), indent=1))
    else:
        from .review import build
        print(build(a.run_dir))


if __name__ == "__main__":
    main()
