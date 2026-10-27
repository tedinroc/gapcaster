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
    r.add_argument("--lang", default="en")
    r.add_argument("--backend", default="mlx", choices=["mlx", "openai"])
    r.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    r.add_argument("--inject", type=int, default=0, help="run the injected-error test on N lines")
    v = sub.add_parser("review", help="build the review page for a run")
    v.add_argument("run_dir")
    a = ap.parse_args()
    if a.cmd == "run":
        from .pipeline import run
        s = Settings(lang=a.lang, llm_backend=a.backend, llm_base_url=a.base_url)
        doc = run(a.video, a.out, a.start, a.end, a.cast, a.ref_srt, s, inject=a.inject)
        from .review import build
        build(a.out)
        print(json.dumps({k: v for k, v in doc["metrics"].items() if k != "injected_error_test"}, indent=1))
    else:
        from .review import build
        print(build(a.run_dir))


if __name__ == "__main__":
    main()
