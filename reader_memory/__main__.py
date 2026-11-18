from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def offline_environment():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(root / "offline" / "tiktoken"))
    os.environ.setdefault("NLTK_DATA", str(root / "offline" / "nltk_data"))


def parser():
    result = argparse.ArgumentParser(description="Offline reader-conditioned memory research prototype")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    commands.add_parser("demo", help="CPU-only fixture integration check; not model results")
    server = commands.add_parser("serve", help="Serve the local reader on loopback")
    server.add_argument("--model-path", default="models/Qwen3.5-4B")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--device", default="cuda")
    server.add_argument("--max-input-tokens", type=int, default=12288)
    smoke = commands.add_parser("smoke", help="Real GPU smoke test with synthetic development data")
    smoke.add_argument("--model-path", default="models/Qwen3.5-4B")
    smoke.add_argument("--out-dir", default="runs/gpu_smoke")
    for name in ["build", "run", "judge"]:
        command = commands.add_parser(name)
        command.add_argument("--data", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
        command.add_argument("--model", default="Qwen3.5-4B")
        if name != "judge":
            command.add_argument("--dataset", choices=["longmemeval", "locomo"], default="longmemeval")
            command.add_argument("--limit", type=int, default=2, help="0 means all; default is only two questions")
        if name == "build":
            command.add_argument("--cache-dir", default="cache/write")
            command.add_argument("--window-turns", type=int, default=8)
            command.add_argument("--raw-only", action="store_true")
        if name == "run":
            command.add_argument("--memories", required=True)
            command.add_argument("--tokenizer-path", default="models/Qwen3.5-4B")
            command.add_argument("--strategy", choices=["raw", "fact", "summary", "mixed", "reader-aware", "llmlingua2"], default="mixed")
            command.add_argument("--budget", type=int, default=2048)
            command.add_argument("--top-k", type=int, default=24)
            command.add_argument("--profile")
            command.add_argument("--compressor-path", default="models/llmlingua2-small")
            command.add_argument("--dry-run", action="store_true")
        if name == "judge":
            command.add_argument("--predictions", required=True)
    return result


def demo():
    from .backend import FixtureBackend
    from .memory import build_groups
    from .retrieval import BM25
    from .schema import Turn
    from .selection import allocate, render_evidence
    from .tokenization import ApproximateCounter
    turns = [Turn("s0:t0", "user", "I previously lived in Shanghai.", "2025-01-01"),
             Turn("s0:t1", "user", "I moved to Hangzhou in 2026.", "2026-10-01")]
    groups = build_groups(turns, FixtureBackend(), "FIXTURE-NOT-A-MODEL")
    choices, statistics = allocate("Where do I currently live?", BM25(groups).search("Where do I currently live?"), ApproximateCounter(), 200)
    print(json.dumps({"mode": "synthetic-fixture-only", "statistics": statistics, "evidence": render_evidence(choices)}, ensure_ascii=False, indent=2))


def smoke(args):
    from .backend import TransformersBackend
    from .experiment import answer_messages
    from .memory import build_groups
    from .retrieval import BM25
    from .schema import Turn, write_json
    from .selection import allocate, render_evidence
    from .tokenization import TokenCounter
    backend = TransformersBackend(args.model_path)
    turns = [Turn("dev0:t0", "user", "I lived in Shanghai until 2025. In January 2026 I moved to Hangzhou. I currently live in Hangzhou.", "2026-10-01"),
             Turn("dev0:t1", "assistant", "I understand that Hangzhou is your current city and Shanghai is your previous city.", "2026-10-01")]
    groups = build_groups(turns, backend, backend.model_name)
    counter = TokenCounter(args.model_path)
    selected, statistics = allocate("Where do I currently live?", BM25(groups).search("Where do I currently live?"), counter, 512)
    response = backend.complete(answer_messages("Where do I currently live?", "2026-10-25", render_evidence(selected)), max_tokens=96)
    if not response["text"] or not any(city in response["text"].lower() for city in ["hangzhou", "杭州"]) or statistics["evidence_tokens"] > 512:
        raise RuntimeError("GPU smoke failed")
    result = {"status": "passed", "model": backend.model_name, "synthetic_development_only": True,
              "budget": statistics, "response": response, "groups": [group.to_dict() for group in groups],
              "gpu_peak_allocated_bytes": backend.torch.cuda.max_memory_allocated(),
              "gpu_peak_reserved_bytes": backend.torch.cuda.max_memory_reserved()}
    write_json(Path(args.out_dir) / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    offline_environment()
    args = parser().parse_args()
    if args.command == "doctor":
        print("Python", sys.version)
        for name in ["torch", "transformers", "accelerate", "llmlingua"]:
            try:
                print(name, importlib.metadata.version(name))
            except importlib.metadata.PackageNotFoundError:
                print(name, "MISSING")
        try:
            import torch
            print("CUDA available", torch.cuda.is_available())
            if torch.cuda.is_available():
                print("GPU", torch.cuda.get_device_name(0), "VRAM", torch.cuda.get_device_properties(0).total_memory)
        except ImportError:
            pass
        return
    if args.command == "demo":
        return demo()
    if args.command == "serve":
        from .backend import TransformersBackend
        from .server import serve
        return serve(TransformersBackend(args.model_path, args.device, args.max_input_tokens), args.host, args.port)
    if args.command == "smoke":
        return smoke(args)
    from .backend import HTTPBackend
    from .experiment import build_dataset, run_dataset, judge_longmemeval
    backend = HTTPBackend(args.endpoint, args.model)
    if args.command == "build":
        return build_dataset(args.dataset, args.data, args.output, backend, args.model, args.limit or None, args.cache_dir, args.window_turns, args.raw_only)
    if args.command == "judge":
        return judge_longmemeval(args.data, args.predictions, args.output, backend, args.model)
    from .tokenization import TokenCounter
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8")) if args.profile else None
    compressor = None
    if args.strategy == "llmlingua2":
        from .baselines import LLMLingua2Compressor
        compressor = LLMLingua2Compressor(args.compressor_path)
    return run_dataset(args.dataset, args.data, args.memories, args.output, backend, args.model,
                       TokenCounter(args.tokenizer_path), args.strategy, args.budget, profile, args.top_k,
                       args.limit or None, compressor, args.dry_run)


if __name__ == "__main__":
    main()
