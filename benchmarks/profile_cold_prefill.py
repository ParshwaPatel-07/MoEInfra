#!/usr/bin/env python3
"""Profile one real cold MoEInfra inference request with cProfile.

This does NOT modify project code. It initializes the runtime first, then
profiles exactly one engine.generate() call so runtime initialization and
benchmark setup do not contaminate the profile.

Usage:
  python -m benchmarks.profile_cold_prefill --config config.yaml --gc-mode normal
  python -m benchmarks.profile_cold_prefill --config config.yaml --gc-mode disabled

The script uses max_new_tokens=1 so the measured workload is dominated by
prefill/TTFT rather than decode.
"""
from __future__ import annotations

import argparse
import cProfile
import gc
import io
import os
import pstats
import time

import torch
from transformers import AutoTokenizer

from config_loader import load_config
from engine.engine import InferenceEngine
from engine.types import ForwardRequest

MODEL_PATH = (
    "/kaggle/input/models/mistral-ai/mixtral/"
    "pytorch/8x7b-instruct-v0.1-hf/1"
)
STORE1 = (
    "/kaggle/input/datasets/parshwapatel07/"
    "mixtral-8x7b-nf4-experts/experts"
)
STORE2 = (
    "/kaggle/input/datasets/parshwapatel07/"
    "mixtral-8x7b-nf4-experts-part2/experts"
)
PROMPT = (
    "Explain how mixture-of-experts language models reduce inference cost "
    "compared with dense language models, and describe the role of expert "
    "routing and expert caching in practical inference systems."
)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_engine(config_path: str, layers: int) -> tuple[InferenceEngine, torch.Tensor]:
    config = load_config(config_path)
    config["model"]["name"] = MODEL_PATH
    config["model"]["device"] = "cuda:0"
    config["model"]["dtype"] = "float16"
    config["model"]["num_layers"] = min(
        int(config["model"].get("num_layers", 32)), layers
    )
    config["model"]["expert_store_paths"] = [STORE1, STORE2]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
    inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = inputs["input_ids"].to("cuda:0")

    engine = InferenceEngine(config)
    cuda_sync()
    engine._initialize_runtime()
    cuda_sync()
    return engine, input_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--gc-mode", choices=("normal", "disabled"), default="normal")
    parser.add_argument("--top", type=int, default=80)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.layers < 1:
        raise ValueError("--layers must be >= 1")

    print("=" * 72)
    print("MoEInfra — REAL COLD PREFILL cProfile")
    print("=" * 72)
    print(f"Layers:        {args.layers}")
    print("Max new:       1")
    print(f"GC mode:       {args.gc_mode}")
    print()

    print("Initializing runtime outside profiler...")
    init_t0 = time.perf_counter()
    engine, input_ids = build_engine(args.config, args.layers)
    init_ms = (time.perf_counter() - init_t0) * 1000.0
    print(f"Initialization: {init_ms:,.2f} ms")
    print(f"Prompt tokens:  {input_ids.shape[1]}")
    print()

    # Ensure this is genuinely cold with respect to the GPU expert cache.
    # We do not run a warm request before profiling.
    gc_was_enabled = gc.isenabled()
    if args.gc_mode == "disabled":
        gc.disable()

    profiler = cProfile.Profile()
    cuda_sync()
    wall_t0 = time.perf_counter()
    error = None
    result = None

    try:
        profiler.enable()
        result = engine.generate(
            ForwardRequest(
                input_ids=input_ids,
                max_new_tokens=1,
            )
        )
    except Exception as exc:  # report profile even if inference fails
        error = exc
    finally:
        profiler.disable()
        cuda_sync()
        wall_ms = (time.perf_counter() - wall_t0) * 1000.0
        if gc_was_enabled and args.gc_mode == "disabled":
            gc.enable()

    print(f"Profiled wall time: {wall_ms:,.2f} ms")
    if result is not None:
        print(f"Reported TTFT:      {result.ttft_ms:,.2f} ms")
        print(f"Generated tokens:   {len(result.token_ids)}")
    if error is not None:
        print(f"Inference error:    {type(error).__name__}: {error}")
    print()

    output_path = os.path.abspath(
        f"benchmarks/results/cold_prefill_{args.gc_mode}.prof"
    )
    text_path = os.path.abspath(
        f"benchmarks/results/cold_prefill_{args.gc_mode}_top{args.top}.txt"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    profiler.dump_stats(output_path)

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
    stats.print_stats(args.top)
    text = stream.getvalue()
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Profile data:      {output_path}")
    print(f"Top functions:     {text_path}")
    print()
    print(text)


if __name__ == "__main__":
    main()
