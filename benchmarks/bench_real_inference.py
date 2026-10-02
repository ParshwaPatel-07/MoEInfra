#!/usr/bin/env python3
"""Standalone real-inference benchmark for MoEInfra.

This is intentionally NOT a pytest test. The script keeps the benchmark path
as small as practical so that the numbers describe MoEInfra inference rather
than pytest startup/collection overhead.

It reports:
  * runtime initialization time
  * cold inference (first request after runtime initialization)
  * warm inference runs on the same persistent engine
  * TTFT / decode time / decode TPS
  * generated tokens and decoded text
  * GPU peak memory
  * cache statistics

Usage on Kaggle:
    python benchmarks/bench_real_inference.py

Optional environment variables:
    MOEINFRA_PROFILE_LAYERS   Number of layers to use (default: 32)
    MOEINFRA_NEW_TOKENS       Number of generated tokens (default: 32)
    MOEINFRA_WARM_RUNS        Number of warm runs (default: 3)
    MOEINFRA_GPU_SLOTS        GPU expert cache slots (default: leave config value)
    MOEINFRA_CPU_SLOTS        CPU expert cache slots (default: leave config value)
"""

from __future__ import annotations

import gc
import os
import statistics
import time
from pathlib import Path

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

# Fixed prompt = reproducible benchmark input.
PROMPT = (
    "Explain how mixture-of-experts language models reduce inference cost "
    "compared with dense language models, and describe the role of expert "
    "routing and expert caching in practical inference systems."
)


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stats_dict(stats) -> dict:
    """Best-effort conversion of CacheStats/dataclass/object to a dict."""
    if hasattr(stats, "__dict__"):
        return dict(vars(stats))

    fields = (
        "hits",
        "misses",
        "evictions",
        "gpu_slots_used",
        "cpu_slots_used",
    )
    return {name: getattr(stats, name) for name in fields if hasattr(stats, name)}


def _fmt_ms(value: float) -> str:
    return f"{value:,.2f} ms"


def _run(engine: InferenceEngine, input_ids: torch.Tensor, max_new_tokens: int):
    """Run one inference and return wall time plus ForwardResult."""
    _cuda_sync()
    t0 = time.perf_counter()
    result = engine.generate(
        ForwardRequest(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )
    )
    _cuda_sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms, result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-inference benchmark")

    num_layers = min(int(os.environ.get("MOEINFRA_PROFILE_LAYERS", "32")), 32)
    max_new_tokens = int(os.environ.get("MOEINFRA_NEW_TOKENS", "32"))
    warm_runs = max(int(os.environ.get("MOEINFRA_WARM_RUNS", "3")), 1)

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

    config = load_config("config.yaml")
    config["model"]["name"] = MODEL_PATH
    config["model"]["device"] = "cuda:0"
    config["model"]["dtype"] = "float16"
    config["model"]["num_layers"] = num_layers
    config["model"]["expert_store_paths"] = [STORE1, STORE2]

    # Only override cache sizes when explicitly requested. Otherwise the
    # benchmark uses the project's configured baseline.
    if "MOEINFRA_GPU_SLOTS" in os.environ:
        config.setdefault("cache", {})["gpu_slots"] = int(
            os.environ["MOEINFRA_GPU_SLOTS"]
        )
    if "MOEINFRA_CPU_SLOTS" in os.environ:
        config.setdefault("cache", {})["cpu_slots"] = int(
            os.environ["MOEINFRA_CPU_SLOTS"]
        )

    print("=" * 68)
    print("MoEInfra — REAL INFERENCE BASELINE")
    print("=" * 68)
    print(f"Model:             Mixtral-8x7B-Instruct-v0.1")
    print(f"Layers:            {num_layers}/32")
    print(f"Device:            cuda:0")
    print(f"Prompt tokens:     {input_ids.shape[1]}")
    print(f"Max new tokens:    {max_new_tokens}")
    print(f"Warm runs:         {warm_runs}")
    print()
    print("Prompt:")
    print(PROMPT)
    print()

    # Runtime initialization is deliberately timed separately from inference.
    engine_construct_t0 = time.perf_counter()
    engine = InferenceEngine(config)
    engine_construct_ms = (time.perf_counter() - engine_construct_t0) * 1000.0

    try:
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()

        _cuda_sync()
        init_t0 = time.perf_counter()
        engine._initialize_runtime()
        _cuda_sync()
        init_ms = (time.perf_counter() - init_t0) * 1000.0

        print("RUNTIME INITIALIZATION")
        print(f"  Engine construction: {_fmt_ms(engine_construct_ms)}")
        print(f"  Runtime initialization: {_fmt_ms(init_ms)}")
        print()

        # First inference exercises the current engine with a cold GPU expert
        # cache. This is the most useful measurement for the existing baseline.
        cold_wall_ms, cold_result = _run(
            engine,
            input_ids,
            max_new_tokens,
        )

        cold_gpu_peak_mib = torch.cuda.max_memory_allocated() / (1024**2)

        print("COLD INFERENCE")
        print(f"  Wall time:        {_fmt_ms(cold_wall_ms)}")
        print(f"  TTFT / prefill:   {_fmt_ms(cold_result.ttft_ms)}")
        print(f"  Decode time:      {_fmt_ms(cold_result.stage_timings['decode'])}")
        print(f"  Decode TPS:       {cold_result.tps:,.4f}")
        print(f"  Generated tokens: {len(cold_result.token_ids)}")
        print(f"  GPU peak allocated:{cold_gpu_peak_mib:,.2f} MiB")
        print(f"  Cache stats:      {_stats_dict(engine._cache_manager.stats())}")
        print()

        print("COLD GENERATED TEXT")
        print(tokenizer.decode(cold_result.token_ids, skip_special_tokens=False))
        print()

        warm_wall_ms: list[float] = []
        warm_ttft_ms: list[float] = []
        warm_decode_ms: list[float] = []
        warm_tps: list[float] = []

        print("WARM INFERENCE")
        for run_idx in range(1, warm_runs + 1):
            wall_ms, result = _run(
                engine,
                input_ids,
                max_new_tokens,
            )
            warm_wall_ms.append(wall_ms)
            warm_ttft_ms.append(result.ttft_ms)
            warm_decode_ms.append(result.stage_timings["decode"])
            warm_tps.append(result.tps)

            print(
                f"  Run {run_idx}: "
                f"wall={wall_ms:,.2f} ms | "
                f"TTFT={result.ttft_ms:,.2f} ms | "
                f"decode={result.stage_timings['decode']:,.2f} ms | "
                f"TPS={result.tps:,.4f}"
            )

        print()
        print("WARM MEDIAN")
        print(f"  Wall time:        {_fmt_ms(statistics.median(warm_wall_ms))}")
        print(f"  TTFT / prefill:   {_fmt_ms(statistics.median(warm_ttft_ms))}")
        print(f"  Decode time:      {_fmt_ms(statistics.median(warm_decode_ms))}")
        print(f"  Decode TPS:       {statistics.median(warm_tps):,.4f}")
        print()

        print("FINAL CACHE STATE")
        print(f"  {_stats_dict(engine._cache_manager.stats())}")
        print()
        print("GPU MEMORY")
        print(
            f"  Peak allocated:   "
            f"{torch.cuda.max_memory_allocated() / (1024**2):,.2f} MiB"
        )
        print(
            f"  Current allocated: "
            f"{torch.cuda.memory_allocated() / (1024**2):,.2f} MiB"
        )
        print()
        print("=" * 68)
        print("BASELINE COMPLETE")
        print("=" * 68)

    finally:
        engine.shutdown()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
