#!/usr/bin/env python3
"""Standalone real-inference benchmark for MoEInfra.

The benchmark explicitly separates expert-store preparation from inference.

Startup preparation:
    1. Copy layers 8-31 from /kaggle/input -> /kaggle/working.
    2. Warm/read all local expert files once.
    3. Load layers 0-7 completely into CPU RAM.
    4. Verify all 256 experts are available.
    5. Register RAM-resident experts with ModelLoader.

Only after those steps do we start measuring inference.

This prevents network-mounted /kaggle/input I/O from silently contaminating
TTFT or prefill measurements.
"""

from __future__ import annotations

import gc
import os
import statistics
import sys
import time
from pathlib import Path

# Make direct execution work:
#     python benchmarks/bench_real_inference.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer

from config_loader import load_config
from engine.engine import InferenceEngine
from engine.types import ForwardRequest
from benchmarks.expert_store_bootstrap import (
    LOCAL_STORE,
    prepare_expert_store,
)


MODEL_PATH = (
    "/kaggle/input/models/mistral-ai/mixtral/"
    "pytorch/8x7b-instruct-v0.1-hf/1"
)

# Fixed prompt for reproducible benchmarking.
PROMPT = (
    "Explain how mixture-of-experts language models reduce inference cost "
    "compared with dense language models, and describe the role of expert "
    "routing and expert caching in practical inference systems."
)


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stats_dict(stats) -> dict:
    """Best-effort conversion of cache statistics to a dictionary."""
    if hasattr(stats, "__dict__"):
        return dict(vars(stats))

    fields = (
        "hits",
        "misses",
        "evictions",
        "gpu_slots_used",
        "cpu_slots_used",
    )

    return {
        name: getattr(stats, name)
        for name in fields
        if hasattr(stats, name)
    }


def _fmt_ms(value: float) -> str:
    return f"{value:,.2f} ms"


def _run(
    engine: InferenceEngine,
    input_ids: torch.Tensor,
    max_new_tokens: int,
):
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
        raise RuntimeError(
            "CUDA is required for the real-inference benchmark."
        )

    # ---------------------------------------------------------------
    # Benchmark configuration
    # ---------------------------------------------------------------

    num_layers = min(
        int(os.environ.get("MOEINFRA_PROFILE_LAYERS", "32")),
        32,
    )

    max_new_tokens = int(
        os.environ.get("MOEINFRA_NEW_TOKENS", "32")
    )

    warm_runs = max(
        int(os.environ.get("MOEINFRA_WARM_RUNS", "3")),
        1,
    )

    # ---------------------------------------------------------------
    # Tokenizer / input preparation
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Print benchmark configuration
    # ---------------------------------------------------------------

    print("=" * 72)
    print("MoEInfra — REAL INFERENCE BASELINE")
    print("=" * 72)
    print(f"Model:             Mixtral-8x7B-Instruct-v0.1")
    print(f"Layers:            {num_layers}/32")
    print(f"Device:            cuda:0")
    print(f"Prompt tokens:     {input_ids.shape[1]}")
    print(f"Max new tokens:    {max_new_tokens}")
    print(f"Warm runs:         {warm_runs}")
    print(f"Local expert store:{LOCAL_STORE}")
    print()
    print("Prompt:")
    print(PROMPT)
    print()

    # ---------------------------------------------------------------
    # EXPERT STORE PREPARATION
    #
    # IMPORTANT:
    # This entire section is startup work.
    # No inference timing starts before it completes.
    # ---------------------------------------------------------------

    print("=" * 72)
    print("EXPERT STORE PREPARATION")
    print("=" * 72)

    preparation_t0 = time.perf_counter()

    ram_experts = prepare_expert_store()

    preparation_ms = (
        time.perf_counter() - preparation_t0
    ) * 1000.0

    print()
    print("EXPERT STORE READY")
    print(
        f"  RAM experts:      {len(ram_experts)}"
    )
    print(
        f"  Local store:      {LOCAL_STORE}"
    )
    print(
        f"  Startup time:     {_fmt_ms(preparation_ms)}"
    )
    print()

    # ---------------------------------------------------------------
    # Engine configuration
    # ---------------------------------------------------------------

    config = load_config("config.yaml")

    config["model"]["name"] = MODEL_PATH
    config["model"]["device"] = "cuda:0"
    config["model"]["dtype"] = "float16"
    config["model"]["num_layers"] = num_layers

    # IMPORTANT:
    # Only the local working-store path is visible to ModelLoader.
    #
    # Layers 0-7 are supplied directly through the preloaded CPU expert
    # dictionary. Layers 8-31 come from this local filesystem path.
    #
    # /kaggle/input is therefore NOT part of the inference-time
    # expert-store search path.
    config["model"]["expert_store_paths"] = [
        str(LOCAL_STORE),
    ]

    # Optional cache overrides.
    if "MOEINFRA_GPU_SLOTS" in os.environ:
        config.setdefault("cache", {})["gpu_slots"] = int(
            os.environ["MOEINFRA_GPU_SLOTS"]
        )

    if "MOEINFRA_CPU_SLOTS" in os.environ:
        config.setdefault("cache", {})["cpu_slots"] = int(
            os.environ["MOEINFRA_CPU_SLOTS"]
        )

    print("ENGINE CONFIGURATION")
    print(
        f"  GPU cache slots:  "
        f"{config.get('cache', {}).get('gpu_slots', 8)}"
    )
    print(
        f"  CPU cache slots:  "
        f"{config.get('cache', {}).get('cpu_slots', 32)}"
    )
    print(
        f"  Expert store:     {config['model']['expert_store_paths']}"
    )
    print()

    # ---------------------------------------------------------------
    # Engine construction
    # ---------------------------------------------------------------

    engine_construct_t0 = time.perf_counter()

    engine = InferenceEngine(config)

    engine_construct_ms = (
        time.perf_counter() - engine_construct_t0
    ) * 1000.0

    try:
        # -----------------------------------------------------------
        # Register permanent RAM-resident experts.
        #
        # These objects remain alive for the lifetime of this engine.
        # -----------------------------------------------------------

        if not hasattr(
            engine._model,
            "set_preloaded_cpu_experts",
        ):
            raise RuntimeError(
                "ModelLoader is missing "
                "set_preloaded_cpu_experts(). "
                "Apply the ModelLoader change before running this benchmark."
            )

        engine._model.set_preloaded_cpu_experts(
            ram_experts
        )

        # -----------------------------------------------------------
        # Runtime initialization
        # -----------------------------------------------------------

        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()

        _cuda_sync()

        init_t0 = time.perf_counter()

        engine._initialize_runtime()

        _cuda_sync()

        init_ms = (
            time.perf_counter() - init_t0
        ) * 1000.0

        print("=" * 72)
        print("RUNTIME INITIALIZATION")
        print("=" * 72)
        print(
            f"  Engine construction:     "
            f"{_fmt_ms(engine_construct_ms)}"
        )
        print(
            f"  Runtime initialization:  "
            f"{_fmt_ms(init_ms)}"
        )
        print()

        # -----------------------------------------------------------
        # COLD INFERENCE
        #
        # This is now a genuine engine cold run:
        # expert storage has already been prepared.
        # -----------------------------------------------------------

        cold_wall_ms, cold_result = _run(
            engine,
            input_ids,
            max_new_tokens,
        )

        cold_gpu_peak_mib = (
            torch.cuda.max_memory_allocated()
            / (1024**2)
        )

        print("=" * 72)
        print("COLD INFERENCE")
        print("=" * 72)

        print(
            f"  Wall time:            "
            f"{_fmt_ms(cold_wall_ms)}"
        )

        print(
            f"  TTFT / prefill:       "
            f"{_fmt_ms(cold_result.ttft_ms)}"
        )

        print(
            f"  Decode time:          "
            f"{_fmt_ms(cold_result.stage_timings['decode'])}"
        )

        print(
            f"  Decode TPS:           "
            f"{cold_result.tps:,.4f}"
        )

        print(
            f"  Generated tokens:     "
            f"{len(cold_result.token_ids)}"
        )

        print(
            f"  GPU peak allocated:   "
            f"{cold_gpu_peak_mib:,.2f} MiB"
        )

        print(
            f"  Cache stats:          "
            f"{_stats_dict(engine._cache_manager.stats())}"
        )

        print()

        print("COLD GENERATED TEXT")
        print(
            tokenizer.decode(
                cold_result.token_ids,
                skip_special_tokens=False,
            )
        )
        print()

        # -----------------------------------------------------------
        # WARM INFERENCE
        # -----------------------------------------------------------

        warm_wall_ms: list[float] = []
        warm_ttft_ms: list[float] = []
        warm_decode_ms: list[float] = []
        warm_tps: list[float] = []

        print("=" * 72)
        print("WARM INFERENCE")
        print("=" * 72)

        for run_idx in range(1, warm_runs + 1):

            wall_ms, result = _run(
                engine,
                input_ids,
                max_new_tokens,
            )

            warm_wall_ms.append(wall_ms)
            warm_ttft_ms.append(result.ttft_ms)
            warm_decode_ms.append(
                result.stage_timings["decode"]
            )
            warm_tps.append(result.tps)

            print(
                f"  Run {run_idx}: "
                f"wall={wall_ms:,.2f} ms | "
                f"TTFT={result.ttft_ms:,.2f} ms | "
                f"decode={result.stage_timings['decode']:,.2f} ms | "
                f"TPS={result.tps:,.4f}"
            )

        print()

        # -----------------------------------------------------------
        # Warm medians
        # -----------------------------------------------------------

        print("WARM MEDIAN")

        print(
            f"  Wall time:            "
            f"{_fmt_ms(statistics.median(warm_wall_ms))}"
        )

        print(
            f"  TTFT / prefill:       "
            f"{_fmt_ms(statistics.median(warm_ttft_ms))}"
        )

        print(
            f"  Decode time:          "
            f"{_fmt_ms(statistics.median(warm_decode_ms))}"
        )

        print(
            f"  Decode TPS:           "
            f"{statistics.median(warm_tps):,.4f}"
        )

        print()

        # -----------------------------------------------------------
        # Final state
        # -----------------------------------------------------------

        print("=" * 72)
        print("FINAL CACHE STATE")
        print("=" * 72)

        print(
            f"  {_stats_dict(engine._cache_manager.stats())}"
        )

        print()

        print("GPU MEMORY")

        print(
            f"  Peak allocated:       "
            f"{torch.cuda.max_memory_allocated() / (1024**2):,.2f} MiB"
        )

        print(
            f"  Current allocated:    "
            f"{torch.cuda.memory_allocated() / (1024**2):,.2f} MiB"
        )

        print()

        # -----------------------------------------------------------
        # Final summary
        # -----------------------------------------------------------

        print("=" * 72)
        print("BENCHMARK SUMMARY")
        print("=" * 72)

        print(
            f"  Expert-store startup: "
            f"{_fmt_ms(preparation_ms)}"
        )

        print(
            f"  Runtime initialization:"
            f" {_fmt_ms(init_ms)}"
        )

        print(
            f"  Cold inference:       "
            f"{_fmt_ms(cold_wall_ms)}"
        )

        print(
            f"  Cold TTFT:            "
            f"{_fmt_ms(cold_result.ttft_ms)}"
        )

        print(
            f"  Warm median inference:"
            f" {_fmt_ms(statistics.median(warm_wall_ms))}"
        )

        print(
            f"  Warm median TTFT:     "
            f"{_fmt_ms(statistics.median(warm_ttft_ms))}"
        )

        print("=" * 72)

    finally:
        engine.shutdown()

        del engine
        del ram_experts

        gc.collect()

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()