#!/usr/bin/env python3
"""Microbenchmark the two remaining likely bottlenecks:

1. CPU-resident NF4 expert -> GPU transfer + reconstruction.
2. GPU-resident NF4 expert forward compute.

This deliberately avoids a full 32-layer inference run.

Usage:
    python -m benchmarks.bench_transfer_vs_expert_compute

The runtime initialization is reported separately because it is not part of
expert transfer or expert execution.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time

import torch

from config_loader import load_config
from engine.engine import InferenceEngine


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


def cuda_sync() -> None:
    torch.cuda.synchronize("cuda:0")


def timed_cuda(fn, repeats: int) -> list[float]:
    values: list[float] = []

    for _ in range(repeats):
        cuda_sync()
        t0 = time.perf_counter()
        fn()
        cuda_sync()
        values.append((time.perf_counter() - t0) * 1000.0)

    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--compute-repeats", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    config = load_config("config.yaml")
    config["model"]["name"] = MODEL_PATH
    config["model"]["device"] = "cuda:0"
    config["model"]["dtype"] = "float16"
    config["model"]["expert_store_paths"] = [STORE1, STORE2]
    config["model"]["num_layers"] = min(
        int(config["model"].get("num_layers", 32)),
        32,
    )

    print("=" * 72)
    print("MoEInfra — TRANSFER VS EXPERT COMPUTE MICROBENCHMARK")
    print("=" * 72)
    print(f"Layer:             {args.layer}")
    print(f"Expert:            {args.expert}")
    print()

    engine = InferenceEngine(config)

    try:
        cuda_sync()
        t0 = time.perf_counter()
        engine._initialize_runtime()
        cuda_sync()
        init_ms = (time.perf_counter() - t0) * 1000.0

        print(f"Runtime initialization: {init_ms:,.2f} ms")
        print()

        moe = engine._decoder_layers[args.layer].mlp.moe_layer

        # Load the prequantized CPU representation once. This is already known
        # to be fast from bench_expert_load.py, so transfer timing below excludes
        # safetensors I/O and PrequantizedNF4Expert construction.
        cpu_expert = engine._model.load_expert(
            args.layer,
            args.expert,
        )
        engine._cache_manager.put(
            args.layer,
            args.expert,
            cpu_expert,
            device="cpu",
        )

        print("TRANSFER + GPU RECONSTRUCTION")
        transfer_ms: list[float] = []

        for i in range(3):
            # Ensure there is no existing GPU copy.
            engine._cache_manager.demote_to_cpu(
                args.layer,
                args.expert,
            )

            values = timed_cuda(
                lambda: moe._ensure_expert_on_gpu(args.expert),
                repeats=1,
            )
            transfer_ms.extend(values)

            # Keep CPU backing and remove the temporary GPU copy.
            engine._cache_manager.demote_to_cpu(
                args.layer,
                args.expert,
            )

            print(f"  Trial {i + 1}: {values[0]:,.3f} ms")

        print(
            f"  Median: {statistics.median(transfer_ms):,.3f} ms"
        )
        print()

        # Re-establish the GPU expert for compute measurement.
        moe._ensure_expert_on_gpu(args.expert)
        expert = engine._cache_manager.peek_gpu(
            args.layer,
            args.expert,
        )
        if expert is None:
            raise RuntimeError("Expert was not resident on GPU.")

        print("EXPERT FORWARD COMPUTE")

        # Measure several token counts because actual routed token batches
        # during prefill are much smaller than the full 40-token prompt for
        # most experts.
        for tokens in (8, 16, 32, 40):
            x = torch.randn(
                tokens,
                config["model"].get("hidden_size", 4096),
                device="cuda:0",
                dtype=torch.float16,
            )

            # One warm-up call.
            with torch.no_grad():
                expert(x)
            cuda_sync()

            values = timed_cuda(
                lambda: expert(x),
                repeats=args.compute_repeats,
            )

            print(
                f"  {tokens:2d} tokens: "
                f"median={statistics.median(values):,.3f} ms | "
                f"min={min(values):,.3f} ms | "
                f"max={max(values):,.3f} ms"
            )

            del x

        del expert
        gc.collect()
        torch.cuda.empty_cache()

        print()
        print("INTERPRETATION")
        print(
            "  If transfer+reconstruction is ~10 ms but expert forward is "
            "hundreds/thousands of ms, the bottleneck is GPU expert compute."
        )
        print(
            "  If transfer+reconstruction is ~seconds, the bottleneck is "
            "the transfer/reconstruction path."
        )

    finally:
        engine.shutdown()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
