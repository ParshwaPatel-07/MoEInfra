#!/usr/bin/env python3
"""A-vs-B first-forward experiment for fresh NF4 expert instances.

Hypotheses:
  H_global:
      First forward of expert A is slow due to shape/dtype/global lazy
      initialization, but first forward of a new expert B with identical
      shapes is fast.

  H_instance:
      First forward of every newly reconstructed Linear4bit/Params4bit expert
      is slow, even when B has the exact same shapes as A.

The experiment measures:
    - CPU expert load (informational)
    - GPU transfer + reconstruction
    - FIRST forward of expert A
    - FIRST forward of a distinct expert B
    - SECOND forward of A
    - CUDA allocated/reserved memory before/after each forward

Run:
    python -m benchmarks.bench_first_forward_ab
"""

from __future__ import annotations

import argparse
import gc
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


def cuda_mem_mib() -> tuple[float, float]:
    return (
        torch.cuda.memory_allocated() / (1024**2),
        torch.cuda.memory_reserved() / (1024**2),
    )


def timed(fn) -> float:
    cuda_sync()
    t0 = time.perf_counter()
    fn()
    cuda_sync()
    return (time.perf_counter() - t0) * 1000.0


def run_forward(expert, x: torch.Tensor) -> tuple[float, float, float]:
    """Measure one forward and memory before/after."""
    allocated_before, reserved_before = cuda_mem_mib()

    cuda_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        y = expert(x)
    cuda_sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    allocated_after, reserved_after = cuda_mem_mib()

    # Keep only the output long enough to ensure the kernel completed, then
    # release it so the next measurement is not polluted by its allocation.
    del y

    return (
        elapsed_ms,
        allocated_after - allocated_before,
        reserved_after - reserved_before,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert-a", type=int, default=0)
    parser.add_argument("--expert-b", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=40)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if args.expert_a == args.expert_b:
        raise ValueError("--expert-a and --expert-b must be different.")

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
    print("MoEInfra — FIRST FORWARD A-vs-B EXPERIMENT")
    print("=" * 72)
    print(f"Layer:             {args.layer}")
    print(f"Expert A:          {args.expert_a}")
    print(f"Expert B:          {args.expert_b}")
    print(f"Tokens:            {args.tokens}")
    print()

    engine = InferenceEngine(config)

    try:
        # Build runtime before the actual experiment. This excludes model
        # construction/runtime initialization from the measurements below.
        init_ms = timed(engine._initialize_runtime)
        print(f"Runtime initialization: {init_ms:,.2f} ms")
        print()

        # Warm up the CUDA context with an unrelated operation so CUDA context
        # creation itself is not attributed to Expert A.
        x_warm = torch.zeros(
            (1,),
            device="cuda:0",
            dtype=torch.float16,
        )
        timed(lambda: x_warm.add_(1))
        del x_warm

        moe = engine._decoder_layers[args.layer].mlp.moe_layer

        # Use the same token shape for A, B, and A-again. This makes the
        # shape/dtype identical across all three forward measurements.
        x = torch.randn(
            args.tokens,
            config["model"].get("hidden_size", 4096),
            device="cuda:0",
            dtype=torch.float16,
        )

        print("BASELINE GPU MEMORY")
        alloc0, reserved0 = cuda_mem_mib()
        print(f"  allocated: {alloc0:,.2f} MiB")
        print(f"  reserved:  {reserved0:,.2f} MiB")
        print()

        # -------------------------
        # Expert A: fresh instance.
        # -------------------------
        t0 = time.perf_counter()
        expert_a = moe._ensure_expert_on_gpu(args.expert_a)
        transfer_a_ms = (time.perf_counter() - t0) * 1000.0
        cuda_sync()

        print("EXPERT A — FRESH INSTANCE")
        print(f"  Transfer + reconstruction: {transfer_a_ms:,.3f} ms")

        first_a_ms, first_a_alloc_delta, first_a_reserved_delta = run_forward(
            expert_a,
            x,
        )
        print(f"  FIRST forward:             {first_a_ms:,.3f} ms")
        print(f"  Allocated delta:           {first_a_alloc_delta:,.3f} MiB")
        print(f"  Reserved delta:            {first_a_reserved_delta:,.3f} MiB")
        print()

        # -------------------------
        # Expert B: distinct fresh object, identical shapes.
        # -------------------------
        t0 = time.perf_counter()
        expert_b = moe._ensure_expert_on_gpu(args.expert_b)
        transfer_b_ms = (time.perf_counter() - t0) * 1000.0
        cuda_sync()

        print("EXPERT B — DISTINCT FRESH INSTANCE")
        print(f"  Transfer + reconstruction: {transfer_b_ms:,.3f} ms")

        first_b_ms, first_b_alloc_delta, first_b_reserved_delta = run_forward(
            expert_b,
            x,
        )
        print(f"  FIRST forward:             {first_b_ms:,.3f} ms")
        print(f"  Allocated delta:           {first_b_alloc_delta:,.3f} MiB")
        print(f"  Reserved delta:            {first_b_reserved_delta:,.3f} MiB")
        print()

        # -------------------------
        # Expert A: same object.
        # -------------------------
        print("EXPERT A — SAME INSTANCE AGAIN")

        second_a_ms, second_a_alloc_delta, second_a_reserved_delta = run_forward(
            expert_a,
            x,
        )
        print(f"  SECOND forward:            {second_a_ms:,.3f} ms")
        print(f"  Allocated delta:           {second_a_alloc_delta:,.3f} MiB")
        print(f"  Reserved delta:            {second_a_reserved_delta:,.3f} MiB")
        print()

        # -------------------------
        # Summary / hypothesis test.
        # -------------------------
        print("SUMMARY")
        print(f"  A first forward:  {first_a_ms:,.3f} ms")
        print(f"  B first forward:  {first_b_ms:,.3f} ms")
        print(f"  A second forward: {second_a_ms:,.3f} ms")
        print()

        ratio_b_to_a = (
            first_b_ms / first_a_ms
            if first_a_ms > 0
            else float("inf")
        )
        ratio_a2_to_a1 = (
            second_a_ms / first_a_ms
            if first_a_ms > 0
            else float("inf")
        )

        print(f"  B-first / A-first:     {ratio_b_to_a:,.3f}x")
        print(f"  A-second / A-first:    {ratio_a2_to_a1:,.3f}x")
        print()

        print("INTERPRETATION")
        if first_a_ms > 500 and first_b_ms < first_a_ms * 0.25:
            print(
                "  Evidence favors GLOBAL/SHAPE-KEYED initialization: "
                "A is slow, but a new same-shape B is substantially faster."
            )
        elif first_a_ms > 500 and first_b_ms > first_a_ms * 0.5:
            print(
                "  Evidence favors PER-INSTANCE initialization/allocation: "
                "the new same-shape B is also slow."
            )
        else:
            print(
                "  No decisive pattern yet. Inspect A/B timings directly."
            )

        print()
        print("GPU MEMORY AFTER EXPERIMENT")
        alloc, reserved = cuda_mem_mib()
        print(f"  allocated: {alloc:,.2f} MiB")
        print(f"  reserved:  {reserved:,.2f} MiB")

        del x
        del expert_a
        del expert_b
        gc.collect()

    finally:
        engine.shutdown()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
