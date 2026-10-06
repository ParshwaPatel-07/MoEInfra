#!/usr/bin/env python3
"""Real prefill Tensor.copy_ diagnostic, robust to None NF4 state fields."""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from config_loader import load_config
from engine.engine import InferenceEngine
from engine.types import ForwardRequest
from transfer.reusable_staging import ReusablePinnedStagingPool

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

OUT = Path("benchmarks/results/stage_copy_diagnostics_v3.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--gc-mode",
        choices=("normal", "disabled"),
        default="normal",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    records: list[dict] = []
    none_calls = 0

    # Current implementation is a @staticmethod:
    # _copy_into_pinned(source, existing=None)
    original_copy = ReusablePinnedStagingPool._copy_into_pinned

    def timed_copy(source, existing=None):
        nonlocal none_calls

        if source is None:
            none_calls += 1
            return original_copy(source, existing)

        nbytes = source.numel() * source.element_size()
        t0 = time.perf_counter()
        result = original_copy(source, existing)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        records.append(
            {
                "bytes": nbytes,
                "mib": nbytes / (1024**2),
                "dtype": str(source.dtype),
                "shape": str(tuple(source.shape)),
                "allocated": existing is None,
                "ms": elapsed_ms,
            }
        )
        return result

    ReusablePinnedStagingPool._copy_into_pinned = staticmethod(timed_copy)

    gc_previous = gc.isenabled()
    if args.gc_mode == "disabled":
        gc.disable()

    try:
        config = load_config(args.config)
        config["model"]["name"] = MODEL_PATH
        config["model"]["device"] = "cuda:0"
        config["model"]["dtype"] = "float16"
        config["model"]["num_layers"] = min(
            int(config["model"].get("num_layers", 32)),
            32,
        )
        config["model"]["expert_store_paths"] = [STORE1, STORE2]

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
        input_ids = tokenizer(
            PROMPT,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"].to("cuda:0")

        print("=" * 72)
        print("MoEInfra — REAL PREFILL COPY DIAGNOSTIC v3")
        print("=" * 72)
        print(f"GC mode:       {args.gc_mode}")
        print(f"Layers:        {config['model']['num_layers']}")
        print(f"Prompt tokens: {input_ids.shape[1]}")
        print()

        engine = InferenceEngine(config)

        torch.cuda.synchronize()
        init_t0 = time.perf_counter()
        engine._initialize_runtime()
        torch.cuda.synchronize()
        init_ms = (time.perf_counter() - init_t0) * 1000.0
        print(f"Runtime initialization: {init_ms:,.2f} ms")

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        engine.generate(
            ForwardRequest(
                input_ids=input_ids,
                max_new_tokens=1,
            )
        )

        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0

        total_copy_ms = sum(r["ms"] for r in records)
        total_bytes = sum(r["bytes"] for r in records)

        alloc = [r["ms"] for r in records if r["allocated"]]
        reuse = [r["ms"] for r in records if not r["allocated"]]

        print()
        print("=" * 72)
        print("RESULT")
        print("=" * 72)
        print(f"Real cold wall time: {wall_ms:,.2f} ms")
        print(f"copy_ calls measured: {len(records)}")
        print(f"None-state calls:      {none_calls}")
        print(f"Total copy time:       {total_copy_ms:,.2f} ms")
        print(f"Copied data:           {total_bytes / 1024**3:,.2f} GiB")

        bw = (
            (total_bytes / 1024**3) / (total_copy_ms / 1000.0)
            if total_copy_ms > 0
            else 0.0
        )
        print(f"Effective copy BW:     {bw:,.2f} GiB/s")

        if alloc:
            print(
                f"Allocation copies:     {len(alloc)} | "
                f"median={statistics.median(alloc):,.3f} ms | "
                f"max={max(alloc):,.3f} ms"
            )

        if reuse:
            print(
                f"Reuse copies:          {len(reuse)} | "
                f"median={statistics.median(reuse):,.3f} ms | "
                f"max={max(reuse):,.3f} ms"
            )

        by_dtype: dict[str, list[float]] = {}
        by_size: dict[str, list[float]] = {}

        for r in records:
            by_dtype.setdefault(r["dtype"], []).append(r["ms"])
            size_key = f"{r['mib']:.3f} MiB"
            by_size.setdefault(size_key, []).append(r["ms"])

        print()
        print("BY DTYPE")
        for dtype, vals in by_dtype.items():
            print(
                f"  {dtype}: n={len(vals)} "
                f"total={sum(vals):,.2f} ms "
                f"median={statistics.median(vals):,.3f} ms "
                f"max={max(vals):,.3f} ms"
            )

        print()
        print("SLOWEST 20 COPY OPERATIONS")
        for idx, r in enumerate(
            sorted(records, key=lambda x: x["ms"], reverse=True)[:20],
            start=1,
        ):
            print(
                f"  {idx:02d}: {r['ms']:,.3f} ms | "
                f"{r['mib']:.3f} MiB | "
                f"{r['dtype']} | "
                f"allocated={r['allocated']} | "
                f"shape={r['shape']}"
            )

        with OUT.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "bytes",
                    "mib",
                    "dtype",
                    "shape",
                    "allocated",
                    "ms",
                ],
            )
            writer.writeheader()
            writer.writerows(records)

        print(f"\nDetailed records saved to: {OUT}")

    finally:
        ReusablePinnedStagingPool._copy_into_pinned = original_copy
        if gc_previous and not gc.isenabled():
            gc.enable()


if __name__ == "__main__":
    main()
