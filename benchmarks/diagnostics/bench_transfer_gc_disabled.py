#!/usr/bin/env python3
"""Long transfer/reconstruction stress test for progressive degradation.

This deliberately reproduces repeated real expert turnover:

    CPU NF4 expert
      -> pinned staging
      -> H2D
      -> GPU NF4 reconstruction
      -> GPU cache insert
      -> demote/remove GPU residency
      -> release temporary references

It records each cycle individually instead of averaging away spikes.

Two passes are run:
    1. normal Python lifecycle
    2. explicit gc.collect() after every cycle

The experiment is intended to distinguish:
    - progressive CUDA allocator fragmentation
    - progressive Python-object/GC pressure
    - flat transfer behavior with isolated environmental spikes

Usage:
    python -m benchmarks.bench_transfer_degradation

Optional:
    python -m benchmarks.bench_transfer_degradation --cycles 40
    python -m benchmarks.bench_transfer_degradation --cycles 60 --gc-each
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import time
from pathlib import Path

import torch

from config_loader import load_config
from engine.engine import InferenceEngine
from transfer.nf4_reconstruct import ReconstructedNF4Expert
from transfer.nf4_transfer import transfer_staged_expert_to_gpu


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


def mem_snapshot(engine: InferenceEngine) -> dict[str, float]:
    stats = torch.cuda.memory_stats("cuda:0")
    cache_stats = engine._cache_manager.stats()

    gpu_cache = getattr(engine._cache_manager, "_gpu_cache", {})
    gpu_expert_mib = sum(
        getattr(entry, "size_bytes", 0)
        for entry in gpu_cache.values()
    ) / (1024**2)

    allocated = torch.cuda.memory_allocated("cuda:0") / (1024**2)
    reserved = torch.cuda.memory_reserved("cuda:0") / (1024**2)
    free, _ = torch.cuda.mem_get_info("cuda:0")

    return {
        "allocated_mib": allocated,
        "reserved_mib": reserved,
        "free_mib": free / (1024**2),
        "inactive_split_mib": (
            stats.get("inactive_split_bytes.all.current", 0)
            / (1024**2)
        ),
        "gpu_expert_mib": gpu_expert_mib,
        "gpu_slots": cache_stats.gpu_slots_used,
    }


def build_config() -> dict:
    config = load_config("config.yaml")
    config["model"]["name"] = MODEL_PATH
    config["model"]["device"] = "cuda:0"
    config["model"]["dtype"] = "float16"
    config["model"]["expert_store_paths"] = [STORE1, STORE2]
    config["model"]["num_layers"] = min(
        int(config["model"].get("num_layers", 32)),
        32,
    )
    return config


def benchmark_pass(
    engine: InferenceEngine,
    pairs: list[tuple[int, int]],
    *,
    explicit_gc: bool,
    pass_name: str,
    csv_path: Path,
) -> list[dict]:
    rows: list[dict] = []
    cache = engine._cache_manager
    scheduler = engine._transfer_scheduler
    staging_pool = engine._staging_pool

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "pass",
            "cycle",
            "layer",
            "expert",
            "load_ms",
            "stage_ms",
            "h2d_ms",
            "reconstruct_ms",
            "cache_put_ms",
            "demote_ms",
            "total_ms",
            "allocated_mib",
            "reserved_mib",
            "free_mib",
            "inactive_split_mib",
            "gpu_expert_mib",
            "gpu_slots",
            "gc_count0",
            "gc_count1",
            "gc_count2",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cycle, (layer_id, expert_id) in enumerate(pairs, start=1):
            # Keep the GPU cache empty at the start of every cycle.
            if cache.is_gpu_resident(layer_id, expert_id):
                cache.demote_to_cpu(layer_id, expert_id)

            cuda_sync()
            total_t0 = time.perf_counter()

            # 1) CPU expert acquisition.
            load_t0 = time.perf_counter()
            cpu_expert = cache.get(layer_id, expert_id)
            if cpu_expert is None:
                cpu_expert = engine._model.load_expert(
                    layer_id,
                    expert_id,
                )
                cache.put(
                    layer_id,
                    expert_id,
                    cpu_expert,
                    device="cpu",
                )
            load_ms = (time.perf_counter() - load_t0) * 1000.0

            # 2) Stage into reusable pinned CPU buffers.
            slot = staging_pool.acquire()
            if slot is None:
                raise RuntimeError(
                    "No staging slot available during stress test."
                )

            stage_t0 = time.perf_counter()
            staging_pool.stage_expert(
                slot,
                cpu_expert,
                layer_id,
                expert_id,
            )
            stage_ms = (time.perf_counter() - stage_t0) * 1000.0

            # 3) H2D copy.
            h2d_t0 = time.perf_counter()
            gpu_state = transfer_staged_expert_to_gpu(
                slot,
                scheduler._transfer_stream,
            )
            scheduler._transfer_stream.synchronize()
            h2d_ms = (time.perf_counter() - h2d_t0) * 1000.0

            # 4) GPU reconstruction.
            reconstruct_t0 = time.perf_counter()
            gpu_expert = ReconstructedNF4Expert(
                cpu_expert,
                gpu_state,
            )
            cuda_sync()
            reconstruct_ms = (
                time.perf_counter() - reconstruct_t0
            ) * 1000.0

            # 5) Publish to the actual GPU cache.
            put_t0 = time.perf_counter()
            cache.put(
                layer_id,
                expert_id,
                gpu_expert,
                device="cuda:0",
            )
            cache_put_ms = (time.perf_counter() - put_t0) * 1000.0

            # 6) Immediately evict/demote this GPU copy, matching the intended
            # turnover pattern being stress-tested.
            demote_t0 = time.perf_counter()
            cache.demote_to_cpu(layer_id, expert_id)
            demote_ms = (time.perf_counter() - demote_t0) * 1000.0

            # Make every temporary GPU-containing object unreachable before
            # observing the next cycle.
            del gpu_expert
            del gpu_state

            staging_pool.mark_transfer_complete(
                slot,
                scheduler._transfer_stream,
            )
            staging_pool.release(slot)

            del slot
            del cpu_expert

            if explicit_gc:
                gc.collect()

            cuda_sync()

            total_ms = (time.perf_counter() - total_t0) * 1000.0
            mem = mem_snapshot(engine)
            gc0, gc1, gc2 = gc.get_count()

            row = {
                "pass": pass_name,
                "cycle": cycle,
                "layer": layer_id,
                "expert": expert_id,
                "load_ms": load_ms,
                "stage_ms": stage_ms,
                "h2d_ms": h2d_ms,
                "reconstruct_ms": reconstruct_ms,
                "cache_put_ms": cache_put_ms,
                "demote_ms": demote_ms,
                "total_ms": total_ms,
                **mem,
                "gc_count0": gc0,
                "gc_count1": gc1,
                "gc_count2": gc2,
            }
            rows.append(row)
            writer.writerow(row)
            f.flush()

            print(
                f"  {cycle:02d}/{len(pairs)} "
                f"L{layer_id:02d}E{expert_id}: "
                f"total={total_ms:8.2f} ms | "
                f"load={load_ms:6.2f} | "
                f"stage={stage_ms:7.2f} | "
                f"h2d={h2d_ms:7.2f} | "
                f"recon={reconstruct_ms:7.2f} | "
                f"alloc={mem['allocated_mib']:8.1f} MiB | "
                f"reserved={mem['reserved_mib']:8.1f} MiB | "
                f"inactive={mem['inactive_split_mib']:7.1f} MiB"
            )

    return rows


def summarize(rows: list[dict], name: str) -> None:
    totals = [r["total_ms"] for r in rows]
    h2d = [r["h2d_ms"] for r in rows]
    recon = [r["reconstruct_ms"] for r in rows]
    alloc = [r["allocated_mib"] for r in rows]
    reserved = [r["reserved_mib"] for r in rows]
    inactive = [r["inactive_split_mib"] for r in rows]

    print()
    print(f"{name} SUMMARY")
    print(f"  total median:       {statistics.median(totals):,.2f} ms")
    print(f"  total max:          {max(totals):,.2f} ms")
    print(f"  H2D median:         {statistics.median(h2d):,.2f} ms")
    print(f"  H2D max:            {max(h2d):,.2f} ms")
    print(f"  reconstruct median: {statistics.median(recon):,.2f} ms")
    print(f"  reconstruct max:    {max(recon):,.2f} ms")
    print(f"  allocated first:    {alloc[0]:,.2f} MiB")
    print(f"  allocated last:     {alloc[-1]:,.2f} MiB")
    print(f"  allocated max:      {max(alloc):,.2f} MiB")
    print(f"  reserved first:     {reserved[0]:,.2f} MiB")
    print(f"  reserved last:      {reserved[-1]:,.2f} MiB")
    print(f"  reserved max:       {max(reserved):,.2f} MiB")
    print(f"  inactive max:       {max(inactive):,.2f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cycles",
        type=int,
        default=40,
        help="Number of distinct expert turnover cycles.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Directory for CSV output.",
    )
    parser.add_argument(
        "--gc-each",
        action="store_true",
        help="Run only the explicit-GC pass.",
    )
    parser.add_argument(
        "--gc-disabled",
        action="store_true",
        help="Run only the pass with automatic cyclic GC disabled.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if args.cycles < 2:
        raise ValueError("--cycles must be >= 2")
    if args.cycles > 256:
        raise ValueError("--cycles cannot exceed 256 unique experts")

    # Deterministic unique layer/expert pairs. This walks complete layer
    # expert sets first, which resembles real prefill turnover.
    pairs = [
        (idx // 8, idx % 8)
        for idx in range(args.cycles)
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MoEInfra — TRANSFER DEGRADATION STRESS TEST")
    print("=" * 72)
    print(f"Cycles:            {args.cycles}")
    print("Pattern:           load -> stage -> H2D -> reconstruct -> demote")
    print("GC comparison:     normal vs explicit gc.collect()")
    print()

    def new_engine() -> InferenceEngine:
        engine = InferenceEngine(build_config())
        cuda_sync()
        t0 = time.perf_counter()
        engine._initialize_runtime()
        cuda_sync()
        init_ms = (time.perf_counter() - t0) * 1000.0
        print(f"Runtime initialization: {init_ms:,.2f} ms")
        return engine

    # Fresh engine for each pass so the passes don't contaminate one another.
    if not args.gc_each and not args.gc_disabled:
        engine = new_engine()
        try:
            normal_rows = benchmark_pass(
                engine,
                pairs,
                explicit_gc=False,
                pass_name="normal_gc",
                csv_path=output_dir / "transfer_degradation_normal.csv",
            )
            summarize(normal_rows, "NORMAL-GC")
        finally:
            engine.shutdown()
            del engine
            gc.collect()
            torch.cuda.empty_cache()

    if not args.gc_each and args.gc_disabled:
        print()
        print("=" * 72)
        print("AUTO-GC-DISABLED PASS")
        print("=" * 72)
        gc.collect()
        was_enabled = gc.isenabled()
        gc.disable()
        print(f"Automatic GC enabled before pass: {was_enabled}")
        try:
            engine = new_engine()
            try:
                disabled_rows = benchmark_pass(
                    engine,
                    pairs,
                    explicit_gc=False,
                    pass_name="gc_disabled",
                    csv_path=output_dir / "transfer_degradation_gc_disabled.csv",
                )
                summarize(disabled_rows, "AUTO-GC-DISABLED")
            finally:
                engine.shutdown()
                del engine
                torch.cuda.empty_cache()
        finally:
            if was_enabled:
                gc.enable()
            gc.collect()

    if not args.gc_disabled:
        print()
        print("=" * 72)
        print("EXPLICIT-GC PASS")
        print("=" * 72)

        engine = new_engine()
        try:
            gc_rows = benchmark_pass(
                engine,
                pairs,
                explicit_gc=True,
                pass_name="explicit_gc",
                csv_path=output_dir / "transfer_degradation_explicit_gc.csv",
            )
            summarize(gc_rows, "EXPLICIT-GC")
        finally:
            engine.shutdown()
            del engine
            gc.collect()
            torch.cuda.empty_cache()

    print()
    print("CSV RESULTS")
    if not args.gc_each and not args.gc_disabled:
        print(f"  {output_dir / 'transfer_degradation_normal.csv'}")
    if args.gc_disabled:
        print(f"  {output_dir / 'transfer_degradation_gc_disabled.csv'}")
    else:
        print(f"  {output_dir / 'transfer_degradation_explicit_gc.csv'}")
    print()
    print("Interpretation:")
    print(
        "  Rising latency + rising inactive_split/reserved memory suggests "
        "allocator fragmentation."
    )
    print(
        "  Rising latency that disappears with explicit GC suggests Python "
        "object/GC pressure."
    )
    print(
        "  Flat latency with isolated spikes suggests external/environmental "
        "noise rather than progressive degradation."
    )


if __name__ == "__main__":
    main()
