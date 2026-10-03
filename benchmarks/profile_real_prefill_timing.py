#!/usr/bin/env python3
"""Low-overhead timing of the REAL MoEInfra cold prefill.

No cProfile and no source edits. Runtime monkey-patching measures:
  - ModelLoader.load_expert()
  - ReusablePinnedStagingPool.stage_expert()
  - actual H2D CUDA-stream time
  - ReconstructedNF4Expert construction
  - ReconstructedNF4Expert.forward() CUDA time
  - complete MoELayer._forward_tokens() wall time

Usage:
  python -m benchmarks.profile_real_prefill_timing --gc-mode normal
  python -m benchmarks.profile_real_prefill_timing --gc-mode disabled
"""
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
from model.loader import ModelLoader
from model.moe_layer import MoELayer
from transfer import scheduler as scheduler_module
from transfer.nf4_reconstruct import ReconstructedNF4Expert
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
RESULT_DIR = Path("benchmarks/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _cuda_sync() -> None:
    torch.cuda.synchronize()


def _stats_dict(stats) -> dict:
    if hasattr(stats, "__dict__"):
        return dict(vars(stats))
    names = ("hits", "misses", "evictions", "gpu_slots_used", "cpu_slots_used")
    return {name: getattr(stats, name) for name in names if hasattr(stats, name)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--gc-mode", choices=("normal", "disabled"), default="normal")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    timing = {
        "load_ms": 0.0,
        "load_calls": 0,
        "stage_ms": 0.0,
        "stage_calls": 0,
        "h2d_ms": 0.0,
        "h2d_calls": 0,
        "reconstruct_ms": 0.0,
        "reconstruct_calls": 0,
        "expert_forward_events": [],
        "expert_forward_calls": 0,
    }
    transfers: list[dict] = []
    layer_totals: list[dict] = []
    pending_h2d: list[dict] = []

    original_load_expert = ModelLoader.load_expert
    original_stage_expert = ReusablePinnedStagingPool.stage_expert
    original_transfer = scheduler_module.transfer_staged_expert_to_gpu
    original_reconstruct = scheduler_module.ReconstructedNF4Expert
    original_layer_forward = MoELayer._forward_tokens
    original_expert_forward = ReconstructedNF4Expert.forward

    def timed_load_expert(self, layer_id, expert_id):
        t0 = time.perf_counter()
        out = original_load_expert(self, layer_id, expert_id)
        timing["load_ms"] += (time.perf_counter() - t0) * 1000.0
        timing["load_calls"] += 1
        return out

    def timed_stage_expert(self, slot, expert, layer_id, expert_id):
        t0 = time.perf_counter()
        out = original_stage_expert(self, slot, expert, layer_id, expert_id)
        elapsed = (time.perf_counter() - t0) * 1000.0
        timing["stage_ms"] += elapsed
        timing["stage_calls"] += 1
        transfers.append({
            "layer": layer_id,
            "expert": expert_id,
            "stage_ms": elapsed,
            "h2d_ms": None,
            "reconstruct_ms": None,
        })
        return out

    def timed_transfer(slot, stream):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start_event.record(stream)
            gpu_state = original_transfer(slot, stream)
            end_event.record(stream)
        pending_h2d.append({
            "key": slot.expert_key,
            "start": start_event,
            "end": end_event,
        })
        return gpu_state

    def timed_reconstruct(*args, **kwargs):
        t0 = time.perf_counter()
        out = original_reconstruct(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        timing["reconstruct_ms"] += elapsed
        timing["reconstruct_calls"] += 1

        if pending_h2d:
            pending = pending_h2d.pop(0)
            h2d_ms = pending["start"].elapsed_time(pending["end"])
            timing["h2d_ms"] += h2d_ms
            timing["h2d_calls"] += 1
            layer_id, expert_id = pending["key"]
            for row in reversed(transfers):
                if (
                    row["layer"] == layer_id
                    and row["expert"] == expert_id
                    and row["h2d_ms"] is None
                ):
                    row["h2d_ms"] = h2d_ms
                    row["reconstruct_ms"] = elapsed
                    break
        return out

    def timed_layer_forward(self, hidden_states):
        t0 = time.perf_counter()
        try:
            return original_layer_forward(self, hidden_states)
        finally:
            layer_totals.append({
                "layer": self.layer_id,
                "wall_ms": (time.perf_counter() - t0) * 1000.0,
            })

    def timed_expert_forward(self, x):
        stream = torch.cuda.current_stream()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start_event.record(stream)
            out = original_expert_forward(self, x)
            end_event.record(stream)
        timing["expert_forward_calls"] += 1
        timing["expert_forward_events"].append((start_event, end_event))
        return out

    ModelLoader.load_expert = timed_load_expert
    ReusablePinnedStagingPool.stage_expert = timed_stage_expert
    scheduler_module.transfer_staged_expert_to_gpu = timed_transfer
    scheduler_module.ReconstructedNF4Expert = timed_reconstruct
    MoELayer._forward_tokens = timed_layer_forward
    ReconstructedNF4Expert.forward = timed_expert_forward

    gc_previous = gc.isenabled()
    if args.gc_mode == "disabled":
        gc.disable()

    try:
        config = load_config(args.config)
        num_layers = min(int(config["model"].get("num_layers", 32)), 32)
        config["model"]["name"] = MODEL_PATH
        config["model"]["device"] = "cuda:0"
        config["model"]["dtype"] = "float16"
        config["model"]["num_layers"] = num_layers
        config["model"]["expert_store_paths"] = [STORE1, STORE2]

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        inputs = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].to("cuda:0")

        print("=" * 72)
        print("MoEInfra — REAL COLD PREFILL LOW-OVERHEAD TIMING")
        print("=" * 72)
        print(f"Layers:          {num_layers}")
        print(f"Prompt tokens:   {input_ids.shape[1]}")
        print(f"Max new tokens:  {args.max_new_tokens}")
        print(f"GC mode:         {args.gc_mode}")
        print()

        engine = InferenceEngine(config)
        _cuda_sync()
        init_t0 = time.perf_counter()
        engine._initialize_runtime()
        _cuda_sync()
        init_ms = (time.perf_counter() - init_t0) * 1000.0
        print(f"Runtime initialization: {init_ms:,.2f} ms")
        print("Running REAL cold prefill...")

        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()
        _cuda_sync()
        start = time.perf_counter()
        result = engine.generate(
            ForwardRequest(input_ids=input_ids, max_new_tokens=args.max_new_tokens)
        )
        _cuda_sync()
        wall_ms = (time.perf_counter() - start) * 1000.0

        expert_gpu_times = [
            s.elapsed_time(e) for s, e in timing["expert_forward_events"]
        ]
        cache_stats = _stats_dict(engine._cache_manager.stats())

        measured = (
            timing["load_ms"]
            + timing["stage_ms"]
            + timing["h2d_ms"]
            + timing["reconstruct_ms"]
            + sum(expert_gpu_times)
        )

        print("\n" + "=" * 72)
        print("RESULT")
        print("=" * 72)
        print(f"Real cold wall time:     {wall_ms:,.2f} ms")
        print(f"Reported TTFT/prefill:   {result.ttft_ms:,.2f} ms")
        print("\nTransfer pipeline:")
        print(f"  load_expert:            {timing['load_ms']:,.2f} ms ({timing['load_calls']} calls)")
        print(f"  stage_expert:           {timing['stage_ms']:,.2f} ms ({timing['stage_calls']} calls)")
        print(f"  H2D CUDA time:          {timing['h2d_ms']:,.2f} ms ({timing['h2d_calls']} calls)")
        print(f"  reconstruction:         {timing['reconstruct_ms']:,.2f} ms ({timing['reconstruct_calls']} calls)")
        print("\nExpert compute:")
        print(f"  expert forwards:        {timing['expert_forward_calls']}")
        if expert_gpu_times:
            print(f"  total expert GPU time:  {sum(expert_gpu_times):,.2f} ms")
            print(f"  median expert GPU time: {statistics.median(expert_gpu_times):,.2f} ms")
            print(f"  max expert GPU time:    {max(expert_gpu_times):,.2f} ms")
        print("\nUnattributed / other wall time:")
        print(f"  wall - measured parts:  {wall_ms - measured:,.2f} ms")
        print("\nGPU memory:")
        print(f"  peak allocated: {torch.cuda.max_memory_allocated() / 1024**2:,.2f} MiB")
        print(f"  peak reserved:  {torch.cuda.max_memory_reserved() / 1024**2:,.2f} MiB")
        print(f"\nCache stats: {cache_stats}")
        print("\nPER-LAYER WALL TIME")
        for row in layer_totals:
            print(f"  L{row['layer']:02d}: {row['wall_ms']:,.2f} ms")

        suffix = args.gc_mode
        transfer_csv = RESULT_DIR / f"real_prefill_timing_{suffix}.csv"
        with transfer_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["layer", "expert", "stage_ms", "h2d_ms", "reconstruct_ms"],
            )
            writer.writeheader()
            writer.writerows(transfers)

        layer_csv = RESULT_DIR / f"real_prefill_layers_{suffix}.csv"
        with layer_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["layer", "wall_ms"])
            writer.writeheader()
            writer.writerows(layer_totals)

        print("\nSaved:")
        print(f"  {transfer_csv}")
        print(f"  {layer_csv}")
    finally:
        ModelLoader.load_expert = original_load_expert
        ReusablePinnedStagingPool.stage_expert = original_stage_expert
        scheduler_module.transfer_staged_expert_to_gpu = original_transfer
        scheduler_module.ReconstructedNF4Expert = original_reconstruct
        MoELayer._forward_tokens = original_layer_forward
        ReconstructedNF4Expert.forward = original_expert_forward
        if gc_previous and not gc.isenabled():
            gc.enable()


if __name__ == "__main__":
    main()
