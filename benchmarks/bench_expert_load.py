#!/usr/bin/env python3
"""Microbenchmark the CPU-side prequantized NF4 expert load path.

This isolates:
    1. safetensors file read / tensor materialization
    2. PrequantizedNF4Expert.from_serialized() construction

It intentionally does NOT transfer anything to GPU and does NOT run inference.

Usage on Kaggle:
    python bench_expert_load.py

Optional:
    python bench_expert_load.py --layer 0 --expert 0
    python bench_expert_load.py --layer 0 --expert 0 --repeat 3
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

from safetensors import safe_open

from model.expert import PrequantizedNF4Expert


MODEL_PATH = Path(
    "/kaggle/input/models/mistral-ai/mixtral/"
    "pytorch/8x7b-instruct-v0.1-hf/1"
)

STORE_PATHS = [
    Path(
        "/kaggle/input/datasets/parshwapatel07/"
        "mixtral-8x7b-nf4-experts/experts"
    ),
    Path(
        "/kaggle/input/datasets/parshwapatel07/"
        "mixtral-8x7b-nf4-experts-part2/experts"
    ),
]

HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 14336


def resolve_expert(layer_id: int, expert_id: int) -> Path:
    filename = f"layer_{layer_id:02d}_expert_{expert_id:02d}.safetensors"

    for store in STORE_PATHS:
        path = store / filename
        if path.is_file():
            return path

    raise FileNotFoundError(filename)


def load_state(path: Path) -> tuple[dict, float]:
    t0 = time.perf_counter()

    with safe_open(path, framework="pt", device="cpu") as f:
        state = {key: f.get_tensor(key) for key in f.keys()}

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return state, elapsed_ms


def construct_expert(state: dict) -> tuple[PrequantizedNF4Expert, float]:
    def extract(prefix: str) -> dict:
        return {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }

    t0 = time.perf_counter()

    expert = PrequantizedNF4Expert.from_serialized(
        extract("w1."),
        extract("w2."),
        extract("w3."),
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return expert, elapsed_ms


def run_once(layer_id: int, expert_id: int) -> None:
    path = resolve_expert(layer_id, expert_id)
    file_mib = path.stat().st_size / (1024**2)

    state, read_ms = load_state(path)
    expert, construct_ms = construct_expert(state)

    throughput = (
        file_mib / (read_ms / 1000.0)
        if read_ms > 0
        else 0.0
    )

    print(
        f"L{layer_id:02d} E{expert_id:02d} | "
        f"file={file_mib:,.2f} MiB | "
        f"read={read_ms:,.2f} ms | "
        f"construct={construct_ms:,.3f} ms | "
        f"effective read BW={throughput:,.1f} MiB/s"
    )

    del expert
    del state
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    print("=" * 72)
    print("MoEInfra — EXPERT LOAD MICROBENCHMARK")
    print("=" * 72)
    print(f"Layer:             {args.layer}")
    print(f"Expert:            {args.expert}")
    print(f"Repeats:           {args.repeat}")
    print()

    # Repeat the same expert to expose cold filesystem/page-cache effects.
    for i in range(args.repeat):
        run_once(args.layer, args.expert)

    print()
    print("Try a different expert after this:")
    print(
        "python bench_expert_load.py "
        f"--layer {args.layer} --expert {(args.expert + 1) % 8} --repeat 1"
    )


if __name__ == "__main__":
    main()
