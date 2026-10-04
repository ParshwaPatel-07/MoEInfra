from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Dict, Iterable

import torch
from safetensors import safe_open

from model.expert import PrequantizedNF4Expert


SOURCE_STORES = (
    Path("/kaggle/input/datasets/parshwapatel07/mixtral-8x7b-nf4-experts/experts"),
    Path("/kaggle/input/datasets/parshwapatel07/mixtral-8x7b-nf4-experts-part2/experts"),
)

LOCAL_STORE = Path("/kaggle/working/experts_local")

RAM_LAYERS = range(0, 8)
DISK_LAYERS = range(8, 32)

EXPECTED_EXPERTS = 32 * 8


def expert_filename(layer_id: int, expert_id: int) -> str:
    return f"layer_{layer_id:02d}_expert_{expert_id:02d}.safetensors"


def find_source_file(layer_id: int, expert_id: int) -> Path:
    filename = expert_filename(layer_id, expert_id)

    matches = [
        store / filename
        for store in SOURCE_STORES
        if (store / filename).is_file()
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one source for {filename}, found: {matches}"
        )

    return matches[0]


def copy_disk_experts() -> None:
    """
    Copy layers 8-31 from /kaggle/input to local working storage.

    Existing files with the same size are skipped, making this resumable.
    """
    LOCAL_STORE.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    t0 = time.perf_counter()

    for layer_id in DISK_LAYERS:
        for expert_id in range(8):
            src = find_source_file(layer_id, expert_id)
            dst = LOCAL_STORE / src.name

            if dst.is_file() and dst.stat().st_size == src.stat().st_size:
                skipped += 1
                continue

            print(f"[COPY] L{layer_id:02d} E{expert_id:02d}")
            shutil.copy2(src, dst)
            copied += 1

    elapsed = time.perf_counter() - t0

    print(
        f"Local copy complete: copied={copied}, "
        f"skipped={skipped}, time={elapsed:.1f}s"
    )


def warm_local_disk_store() -> None:
    """
    Sequentially read every local expert file once.

    This intentionally happens during startup, never during inference.
    """
    files = sorted(LOCAL_STORE.glob("layer_*_expert_*.safetensors"))

    expected = len(tuple(DISK_LAYERS)) * 8
    if len(files) != expected:
        raise RuntimeError(
            f"Expected {expected} local expert files, found {len(files)}"
        )

    t0 = time.perf_counter()
    total_bytes = 0

    for path in files:
        with path.open("rb") as f:
            while True:
                chunk = f.read(8 * 1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)

    elapsed = time.perf_counter() - t0

    gb = total_bytes / (1024 ** 3)
    bw = gb / elapsed if elapsed else 0.0

    print(
        f"Local store warmed: {gb:.2f} GiB "
        f"in {elapsed:.2f}s ({bw:.2f} GiB/s)"
    )


def _extract(state: dict[str, torch.Tensor], prefix: str):
    return {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }


def load_ram_experts() -> Dict[tuple[int, int], PrequantizedNF4Expert]:
    """
    Load layers 0-7 completely into CPU RAM.

    The returned objects stay alive for the lifetime of the inference
    process, so these experts never need to be reread from /kaggle/input.
    """
    experts: Dict[tuple[int, int], PrequantizedNF4Expert] = {}

    t0 = time.perf_counter()
    total_bytes = 0

    for layer_id in RAM_LAYERS:
        for expert_id in range(8):
            path = find_source_file(layer_id, expert_id)

            with safe_open(path, framework="pt", device="cpu") as f:
                state = {
                    key: f.get_tensor(key)
                    for key in f.keys()
                }

            expert = PrequantizedNF4Expert.from_serialized(
                _extract(state, "w1."),
                _extract(state, "w2."),
                _extract(state, "w3."),
                hidden_size=4096,
                intermediate_size=14336,
            )

            experts[(layer_id, expert_id)] = expert
            total_bytes += expert.size_bytes

    elapsed = time.perf_counter() - t0

    print(
        f"CPU RAM experts loaded: {len(experts)} experts, "
        f"{total_bytes / (1024 ** 3):.2f} GiB, "
        f"time={elapsed:.2f}s"
    )

    return experts


def verify_store(
    ram_experts: Dict[tuple[int, int], PrequantizedNF4Expert],
) -> None:
    local_files = {
        path.name
        for path in LOCAL_STORE.glob("layer_*_expert_*.safetensors")
    }

    if len(ram_experts) != 64:
        raise RuntimeError(
            f"Expected 64 RAM experts, found {len(ram_experts)}"
        )

    if len(local_files) != 192:
        raise RuntimeError(
            f"Expected 192 local experts, found {len(local_files)}"
        )

    total = len(ram_experts) + len(local_files)

    if total != EXPECTED_EXPERTS:
        raise RuntimeError(
            f"Expected {EXPECTED_EXPERTS} total experts, found {total}"
        )

    for layer_id in RAM_LAYERS:
        for expert_id in range(8):
            if (layer_id, expert_id) not in ram_experts:
                raise RuntimeError(
                    f"Missing RAM expert L{layer_id} E{expert_id}"
                )

    for layer_id in DISK_LAYERS:
        for expert_id in range(8):
            filename = expert_filename(layer_id, expert_id)
            if filename not in local_files:
                raise RuntimeError(
                    f"Missing local expert L{layer_id} E{expert_id}"
                )

    print(
        "EXPERT STORE READY: "
        f"64 RAM + 192 local = {total} experts"
    )


def prepare_expert_store():
    """
    Complete startup preparation.

    Nothing returned by this function should be discarded: the RAM expert
    dictionary must stay alive for the whole inference process.
    """
    copy_disk_experts()
    warm_local_disk_store()
    ram_experts = load_ram_experts()
    verify_store(ram_experts)
    return ram_experts