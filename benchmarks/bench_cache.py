"""Cache throughput benchmark for MoEInfra.

Measures how many ``get`` + ``put`` operations per second the
:class:`~cache.manager.CacheManager` can sustain once implemented.

Run with::

    python benchmarks/bench_cache.py
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

import torch

from cache.manager import CacheManager
from cache.types import EvictionPolicy


def benchmark_cache_throughput(
    gpu_slots: int = 8,
    cpu_slots: int = 32,
    n_iterations: int = 1_000,
    tensor_shape: tuple = (128, 128),
    policy: EvictionPolicy = EvictionPolicy.LRU,
    verbose: bool = True,
) -> dict:
    """Benchmark CacheManager get/put throughput.

    Attempts *n_iterations* round-trip get → put cycles and measures
    aggregate operations-per-second.  Because the core methods raise
    :exc:`NotImplementedError` in the current skeleton, this benchmark
    catches those errors and reports ``0`` throughput until the stubs
    are implemented.

    Args:
        gpu_slots: Number of GPU cache slots to configure.
        cpu_slots: Number of CPU cache slots to configure.
        n_iterations: Number of get/put cycles to execute.
        tensor_shape: Shape of the mock expert tensor.
        policy: Eviction policy to benchmark.
        verbose: Print a human-readable summary when ``True``.

    Returns:
        A dict with keys ``ops_per_sec``, ``iterations``, and
        ``not_implemented``.
    """
    manager = CacheManager(
        gpu_slots=gpu_slots,
        cpu_slots=cpu_slots,
        policy=policy,
    )
    tensor = torch.randn(*tensor_shape)
    not_implemented = False
    completed = 0

    t_start = time.perf_counter()
    for i in range(n_iterations):
        layer_id = i % 32
        expert_id = i % 8
        try:
            manager.get(layer_id, expert_id)
            manager.put(layer_id, expert_id, tensor, "cpu")
            completed += 1
        except NotImplementedError:
            not_implemented = True
            break
        except Exception as exc:
            if verbose:
                print(f"  Unexpected error at iteration {i}: {exc}")
            break
    elapsed = time.perf_counter() - t_start

    ops_per_sec = completed / elapsed if elapsed > 0 else 0.0
    result = {
        "ops_per_sec": ops_per_sec,
        "iterations": completed,
        "not_implemented": not_implemented,
    }

    if verbose:
        if not_implemented:
            print(
                f"[bench_cache] CacheManager stubs not yet implemented. "
                f"Implement get/put to run this benchmark."
            )
        else:
            print(
                f"[bench_cache] {completed}/{n_iterations} iterations | "
                f"{ops_per_sec:,.0f} ops/s | elapsed={elapsed:.3f}s"
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CacheManager throughput.")
    parser.add_argument("--gpu-slots", type=int, default=8)
    parser.add_argument("--cpu-slots", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=1_000)
    args = parser.parse_args()

    benchmark_cache_throughput(
        gpu_slots=args.gpu_slots,
        cpu_slots=args.cpu_slots,
        n_iterations=args.iterations,
    )
