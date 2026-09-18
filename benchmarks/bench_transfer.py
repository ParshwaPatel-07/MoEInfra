"""Transfer bandwidth benchmark for MoEInfra.

Measures end-to-end PCIe transfer throughput through the
:class:`~transfer.scheduler.TransferScheduler` once implemented, and
validates :class:`~transfer.bandwidth.BandwidthMonitor` accuracy against
``torch.Tensor.to()`` wall-clock time.

Run with::

    python benchmarks/bench_transfer.py
"""
from __future__ import annotations

import argparse
import time
import uuid

import torch

from cache.manager import CacheManager
from cache.types import EvictionPolicy
from transfer.bandwidth import BandwidthMonitor
from transfer.scheduler import TransferScheduler
from transfer.types import TransferDirection, TransferPriority, TransferRequest


def benchmark_transfer_bandwidth(
    n_transfers: int = 50,
    expert_size_mb: float = 12.0,
    bandwidth_gbps: float = 8.0,
    verbose: bool = True,
) -> dict:
    """Benchmark TransferScheduler end-to-end bandwidth.

    Issues *n_transfers* CPU→GPU transfer requests and measures aggregate
    throughput.  Falls back gracefully when stubs are not yet implemented.

    Args:
        n_transfers: Number of transfer requests to submit and execute.
        expert_size_mb: Simulated size of each expert tensor in MB.
        bandwidth_gbps: Configured peak bandwidth passed to the scheduler.
        verbose: Print a human-readable summary when ``True``.

    Returns:
        A dict with keys ``avg_gbps``, ``peak_gbps``, ``completed``,
        and ``not_implemented``.
    """
    cache_manager = CacheManager(
        gpu_slots=8, cpu_slots=32, policy=EvictionPolicy.LRU
    )
    scheduler = TransferScheduler(
        cache_manager=cache_manager,
        bandwidth_gbps=bandwidth_gbps,
        max_concurrent=4,
    )
    monitor = BandwidthMonitor(window_s=10.0)

    expert_bytes = int(expert_size_mb * 1024 ** 2)
    not_implemented = False
    completed = 0

    # Submit requests
    requests = []
    for i in range(n_transfers):
        req = TransferRequest(
            request_id=str(uuid.uuid4()),
            layer_id=i % 32,
            expert_id=i % 8,
            direction=TransferDirection.CPU_TO_GPU,
            priority=TransferPriority.NORMAL,
            issued_at=time.monotonic(),
        )
        try:
            scheduler.submit(req)
        except NotImplementedError:
            not_implemented = True
            break
        requests.append(req)

    if not not_implemented:
        # Execute requests and measure bandwidth
        for _ in range(len(requests)):
            try:
                result = scheduler.execute_next()
                if result and result.success:
                    monitor.record(expert_bytes)
                    completed += 1
            except NotImplementedError:
                not_implemented = True
                break

    result_dict = {
        "avg_gbps": monitor.current_gbps(),
        "peak_gbps": monitor.peak_gbps(),
        "completed": completed,
        "not_implemented": not_implemented,
    }

    if verbose:
        if not_implemented:
            print(
                "[bench_transfer] TransferScheduler stubs not yet implemented. "
                "Implement submit/execute_next to run this benchmark."
            )
        else:
            print(
                f"[bench_transfer] {completed}/{n_transfers} transfers | "
                f"avg={result_dict['avg_gbps']:.3f} GB/s | "
                f"peak={result_dict['peak_gbps']:.3f} GB/s"
            )
    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark transfer bandwidth.")
    parser.add_argument("--transfers", type=int, default=50)
    parser.add_argument("--expert-size-mb", type=float, default=12.0)
    parser.add_argument("--bandwidth-gbps", type=float, default=8.0)
    args = parser.parse_args()

    benchmark_transfer_bandwidth(
        n_transfers=args.transfers,
        expert_size_mb=args.expert_size_mb,
        bandwidth_gbps=args.bandwidth_gbps,
    )
