"""Shared pytest fixtures for MoEInfra tests.

Provides lightweight, dependency-injected fixtures that avoid loading real
model weights (which are ~24 GB) during the test suite.
"""
from __future__ import annotations

import pytest
import torch

from cache.manager import CacheManager
from cache.types import EvictionPolicy
from metrics.collector import MetricsCollector
from transfer.scheduler import TransferScheduler


@pytest.fixture
def config() -> dict:
    """Return a minimal configuration dict for testing.

    Returns:
        A dict matching the structure of ``config.yaml``.
    """
    return {
        "model": {
            "name": "mistralai/Mixtral-8x7B-v0.1",
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "num_layers": 32,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "quantization": "int4",
        },
        "cache": {
            "gpu_slots": 8,
            "cpu_slots": 32,
            "eviction_policy": "lru",
        },
        "transfer": {
            "bandwidth_gbps": 8.0,
            "prefetch_depth": 2,
            "max_concurrent_transfers": 4,
        },
        "logging": {
            "level": "DEBUG",
            "file": "/tmp/moeinfra_test.log",
        },
        "metrics": {
            "enabled": True,
            "export_interval_s": 10,
        },
    }


@pytest.fixture
def cache_manager() -> CacheManager:
    """Return a CacheManager configured for unit tests.

    Returns:
        A :class:`~cache.manager.CacheManager` with small slot counts.
    """
    return CacheManager(
        gpu_slots=4,
        cpu_slots=8,
        policy=EvictionPolicy.LRU,
    )


@pytest.fixture
def transfer_scheduler(cache_manager: CacheManager) -> TransferScheduler:
    """Return a TransferScheduler wired to the test cache manager.

    Args:
        cache_manager: Injected :class:`~cache.manager.CacheManager` fixture.

    Returns:
        A :class:`~transfer.scheduler.TransferScheduler` instance.
    """
    return TransferScheduler(
        cache_manager=cache_manager,
        bandwidth_gbps=8.0,
        max_concurrent=2,
    )


@pytest.fixture
def metrics_collector() -> MetricsCollector:
    """Return an enabled MetricsCollector for testing.

    Returns:
        A :class:`~metrics.collector.MetricsCollector` instance.
    """
    return MetricsCollector(enabled=True)


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Return a small float32 tensor for use as a mock expert weight.

    Returns:
        A :class:`torch.Tensor` of shape ``(64, 64)`` on CPU.
    """
    return torch.randn(64, 64)
