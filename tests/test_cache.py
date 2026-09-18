"""Tests for the cache subsystem.

Covers :class:`~cache.manager.CacheManager`,
:class:`~cache.types.CacheEntry`, and :class:`~cache.types.CacheStats`.
"""
from __future__ import annotations

import time

import pytest
import torch

from cache.manager import CacheManager
from cache.types import CacheEntry, CacheStats, EvictionPolicy


# ── Construction ──────────────────────────────────────────────────────────

class TestCacheManagerInit:
    """CacheManager initialisation tests."""

    def test_default_construction(self, cache_manager: CacheManager) -> None:
        """CacheManager should store slot counts and policy on init."""
        assert cache_manager.gpu_slots == 4
        assert cache_manager.cpu_slots == 8
        assert cache_manager.policy == EvictionPolicy.LRU

    def test_repr_contains_slots(self, cache_manager: CacheManager) -> None:
        """__repr__ should include slot info."""
        r = repr(cache_manager)
        assert "CacheManager" in r
        assert "gpu=" in r
        assert "cpu=" in r

    def test_stats_initially_zero(self, cache_manager: CacheManager) -> None:
        """All stats counters should be zero after init."""
        s = cache_manager.stats()
        assert s.hits == 0
        assert s.misses == 0
        assert s.evictions == 0
        assert s.hit_rate == 0.0


# ── CacheManager behaviour ────────────────────────────────────────────────

class TestCacheManagerBehaviour:
    """Behavioural tests for the implemented CacheManager methods."""

    def test_get_miss_returns_none(self, cache_manager: CacheManager) -> None:
        """get() on an empty cache should return None and increment misses."""
        result = cache_manager.get(layer_id=0, expert_id=0)
        assert result is None
        assert cache_manager.stats().misses == 1

    def test_put_then_get_hit(
        self, cache_manager: CacheManager, sample_tensor: torch.Tensor
    ) -> None:
        """put() followed by get() should return the stored tensor."""
        cache_manager.put(0, 0, sample_tensor, "cpu")
        result = cache_manager.get(0, 0)
        assert result is not None
        assert result.shape == sample_tensor.shape
        assert cache_manager.stats().hits == 1

    def test_put_updates_slot_count(
        self, cache_manager: CacheManager, sample_tensor: torch.Tensor
    ) -> None:
        """After put(), cpu_slots_used should reflect the new entry."""
        cache_manager.put(1, 2, sample_tensor, "cpu")
        assert cache_manager.stats().cpu_slots_used == 1

    def test_evict_lru_removes_entry(
        self, cache_manager: CacheManager, sample_tensor: torch.Tensor
    ) -> None:
        """evict() should remove one CPU entry and increment evictions."""
        cache_manager.put(0, 0, sample_tensor, "cpu")
        evicted = cache_manager.evict("cpu")
        assert evicted is not None
        assert evicted.expert_id == 0
        assert cache_manager.stats().evictions == 1

    def test_evict_empty_returns_none(self, cache_manager: CacheManager) -> None:
        """evict() on an empty tier should return None."""
        result = cache_manager.evict("cpu")
        assert result is None

    def test_cache_full_evicts_on_put(
        self, cache_manager: CacheManager
    ) -> None:
        """put() into a full CPU tier should auto-evict to make room."""
        # cpu_slots = 8 in the fixture
        for i in range(8):
            cache_manager.put(0, i, torch.randn(4, 4), "cpu")
        # One more insert should trigger eviction
        cache_manager.put(0, 99, torch.randn(4, 4), "cpu")
        assert cache_manager.stats().evictions == 1
        assert cache_manager.stats().cpu_slots_used == 8  # still full

    def test_promote_to_gpu_not_found(self, cache_manager: CacheManager) -> None:
        """promote_to_gpu() returns False when the expert is not in CPU cache."""
        result = cache_manager.promote_to_gpu(layer_id=0, expert_id=99)
        assert result is False


# ── Types ─────────────────────────────────────────────────────────────────

class TestCacheEntry:
    """CacheEntry dataclass tests."""

    def _make_entry(self, device: str = "cpu") -> CacheEntry:
        return CacheEntry(
            expert_id=0,
            layer_id=1,
            device=device,
            tensor=None,
            last_access=time.monotonic() - 5.0,
            access_count=3,
            size_bytes=1024,
        )

    def test_is_on_gpu_cpu(self) -> None:
        """is_on_gpu should return False for CPU entries."""
        entry = self._make_entry(device="cpu")
        assert not entry.is_on_gpu

    def test_is_on_gpu_cuda(self) -> None:
        """is_on_gpu should return True for CUDA entries."""
        entry = self._make_entry(device="cuda:0")
        assert entry.is_on_gpu

    def test_age_positive(self) -> None:
        """age should be >= 5 seconds for an entry last accessed 5s ago."""
        entry = self._make_entry()
        assert entry.age >= 5.0


class TestCacheStats:
    """CacheStats dataclass tests."""

    def test_hit_rate_no_requests(self) -> None:
        """hit_rate should be 0.0 when no requests have been made."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self) -> None:
        """hit_rate should equal hits / (hits + misses)."""
        stats = CacheStats(hits=3, misses=1)
        assert stats.hit_rate == pytest.approx(0.75)
