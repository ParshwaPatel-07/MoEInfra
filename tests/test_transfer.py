"""Tests for the transfer subsystem.

Covers :class:`~transfer.bandwidth.BandwidthMonitor`,
:class:`~transfer.scheduler.TransferScheduler`, and the type definitions.
"""
from __future__ import annotations

import time
import uuid

import pytest
import torch

from transfer.bandwidth import BandwidthMonitor
from transfer.scheduler import TransferScheduler
from transfer.types import (
    TransferDirection,
    TransferPriority,
    TransferRequest,
)


# ── BandwidthMonitor ──────────────────────────────────────────────────────

class TestBandwidthMonitor:
    """BandwidthMonitor tests."""

    def test_initial_current_gbps_zero(self) -> None:
        """current_gbps should be 0.0 with fewer than two samples."""
        mon = BandwidthMonitor(window_s=5.0)
        assert mon.current_gbps() == 0.0

    def test_record_and_peak(self) -> None:
        """Recording samples should raise peak_gbps above zero."""
        mon = BandwidthMonitor(window_s=5.0)
        # Simulate two transfers of 1 GB each
        mon.record(1 * 1024 ** 3)
        time.sleep(0.01)
        mon.record(1 * 1024 ** 3)
        assert mon.peak_gbps() > 0.0

    def test_reset_clears_peak(self) -> None:
        """reset() should zero peak_gbps and all samples."""
        mon = BandwidthMonitor(window_s=5.0)
        mon.record(1024)
        time.sleep(0.01)
        mon.record(1024)
        mon.reset()
        assert mon.peak_gbps() == 0.0
        assert mon.current_gbps() == 0.0


# ── TransferScheduler behaviour ───────────────────────────────────────────

class TestTransferSchedulerBehaviour:
    """Behavioural tests for the implemented TransferScheduler."""

    def _make_request(
        self,
        layer_id: int = 0,
        expert_id: int = 0,
        priority: TransferPriority = TransferPriority.NORMAL,
    ) -> TransferRequest:
        return TransferRequest(
            request_id=str(uuid.uuid4()),
            layer_id=layer_id,
            expert_id=expert_id,
            direction=TransferDirection.CPU_TO_GPU,
            priority=priority,
            issued_at=time.monotonic(),
        )

    def test_submit_increments_pending(
        self, transfer_scheduler: TransferScheduler
    ) -> None:
        """submit() should add one entry to the pending queue."""
        assert transfer_scheduler.pending_count() == 0
        req = self._make_request()
        req.tensor = torch.randn(4, 4)
        transfer_scheduler.submit(req)
        assert transfer_scheduler.pending_count() == 1

    def test_submit_dedup_same_triple(
        self, transfer_scheduler: TransferScheduler
    ) -> None:
        """Submitting duplicate (layer, expert, direction) should be ignored."""
        req1 = self._make_request(layer_id=0, expert_id=0)
        req1.tensor = torch.randn(4, 4)
        req2 = self._make_request(layer_id=0, expert_id=0)  # same triple
        req2.tensor = torch.randn(4, 4)
        transfer_scheduler.submit(req1)
        transfer_scheduler.submit(req2)
        assert transfer_scheduler.pending_count() == 1

    def test_execute_next_empty_returns_none(
        self, transfer_scheduler: TransferScheduler
    ) -> None:
        """execute_next() on empty queue should return None."""
        assert transfer_scheduler.execute_next() is None

    def test_cancel_removes_pending(
        self, transfer_scheduler: TransferScheduler
    ) -> None:
        """cancel() should remove the request and return True."""
        req = self._make_request()
        req.tensor = torch.randn(4, 4)
        transfer_scheduler.submit(req)
        assert transfer_scheduler.cancel(req.request_id) is True
        assert transfer_scheduler.pending_count() == 0

    def test_cancel_unknown_returns_false(
        self, transfer_scheduler: TransferScheduler
    ) -> None:
        """cancel() with unknown id should return False."""
        assert transfer_scheduler.cancel("nonexistent-id") is False

    def test_priority_ordering(
        self, transfer_scheduler: TransferScheduler
    ) -> None:
        """HIGH priority request should be executed before LOW priority."""
        low_req = self._make_request(expert_id=0, priority=TransferPriority.LOW)
        low_req.tensor = torch.randn(4, 4)
        high_req = self._make_request(expert_id=1, priority=TransferPriority.HIGH)
        high_req.tensor = torch.randn(4, 4)
        # Submit LOW first, then HIGH — HIGH should come out first
        transfer_scheduler.submit(low_req)
        transfer_scheduler.submit(high_req)
        result = transfer_scheduler.execute_next()
        assert result is not None
        assert result.request_id == high_req.request_id
