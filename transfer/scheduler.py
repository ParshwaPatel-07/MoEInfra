"""Expert-transfer scheduler for MoEInfra.

Manages a priority queue of :class:`~transfer.types.TransferRequest` objects
and executes them synchronously against the two-level cache manager.

Design choices
--------------
* **heapq over a list**: Python's ``heapq`` gives O(log n) insert and
  O(log n) pop-min.  Requests are ordered by ``(priority.value, issued_at)``
  so HIGH=0 always precedes NORMAL=1 and LOW=2, with FIFO tie-breaking via
  timestamp.
* **Deduplication on submit**: before inserting we check ``_queue`` and
  ``_in_flight`` for the same ``(layer_id, expert_id, direction)`` triple or
  matching ``request_id``.  Duplicates are silently dropped to avoid redundant
  PCIe moves.
* **Synchronous execution** (``execute_next``): the spec says "synchronous
  version first, then async".  We call ``tensor.to(device,
  non_blocking=False)`` so the method blocks until the DMA finishes, then
  calls ``cache_manager.put``.  This keeps the state machine simple and
  correct before introducing CUDA streams.
* **``drain`` delegates to ``execute_next``**: avoids code duplication and
  automatically inherits any future optimisation to ``execute_next``.
* **``cancel`` rebuilds heap**: the queue is small (prefetch_depth *
  num_experts_per_tok ≤ ~16 entries), so an O(n) rebuild is acceptable.
* **BandwidthMonitor**: every completed transfer calls
  ``self._bw_monitor.record`` so callers can query rolling throughput.
"""
from __future__ import annotations

import heapq
import logging
import time
from typing import List, Optional

import torch

from transfer.bandwidth import BandwidthMonitor
from transfer.types import TransferDirection, TransferRequest, TransferResult


class TransferScheduler:
    """Priority-queue-based PCIe transfer scheduler.

    Accepts :class:`~transfer.types.TransferRequest` submissions, orders them
    by ``(priority.value, issued_at)``, and executes them synchronously one at
    a time against the :class:`~cache.manager.CacheManager`.

    Attributes:
        bandwidth_gbps: Configured peak PCIe bandwidth in GB/s.
        max_concurrent: Reserved for the future async implementation.
    """

    def __init__(
        self,
        cache_manager,
        bandwidth_gbps: float,
        max_concurrent: int,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialise the scheduler.

        Args:
            cache_manager: A :class:`~cache.manager.CacheManager` instance the
                scheduler will call to store transferred tensors.
            bandwidth_gbps: Expected peak PCIe bandwidth in GB/s.
            max_concurrent: Reserved for a future async implementation.
            logger: Optional pre-configured logger.
        """
        self._cache_manager = cache_manager
        self.bandwidth_gbps: float = bandwidth_gbps
        self.max_concurrent: int = max_concurrent
        self._logger: logging.Logger = logger or logging.getLogger(__name__)
        self._bw_monitor: BandwidthMonitor = BandwidthMonitor(window_s=5.0)

        # Min-heap entries: (sort_key_tuple, TransferRequest)
        # sort_key = (priority.value, issued_at) — lower = higher priority
        self._heap: list = []
        # Flat list kept in sync for O(1) membership queries + cancel
        self._queue: list[TransferRequest] = []
        self._in_flight: list[TransferRequest] = []

        self._logger.debug(
            "TransferScheduler initialised: bandwidth=%.1f GB/s, max_concurrent=%d",
            bandwidth_gbps,
            max_concurrent,
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _is_duplicate(self, request: TransferRequest) -> bool:
        """Return True if an equivalent request is already queued or in-flight.

        Equivalence: same ``request_id`` OR same
        ``(layer_id, expert_id, direction)`` triple.

        Args:
            request: The candidate request.

        Returns:
            ``True`` if a duplicate exists.
        """
        triple = (request.layer_id, request.expert_id, request.direction)
        for existing in (*self._queue, *self._in_flight):
            if existing.request_id == request.request_id:
                return True
            if (existing.layer_id, existing.expert_id, existing.direction) == triple:
                return True
        return False

    @staticmethod
    def _sort_key(request: TransferRequest) -> tuple:
        """Heap sort key: ``(priority.value, issued_at)``."""
        return (request.priority.value, request.issued_at)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def submit(self, request: TransferRequest) -> None:
        """Enqueue a transfer request.

        Inserts *request* into the min-heap ordered by
        ``(priority.value, issued_at)``.  Silently drops duplicates (same
        ``request_id`` or same ``(layer_id, expert_id, direction)`` triple).

        Args:
            request: The :class:`~transfer.types.TransferRequest` to enqueue.
        """
        if self._is_duplicate(request):
            self._logger.debug(
                "submit: duplicate dropped  id=%s  layer=%d expert=%d dir=%s",
                request.request_id,
                request.layer_id,
                request.expert_id,
                request.direction.value,
            )
            return

        heapq.heappush(self._heap, (self._sort_key(request), request))
        self._queue.append(request)

        self._logger.debug(
            "submit: queued  id=%s  layer=%d expert=%d dir=%s priority=%s",
            request.request_id,
            request.layer_id,
            request.expert_id,
            request.direction.value,
            request.priority.name,
        )

    def execute_next(self) -> Optional[TransferResult]:
        """Dequeue and execute the highest-priority pending request.

        Pops the lowest sort-key entry from the heap, moves the tensor via
        ``tensor.to(device, non_blocking=False)`` (blocking until DMA is
        done), calls ``cache_manager.put`` to register the result, and records
        the bandwidth sample.

        Returns:
            A :class:`~transfer.types.TransferResult` on success or failure,
            or ``None`` if the queue is empty.
        """
        if not self._heap:
            return None

        _, request = heapq.heappop(self._heap)
        self._queue.remove(request)
        self._in_flight.append(request)

        self._logger.debug(
            "execute_next: starting  id=%s  layer=%d expert=%d dir=%s",
            request.request_id, request.layer_id,
            request.expert_id, request.direction.value,
        )

        t_start = time.perf_counter()
        error_msg: Optional[str] = None
        bytes_moved: int = 0

        try:
            target_device = (
                "cuda:0"
                if request.direction == TransferDirection.CPU_TO_GPU
                else "cpu"
            )

            tensor = request.tensor
            if tensor is None:
                # Fetch from the cache (may be None if not resident)
                tensor = self._cache_manager.get(request.layer_id, request.expert_id)
            if tensor is None:
                raise RuntimeError(
                    f"No tensor for layer={request.layer_id} expert={request.expert_id}"
                )

            bytes_moved = tensor.nelement() * tensor.element_size()
            moved = tensor.to(target_device, non_blocking=False)

            self._cache_manager.put(
                request.layer_id, request.expert_id, moved, target_device
            )
        except Exception as exc:  # pylint: disable=broad-except
            error_msg = str(exc)
            self._logger.error(
                "execute_next: FAILED  id=%s  error=%s",
                request.request_id, error_msg,
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self._in_flight.remove(request)

        if error_msg is None:
            self._bw_monitor.record(bytes_moved)
            self._logger.debug(
                "execute_next: done  id=%s  bytes=%d  elapsed=%.2f ms",
                request.request_id, bytes_moved, elapsed_ms,
            )

        return TransferResult(
            request_id=request.request_id,
            success=error_msg is None,
            elapsed_ms=elapsed_ms,
            bytes_transferred=bytes_moved,
            error=error_msg,
        )

    def drain(self) -> List[TransferResult]:
        """Execute all pending requests and return their results.

        Delegates to :meth:`execute_next` so future optimisations propagate
        automatically.

        Returns:
            List of :class:`~transfer.types.TransferResult` in execution order
            (highest priority first).
        """
        results: List[TransferResult] = []
        while self._heap:
            result = self.execute_next()
            if result is not None:
                results.append(result)
        return results

    def cancel(self, request_id: str) -> bool:
        """Remove a pending request from the queue without executing it.

        Has no effect on in-flight or completed requests.  Rebuilds the heap
        after removal to restore the heap invariant.

        Args:
            request_id: The ``request_id`` of the request to cancel.

        Returns:
            ``True`` if found and removed, ``False`` otherwise.
        """
        target: Optional[TransferRequest] = None
        for req in self._queue:
            if req.request_id == request_id:
                target = req
                break

        if target is None:
            self._logger.debug("cancel: request_id=%s not found", request_id)
            return False

        self._queue.remove(target)
        # Rebuild heap to maintain the invariant (queue is small, O(n) is fine)
        self._heap = [(self._sort_key(r), r) for r in self._queue]
        heapq.heapify(self._heap)

        self._logger.debug(
            "cancel: removed  id=%s  layer=%d expert=%d",
            target.request_id, target.layer_id, target.expert_id,
        )
        return True

    # ------------------------------------------------------------------ #
    # Fully-implemented helpers                                            #
    # ------------------------------------------------------------------ #

    def pending_count(self) -> int:
        """Return the number of requests currently waiting in the queue.

        Returns:
            Queue length as an integer.
        """
        return len(self._queue)

    @property
    def bandwidth_monitor(self) -> BandwidthMonitor:
        """Expose the internal :class:`~transfer.bandwidth.BandwidthMonitor`.

        Returns:
            The rolling-window bandwidth tracker.
        """
        return self._bw_monitor

    def __repr__(self) -> str:
        return (
            f"TransferScheduler("
            f"pending={len(self._queue)}, "
            f"in_flight={len(self._in_flight)}, "
            f"bandwidth={self.bandwidth_gbps:.1f} GB/s)"
        )

