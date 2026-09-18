"""Metrics reporter for MoEInfra.

:class:`MetricsReporter` runs a background thread that periodically calls
:meth:`~metrics.collector.MetricsCollector.snapshot` and logs a human-readable
summary at configurable intervals.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from metrics.collector import MetricsCollector
from metrics.types import MetricsSummary


class MetricsReporter:
    """Periodic background metrics reporter.

    Starts a daemon thread that wakes every *interval_s* seconds, calls
    :meth:`~metrics.collector.MetricsCollector.snapshot`, and logs the
    result.

    Attributes:
        interval_s: Seconds between successive report emissions.
    """

    def __init__(
        self,
        collector: MetricsCollector,
        interval_s: float = 10.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialise the reporter.

        Args:
            collector: The :class:`~metrics.collector.MetricsCollector` to
                periodically snapshot.
            interval_s: How often (in seconds) to emit a metrics report.
            logger: Optional pre-configured logger.
        """
        self._collector: MetricsCollector = collector
        self.interval_s: float = interval_s
        self._logger: logging.Logger = logger or logging.getLogger(__name__)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background reporting thread.

        Returns:
            None
        """
        if self._thread and self._thread.is_alive():
            self._logger.warning("MetricsReporter already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="metrics-reporter",
            daemon=True,
        )
        self._thread.start()
        self._logger.info(
            "MetricsReporter started (interval=%.1fs).", self.interval_s
        )

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to join.

        Returns:
            None
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s + 2.0)
        self._logger.info("MetricsReporter stopped.")

    def report(self) -> MetricsSummary:
        """Take a snapshot and log it immediately.

        Returns:
            The :class:`~metrics.types.MetricsSummary` that was logged.
        """
        summary = self._collector.snapshot()
        self._logger.info(
            "METRICS | hit_rate=%.3f | hits=%d | misses=%d | evictions=%d "
            "| transfers=%d | bytes=%d | bandwidth=%.3f GB/s "
            "| ttft_avg=%s ms | tps_avg=%s | gpu_slots=%d | cpu_slots=%d",
            summary.hit_rate,
            summary.cache_hits,
            summary.cache_misses,
            summary.evictions,
            summary.transfers_completed,
            summary.total_bytes_transferred,
            summary.transfer_bandwidth_gbps,
            f"{summary.ttft_ms_avg:.1f}" if summary.ttft_ms_avg is not None else "N/A",
            f"{summary.tps_avg:.2f}" if summary.tps_avg is not None else "N/A",
            summary.gpu_slots_used,
            summary.cpu_slots_used,
        )
        return summary

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        """Background reporting loop.  Runs until :attr:`_stop_event` is set."""
        while not self._stop_event.wait(timeout=self.interval_s):
            try:
                self.report()
            except Exception:
                self._logger.exception("Unexpected error in MetricsReporter loop.")
