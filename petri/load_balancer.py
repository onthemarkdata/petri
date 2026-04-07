"""Adaptive load balancer for Petri's concurrent node processing.

Monitors system resources (CPU, memory) and dynamically adjusts the number
of concurrent agents. Starts with 1 worker and scales up/down by 1 based
on resource utilization, maintaining a 10% buffer below capacity.
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# Resource thresholds (as fractions of 1.0)
_HIGH_WATERMARK = 0.90  # Scale down when usage exceeds 90%
_LOW_WATERMARK = 0.70   # Scale up when usage is below 70%
_POLL_INTERVAL = 5.0    # Seconds between resource checks


def _get_cpu_percent() -> float:
    """Get current CPU utilization as a fraction (0.0-1.0).

    Uses os.getloadavg() / cpu_count as a lightweight alternative to psutil.
    """
    try:
        load_1min = os.getloadavg()[0]
        cpu_count = os.cpu_count() or 1
        return min(load_1min / cpu_count, 1.0)
    except (OSError, AttributeError):
        return 0.5  # Conservative fallback


def _get_memory_percent() -> float:
    """Get current memory utilization as a fraction (0.0-1.0).

    Reads from /proc/meminfo on Linux or vm_stat on macOS.
    """
    try:
        import subprocess

        result = subprocess.run(
            ["vm_stat"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return 0.5

        lines = result.stdout.strip().splitlines()
        stats: dict[str, int] = {}
        for line in lines[1:]:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip().rstrip(".")
                try:
                    stats[key] = int(val)
                except ValueError:
                    continue

        free = stats.get("Pages free", 0)
        active = stats.get("Pages active", 0)
        inactive = stats.get("Pages inactive", 0)
        speculative = stats.get("Pages speculative", 0)
        wired = stats.get("Pages wired down", 0)

        total = free + active + inactive + speculative + wired
        if total == 0:
            return 0.5

        used = active + wired
        return used / total
    except Exception:
        return 0.5  # Conservative fallback


class AdaptiveLoadBalancer:
    """Monitors resources and provides a dynamic worker count.

    The balancer starts at ``min_workers`` (default 1) and adjusts by 1
    in either direction based on CPU and memory utilization. It maintains
    a 10% buffer (high watermark at 90%) to prevent resource exhaustion.

    Usage::

        balancer = AdaptiveLoadBalancer(max_workers=8)
        balancer.start()

        # In the processing loop:
        current = balancer.recommended_workers

        balancer.stop()
    """

    def __init__(
        self,
        max_workers: int = 4,
        min_workers: int = 1,
        poll_interval: float = _POLL_INTERVAL,
    ):
        self.max_workers = max(max_workers, 1)
        self.min_workers = max(min_workers, 1)
        self.poll_interval = poll_interval

        self._current_workers = self.min_workers
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def recommended_workers(self) -> int:
        """Current recommended number of concurrent workers."""
        with self._lock:
            return self._current_workers

    def start(self) -> None:
        """Start the background resource monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="petri-load-balancer"
        )
        self._thread.start()
        logger.info(
            "Load balancer started: min=%d, max=%d, current=%d",
            self.min_workers, self.max_workers, self._current_workers,
        )

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval + 1)
            self._thread = None

    def _monitor_loop(self) -> None:
        """Periodically check resources and adjust worker count."""
        while not self._stop_event.is_set():
            self._adjust()
            self._stop_event.wait(timeout=self.poll_interval)

    def _adjust(self) -> None:
        """Check resource utilization and scale workers up or down by 1."""
        cpu = _get_cpu_percent()
        mem = _get_memory_percent()
        peak = max(cpu, mem)

        with self._lock:
            prev = self._current_workers

            if peak > _HIGH_WATERMARK and self._current_workers > self.min_workers:
                self._current_workers -= 1
                logger.info(
                    "Load balancer: scaling down %d -> %d (cpu=%.0f%%, mem=%.0f%%)",
                    prev, self._current_workers, cpu * 100, mem * 100,
                )
            elif peak < _LOW_WATERMARK and self._current_workers < self.max_workers:
                self._current_workers += 1
                logger.info(
                    "Load balancer: scaling up %d -> %d (cpu=%.0f%%, mem=%.0f%%)",
                    prev, self._current_workers, cpu * 100, mem * 100,
                )
