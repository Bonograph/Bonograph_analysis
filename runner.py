"""Run analyses on a background thread, so the window stays responsive.

The sheet's data is read on the main thread first (a plain copy), then the
calculation - pure numpy/scipy, no Qt - runs on a worker thread. When it
finishes, ``on_done`` is called back on the main thread, where the results
window is built. A run can be cancelled: the calculation can't be stopped
part way, but its result is then simply dropped.

Hand-off is by polling a Future from a main-thread timer rather than by a
cross-thread Qt signal: Python callables connected to signals don't always
land on the thread you expect, and a poll every 30 ms costs nothing.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
import traceback
from typing import Callable

_executor: concurrent.futures.ThreadPoolExecutor | None = None
_lock = threading.Lock()
_jobs: list["Job"] = []


def _pool() -> concurrent.futures.ThreadPoolExecutor:
    global _executor
    with _lock:
        if _executor is None:
            _executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="ape-analysis")
        return _executor


def _call(fn):
    """Runs on the worker: the result, or the error with its traceback text."""
    try:
        return fn(), None
    except BaseException as exc:  # noqa: BLE001 - handed back to the main thread
        return None, (exc, traceback.format_exc())


class Job:
    """One analysis run. ``on_done(result, error)`` is called on the main
    thread; ``error`` is (exception, traceback text) or None."""

    def __init__(self, fn: Callable, on_done: Callable, label: str = ""):
        from PySide6.QtCore import QTimer

        self.label = label
        self.started = time.perf_counter()
        self.cancelled = False
        self.finished = False
        self._on_done = on_done
        self._future = _pool().submit(_call, fn)
        self._timer = QTimer()
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        _jobs.append(self)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def cancel(self) -> None:
        """Drop the result when it arrives."""
        self.cancelled = True

    def _poll(self) -> None:
        if not self._future.done():
            return
        self._timer.stop()
        self.finished = True
        if self in _jobs:
            _jobs.remove(self)
        if self.cancelled:
            return
        result, error = self._future.result()
        try:
            self._on_done(result, error)
        except Exception:  # noqa: BLE001 - a failure showing results must not take the app down
            try:
                import sys

                import crash_guard

                crash_guard.report(*sys.exc_info(), where=f"showing {self.label}")
            except Exception:  # noqa: BLE001
                traceback.print_exc()


def submit(fn: Callable, on_done: Callable, label: str = "") -> Job:
    return Job(fn, on_done, label)


def pending() -> int:
    """Runs still in progress (not counting cancelled ones)."""
    return sum(1 for j in _jobs if not j.finished and not j.cancelled)


def wait_all(timeout: float = 60.0) -> bool:
    """Block (with the event loop running) until every run has finished and
    been handled - for tests and for closing the app. True if all done."""
    from PySide6.QtCore import QEventLoop, QTimer

    if not _jobs:
        return True
    deadline = time.perf_counter() + timeout
    loop = QEventLoop()

    def check():
        if not _jobs or time.perf_counter() > deadline:
            loop.quit()
        else:
            QTimer.singleShot(20, check)

    QTimer.singleShot(0, check)
    loop.exec()
    return not _jobs
