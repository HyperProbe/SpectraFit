"""Worker processes for fitting on several processor cores.

Blocks of pixels are independent, so each can be fitted in its own process. The
pool uses the "spawn" start method (safe on macOS and next to a Qt window), and
every worker limits its maths libraries to one thread so the cores are not
oversubscribed.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS")


def available_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))  # respects limits set by a cluster scheduler (Linux)
    except AttributeError:
        return os.cpu_count() or 1


def resolve_cores(setting: Optional[int]) -> int:
    """Number of cores to use: the setting, or all but one when it is empty or 0."""
    total = available_cores()
    if not setting or setting <= 0:
        return max(1, total - 1)
    return max(1, min(int(setting), total))


def _watch_parent(parent_pid: int) -> None:
    """Worker start-up: exit as soon as the main program is gone (crash, force-quit), instead of idling forever."""
    import threading
    import time

    def watch():
        while True:
            if os.getppid() != parent_pid:
                os._exit(0)
            time.sleep(2.0)

    threading.Thread(target=watch, daemon=True).start()


def _noop(_=None) -> int:
    import time

    time.sleep(0.05)  # keep each worker busy briefly so every one of them gets started
    return os.getpid()


def start_pool(n_workers: int) -> Optional[ProcessPoolExecutor]:
    """Start `n_workers` worker processes (None for a single core)."""
    if n_workers <= 1:
        return None
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    try:
        for k in _THREAD_VARS:  # inherited by the workers when they start
            os.environ[k] = "1"
        pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_watch_parent, initargs=(os.getpid(),))
        # start all workers now, while the variables are set (processes otherwise start lazily)
        try:
            list(pool.map(_noop, range(n_workers)))
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return pool
