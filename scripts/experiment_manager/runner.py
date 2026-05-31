"""Parallel experiment execution: process registry and GPU dispatch.

After setup (see ``setup.py``), each case directory is self-contained with a
``config.yaml``.  The runner only needs a list of directory paths and dispatches
``fit --config <dir>/config.yaml`` across available GPUs.
"""

from __future__ import annotations

import os
import queue
import shlex
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from ._logging import Logger
from ._utils import _write_json


class ProcessRegistry:
    """Tracks running subprocesses and supports graceful/forced termination."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._procs: dict[int, tuple[subprocess.Popen, str, str]] = {}

    def register(self, proc: subprocess.Popen, task_key: str, gpu_id: str) -> None:
        with self._lock:
            self._procs[proc.pid] = (proc, task_key, gpu_id)

    def unregister(self, pid: int) -> None:
        with self._lock:
            self._procs.pop(pid, None)

    def terminate_all(self, logger: Logger, reason: str, force: bool = False) -> None:
        with self._lock:
            snapshot = list(self._procs.values())

        for proc, task_key, gpu_id in snapshot:
            if proc.poll() is not None:
                continue
            sig = signal.SIGKILL if force else signal.SIGTERM
            try:
                os.killpg(proc.pid, sig)
                logger.warn(
                    f"Send {sig.name} to {task_key} gpu={gpu_id}, reason={reason}"
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"killpg failed for {task_key}: {e}")


# ---------------------------------------------------------------------------
# Single case execution
# ---------------------------------------------------------------------------


def _run_case_dir(
    case_dir: Path,
    gpu_id: str,
    runner_cmd: list[str],
    repo_root: Path,
    logger: Logger,
    stop_requested: threading.Event,
    proc_registry: ProcessRegistry,
    retries: int = 0,
    dry_run: bool = False,
) -> tuple[bool, int, float]:
    """Run ``fit --config <case_dir>/config.yaml``.

    Returns ``(success, return_code, duration_sec)``.
    """
    import time

    config_path = case_dir / "config.yaml"
    case_key = str(case_dir.relative_to(case_dir.parents[2]))  # run/case
    run_log = case_dir / "run.log"

    cmd = [*runner_cmd, str(repo_root / "main.py"), "fit", "--config", str(config_path)]

    if dry_run:
        with run_log.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DRY-RUN gpu={gpu_id}\n")
            f.write(" ".join(cmd) + "\n")
        logger.info(f"DRY-RUN {case_key} gpu={gpu_id}")
        return True, 0, 0.0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["PYTHONUNBUFFERED"] = "1"

    start = time.time()
    attempt = 0

    while True:
        if stop_requested.is_set():
            return False, 130, time.time() - start

        with run_log.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] attempt={attempt} gpu={gpu_id}\n")
            f.write(" ".join(cmd) + "\n")
            proc = subprocess.Popen(  # noqa: S603
                cmd, cwd=repo_root, env=env,
                stdout=f, stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        proc_registry.register(proc, case_key, gpu_id)
        rc = proc.wait()
        proc_registry.unregister(proc.pid)

        if rc == 0:
            break
        if attempt >= retries:
            return False, rc, time.time() - start
        attempt += 1

    return True, 0, time.time() - start


# ---------------------------------------------------------------------------
# Multi-GPU dispatch
# ---------------------------------------------------------------------------


def execute_dirs(
    dirs: list[Path],
    gpus: list[str],
    runner_cmd: list[str],
    repo_root: Path,
    logger: Logger,
    retries: int = 0,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> dict[Path, tuple[bool, int, float]]:
    """Dispatch case directories to GPU workers via a shared queue.

    Returns ``{case_dir: (success, return_code, duration_sec)}``.
    """
    from .filesystem import _has_checkpoint

    # Filter out completed cases
    pending: list[Path] = []
    skipped: dict[Path, tuple[bool, int, float]] = {}
    for d in dirs:
        if skip_existing and _has_checkpoint(d):
            logger.info(f"Skip existing: {d.name}")
            skipped[d] = (True, 0, 0.0)
        else:
            pending.append(d)

    if not pending:
        logger.info("All cases already completed, nothing to run")
        return skipped

    stop_requested = threading.Event()
    registry = ProcessRegistry()
    task_queue: "queue.Queue[Path | None]" = queue.Queue()
    results: dict[Path, tuple[bool, int, float]] = dict(skipped)
    result_lock = threading.Lock()
    signal_state = {"count": 0}
    interrupted = False

    for d in pending:
        task_queue.put(d)
    for _ in gpus:
        task_queue.put(None)

    def on_signal(signum: int, _frame: object) -> None:
        signal_state["count"] += 1
        stop_requested.set()
        if signal_state["count"] == 1:
            registry.terminate_all(logger, f"signal_{signum}", force=False)
        else:
            registry.terminate_all(logger, f"signal_{signum}_force", force=True)
        raise KeyboardInterrupt

    def worker(gid: str) -> None:
        while True:
            if stop_requested.is_set():
                break
            try:
                case_dir = task_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if case_dir is None:
                task_queue.task_done()
                break

            success, rc, dur = _run_case_dir(
                case_dir, gid, runner_cmd, repo_root, logger,
                stop_requested, registry, retries, dry_run,
            )
            with result_lock:
                results[case_dir] = (success, rc, dur)
            logger.info(
                f"{'OK' if success else 'FAIL'}: {case_dir.name} rc={rc} dur={dur:.0f}s gpu={gid}"
            )
            task_queue.task_done()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    logger.info(f"Dispatching {len(pending)} cases across {len(gpus)} GPUs: {gpus}")

    worker_threads = [
        threading.Thread(target=worker, args=(gid,), daemon=False) for gid in gpus
    ]
    try:
        for t in worker_threads:
            t.start()
        while any(t.is_alive() for t in worker_threads):
            for t in worker_threads:
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        interrupted = True
        stop_requested.set()
        registry.terminate_all(logger, "keyboard_interrupt", force=True)
        for t in worker_threads:
            t.join(timeout=1.0)
        logger.warn("Interrupted by Ctrl+C")

    if interrupted:
        raise KeyboardInterrupt

    n_ok = sum(1 for s, _, _ in results.values() if s)
    n_fail = len(results) - n_ok
    logger.info(f"Done: {n_ok} success, {n_fail} failed, {len(skipped)} skipped")
    return results
