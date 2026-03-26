#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import os
import queue
import signal
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List


DEFAULT_DATA_ROOT = "/media/data3/sj/Data/gen4d_outputs/ASOCA"
DEFUALT_OUTPUT_ROOT = "/media/data3/sj/Data/gen4d_outputs/ASOCA_recon"
DEFAULT_CONFIG = "configs/rotate_xray_3dgr/rotate_xray_3dgr_deformable_xray_render.yaml"
DEFAULT_GPUS = "2,3"
DEFAULT_RUNNER = "pixi run python"


@dataclass(frozen=True)
class Task:
	case_path: Path
	exp_name: str


@dataclass(frozen=True)
class Result:
	task: Task
	gpu_id: str
	success: bool
	status: str
	return_code: int
	run_log: Path


class BatchLogger:
	def __init__(self, batch_log_path: Path, error_log_path: Path) -> None:
		self.batch_log_path = batch_log_path
		self.error_log_path = error_log_path
		self._lock = threading.Lock()

	def log(self, level: str, msg: str) -> None:
		line = f"[{self._ts()}] [{level}] {msg}"
		with self._lock:
			print(line)
			self.batch_log_path.parent.mkdir(parents=True, exist_ok=True)
			with self.batch_log_path.open("a", encoding="utf-8") as f:
				f.write(line + "\n")

	def error(self, msg: str) -> None:
		line = f"[{self._ts()}] {msg}"
		with self._lock:
			print(line)
			self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
			with self.error_log_path.open("a", encoding="utf-8") as f:
				f.write(line + "\n")

	@staticmethod
	def _ts() -> str:
		return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ProcessRegistry:
	def __init__(self) -> None:
		self._lock = threading.Lock()
		self._procs: dict[int, tuple[subprocess.Popen, str, str, Path]] = {}

	def register(self, proc: subprocess.Popen, exp_name: str, gpu_id: str, run_log: Path) -> None:
		with self._lock:
			self._procs[proc.pid] = (proc, exp_name, gpu_id, run_log)

	def unregister(self, pid: int) -> None:
		with self._lock:
			self._procs.pop(pid, None)

	def terminate_all(self, logger: BatchLogger, reason: str, force: bool = False) -> None:
		with self._lock:
			snapshot = list(self._procs.values())

		for proc, exp_name, gpu_id, _run_log in snapshot:
			if proc.poll() is not None:
				continue

			sig = signal.SIGKILL if force else signal.SIGTERM
			try:
				os.killpg(proc.pid, sig)
				logger.log(
					"WARN",
					f"Send {sig.name} to exp={exp_name} gpu={gpu_id} pid={proc.pid}, reason={reason}",
				)
			except Exception as e:  # noqa: BLE001
				logger.log(
					"WARN",
					f"Failed to kill process group for pid={proc.pid} ({exp_name}): {e}, fallback to proc.kill/terminate",
				)
				try:
					if force:
						proc.kill()
					else:
						proc.terminate()
				except Exception:  # noqa: BLE001
					pass


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Batch ASOCA training with per-GPU parallel workers and fault isolation.",
	)
	parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
	parser.add_argument("--config", default=DEFAULT_CONFIG)
	parser.add_argument("--output-root", default=DEFUALT_OUTPUT_ROOT)
	parser.add_argument("--gpus", default=DEFAULT_GPUS, help="Comma separated physical GPU ids, e.g. 2,3")
	parser.add_argument("--retries", type=int, default=0)
	parser.add_argument(
		"--runner",
		default=DEFAULT_RUNNER,
		help="Command prefix to run training, e.g. 'pixi run python' or '/usr/bin/python3'",
	)
	parser.add_argument("--dry-run", action="store_true")
	parser.add_argument(
		"extra_args",
		nargs=argparse.REMAINDER,
		help="Extra args forwarded to main.py fit. Use '-- --max_steps 20000'.",
	)
	args = parser.parse_args()
	if args.retries < 0:
		parser.error("--retries must be >= 0")
	return args


def normalize_extra_args(extra_args: List[str]) -> List[str]:
	if extra_args and extra_args[0] == "--":
		return extra_args[1:]
	return extra_args


def parse_runner(raw: str) -> List[str]:
	runner = shlex.split(raw)
	if not runner:
		raise ValueError("Runner command is empty")
	return runner


def parse_gpu_ids(raw: str) -> List[str]:
	ids = [x.strip() for x in raw.split(",") if x.strip()]
	if not ids:
		raise ValueError(f"No valid GPU ids parsed from: {raw}")
	return ids


def case_is_valid(case_path: Path) -> tuple[bool, str]:
	missing = []

	if not (case_path / "rotate_dsa.json").is_file():
		missing.append("rotate_dsa.json")
	if not (case_path / "depth_map.npz").is_file():
		missing.append("depth_map.npz")
	if not any((case_path / "rotate_dsa").glob("*.png")):
		missing.append("rotate_dsa/*.png")
	if not any((case_path / "label").glob("*.png")):
		missing.append("label/*.png")

	if missing:
		return False, "missing: " + ", ".join(missing)
	return True, ""


def task_is_done(output_root: Path, exp_name: str) -> bool:
	out_dir = output_root / exp_name
	return (out_dir / "checkpoints").exists() or (out_dir / "point_cloud").exists()


def discover_tasks(
	data_root: Path,
	output_root: Path,
	logger: BatchLogger,
) -> tuple[List[Task], List[str], List[str]]:
	tasks: List[Task] = []
	skipped: List[str] = []
	invalid: List[str] = []

	case_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
	for case_path in case_dirs:
		exp_name = f"{case_path.name}"

		ok, reason = case_is_valid(case_path)
		if not ok:
			msg = f"{case_path}|{reason}"
			invalid.append(msg)
			logger.log("WARN", f"Invalid case: {msg}")
			continue

		if task_is_done(output_root, exp_name):
			msg = f"{case_path}|{exp_name}|existing_output"
			skipped.append(msg)
			logger.log("INFO", f"Skip existing output: {exp_name}")
			continue

		tasks.append(Task(case_path=case_path, exp_name=exp_name))

	return tasks, skipped, invalid


def write_run_log_header(run_log: Path, gpu_id: str, cmd: List[str]) -> None:
	run_log.parent.mkdir(parents=True, exist_ok=True)
	with run_log.open("a", encoding="utf-8") as f:
		f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start gpu={gpu_id}\n")
		f.write("Command: " + " ".join(cmd) + "\n")


def tail_lines(path: Path, n: int = 30) -> List[str]:
	if not path.exists():
		return ["run.log not found"]
	try:
		with path.open("r", encoding="utf-8", errors="replace") as f:
			lines = f.readlines()
		return [ln.rstrip("\n") for ln in lines[-n:]]
	except Exception as e:  # noqa: BLE001
		return [f"failed to read run.log tail: {e}"]


def run_single_task(
	task: Task,
	gpu_id: str,
	repo_root: Path,
	config_path: Path,
	output_root: Path,
	runner: List[str],
	retries: int,
	dry_run: bool,
	stop_requested: threading.Event,
	proc_registry: ProcessRegistry,
	logger: BatchLogger,
	extra_args: List[str],
) -> Result:
	out_dir = output_root / task.exp_name
	run_log = out_dir / "run.log"

	cmd = [
		*runner,
		str(repo_root / "main.py"),
		"fit",
		"--config",
		str(config_path),
		"--output",
		str(output_root),
		"--data.path",
		str(task.case_path),
		"-n",
		task.exp_name,
		"--trainer.devices",
		"1",
	] + extra_args

	display_cmd = [f"CUDA_VISIBLE_DEVICES={gpu_id}"] + cmd
	write_run_log_header(run_log, gpu_id, display_cmd)

	if dry_run:
		with run_log.open("a", encoding="utf-8") as f:
			f.write("Dry-run: command not executed.\n")
		return Result(task, gpu_id, True, "DRY_RUN_SUCCESS", 0, run_log)

	env = os.environ.copy()
	env["CUDA_VISIBLE_DEVICES"] = gpu_id
	env["PYTHONUNBUFFERED"] = "1"

	attempt = 0
	while True:
		if stop_requested.is_set():
			return Result(task, gpu_id, False, "INTERRUPTED:stop_requested", 130, run_log)

		with run_log.open("a", encoding="utf-8") as f:
			f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] attempt={attempt}\n")

		try:
			with run_log.open("a", encoding="utf-8") as f:
				proc = subprocess.Popen(  # noqa: S603
					cmd,
					cwd=repo_root,
					env=env,
					stdout=f,
					stderr=subprocess.STDOUT,
					start_new_session=True,
				)

			proc_registry.register(proc, task.exp_name, gpu_id, run_log)
			interrupted = False
			while True:
				if stop_requested.is_set():
					proc_registry.terminate_all(logger, reason="stop_requested", force=False)
					try:
						rc = proc.wait(timeout=5)
					except subprocess.TimeoutExpired:
						proc_registry.terminate_all(logger, reason="escalate_kill", force=True)
						rc = proc.wait(timeout=5)
					interrupted = True
					break

				try:
					rc = proc.wait(timeout=1)
					break
				except subprocess.TimeoutExpired:
					continue

			proc_registry.unregister(proc.pid)
			if interrupted:
				return Result(task, gpu_id, False, f"INTERRUPTED:{rc}", rc, run_log)
		except Exception as e:  # noqa: BLE001
			with run_log.open("a", encoding="utf-8") as f:
				f.write(f"Exception during subprocess.run: {e}\n")
			rc = -99

		if rc == 0:
			return Result(task, gpu_id, True, "SUCCESS", rc, run_log)

		if attempt >= retries:
			return Result(task, gpu_id, False, f"FAILED:{rc}", rc, run_log)

		attempt += 1
		with run_log.open("a", encoding="utf-8") as f:
			f.write(f"Retrying, next_attempt={attempt}\n")


def worker_loop(
	gpu_id: str,
	task_queue: "queue.Queue[Task | None]",
	results: List[Result],
	result_lock: threading.Lock,
	logger: BatchLogger,
	repo_root: Path,
	config_path: Path,
	output_root: Path,
	runner: List[str],
	retries: int,
	dry_run: bool,
	stop_requested: threading.Event,
	proc_registry: ProcessRegistry,
	extra_args: List[str],
) -> None:
	while True:
		if stop_requested.is_set():
			break

		try:
			task = task_queue.get(timeout=0.5)
		except queue.Empty:
			continue

		if task is None:
			task_queue.task_done()
			break

		if stop_requested.is_set():
			logger.log("WARN", f"Stop requested, skip unscheduled task exp={task.exp_name}")
			task_queue.task_done()
			continue

		logger.log("INFO", f"Scheduled exp={task.exp_name} on GPU {gpu_id}")
		result = run_single_task(
			task=task,
			gpu_id=gpu_id,
			repo_root=repo_root,
			config_path=config_path,
			output_root=output_root,
			runner=runner,
			retries=retries,
			dry_run=dry_run,
			stop_requested=stop_requested,
			proc_registry=proc_registry,
			logger=logger,
			extra_args=extra_args,
		)

		with result_lock:
			results.append(result)

		if result.success:
			logger.log("INFO", f"Finished exp={task.exp_name} on GPU {gpu_id} status={result.status}")
		else:
			logger.log("ERROR", f"Failed exp={task.exp_name} on GPU {gpu_id} status={result.status}")
			logger.error(
				f"exp={task.exp_name} case={task.case_path} gpu={gpu_id} status={result.status} return_code={result.return_code}"
			)
			logger.error("last_30_lines:")
			for line in tail_lines(result.run_log, n=30):
				logger.error("    " + line)

		task_queue.task_done()


def main() -> int:
	args = parse_args()
	extra_args = normalize_extra_args(args.extra_args)

	repo_root = Path(__file__).resolve().parent.parent
	data_root = Path(args.data_root).resolve()
	config_path = Path(args.config)
	if not config_path.is_absolute():
		config_path = (repo_root / config_path).resolve()

	output_root = Path(args.output_root).resolve() if args.output_root else (repo_root / "outputs")
	output_root.mkdir(parents=True, exist_ok=True)

	if not data_root.is_dir():
		print(f"[FATAL] Data root not found: {data_root}", file=sys.stderr)
		return 2
	if not config_path.is_file():
		print(f"[FATAL] Config not found: {config_path}", file=sys.stderr)
		return 2

	try:
		gpu_ids = parse_gpu_ids(args.gpus)
	except ValueError as e:
		print(f"[FATAL] {e}", file=sys.stderr)
		return 2

	try:
		runner = parse_runner(args.runner)
	except ValueError as e:
		print(f"[FATAL] {e}", file=sys.stderr)
		return 2

	log_dir = Path(__file__).resolve().parent / "logs"
	log_dir.mkdir(parents=True, exist_ok=True)
	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	batch_log = log_dir / f"batch_train_{ts}.log"
	error_log = log_dir / f"errors_{ts}.log"
	logger = BatchLogger(batch_log, error_log)

	logger.log("INFO", f"Repo root: {repo_root}")
	logger.log("INFO", f"Data root: {data_root}")
	logger.log("INFO", f"Config: {config_path}")
	logger.log("INFO", f"Output root: {output_root}")
	logger.log("INFO", f"GPUs: {', '.join(gpu_ids)}")
	logger.log("INFO", f"Retries: {args.retries}")
	logger.log("INFO", f"Runner: {' '.join(runner)}")
	if extra_args:
		logger.log("INFO", f"Extra args: {' '.join(extra_args)}")
	if args.dry_run:
		logger.log("INFO", "Dry-run mode enabled.")

	tasks, skipped, invalid = discover_tasks(
		data_root=data_root,
		output_root=output_root,
		logger=logger,
	)

	total_cases = len([p for p in data_root.iterdir() if p.is_dir()])
	logger.log(
		"INFO",
		f"Discovery summary: total={total_cases} queued={len(tasks)} skipped={len(skipped)} invalid={len(invalid)}",
	)

	if not tasks:
		logger.log("WARN", "No queued cases. Exit.")
		print(f"Batch log: {batch_log}")
		print(f"Error log: {error_log}")
		return 0

	task_queue: "queue.Queue[Task | None]" = queue.Queue()
	for task in tasks:
		task_queue.put(task)

	for _ in gpu_ids:
		task_queue.put(None)

	stop_requested = threading.Event()
	proc_registry = ProcessRegistry()
	signal_count = {"count": 0}

	def on_signal(signum: int, _frame: object) -> None:
		signal_count["count"] += 1
		stop_requested.set()
		if signal_count["count"] == 1:
			logger.log("WARN", f"Received signal {signum}. Stopping all running tasks now.")
			proc_registry.terminate_all(logger, reason=f"signal_{signum}", force=False)
		else:
			logger.log("WARN", f"Received signal {signum} again. Force killing all running tasks.")
			proc_registry.terminate_all(logger, reason=f"signal_{signum}_force", force=True)

	signal.signal(signal.SIGINT, on_signal)
	signal.signal(signal.SIGTERM, on_signal)

	results: List[Result] = []
	result_lock = threading.Lock()

	with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
		futures = [
			executor.submit(
				worker_loop,
				gpu_id,
				task_queue,
				results,
				result_lock,
				logger,
				repo_root,
				config_path,
				output_root,
				runner,
				args.retries,
				args.dry_run,
				stop_requested,
				proc_registry,
				extra_args,
			)
			for gpu_id in gpu_ids
		]

		for fut in concurrent.futures.as_completed(futures):
			try:
				fut.result()
			except Exception as e:  # noqa: BLE001
				logger.log("ERROR", f"Worker thread crashed: {e}")

	success = [r for r in results if r.success]
	failed = [r for r in results if not r.success]

	logger.log(
		"INFO",
		"Batch finished. "
		f"total={total_cases} queued={len(tasks)} success={len(success)} failed={len(failed)} "
		f"skipped={len(skipped)} invalid={len(invalid)} stop_requested={int(stop_requested.is_set())}",
	)

	if skipped:
		logger.log("INFO", "Skipped list:")
		for item in skipped:
			logger.log("INFO", item)
	if invalid:
		logger.log("INFO", "Invalid list:")
		for item in invalid:
			logger.log("INFO", item)
	if failed:
		logger.log("INFO", "Failed list:")
		for item in failed:
			logger.log("INFO", f"{item.task.case_path}|{item.task.exp_name}|gpu={item.gpu_id}|{item.status}")

	print(f"Batch log: {batch_log}")
	print(f"Error log: {error_log}")
	if stop_requested.is_set():
		return 130
	return 1 if failed else 0


if __name__ == "__main__":
	raise SystemExit(main())
