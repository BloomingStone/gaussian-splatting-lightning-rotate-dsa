#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf


@dataclass(frozen=True)
class RunSpec:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class CaseTask:
    run_spec: RunSpec
    case_path: Path


@dataclass(frozen=True)
class CaseResult:
    run_name: str
    case_name: str
    case_path: str
    gpu_id: str
    success: bool
    status: str
    return_code: int
    duration_sec: float
    output_dir: str
    run_log: str
    metrics_2d: dict[str, float] | None = None
    metrics_3d: dict[str, Any] | None = None
    metrics_2d_file: str | None = None
    metrics_3d_file: str | None = None


@dataclass(frozen=True)
class ScheduledTask:
    run_name: str
    run_root: Path
    case_task: CaseTask


@dataclass
class RunState:
    run_spec: RunSpec
    run_root: Path
    total_cases: int
    completed_cases: int = 0
    results: list[CaseResult] = field(default_factory=list)
    summary_written: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    done_event: threading.Event = field(default_factory=threading.Event)


class Logger:
    _LEVELS = {
        "DEBUG": 10,
        "INFO": 20,
        "WARN": 30,
        "ERROR": 40,
    }

    _COLORS = {
        "DEBUG": "\033[36m",  # cyan
        "INFO": "\033[32m",   # green
        "WARN": "\033[33m",   # yellow
        "ERROR": "\033[31m",  # red
    }
    _RESET = "\033[0m"

    def __init__(self, path: Path, level: str = "INFO") -> None:
        self.path = path
        self._lock = threading.Lock()
        level_upper = str(level).upper()
        if level_upper not in self._LEVELS:
            level_upper = "INFO"
        self.level = level_upper

    def _enabled(self, level: str) -> bool:
        return self._LEVELS[level] >= self._LEVELS[self.level]

    def _console_line(self, level: str, plain_line: str) -> str:
        if not sys.stdout.isatty():
            return plain_line
        color = self._COLORS[level]
        return f"{color}{plain_line}{self._RESET}"

    def _log(self, level: str, msg: str) -> None:
        if not self._enabled(level):
            return
        plain_line = f"[{self._ts()}] [{level}] {msg}"
        console_line = self._console_line(level, plain_line)
        with self._lock:
            print(console_line)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(plain_line + "\n")

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def info(self, msg: str) -> None:
        self._log("INFO", msg)

    def warn(self, msg: str) -> None:
        self._log("WARN", msg)

    def error(self, msg: str) -> None:
        self._log("ERROR", msg)
    
    def debug(self, msg: str) -> None:
        self._log("DEBUG", msg)



class ProcessRegistry:
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
                logger.warn(f"Send {sig.name} to {task_key} gpu={gpu_id}, reason={reason}")
            except Exception as e:  # noqa: BLE001
                logger.error(f"killpg failed for {task_key}: {e}")


def _hash_params(data: dict[str, Any]) -> str:
    text = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _sanitize(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("_", "-", ".", "="):
            out.append(ch)
        else:
            out.append("-")
    return "".join(out)


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def _discover_cases(data_root: Path, include_pattern: str, exclude_pattern: str, logger: Logger) -> list[Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    cases: list[Path] = []
    for p in sorted([x for x in data_root.iterdir() if x.is_dir()]):
        name = p.name
        if include_pattern and include_pattern not in name:
            continue
        if exclude_pattern and exclude_pattern in name:
            continue

        if not (p / "rotate_dsa.json").is_file():
            logger.error(f"Skip invalid case {name}: missing rotate_dsa.json")
            continue
        if not (p / "depth_map.npz").is_file():
            logger.error(f"Skip invalid case {name}: missing depth_map.npz")
            continue
        if not any((p / "rotate_dsa").glob("*.png")):
            logger.error(f"Skip invalid case {name}: missing rotate_dsa/*.png")
            continue
        if not any((p / "label").glob("*.png")):
            logger.error(f"Skip invalid case {name}: missing label/*.png")
            continue

        cases.append(p)
    return cases


def _expand_sweeps(cfg: DictConfig) -> list[RunSpec]:
    sweeps = _to_plain(cfg.sweeps)
    run_specs: list[RunSpec] = []

    for sweep in sweeps:
        sweep_name = str(sweep["name"])
        mode = str(sweep.get("mode", "grid"))
        fixed = dict(sweep.get("fixed", {}))
        grid: dict[str, list[Any]] = {k: list(v) for k, v in dict(sweep.get("grid", {})).items()}
        run_counter = 0

        if mode == "grid":
            keys = sorted(grid.keys())
            combinations: list[dict[str, Any]] = [dict()]
            for k in keys:
                next_rows: list[dict[str, Any]] = []
                for row in combinations:
                    for v in grid[k]:
                        new_row = dict(row)
                        new_row[k] = v
                        next_rows.append(new_row)
                combinations = next_rows

            for row in combinations:
                params = {**fixed, **row}
                token = _hash_params(params)
                run_counter += 1
                run_specs.append(RunSpec(name=f"{_sanitize(sweep_name)}__r{run_counter:03d}__{token}", params=params))
            continue

        if mode == "single-variable":
            for k, values in sorted(grid.items()):
                for v in values:
                    params = dict(fixed)
                    params[k] = v
                    token = _hash_params(params)
                    run_counter += 1
                    run_specs.append(
                        RunSpec(
                            name=f"{_sanitize(sweep_name)}__sv-{_sanitize(k)}-{_sanitize(str(v))}__r{run_counter:03d}__{token}",
                            params=params,
                        )
                    )
            continue

        raise ValueError(f"Unknown sweep mode: {mode}")

    return run_specs


def _override_cli_args(overrides: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for k, v in sorted(overrides.items()):
        args.append(f"--{k}")
        if isinstance(v, bool):
            args.append("true" if v else "false")
        else:
            args.append(str(v))
    return args


def _extract_step(path: Path) -> int:
    stem = path.stem
    idx = stem.find("step=")
    if idx < 0:
        return -1
    idx += len("step=")
    digits = []
    while idx < len(stem) and stem[idx].isdigit():
        digits.append(stem[idx])
        idx += 1
    if not digits:
        return -1
    return int("".join(digits))


def _latest_by_step(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return sorted(paths, key=lambda p: (_extract_step(p), p.name))[-1]


def _read_2d_metrics(case_output_dir: Path, logger: Logger) -> tuple[dict[str, float] | None, Path | None]:
    metrics_dir = case_output_dir / "metrics"
    if not metrics_dir.exists():
        logger.error(f"Metrics directory not found for case={case_output_dir.name}")
        return None, None

    fp = _latest_by_step(sorted(metrics_dir.glob("test-step=*.csv")))
    if fp is None:
        fp = _latest_by_step(sorted(metrics_dir.glob("val-step=*.csv")))
    if fp is None:
        logger.error(f"Latest metrics file not found for case={case_output_dir.name}")
        return None, None

    with fp.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if len(rows) < 2:
        logger.error(f"Insufficient data in metrics file for case={case_output_dir.name}")
        return None, fp

    header = rows[0]
    mean_row = None
    for row in rows[1:]:
        if row and row[0].strip().upper() == "MEAN":
            mean_row = row
            break
    if mean_row is None:
        return None, fp

    out: dict[str, float] = {}
    for i, k in enumerate(header[1:], start=1):
        try:
            out[k] = float(mean_row[i])
        except Exception:  # noqa: BLE001
            continue
    return out, fp


def _read_case_type(case_path: Path) -> str | None:
    jp = case_path / "rotate_dsa.json"
    if not jp.exists():
        return None
    try:
        with jp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        ct = str(data.get("coronary_type", "")).upper()
        if ct in ("LCA", "RCA"):
            return ct
    except Exception:  # noqa: BLE001
        return None
    return None


def _find_gt(cfg: DictConfig, case_path: Path, logger: Logger) -> Path | None:
    try:
        save_phase = cfg.train.static_overrides["model.saver.init_args.save_phase"]
    except AttributeError:
        logger.error("save_phase not found in config(cfg.train.static_overrides[\"model.saver.init_args.save_phase\"]), return None")
        return None
    
    ct = _read_case_type(case_path)
    if ct is None:
        logger.error(f"Can't find ground truth for case={case_path.name}: invalid coronary type")
        return None
    
    if abs(save_phase) < 1e-6:
        p = case_path / f"{ct}_label.nii.gz"
    elif 0<= save_phase < 1:
        phase_str = f"{save_phase:.2f}".replace(".", "_")
        p = case_path / f"{ct}_{phase_str}_label.nii.gz"
    else:
        logger.error(f"Can't find ground truth for case={case_path.name}: invalid save_phase={save_phase}")
        return None
    
    if not p.exists():
        logger.error(f"Can't find ground truth for case={case_path.name}: {str(p)} not found")
        return None
    
    logger.debug(f"Found ground truth for case={case_path.name}: {p}")
    return p


def _find_pred(case_output_dir: Path, logger: Logger) -> Path | None:
    ckpt = case_output_dir / "checkpoints"
    if not ckpt.exists():
        logger.error(f"Can't find predicted labels for case={case_output_dir.name}: no checkpoints found")
        return None
    return _latest_by_step(sorted(ckpt.glob("volume__epoch=*-step=*.nii.gz")))


def _has_checkpoint(case_output_dir: Path) -> bool:
    ckpt_dir = case_output_dir / "checkpoints"
    if not ckpt_dir.exists():
        return False
    return any(ckpt_dir.glob("*.ckpt"))


def _is_reconstruction_ready(case_output_dir: Path) -> bool:
    return _has_checkpoint(case_output_dir)


def _restore_existing_case_result(
    run_spec: RunSpec,
    case_path: Path,
    gpu_id: str,
    case_output_dir: Path,
) -> CaseResult:
    status_file = case_output_dir / "run_status.json"
    if status_file.exists():
        with status_file.open("r", encoding="utf-8") as f:
            prev = json.load(f)
        return CaseResult(
            run_name=run_spec.name,
            case_name=case_path.name,
            case_path=str(case_path),
            gpu_id=gpu_id,
            success=bool(prev.get("success", True)),
            status="SKIPPED_EXISTING",
            return_code=int(prev.get("return_code", 0)),
            duration_sec=float(prev.get("duration_sec", 0.0)),
            output_dir=str(case_output_dir),
            run_log=str(case_output_dir / "manager_run.log"),
            metrics_2d=prev.get("metrics_2d"),
            metrics_3d=prev.get("metrics_3d"),
            metrics_2d_file=prev.get("metrics_2d_file"),
            metrics_3d_file=prev.get("metrics_3d_file"),
        )

    return CaseResult(
        run_name=run_spec.name,
        case_name=case_path.name,
        case_path=str(case_path),
        gpu_id=gpu_id,
        success=True,
        status="SKIPPED_EXISTING",
        return_code=0,
        duration_sec=0.0,
        output_dir=str(case_output_dir),
        run_log=str(case_output_dir / "manager_run.log"),
    )


def _import_metrics3d_api(repo_root: Path):
    metrics_dir = repo_root / "scripts" / "metrics_3D"
    metrics_dir_str = str(metrics_dir)
    if metrics_dir_str not in sys.path:
        sys.path.insert(0, metrics_dir_str)

    # Local import to keep the manager independent from metrics_3D module load time.
    from metric import EvaluationConfig, FragiParams, run_evaluation  # type: ignore
    from nii_io import NiiLoader  # type: ignore

    return EvaluationConfig, FragiParams, run_evaluation, NiiLoader


def _build_eval3d_config(repo_root: Path, cfg: DictConfig, logger: Logger):
    EvaluationConfig, FragiParams, _, _ = _import_metrics3d_api(repo_root)
    raw = _to_plain(cfg.train.get("eval3d", {}))
    raw_dict = dict(raw) if isinstance(raw, dict) else {}

    valid_keys = set(EvaluationConfig.__dataclass_fields__.keys())
    kwargs: dict[str, Any] = {}

    for k, v in raw_dict.items():
        if k not in valid_keys:
            logger.warn(f"Ignore unknown eval3d key: {k}")
            continue
        if k == "fragi_params" and isinstance(v, dict):
            kwargs[k] = FragiParams(**v)
        else:
            kwargs[k] = v

    try:
        return EvaluationConfig(**kwargs)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to build EvaluationConfig from yaml, fallback to defaults. error={e}")
        return EvaluationConfig()


def _run_3d_eval(
    repo_root: Path, 
    case_path: Path, 
    case_output_dir: Path, 
    cfg: DictConfig,
    dry_run: bool, 
    logger: Logger
) -> tuple[dict[str, Any] | None, Path | None]:
    pred = _find_pred(case_output_dir, logger)
    gt = _find_gt(cfg, case_path, logger)
    if pred is None or gt is None:
        logger.error(f"Skip 3D eval case={case_path.name}: pred={pred}, gt={gt}")
        return None, None

    out_dir = case_output_dir / "metrics_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_config = _build_eval3d_config(repo_root, cfg, logger)
    out_json = out_dir / f"{pred.stem}_evaluation.json"

    if dry_run:
        with (out_dir / "metrics3d.log").open("a", encoding="utf-8") as f:
            f.write("Dry-run in-process 3D eval\n")
            f.write(f"pred={pred}\n")
            f.write(f"gt={gt}\n")
            f.write("eval_config=" + repr(eval_config) + "\n")
        return None, None

    with (out_dir / "metrics3d.log").open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] in-process run_evaluation start\n")
        f.write(f"pred={pred}\n")
        f.write(f"gt={gt}\n")
        f.write("eval_config=" + repr(eval_config) + "\n")
        try:
            _, _, run_evaluation, NiiLoader = _import_metrics3d_api(repo_root)
            load_res = NiiLoader._load_both_cp(pred, gt)
            metric, _pred_gt_free, _pred_oracle = run_evaluation(
                load_res.pred,
                load_res.gt,
                load_res.spacing,
                eval_config,
            )
            _write_json(out_json, metric)
            f.write("exit=0\n")
            return metric, out_json
        except Exception as e:  # noqa: BLE001
            f.write(f"exit=1 error={e}\n")
            f.write(traceback.format_exc() + "\n")
            logger.error(f"3D eval failed for case={case_path.name}, error={e}, log={out_dir / 'metrics3d.log'}")
            return None, None


def _extract_by_pipeline(metrics_3d: dict[str, Any] | None, pipeline: Literal["pipeline_a", "pipeline_b"]) -> dict[str, float]:
    if metrics_3d is None:
        return {}
    out: dict[str, float] = {}
    metrics = metrics_3d.get(pipeline, {}).get("metrics", {})
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            try:
                out[k] = float(v)
            except Exception:  # noqa: BLE001
                continue
    return out


def _agg(items: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not items:
        return {}
    keys: set[str] = set()
    for d in items:
        keys.update(d.keys())
    out: dict[str, dict[str, float]] = {}
    for k in sorted(keys):
        vals = [d[k] for d in items if k in d]
        if not vals:
            continue
        out[k] = {
            "mean": mean(vals),
            "std": pstdev(vals) if len(vals) > 1 else 0.0,
            "count": float(len(vals)),
        }
    return out


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _run_single_case(
    task: CaseTask,
    cfg: DictConfig,
    gpu_id: str,
    repo_root: Path,
    run_root: Path,
    logger: Logger,
    stop_requested: threading.Event,
    proc_registry: ProcessRegistry,
) -> CaseResult:
    case_name = task.case_path.name
    case_output_dir = run_root / "cases" / case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Start case={case_name} for {task.run_spec.name} in gpu={gpu_id}")

    run_log = case_output_dir / "manager_run.log"

    static_overrides = dict(_to_plain(cfg.train.static_overrides))
    param_overrides = task.run_spec.params
    common_overrides = {
        **static_overrides,
        **param_overrides,
        "data.path": str(task.case_path),
        "trainer.devices": 1,
        "data.parser.init_args.seed": int(cfg.train.seed),
    }

    runner = shlex.split(str(cfg.train.runner))
    fit_cmd = [
        *runner,
        str(repo_root / "main.py"),
        "fit",
        "--config",
        str(Path(str(cfg.train.base_config)).resolve() if Path(str(cfg.train.base_config)).is_absolute() else (repo_root / str(cfg.train.base_config)).resolve()),
        "--output",
        str(run_root / "cases"),
        "-n",
        case_name,
        *_override_cli_args(common_overrides),
    ]

    extra_args = [str(x) for x in list(_to_plain(cfg.train.extra_args))]
    fit_cmd.extend(extra_args)

    eval_only = bool(cfg.train.get("eval_only", False))
    run_test = bool(cfg.train.run_test)
    test_cmd = None
    if run_test:
        test_cmd = [
            *runner,
            str(repo_root / "main.py"),
            "test",
            "--config",
            str(Path(str(cfg.train.base_config)).resolve() if Path(str(cfg.train.base_config)).is_absolute() else (repo_root / str(cfg.train.base_config)).resolve()),
            "--output",
            str(run_root / "cases"),
            "-n",
            case_name,
            *_override_cli_args(common_overrides),
            *extra_args,
        ]

    _write_json(case_output_dir / "resolved_overrides.json", common_overrides)

    with run_log.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] case={case_name} gpu={gpu_id}\n")
        f.write("FIT: " + " ".join([f"CUDA_VISIBLE_DEVICES={gpu_id}"] + fit_cmd) + "\n")
        if test_cmd is not None:
            f.write("TEST: " + " ".join([f"CUDA_VISIBLE_DEVICES={gpu_id}"] + test_cmd) + "\n")

    if bool(cfg.train.dry_run):
        result = CaseResult(
            run_name=task.run_spec.name,
            case_name=case_name,
            case_path=str(task.case_path),
            gpu_id=gpu_id,
            success=True,
            status="DRY_RUN_SUCCESS",
            return_code=0,
            duration_sec=0.0,
            output_dir=str(case_output_dir),
            run_log=str(run_log),
        )
        _write_json(case_output_dir / "run_status.json", result.__dict__)
        return result

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["PYTHONUNBUFFERED"] = "1"

    retries = int(cfg.train.retries)
    start = time.time()
    rc = 0 if eval_only else -1
    attempt = 0

    if eval_only:
        if not _is_reconstruction_ready(case_output_dir):
            duration = time.time() - start
            result = CaseResult(
                run_name=task.run_spec.name,
                case_name=case_name,
                case_path=str(task.case_path),
                gpu_id=gpu_id,
                success=False,
                status="EVAL_ONLY_NO_CHECKPOINT",
                return_code=2,
                duration_sec=duration,
                output_dir=str(case_output_dir),
                run_log=str(run_log),
            )
            _write_json(case_output_dir / "run_status.json", result.__dict__)
            return result
        logger.info(f"Eval-only mode: skip FIT and rerun test/metrics for case={case_name}")
        with run_log.open("a", encoding="utf-8") as f:
            f.write("Skip FIT: eval_only=true and checkpoint exists.\n")
    else:
        while True:
            if stop_requested.is_set():
                duration = time.time() - start
                result = CaseResult(
                    run_name=task.run_spec.name,
                    case_name=case_name,
                    case_path=str(task.case_path),
                    gpu_id=gpu_id,
                    success=False,
                    status="INTERRUPTED",
                    return_code=130,
                    duration_sec=duration,
                    output_dir=str(case_output_dir),
                    run_log=str(run_log),
                )
                _write_json(case_output_dir / "run_status.json", result.__dict__)
                return result

            with run_log.open("a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] attempt={attempt}\n")
                proc = subprocess.Popen(fit_cmd, cwd=repo_root, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)  # noqa: S603

            proc_registry.register(proc, f"{task.run_spec.name}/{case_name}", gpu_id)
            rc = proc.wait()
            proc_registry.unregister(proc.pid)

            if rc == 0:
                break
            if attempt >= retries:
                duration = time.time() - start
                result = CaseResult(
                    run_name=task.run_spec.name,
                    case_name=case_name,
                    case_path=str(task.case_path),
                    gpu_id=gpu_id,
                    success=False,
                    status=f"FIT_FAILED:{rc}",
                    return_code=rc,
                    duration_sec=duration,
                    output_dir=str(case_output_dir),
                    run_log=str(run_log),
                )
                _write_json(case_output_dir / "run_status.json", result.__dict__)
                return result
            attempt += 1

    if test_cmd is not None:
        if _has_checkpoint(case_output_dir):
            with run_log.open("a", encoding="utf-8") as f:
                proc = subprocess.Popen(test_cmd, cwd=repo_root, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)  # noqa: S603
            proc_registry.register(proc, f"{task.run_spec.name}/{case_name}::test", gpu_id)
            rc_test = proc.wait()
            proc_registry.unregister(proc.pid)
            if rc_test != 0:
                rc = rc_test
        else:
            logger.warn(f"Skip test for case={case_name}: no checkpoint found under {case_output_dir / 'checkpoints'}")
            with run_log.open("a", encoding="utf-8") as f:
                f.write("Skip TEST: no checkpoint found.\n")

    metrics_2d, metrics_2d_file = _read_2d_metrics(case_output_dir, logger)
    metrics_3d = None
    metrics_3d_file = None
    if bool(cfg.train.get("run_3d_eval", True)):
        metrics_3d, metrics_3d_file = _run_3d_eval(repo_root, task.case_path, case_output_dir, cfg, bool(cfg.train.dry_run), logger)
    else:
        logger.warn(f"3D evaluation is disabled by config(train.run_3d_eval=false), skip 3D eval for case={case_name}")
    
    duration = time.time() - start
    success = rc == 0
    result = CaseResult(
        run_name=task.run_spec.name,
        case_name=case_name,
        case_path=str(task.case_path),
        gpu_id=gpu_id,
        success=success,
        status="SUCCESS" if success else f"FAILED:{rc}",
        return_code=rc,
        duration_sec=duration,
        output_dir=str(case_output_dir),
        run_log=str(run_log),
        metrics_2d=metrics_2d,
        metrics_3d=metrics_3d,
        metrics_2d_file=str(metrics_2d_file) if metrics_2d_file else None,
        metrics_3d_file=str(metrics_3d_file) if metrics_3d_file else None,
    )
    _write_json(case_output_dir / "run_status.json", result.__dict__)
    return result


def _write_run_summary(run_root: Path, run_spec: RunSpec, case_results: list[CaseResult], logger: Logger) -> None:
    rows: list[dict[str, Any]] = []
    list_2d: list[dict[str, float]] = []
    list_3d_a: list[dict[str, float]] = []
    list_3d_b: list[dict[str, float]] = []

    for r in case_results:
        row: dict[str, Any] = {
            "run_name": r.run_name,
            "case_name": r.case_name,
            "success": int(r.success),
            "status": r.status,
            "return_code": r.return_code,
            "duration_sec": r.duration_sec,
            "output_dir": r.output_dir,
            "metrics_2d_file": r.metrics_2d_file,
            "metrics_3d_file": r.metrics_3d_file,
        }
        
        if r.metrics_2d:
            for k, v in r.metrics_2d.items():
                row[f"2d_{k}"] = v
            list_2d.append(r.metrics_2d)
        else:
            logger.warn(f"No 2D metrics found for case={r.case_name}, output_dir={r.output_dir}")
        
        m3_a = _extract_by_pipeline(r.metrics_3d, "pipeline_a")
        if m3_a:
            for k, v in m3_a.items():
                row[f"3d_{k}"] = v
            list_3d_a.append(m3_a)
        else:
            logger.warn(f"No 3D metrics for pipeline_a found for case={r.case_name}, output_dir={r.output_dir}")
        
        m3_b = _extract_by_pipeline(r.metrics_3d, "pipeline_b")
        if m3_b:
            for k, v in m3_b.items():
                row[f"3d_by_gt_{k}"] = v
            list_3d_b.append(m3_b)
        else:
            logger.warn(f"No 3D metrics for pipeline_b found for case={r.case_name}, output_dir={r.output_dir}")
        rows.append(row)

    summary = {
        "run_name": run_spec.name,
        "params": run_spec.params,
        "num_cases": len(case_results),
        "num_success": sum(1 for x in case_results if x.success),
        "num_failed": sum(1 for x in case_results if not x.success),
        "agg": {
            "metrics2d": _agg(list_2d),
            "metrics3d_pipeline_a": _agg(list_3d_a),
            "metrics3d_pipeline_b": _agg(list_3d_b),
        },
        "cases": rows,
    }

    _write_json(run_root / "run_summary.json", summary)

    fields: set[str] = set()
    for row in rows:
        fields.update(row.keys())
    with (run_root / "run_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(fields))
        writer.writeheader()
        writer.writerows(rows)


def _execute_one_run(cfg: DictConfig, run_spec: RunSpec, cases: list[Path], results_root: Path, repo_root: Path, logger: Logger) -> list[CaseResult]:
    run_root = results_root / str(cfg.study_name) / run_spec.name
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "run_spec.json", {"name": run_spec.name, "params": run_spec.params})

    parallel_mode = str(_to_plain(cfg.train.get("parallel_mode", "serial"))).strip().lower()

    gpu_id = str(_to_plain(cfg.train.get("gpu", "0")))
    raw_gpus = _to_plain(cfg.train.get("gpus", []))
    gpus = [str(x) for x in raw_gpus] if isinstance(raw_gpus, list) else []
    if not gpu_id and gpus:
        gpu_id = gpus[0]
    if not gpu_id:
        gpu_id = "0"

    if parallel_mode not in ("serial", "multi_gpu"):
        logger.warn(f"Unknown parallel_mode={parallel_mode}, fallback to serial")
        parallel_mode = "serial"

    if parallel_mode == "multi_gpu":
        worker_gpus = gpus if len(gpus) > 0 else [gpu_id]
    else:
        worker_gpus = [gpu_id]

    stop_requested = threading.Event()
    registry = ProcessRegistry()
    results: list[CaseResult] = []
    skip_existing_enabled = bool(cfg.train.skip_existing) and not bool(cfg.train.get("eval_only", False))

    signal_state = {"count": 0}
    interrupted = False
    worker_threads: list[threading.Thread] = []

    def on_signal(signum: int, _frame: object) -> None:
        signal_state["count"] += 1
        stop_requested.set()
        if signal_state["count"] == 1:
            registry.terminate_all(logger, f"signal_{signum}", force=False)
        else:
            registry.terminate_all(logger, f"signal_{signum}_force", force=True)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        if parallel_mode == "serial":
            logger.info(f"Serial mode enabled. Using GPU {worker_gpus[0]}")
            for case in cases:
                if stop_requested.is_set():
                    logger.warn("Stop requested, break remaining cases")
                    break

                task = CaseTask(run_spec=run_spec, case_path=case)

                if skip_existing_enabled:
                    out = run_root / "cases" / task.case_path.name
                    if _is_reconstruction_ready(out):
                        logger.info(f"Skip existing case={task.case_path.name} since checkpoint exists and skip_existing=true")
                        results.append(_restore_existing_case_result(run_spec, task.case_path, worker_gpus[0], out))
                        continue

                res = _run_single_case(task, cfg, worker_gpus[0], repo_root, run_root, logger, stop_requested, registry)
                results.append(res)
        else:
            logger.info(f"Multi-GPU mode enabled. Worker GPUs={worker_gpus}")
            task_queue: "queue.Queue[CaseTask | None]" = queue.Queue()
            for case in cases:
                task_queue.put(CaseTask(run_spec=run_spec, case_path=case))
            for _ in worker_gpus:
                task_queue.put(None)

            result_lock = threading.Lock()

            def worker(gid: str) -> None:
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

                    if skip_existing_enabled:
                        out = run_root / "cases" / task.case_path.name
                        if _is_reconstruction_ready(out):
                            logger.info(f"Skip existing case={task.case_path.name} since checkpoint exists and skip_existing=true")
                            with result_lock:
                                results.append(_restore_existing_case_result(run_spec, task.case_path, gid, out))
                            task_queue.task_done()
                            continue

                    res = _run_single_case(task, cfg, gid, repo_root, run_root, logger, stop_requested, registry)
                    with result_lock:
                        results.append(res)
                    task_queue.task_done()

            worker_threads = [threading.Thread(target=worker, args=(gid,), daemon=False) for gid in worker_gpus]
            for thread in worker_threads:
                thread.start()

            while any(thread.is_alive() for thread in worker_threads):
                for thread in worker_threads:
                    thread.join(timeout=0.5)

    except KeyboardInterrupt:
        interrupted = True
        stop_requested.set()
        registry.terminate_all(logger, "keyboard_interrupt", force=True)
        for thread in worker_threads:
            thread.join(timeout=1.0)
        logger.warn("Interrupted by Ctrl+C, current run stopped.")
    finally:
        _write_run_summary(run_root, run_spec, results, logger)

    if interrupted:
        raise KeyboardInterrupt
    return results


def _execute_all_runs_multi_gpu(
    cfg: DictConfig,
    run_specs: list[RunSpec],
    cases: list[Path],
    results_root: Path,
    repo_root: Path,
    logger: Logger,
) -> list[CaseResult]:
    gpu_id = str(_to_plain(cfg.train.get("gpu", "0")))
    raw_gpus = _to_plain(cfg.train.get("gpus", []))
    gpus = [str(x) for x in raw_gpus] if isinstance(raw_gpus, list) else []
    if not gpu_id and gpus:
        gpu_id = gpus[0]
    if not gpu_id:
        gpu_id = "0"
    worker_gpus = gpus if len(gpus) > 0 else [gpu_id]

    stop_requested = threading.Event()
    registry = ProcessRegistry()
    skip_existing_enabled = bool(cfg.train.skip_existing) and not bool(cfg.train.get("eval_only", False))

    run_states: dict[str, RunState] = {}
    task_queue: "queue.Queue[ScheduledTask | None]" = queue.Queue()
    worker_threads: list[threading.Thread] = []
    signal_state = {"count": 0}
    interrupted = False

    for run_spec in run_specs:
        run_root = results_root / str(cfg.study_name) / run_spec.name
        run_root.mkdir(parents=True, exist_ok=True)
        _write_json(run_root / "run_spec.json", {"name": run_spec.name, "params": run_spec.params})

        run_states[run_spec.name] = RunState(
            run_spec=run_spec,
            run_root=run_root,
            total_cases=len(cases),
        )

        for case in cases:
            task_queue.put(
                ScheduledTask(
                    run_name=run_spec.name,
                    run_root=run_root,
                    case_task=CaseTask(run_spec=run_spec, case_path=case),
                )
            )

    for _ in worker_gpus:
        task_queue.put(None)

    def finalize_run_state(state: RunState) -> None:
        should_write = False
        snapshot: list[CaseResult] = []
        with state.lock:
            if state.completed_cases >= state.total_cases:
                state.done_event.set()
            if state.done_event.is_set() and not state.summary_written:
                state.summary_written = True
                should_write = True
                snapshot = list(state.results)
        if should_write:
            _write_run_summary(state.run_root, state.run_spec, snapshot, logger)

    def on_signal(signum: int, _frame: object) -> None:
        signal_state["count"] += 1
        stop_requested.set()
        if signal_state["count"] == 1:
            registry.terminate_all(logger, f"signal_{signum}", force=False)
        else:
            registry.terminate_all(logger, f"signal_{signum}_force", force=True)
        raise KeyboardInterrupt

    def submit_result(run_name: str, result: CaseResult) -> None:
        state = run_states[run_name]
        with state.lock:
            state.results.append(result)
            state.completed_cases += 1
        finalize_run_state(state)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    logger.info(f"Multi-GPU global queue mode enabled. Worker GPUs={worker_gpus}")

    def worker(gid: str) -> None:
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
                task_queue.task_done()
                continue

            if skip_existing_enabled:
                out = task.run_root / "cases" / task.case_task.case_path.name
                if _is_reconstruction_ready(out):
                    logger.info(
                        f"Skip existing case={task.case_task.case_path.name} run={task.run_name} since checkpoint exists and skip_existing=true"
                    )
                    result = _restore_existing_case_result(
                        task.case_task.run_spec,
                        task.case_task.case_path,
                        gid,
                        out,
                    )
                    submit_result(task.run_name, result)
                    task_queue.task_done()
                    continue

            result = _run_single_case(
                task=task.case_task,
                cfg=cfg,
                gpu_id=gid,
                repo_root=repo_root,
                run_root=task.run_root,
                logger=logger,
                stop_requested=stop_requested,
                proc_registry=registry,
            )
            submit_result(task.run_name, result)
            task_queue.task_done()

    try:
        worker_threads = [threading.Thread(target=worker, args=(gid,), daemon=False) for gid in worker_gpus]
        for thread in worker_threads:
            thread.start()

        while any(thread.is_alive() for thread in worker_threads):
            for thread in worker_threads:
                thread.join(timeout=0.5)
    except KeyboardInterrupt:
        interrupted = True
        stop_requested.set()
        registry.terminate_all(logger, "keyboard_interrupt", force=True)
        for thread in worker_threads:
            thread.join(timeout=1.0)
        logger.warn("Interrupted by Ctrl+C, global queue run stopped.")
    finally:
        for state in run_states.values():
            with state.lock:
                if state.completed_cases >= state.total_cases:
                    state.done_event.set()
            finalize_run_state(state)

    if interrupted:
        raise KeyboardInterrupt

    all_results: list[CaseResult] = []
    for run_spec in run_specs:
        state = run_states[run_spec.name]
        with state.lock:
            all_results.extend(list(state.results))
    return all_results


@hydra.main(version_base=None, config_path=str(Path(__file__).parent.parent / "configs" / "experiments"), config_name="q1")
def main(cfg: DictConfig) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    results_root = Path(str(cfg.train.results_root)).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = Logger(
        results_root / "manager_logs" / f"{cfg.study_name}_{ts}.log",
        level=str(cfg.train.get("log_level", "INFO")),
    )

    logger.info("Hydra config resolved:")
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    data_root = Path(str(cfg.train.data_root)).resolve()
    include_pattern = str(cfg.train.case_filter.include_pattern)
    exclude_pattern = str(cfg.train.case_filter.exclude_pattern)

    cases = _discover_cases(data_root, include_pattern, exclude_pattern, logger)
    if not cases:
        logger.warn("No valid case discovered, exit")
        return

    run_specs = _expand_sweeps(cfg)
    logger.info(f"Study={cfg.study_name}, runs={len(run_specs)}, cases={len(cases)}")

    parallel_mode = str(_to_plain(cfg.train.get("parallel_mode", "serial"))).strip().lower()
    if parallel_mode not in ("serial", "multi_gpu"):
        logger.warn(f"Unknown parallel_mode={parallel_mode}, fallback to serial")
        parallel_mode = "serial"

    all_results: list[CaseResult] = []
    try:
        if parallel_mode == "multi_gpu":
            logger.info("Use global queue scheduler for multi_gpu mode")
            all_results = _execute_all_runs_multi_gpu(cfg, run_specs, cases, results_root, repo_root, logger)
        else:
            for idx, run_spec in enumerate(run_specs, start=1):
                logger.info(f"Run {idx}/{len(run_specs)}: {run_spec.name}")
                rs = _execute_one_run(cfg, run_spec, cases, results_root, repo_root, logger)
                all_results.extend(rs)
    except KeyboardInterrupt:
        logger.warn("Interrupted by Ctrl+C, stop the whole experiment manager.")
        _write_json(
            results_root / str(cfg.study_name) / "study_summary.json",
            {
                "study": str(cfg.study_name),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "num_runs": len(run_specs),
                "num_cases": len(cases),
                "num_case_results": len(all_results),
                "num_success": sum(1 for x in all_results if x.success),
                "num_failed": sum(1 for x in all_results if not x.success),
                "interrupted": True,
            },
        )
        return

    summary = {
        "study": str(cfg.study_name),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "num_runs": len(run_specs),
        "num_cases": len(cases),
        "num_case_results": len(all_results),
        "num_success": sum(1 for x in all_results if x.success),
        "num_failed": sum(1 for x in all_results if not x.success),
    }
    _write_json(results_root / str(cfg.study_name) / "study_summary.json", summary)
    logger.info(f"Done: {summary}")


if __name__ == "__main__":
    # usage: python -m scripts/experiment_manager --config-path configs/experiments --config-name q1 train.result_root=....
    main()
