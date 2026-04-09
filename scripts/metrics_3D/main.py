import sys
import os
import logging
from pathlib import Path
from typing import  Any

import numpy as np
from cyclopts import App
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))  # Adjust as needed to import common types

from metric import EvaluationConfig, run_evaluation, determine_best_threshold, Metrics
from nii_io import NiiLoader, NiiSaver


def _parse_log_level(value: str, default: int = logging.INFO) -> int:
    level_name = value.strip().upper()
    level = getattr(logging, level_name, None)
    if isinstance(level, int):
        return level
    return default


def _configure_logging_from_env() -> None:
    """Configure root and module loggers from environment variables.

    Supported variables:
    - METRIC_LOG_LEVEL: global level, e.g. DEBUG/INFO/WARNING/ERROR
    - METRIC_LOG_FORMAT: logging format
    - METRIC_LOG_DATEFMT: datetime format
    - METRIC_LOG_LEVEL_MAIN: optional override for this module
    - METRIC_LOG_LEVEL_METRIC: optional override for metric module
    - METRIC_LOG_LEVEL_NII_IO: optional override for nii_io module
    """
    global_level = _parse_log_level(os.getenv("METRIC_LOG_LEVEL", "INFO"), logging.INFO)
    log_format = os.getenv(
        "METRIC_LOG_FORMAT",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    date_format = os.getenv("METRIC_LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")

    logging.basicConfig(level=global_level, format=log_format, datefmt=date_format, force=True)

    module_overrides = {
        __name__: os.getenv("METRIC_LOG_LEVEL_MAIN"),
        "metric": os.getenv("METRIC_LOG_LEVEL_METRIC"),
        "nii_io": os.getenv("METRIC_LOG_LEVEL_NII_IO"),
    }
    for logger_name, value in module_overrides.items():
        if value:
            logging.getLogger(logger_name).setLevel(_parse_log_level(value, global_level))


def _write_json(data: dict[str, Any], output_path: Path ) -> None:
    import json
    with output_path.open("w") as f:
        json.dump(data, f, indent=2)


def _aggregate_case_metrics(all_results: dict[str, Any]) -> dict[str, Any]:
    metric_names = ("dice", "precision", "recall", "hd95", "assd", "cldice")
    consistency_names = ("dice_a_vs_b",)

    def _summary(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "sum": float(array.sum()),
            "mean": float(array.mean()) if array.size else float("nan"),
            "std": float(array.std(ddof=0)) if array.size else float("nan"),
        }

    aggregate: dict[str, Any] = {
        "num_cases": int(len(all_results)),
        "pipeline_a": {},
        "pipeline_b": {},
        "consistency": {},
    }

    for section in ("pipeline_a", "pipeline_b"):
        for metric_name in metric_names:
            values = [case[section]["metrics"][metric_name] for case in all_results.values()]
            aggregate[section][metric_name] = _summary(values)

    for metric_name in consistency_names:
        values = [case["consistency"][metric_name] for case in all_results.values()]
        aggregate["consistency"][metric_name] = _summary(values)

    return aggregate


app = App()


@app.command()
def val_once(
    pred_nii_path: Path,
    gt_nii_path: Path,
    output_dir: Path|None = None,
    save_pred_labels: bool = False,
    config: EvaluationConfig = EvaluationConfig(),
):
    """
    结果有两种计算路径: pipeline_a和pipeline_b, 前者是基于pred经过一系列处理 (如闭运算， 到中心距离加权加权, 提取最大连通域,等) 后的结果
    与gt计算, 后者使用 gt 得到范围框后再进行阈值化处理后，与gt计算指标，理论上结果应当接近。两者的结果都输出，并且在aggregate中分别统计。
    
    需要注意gt和pred相位应相同
    """
    load_res = NiiLoader._load_both_cp(pred_nii_path, gt_nii_path)
    metric, pred_gt_free, pred_oracal = run_evaluation(load_res.pred, load_res.gt, load_res.spacing, config)
    print(metric)
    
    if output_dir is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(metric, output_dir / f"{pred_nii_path.stem}_evaluation.json")
    if save_pred_labels:
        NiiSaver.save_nifti(pred_gt_free, load_res.affine, output_dir / f"{pred_nii_path.stem}_pred_gt_free.nii.gz")
        NiiSaver.save_nifti(pred_oracal, load_res.affine, output_dir / f"{pred_nii_path.stem}_pred_oracal.nii.gz")



@app.command()
def val_dir(
    pred_dir: Path,
    gt_dir: Path,
    output_dir: Path|None = None,
    save_pred_labels: bool = False,
    config: EvaluationConfig = EvaluationConfig(),
    n_workers_loader: int = 2,
    prefetch_size: int = 4,
    n_workers_saver: int = 2,
):
    all_results: dict[str, Any] = {}
    case_affines: dict[str, np.ndarray] = {}
    with (
        NiiLoader(pred_dir, gt_dir, num_workers=n_workers_loader, prefetch_size=prefetch_size) as loader,
        tqdm(total=len(loader), desc="Evaluating cases") as pbar,
    ):
        for case_id, load_res in loader:
            metric, pred_gt_free, pred_oracal = run_evaluation(load_res.pred, load_res.gt, load_res.spacing, config)
            metric["case_id"] = case_id
            dice_a = metric["pipeline_a"]["metrics"]["dice"]
            dice_b = metric["pipeline_b"]["metrics"]["dice"]
            all_results[case_id] = metric
            case_affines[case_id] = np.asarray(load_res.affine)
            pbar.set_postfix({"case_id": case_id, "dice_a": f"{dice_a:.3f}", "dice_b": f"{dice_b:.3f}"})
            pbar.update(1)
    
    print(all_results)
    if output_dir is None:
        return

    aggregate = _aggregate_case_metrics(all_results)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        {
            "cases": all_results,
            "aggregate": aggregate,
        },
        output_dir / f"all_cases_evaluation.json",
    )
    if save_pred_labels:
        with NiiSaver(max_workers=n_workers_saver) as saver:
            for case_id, metric in all_results.items():
                pred_gt_free = metric["pipeline_a"]["prediction"]
                pred_oracal = metric["pipeline_b"]["prediction"]
                affine = case_affines[case_id]
                saver.submit(pred_gt_free, affine, output_dir / f"{case_id}_pred_gt_free.nii.gz")
                saver.submit(pred_oracal, affine, output_dir / f"{case_id}_pred_oracal.nii.gz")
    

@app.command()
def find_best(
    pred_dir: Path,
    gt_dir: Path,
    n_choosen_to_optimize: int = 5,
    threshold_start: float = 0.01,
    threshold_end: float = 0.08,
    n_thresholds: int = 40,
    output_dir: Path|None = None,
    objective: Metrics = "dice",
    visualize: bool = False,
):
    with (
        NiiLoader(pred_dir, gt_dir, random_chosen=n_choosen_to_optimize) as loader,
        tqdm(total=len(loader), desc="Finding best thresholds") as pbar,
    ):
        threshold_candidates = [float(round(x, 3)) for x in np.linspace(threshold_start, threshold_end, n_thresholds)]
        summary: dict[str, Any] = {
            "sample_size": len(loader),
            "threshold_candidates": threshold_candidates,
            "cases": [],
        }
        for case_id, load_res in loader:
            best = determine_best_threshold(
                volume=load_res.pred,
                gt_label=load_res.gt,
                spacing=load_res.spacing,
                thresholds=threshold_candidates,
                config=EvaluationConfig(visualize=visualize),
                objective=objective,
            )

            case_summary = {
                "case_id": case_id,
                "best_threshold": best["best_threshold"],
                "best_metrics": best["best_metrics"],
            }
            summary["cases"].append(case_summary)
            
            pbar.update(1)
            pbar.set_postfix({
                "case_id": case_id, 
                "best_threshold": f"{best['best_threshold']:.3f}", 
                f"best_{objective}": f"{best['best_metrics'][objective]:.3f}"
            })

    print(summary)
    if output_dir is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(summary, output_dir / "best_threshold_summary.json")



if __name__ == "__main__":
    _configure_logging_from_env()
    app()