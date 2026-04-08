from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest

import cupy as cp

from metric import EvaluationConfig, determine_best_threshold, run_evaluation
from nii_io import NiiLoader, NiiSaver
import main as main_module


GT_PATH = Path(
    "/media/data3/sj/Data/gen4d_outputs/ASOCA/Diseased_02__LCA/LCA_label.nii.gz"
)
PRED_PATH = Path(
    "/media/data3/sj/Data/gen4d_outputs/ASOCA_recon_1/LCA/"
    "Diseased_02__LCA/checkpoints/volume__epoch=209-step=20000.nii.gz"
)


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


@pytest.fixture(scope="session")
def gpu_ready() -> None:
    try:
        _ = cp.zeros((1,), dtype=cp.float32)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"CUDA/CuPy is unavailable: {exc}")


@pytest.fixture(scope="session")
def sample_paths() -> tuple[Path, Path]:
    if not GT_PATH.exists() or not PRED_PATH.exists():
        pytest.skip("Provided NIfTI sample files are not available.")
    return PRED_PATH, GT_PATH


@pytest.fixture
def sample_dirs(tmp_path: Path, sample_paths: tuple[Path, Path]) -> tuple[Path, Path]:
    pred_src, gt_src = sample_paths
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    pred_case_dir = pred_dir / "LCA" / "Diseased_02__LCA" / "checkpoints"
    gt_case_dir = gt_dir / "Diseased_02__LCA"
    pred_case_dir.mkdir(parents=True, exist_ok=True)
    gt_case_dir.mkdir(parents=True, exist_ok=True)

    _link_or_copy(pred_src, pred_case_dir / pred_src.name)
    _link_or_copy(gt_src, gt_case_dir / gt_src.name)
    return pred_dir, gt_dir


def test_metric_run_evaluation_and_threshold_search(
    gpu_ready: None,
    sample_paths: tuple[Path, Path],
) -> None:
    pred_path, gt_path = sample_paths
    load_res = NiiLoader._load_both_cp(pred_path, gt_path)
    pred, gt, spacing = load_res.pred, load_res.gt, load_res.spacing

    cfg = EvaluationConfig(threshold=0.08)
    out, pred_a, pred_b = run_evaluation(pred, gt, spacing, cfg)
    out_any = cast(dict[str, Any], out)

    assert pred_a.shape == gt.shape
    assert pred_b.shape == gt.shape
    assert out_any["pipeline_a"]["info"]["threshold"] == pytest.approx(0.08)
    assert out_any["pipeline_b"]["info"]["threshold"] == pytest.approx(0.08)

    metrics_a = out_any["pipeline_a"]["metrics"]
    assert 0.0 <= metrics_a["dice"] <= 1.0
    assert 0.0 <= metrics_a["precision"] <= 1.0
    assert 0.0 <= metrics_a["recall"] <= 1.0

    candidates = [0.05, 0.08, 0.11]
    best = determine_best_threshold(
        volume=pred,
        gt_label=gt,
        spacing=spacing,
        thresholds=candidates,
        config=cfg,
        objective="dice",
    )
    best_any = cast(dict[str, Any], best)
    assert best_any["best_threshold"] in candidates
    assert len(best_any["all_results"]) == len(candidates)


def test_nii_loader_iterates_with_prefetch(
    gpu_ready: None,
    sample_dirs: tuple[Path, Path],
) -> None:
    pred_dir, gt_dir = sample_dirs
    with NiiLoader(pred_dir, gt_dir, num_workers=1, prefetch_size=1) as loader:
        items = list(loader)

    assert len(items) == 1
    case_id, pred_cp, gt_cp, spacing = items[0]
    assert case_id == "Diseased_02__LCA"
    assert isinstance(pred_cp, cp.ndarray)
    assert isinstance(gt_cp, cp.ndarray)
    assert pred_cp.shape == gt_cp.shape
    assert len(spacing) == 3


def test_nii_saver_saves_valid_nifti(
    gpu_ready: None,
    sample_paths: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    pred_path, gt_path = sample_paths
    load_res = NiiLoader._load_both_cp(pred_path, gt_path)
    pred, affine = load_res.pred, load_res.affine

    pred_mask = cast(cp.ndarray, pred > 0.08).astype(cp.uint8)
    out_path = tmp_path / "pred_mask.nii.gz"
    NiiSaver.save_nifti(pred_mask, affine, out_path)

    assert out_path.exists()
    reloaded = cast(nib.Nifti1Image, nib.load(str(out_path))).get_fdata()
    assert reloaded.shape == tuple(pred_mask.shape)
    
    with NiiSaver(max_workers=1) as saver:
        out_path2 = tmp_path / "pred_mask_async.nii.gz"
        f1 = saver.submit(pred_mask, affine, out_path2)
        
        out_path3 = tmp_path / "pred_mask_async2.nii.gz"
        f2 = saver.submit(pred_mask, affine, out_path3)
        f1.result()  # Wait for the first save to complete
        f2.result()  # Wait for the second save to complete
    
    assert out_path2.exists()
    assert out_path3.exists()
    reloaded2 = cast(nib.Nifti1Image, nib.load(str(out_path2))).get_fdata()
    reloaded3 = cast(nib.Nifti1Image, nib.load(str(out_path3))).get_fdata()
    assert reloaded2.shape == tuple(pred_mask.shape)
    assert reloaded3.shape == tuple(pred_mask.shape)


def test_main_find_best_outputs_summary(
    gpu_ready: None,
    sample_dirs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    pred_dir, gt_dir = sample_dirs
    out_dir = tmp_path / "best_out"

    main_module.find_best(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        n_choosen_to_optimize=1,
        output_dir=out_dir,
    )

    summary_path = out_dir / "best_threshold_summary.json"
    assert summary_path.exists()

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["sample_size"] == 1
    assert len(summary["cases"]) == 1
    assert summary["cases"][0]["best_threshold"] in summary["threshold_candidates"]


def test_main_val_dir_writes_aggregate_stats(
    gpu_ready: None,
    sample_dirs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    pred_dir, gt_dir = sample_dirs
    out_dir = tmp_path / "val_dir_out"

    main_module.val_dir(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        output_dir=out_dir,
        save_pred_labels=False,
        config=EvaluationConfig(threshold=0.08),
        n_workers_loader=1,
        prefetch_size=1,
        n_workers_saver=1,
    )

    result_path = out_dir / "all_cases_evaluation.json"
    assert result_path.exists()

    with result_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert "cases" in payload
    assert "aggregate" in payload
    assert payload["aggregate"]["num_cases"] == 1
    assert payload["aggregate"]["pipeline_a"]["dice"]["sum"] == pytest.approx(
        payload["aggregate"]["pipeline_a"]["dice"]["mean"]
    )
    assert payload["aggregate"]["pipeline_a"]["dice"]["std"] == pytest.approx(0.0)

