"""Tests for 3D metric utilities — MONAI / PyTorch."""

import math

import numpy as np
import pytest
import torch

from internal.metrics.metric_3d_utils import (
    segment_volume_with_roi,
    compute_all_metrics,
)


# ---------------------------------------------------------------------------
# segment_volume_with_roi
# ---------------------------------------------------------------------------

def test_segment_torch():
    vol = torch.zeros(6, 6, 10)
    vol[2:4, 2:4, 2:8] = 1.0
    aabb = torch.ones_like(vol, dtype=torch.bool)
    pred = segment_volume_with_roi(vol, 0.5, aabb)
    assert isinstance(pred, torch.Tensor)
    assert pred.dtype == torch.bool
    assert pred[2:4, 2:4, 2:8].all()
    assert not pred[0:2, 0:2, 0:2].any()


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

def _tube_mask(shape, device="cpu"):
    """Helper: create a tube-shaped mask."""
    m = torch.zeros(shape, dtype=torch.bool, device=device)
    D, H, W = shape[-3:]
    m[..., D//3:2*D//3, H//3:2*H//3, W//3:2*W//3] = True
    return m


# ── Perfect overlap ───────────────────────────────────────────────────────

def test_perfect_overlap():
    m = _tube_mask((6, 6, 10))
    res = compute_all_metrics(m, m, (1.0, 1.0, 1.0))
    assert res["dice"] == pytest.approx(1.0)
    assert res["precision"] == pytest.approx(1.0)
    assert res["recall"] == pytest.approx(1.0)
    assert res["hd95"] == pytest.approx(0.0)
    assert 0.0 <= res["cldice"] <= 1.0


# ── No overlap ────────────────────────────────────────────────────────────

def test_no_overlap():
    pred = torch.zeros(6, 6, 10, dtype=torch.bool)
    pred[0:2, 0:2, 0:2] = True
    gt = torch.zeros(6, 6, 10, dtype=torch.bool)
    gt[4:6, 4:6, 8:10] = True
    res = compute_all_metrics(pred, gt, (1.0, 1.0, 1.0))
    assert res["dice"] == pytest.approx(0.0)
    assert res["precision"] == pytest.approx(0.0)
    assert res["recall"] == pytest.approx(0.0)


# ── Both empty ────────────────────────────────────────────────────────────

def test_both_empty():
    pred = torch.zeros(4, 4, 4, dtype=torch.bool)
    gt = torch.zeros(4, 4, 4, dtype=torch.bool)
    res = compute_all_metrics(pred, gt, (1.0, 1.0, 1.0))
    assert res["dice"] == pytest.approx(1.0)
    assert res["precision"] == pytest.approx(1.0)
    assert res["recall"] == pytest.approx(1.0)
    assert res["hd95"] == pytest.approx(0.0)
    assert res["cldice"] == pytest.approx(1.0)


# ── One empty ─────────────────────────────────────────────────────────────

def test_pred_only():
    pred = _tube_mask((6, 6, 10))
    gt = torch.zeros(6, 6, 10, dtype=torch.bool)
    res = compute_all_metrics(pred, gt, (1.0, 1.0, 1.0))
    assert math.isinf(res["hd95"])


def test_gt_only():
    pred = torch.zeros(6, 6, 10, dtype=torch.bool)
    gt = _tube_mask((6, 6, 10))
    res = compute_all_metrics(pred, gt, (1.0, 1.0, 1.0))
    assert math.isinf(res["hd95"])


# ── GPU input (if available) ──────────────────────────────────────────────

def test_compute_all_torch_input():
    """Tube-shaped prediction with torch input."""
    m = _tube_mask((6, 6, 10))
    gt_np = m.cpu().numpy()
    res = compute_all_metrics(m, gt_np, (1.0, 1.0, 1.0))
    assert res["dice"] == pytest.approx(1.0)
    assert res["precision"] == pytest.approx(1.0)
    assert res["cldice"] >= 0.0

def test_gt_numpy_input():
    """GT as numpy array is accepted."""
    pred = _tube_mask((6, 6, 6))
    gt_np = pred.cpu().numpy()
    res = compute_all_metrics(pred, gt_np, (1.0, 1.0, 1.0))
    assert res["dice"] == pytest.approx(1.0)
    assert res["hd95"] == pytest.approx(0.0)
    assert "assd" not in res
