from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from internal.dataparsers.xray_dataparser.splitters import (
    FileSpliter,
    PhaseStratifiedSpliter,
    ReconstructionSpliter,
    RenderNewViewsSpliter,
)

from .common import load_test_meta


def test_reconstruction_splitter_uses_all_frames(test_xray_data_root: Path):
    meta = load_test_meta(test_xray_data_root)
    splits = ReconstructionSpliter().split(test_xray_data_root, meta)

    expected = list(range(meta.num_frames))
    assert splits == {"train": expected, "val": expected, "test": expected}


def test_render_new_views_splitter_is_deterministic_per_data_root(test_xray_data_root: Path):
    meta = load_test_meta(test_xray_data_root)
    splitter = RenderNewViewsSpliter(train_ratio=0.6, seed=17, random_loader_mode="random-shuffle")

    split_a = splitter.split(test_xray_data_root, meta)
    split_b = splitter.split(test_xray_data_root, meta)
    split_c = splitter.split(Path("/tmp/data-b"), meta)

    assert split_a == split_b
    assert split_a != split_c
    assert len(split_a["train"]) == int(meta.num_frames * 0.6)
    assert len(split_a["val"]) == meta.num_frames - int(meta.num_frames * 0.6)
    assert split_a["val"] == split_a["test"]
    assert sorted(split_a["train"] + split_a["val"]) == list(range(meta.num_frames))


def test_render_new_views_splitter_random_start_keeps_contiguous_sequence(test_xray_data_root: Path):
    meta = load_test_meta(test_xray_data_root)
    splitter = RenderNewViewsSpliter(train_ratio=0.5, seed=3, random_loader_mode="random-start")

    splits = splitter.split(test_xray_data_root, meta)
    train = np.array(splits["train"])

    assert len(train) == int(meta.num_frames * 0.5)
    assert len(splits["val"]) == meta.num_frames - int(meta.num_frames * 0.5)
    assert np.all(np.diff(train) == 1)


def test_render_new_views_splitter_rejects_unknown_mode(test_xray_data_root: Path):
    meta = load_test_meta(test_xray_data_root)
    splitter = RenderNewViewsSpliter(random_loader_mode="invalid")  # type: ignore

    with pytest.raises(ValueError):
        splitter.split(Path("/tmp/data-d"), meta)


# ───────── PhaseStratifiedSpliter ─────────


def _check_split_validity(
    splits: dict[str, list[int]],
    n_frames: int,
    key_test: str = "test",
) -> None:
    """Assert that train/val/test partition *n_frames* without overlap."""
    assert "train" in splits and "val" in splits and key_test in splits
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits[key_test])
    # no overlap between train and val
    assert train_set.isdisjoint(val_set), "train and val overlap"
    # union covers all frames
    all_frames = set(range(n_frames))
    assert train_set | val_set == all_frames, f"missing frames: {all_frames - (train_set | val_set)}"
    # val == test unless overridden
    assert splits["val"] == splits["test"]


def test_phase_stratified_splitter_basic_validity(test_xray_data_root: Path):
    """Basic validity: union covers all frames, train/val disjoint, val == test."""
    meta = load_test_meta(test_xray_data_root)
    splitter = PhaseStratifiedSpliter(train_ratio=0.8, seed=0, n_bins=20)
    splits = splitter.split(test_xray_data_root, meta)
    _check_split_validity(splits, meta.num_frames)


def test_phase_stratified_splitter_excludes_phase0_when_not_dominant(test_xray_data_root: Path):
    """Phase‑0 frames should NOT appear in training when phase 0 is not the
    most numerous bin.  Use fine bins (n_bins=50) so phase 0 is isolated."""
    meta = load_test_meta(test_xray_data_root)
    splitter = PhaseStratifiedSpliter(train_ratio=0.8, seed=0, n_bins=50)
    splits = splitter.split(test_xray_data_root, meta)

    phases = meta.phase_array
    # ground truth phase‑0 indices
    phase0_indices = set(np.where(phases == 0)[0].tolist())
    train_set = set(splits["train"])

    n0 = len(phase0_indices)
    # find max bin size to decide if phase 0 is dominant
    bin_edges = np.linspace(0, 1, 51)
    bin_ids = np.clip(np.digitize(phases, bin_edges) - 1, 0, 49)
    max_other = max(
        (bin_ids == b).sum()
        for b in range(50)
        if b != 0 and (bin_ids == b).sum() > 0
    )

    if n0 > 0 and n0 <= max_other:
        # phase 0 is NOT dominant → must be absent from training
        assert train_set.isdisjoint(phase0_indices), (
            f"phase‑0 frames {phase0_indices} found in training but phase 0 "
            f"count={n0} ≤ max other bin count={max_other}"
        )


def test_phase_stratified_splitter_uses_phase0_when_dominant(
    test_xray_data_no_flow_root: Path,
):
    """When phase 0 has strictly more frames than any other bin, it SHOULD
    appear in training.  Diseased_17 is known to have many phase‑0 frames."""
    meta = load_test_meta(test_xray_data_no_flow_root)
    phases = meta.phase_array
    phase0_indices = set(np.where(phases == 0)[0].tolist())
    n0 = len(phase0_indices)

    if n0 == 0:
        pytest.skip("No phase‑0 frames in this dataset")

    # verify phase 0 IS the dominant bin with coarse bins
    bin_edges = np.linspace(0, 1, 21)
    bin_ids = np.clip(np.digitize(phases, bin_edges) - 1, 0, 19)
    max_other = max(
        (bin_ids == b).sum()
        for b in range(20)
        if b != 0 and (bin_ids == b).sum() > 0
    )

    if n0 <= max_other:
        pytest.skip("Phase 0 is not dominant in this dataset, cannot test the dominant case")

    splitter = PhaseStratifiedSpliter(train_ratio=0.8, seed=0, n_bins=20)
    splits = splitter.split(test_xray_data_no_flow_root, meta)
    train_set = set(splits["train"])

    assert not train_set.isdisjoint(phase0_indices), (
        f"phase‑0 is dominant (n0={n0} > others≤{max_other}) but none selected in training"
    )


def test_phase_stratified_splitter_deterministic(test_xray_data_root: Path):
    """Same seed + same data_dir → identical splits."""
    meta = load_test_meta(test_xray_data_root)
    splitter = PhaseStratifiedSpliter(train_ratio=0.7, seed=42, n_bins=10)
    a = splitter.split(test_xray_data_root, meta)
    b = splitter.split(test_xray_data_root, meta)
    assert a == b


def test_phase_stratified_splitter_different_seed_different_split(test_xray_data_root: Path):
    """Different seed should (very likely) produce a different split."""
    meta = load_test_meta(test_xray_data_root)
    s0 = PhaseStratifiedSpliter(train_ratio=0.8, seed=0, n_bins=20)
    s1 = PhaseStratifiedSpliter(train_ratio=0.8, seed=1, n_bins=20)
    a = s0.split(test_xray_data_root, meta)
    b = s1.split(test_xray_data_root, meta)
    assert a != b, "different seeds produced identical splits — unlikely but possible"


def test_phase_stratified_splitter_covers_all_frames_with_enough_bins(test_xray_data_root: Path):
    """With many small bins, stratification still covers all frames."""
    meta = load_test_meta(test_xray_data_root)
    splitter = PhaseStratifiedSpliter(train_ratio=0.5, seed=7, n_bins=100)
    splits = splitter.split(test_xray_data_root, meta)
    _check_split_validity(splits, meta.num_frames)


# ───────── FileSpliter ─────────


def test_file_splitter_loads_from_relative_path(test_xray_data_root: Path, tmp_path: Path):
    """Load split from a JSON file stored next to the data."""
    meta = load_test_meta(test_xray_data_root)
    n = meta.num_frames

    dummy_splits = {
        "train": [0, 2, 4],
        "val": [1, 3],
        "test": [1, 3],
    }
    split_path = tmp_path / "my_splits.json"
    with open(split_path, "w") as f:
        json.dump(dummy_splits, f)

    # By default FileSpliter uses `split_file_path` relative to data_dir,
    # but we can monkey-patch by passing absolute path in init_args.
    # Here we mimic a relative path by pointing directly at tmp_path.
    # In practice the file lives under data_dir (e.g. data/Diseased_17/splits.json)
    splitter = FileSpliter(split_file_path=str(split_path))
    result = splitter.split(tmp_path, meta)  # data_dir doesn't matter for absolute path
    assert result == dummy_splits


def test_file_splitter_reads_absolute_path(test_xray_data_root: Path, tmp_path: Path):
    """Absolute split_file_path is used as-is regardless of data_dir."""
    meta = load_test_meta(test_xray_data_root)
    splits = {"train": list(range(5)), "val": list(range(5, 10)), "test": list(range(5, 10))}
    split_path = tmp_path / "abs_splits.json"
    with open(split_path, "w") as f:
        json.dump(splits, f)

    result = FileSpliter(split_file_path=str(split_path)).split(test_xray_data_root, meta)
    assert result == splits


def test_file_splitter_missing_file_raises(test_xray_data_root: Path):
    """Non‑existent file should raise FileNotFoundError."""
    meta = load_test_meta(test_xray_data_root)
    splitter = FileSpliter(split_file_path="nonexistent_splits.json")
    with pytest.raises(FileNotFoundError):
        splitter.split(test_xray_data_root, meta)


def test_file_splitter_loads_with_relative_to_data_dir(test_xray_data_root: Path, tmp_path: Path):
    """When split_file_path is relative, it resolves relative to data_dir."""
    meta = load_test_meta(test_xray_data_root)
    dummy_splits = {
        "train": [0, 1, 2],
        "val": [3, 4],
        "test": [3, 4],
    }
    split_path = test_xray_data_root / "splits.json"
    with open(split_path, "w") as f:
        json.dump(dummy_splits, f)

    try:
        result = FileSpliter(split_file_path="splits.json").split(test_xray_data_root, meta)
        assert result == dummy_splits
    finally:
        split_path.unlink(missing_ok=True)