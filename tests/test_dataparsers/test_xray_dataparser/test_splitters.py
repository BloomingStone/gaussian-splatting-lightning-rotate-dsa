from pathlib import Path

import numpy as np
import pytest

from internal.dataparsers.xray_dataparser.splitters import ReconstructionSpliter, RenderNewViewsSpliter

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