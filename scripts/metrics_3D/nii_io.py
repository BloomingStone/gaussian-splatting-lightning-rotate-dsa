import sys
import nibabel as nib
from pathlib import Path
from concurrent.futures import Future, ProcessPoolExecutor
import logging
from typing import cast
from dataclasses import dataclass

import numpy as np
import cupy as cp

sys.path.append(str(Path(__file__).parent))  # Adjust as needed to import common types

from metric import Array3D, Spacing3D

logger = logging.getLogger(__name__)

@dataclass
class Pair:
    pred_path: Path
    gt_path: Path

@dataclass
class LoadResCPU:
    pred: np.ndarray
    gt: np.ndarray
    spacing: Spacing3D
    affine: np.ndarray

@dataclass
class LoadRes:
    pred: Array3D
    gt: Array3D
    spacing: Spacing3D
    affine: np.ndarray


def _load_nifti_cpu(path: Path) -> tuple[np.ndarray, Spacing3D, np.ndarray]:
    img = nib.load(str(path))
    assert isinstance(img, nib.Nifti1Image), f"Unsupported NIfTI format: {type(img)}"
    data = img.get_fdata(dtype=np.float32)
    spacing = cast(Spacing3D, tuple(float(v) for v in img.header.get_zooms()[:3]))
    affine = np.asarray(img.affine)
    assert affine is not None, "Affine is required to determine spacing and orientation."
    return data, spacing, affine

def _load_both_cpu(pred_path: Path, gt_path: Path) -> LoadResCPU:
    pred, spacing, affine = _load_nifti_cpu(pred_path)
    gt, spacing_, affine_ = _load_nifti_cpu(gt_path)
    assert np.allclose(spacing, spacing_), "Pred and GT spacing must match."
    assert np.allclose(affine, affine_), "Pred and GT affine must match."
    return LoadResCPU(pred, gt, spacing, affine)

def _save_nifti_cpu(data: np.ndarray, affine: np.ndarray, path: Path) -> str:
    img = nib.Nifti1Image(np.asarray(data, dtype=np.uint8), affine)
    nib.save(img, str(path))
    return str(path)


class NiiLoader:
    def __init__(
        self,
        pred_root: Path,
        gt_root: Path,
        *,
        num_workers: int = 2,
        prefetch_size: int = 4,
        val_phase_in_100: int = 0,     # if 50, phase = 50 / 100 = 0.5
        random_chosen: int|None = None,
    ):
        self.pred_root = pred_root
        self.gt_root = gt_root
        pred_files = {p.parents[1].name: p for p in pred_root.rglob("*.nii*") if "dxyz" not in p.name}
        
        if val_phase_in_100 == 0:
            gt_files = {p.parents[0].name: p for p in gt_root.rglob("*.nii*") if "LCA_label" in p.name or "RCA_label" in p.name}
        elif val_phase_in_100 > 0 and val_phase_in_100 < 100:
            gt_files = {p.parents[1].name: p for p in gt_root.rglob("*.nii*") if str(val_phase_in_100) in p.name}
        else:
            raise ValueError("val_phase_in_100 must be in the range [0, 100].")
        
        logger.info(f"Found {len(pred_files)} prediction files and {len(gt_files)} GT files.")
        logger.debug(f"First 5 pred files: {"\t".join(map(str, list(pred_files.values())[:5]))}")
        logger.debug(f"First 5 GT files: {"\t".join(map(str, list(gt_files.values())[:5]))}")

        assert pred_files.keys() == gt_files.keys(), "Case IDs in pred and GT directories must match."
        self.files = {k: Pair(pred_files[k], gt_files[k]) for k in sorted(pred_files.keys())}
        self.case_ids = list(self.files.keys())
        
        if random_chosen is not None and 0 < random_chosen < len(self.files):
            import random
            random.seed(42)  # For reproducibility
            chosen_ids = random.sample(self.case_ids, random_chosen)
            self.files = {k: self.files[k] for k in chosen_ids}
            self.case_ids = list(self.files.keys())
        

        self.num_workers = max(1, int(num_workers))
        self.prefetch_size = max(1, int(prefetch_size))

        self._executor: ProcessPoolExecutor|None = None
        self._futures: dict[int, Future[LoadResCPU]] = {}
        self._next_submit: int = 0
        self._next_yield: int = 0

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> "NiiLoader":
        self.reset()
        return self

    def __next__(self) -> tuple[str, LoadRes]:
        if self._next_yield >= len(self):
            self.close()
            raise StopIteration

        self._submit_prefetch_jobs()
        fut = self._futures.pop(self._next_yield)
        data = fut.result()

        case_id = self.case_ids[self._next_yield]

        self._next_yield += 1
        self._submit_prefetch_jobs()

        # Keep CPU prefetch and move to GPU only when consumed.
        pred_cp = cp.asarray(data.pred)
        gt_cp = cp.asarray(data.gt)
        return case_id, LoadRes(pred_cp, gt_cp, data.spacing, data.affine)

    def reset(self) -> None:
        self.close()
        self._executor = ProcessPoolExecutor(max_workers=self.num_workers)
        self._futures = {}
        self._next_submit = 0
        self._next_yield = 0
        self._submit_prefetch_jobs()

    def close(self) -> None:
        for fut in self._futures.values():
            fut.cancel()
        self._futures.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _submit_prefetch_jobs(self) -> None:
        if self._executor is None:
            return
        while (
            self._next_submit < len(self)
            and len(self._futures) < self.prefetch_size
        ):
            idx = self._next_submit
            pred_path = self.files[self.case_ids[idx]].pred_path
            gt_path = self.files[self.case_ids[idx]].gt_path
            self._futures[idx] = self._executor.submit(_load_both_cpu, pred_path, gt_path)
            self._next_submit += 1
    
    @property
    def case_id(self) -> str:
        if self._next_yield < len(self):
            return self.case_ids[self._next_yield]
        raise IndexError("No more cases to yield.")
    
    @staticmethod
    def _load_nifti(path: Path) -> tuple[np.ndarray, Spacing3D, np.ndarray]:
        return _load_nifti_cpu(path)
    
    @staticmethod
    def _load_both(pred_path: Path, gt_path: Path) -> LoadResCPU:
        return _load_both_cpu(pred_path, gt_path)
    
    @staticmethod
    def _load_both_cp(pred_path: Path, gt_path: Path) -> LoadRes:
        load_res = NiiLoader._load_both(pred_path, gt_path)
        pred_cp = cp.asarray(load_res.pred)
        gt_cp = cp.asarray(load_res.gt)
        return LoadRes(pred_cp, gt_cp, load_res.spacing, load_res.affine)

    def __enter__(self) -> "NiiLoader":
        self.reset()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

class NiiSaver:
    def __init__(self, max_workers: int = 1):
        self.max_workers = max(1, int(max_workers))
        self._executor: ProcessPoolExecutor|None = None
        self._futures: list[Future[str]] = []

    def start(self) -> None:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)

    def submit(self, data: Array3D, affine: np.ndarray, path: Path) -> Future[str]:
        self.start()
        assert self._executor is not None
        payload = cp.asnumpy(data)
        fut = self._executor.submit(_save_nifti_cpu, payload, np.asarray(affine), path)
        self._futures.append(fut)
        return fut

    def close(self) -> None:
        for fut in self._futures:
            fut.cancel()
        self._futures.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "NiiSaver":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def save_nifti(data: Array3D, affine: np.ndarray, path: Path) -> None:
        _save_nifti_cpu(cp.asnumpy(data), np.asarray(affine), path)