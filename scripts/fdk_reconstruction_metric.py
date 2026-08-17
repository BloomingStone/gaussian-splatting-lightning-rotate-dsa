#!/usr/bin/env python
"""FDK (Feldkamp-Davis-Kress) reconstruction + 3D metric computation for X-ray DSA data.

Usage:
    # Sweep threshold on one static case to find optimal threshold:
    pixi run python scripts/fdk_reconstruction_metric.py sweep \\
        --data data/gen_4d_output_all/static/asoca-diseased__Diseased_02__LCA \\
        --output outputs/fdk_metric/threshold_sweep.csv

    # Run FDK reconstruction + metrics on all flow cases with a given threshold:
    pixi run python scripts/fdk_reconstruction_metric.py run \\
        --data-dir data/gen_4d_output_all/flow \\
        --threshold 0.0344 \\
        --output outputs/fdk_metric/results.csv \\
        --workers 4

    # Or combined: sweep from static then evaluate on flow:
    pixi run python scripts/fdk_reconstruction_metric.py all \\
        --static-case data/gen_4d_output_all/static/asoca-diseased__Diseased_02__LCA \\
        --flow-dir data/gen_4d_output_all/flow \\
        --output-dir outputs/fdk_metric
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from internal.dataparsers.xray_dataparser.meta import (
    XRayMetaLoader,
)
from internal.dataparsers.xray_dataparser.conebeam import (
    ConeBeamParams,
    ConeBeamProjector,
    PngOdlTransform,
)
from internal.metrics.metric_3d_utils import SegmentationMetricsComputer


# ---------------------------------------------------------------------------
# FDK reconstruction helpers
# ---------------------------------------------------------------------------

def load_png_projections(image_dir: Path, indices: list[int]) -> np.ndarray:
    """Load PNG projections for given frame indices.

    Returns:
        (N, H, W) float32 array in [0, 1].
    """
    from PIL import Image

    image_paths = sorted(image_dir.glob("*.png"))
    if len(image_paths) == 0:
        raise ValueError(f"No PNG files found in {image_dir}")

    projections = []
    for i in indices:
        img = np.asarray(Image.open(image_paths[i]).convert("L"), dtype=np.float32)
        img /= 255.0
        projections.append(img)
    return np.stack(projections, axis=0)


def preprocess_indices_alphas(
    meta: XRayMeta,
    phase_min: float = 0.0,
    phase_max: float = 0.5,
    use_all_frames: bool = False,
) -> tuple[list[int], list[float]]:
    """Select & sort frames by phase range, return indices and alpha angles in radians.

    When *use_all_frames* is True, all frames are used regardless of phase.

    .. note::

       Alpha angles are negated (DSA convention) and shifted by +π to align
       with ODL's cone-beam geometry (source-detector vs camera definition).
    """
    n = meta.num_frames
    indices_all = list(range(n))
    alphas_all = meta.alphas_radians.tolist()

    # DSA: alpha rotates from Anterior(+Y) to Right(+X) = negative Z rotation.
    # -a corrects rotation direction; +π aligns ODL source-detector convention.
    alphas_all = [-a + np.pi for a in alphas_all]
    phases_all = meta.phase_array.tolist()

    if use_all_frames:
        data = list(zip(indices_all, alphas_all, phases_all))
    else:
        y0, y1 = 1 - phase_min, 1 - phase_max
        phase_min_sym, phase_max_sym = min(y0, y1), max(y0, y1)

        data = [
            (i, a, p) for i, a, p in zip(indices_all, alphas_all, phases_all)
            if (phase_min - 1e-8 <= p <= phase_max + 1e-8)
            or (phase_min_sym - 1e-8 <= p <= phase_max_sym + 1e-8)
        ]

    if not data:
        raise ValueError(
            f"No frames selected (use_all_frames={use_all_frames}, "
            f"phase=[{phase_min},{phase_max}])"
        )

    data.sort(key=lambda x: float(x[1]))
    indices, alphas, _ = zip(*data)
    return list(indices), list(alphas)


def fdk_reconstruct(
    data_dir: Path,
    meta: XRayMeta,
    indices: list[int],
    alphas: list[float],
    use_filter: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Run FDK (FBP) reconstruction on selected frames.

    Returns:
        (D, H, W) float32 volume.
    """
    if verbose:
        print(f"  Loading {len(indices)} projections ...", flush=True)
    image_dir = data_dir / "rotate_dsa"
    projections = load_png_projections(image_dir, indices)
    if verbose:
        print(f"  Projections shape: {projections.shape}", flush=True)

    geom = meta.c_arm_geometry
    if verbose:
        print(f"  Building cone-beam geometry (volume={tuple(meta.volume_size)}, "
              f"detector={projections.shape[1:]}, alphas={len(alphas)}) ...", flush=True)
    projector = ConeBeamProjector(
        param=ConeBeamParams.init_from(
            shape=tuple(meta.volume_size),
            affine=meta.centering_affine,
            alphas=np.asarray(alphas, dtype=np.float64),
            proj_size=projections.shape[1:],
            dh=geom.dely,
            dw=geom.delx,
            dde=geom.sdd - geom.sod,
            dso=geom.sod,
        ),
        img_transform=PngOdlTransform(),
    )

    if verbose:
        print(f"  Running {'FBP' if use_filter else 'adjoint'} reconstruction ...", flush=True)
    volume = projector.backward_proj(projections, use_filter=use_filter)

    # Note: XY flip ([::-1, ::-1, :]) is handled by ConeBeamProjector
    # with default align_ras=True.  Do NOT flip again here.

    if verbose:
        print(f"  Volume shape: {volume.shape}, range: [{volume.min():.4f}, {volume.max():.4f}]", flush=True)

    # XY flip is now handled by ConeBeamProjector with align_ras=True (default)
    return volume


# ---------------------------------------------------------------------------
# Threshold sweep on static data
# ---------------------------------------------------------------------------

def _fast_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice score for two boolean arrays (fast, no GPU)."""
    intersection = float(np.logical_and(pred, gt).sum())
    p_sum = float(pred.sum())
    g_sum = float(gt.sum())
    if p_sum + g_sum == 0:
        return 1.0
    return 2.0 * intersection / (p_sum + g_sum)


def sweep_threshold(
    data_dir: Path,
    output_csv: str | Path | None = None,
    n_thresholds: int = 100,
    phase_min: float = 0.0,
    phase_max: float = 0.5,
    use_all_frames: bool = False,
    use_filter: bool = True,
) -> list[dict]:
    """Sweep binarisation thresholds, only computing **dice** (fast).

    Returns list of dicts with keys: threshold, dice.
    Best threshold (max dice) is printed at the end.
    """
    meta = XRayMetaLoader().load(data_dir)

    if meta.label_3d_info is None or meta.label_3d_info.data is None:
        raise ValueError(f"No GT label available at {data_dir}")

    label_info = meta.label_3d_info
    gt_label = label_info.data.astype(bool)
    aabb_mask = label_info.aabb.astype(bool)

    print(f"  Selecting frames (phase=[{phase_min}, {phase_max}], use_all={use_all_frames}) ...", flush=True)
    indices, alphas = preprocess_indices_alphas(
        meta, phase_min=phase_min, phase_max=phase_max,
        use_all_frames=use_all_frames,
    )
    print(f"  Selected {len(indices)} frames", flush=True)
    volume = fdk_reconstruct(data_dir, meta, indices, alphas, use_filter=use_filter)

    vmin, vmax = float(volume.min()), float(volume.max())
    thresholds = np.linspace(vmin, vmax, n_thresholds + 2)[1:-1]

    results: list[dict] = []
    for i, thr in enumerate(thresholds):
        pred = (volume > thr) & aabb_mask
        dice = _fast_dice(pred, gt_label)
        results.append({"threshold": float(thr), "dice": dice})
        if (i + 1) % max(1, n_thresholds // 10) == 0 or i == 0:
            print(f"  [{i+1:3d}/{n_thresholds}] thr={thr:.6f}  dice={dice:.4f}", flush=True)

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["threshold", "dice"])
            writer.writeheader()
            writer.writerows(results)

    best_idx = int(np.argmax([r["dice"] for r in results]))
    best = results[best_idx]
    print(f"  Best threshold: {best['threshold']:.6f} (dice={best['dice']:.4f})", flush=True)

    return results


# ---------------------------------------------------------------------------
# Single-case evaluation
# ---------------------------------------------------------------------------

def evaluate_case(
    data_dir: Path,
    threshold: float,
    phase_min: float = 0.0,
    phase_max: float = 0.5,
    use_all_frames: bool = False,
    use_filter: bool = True,
) -> dict:
    """FDK reconstruct *data_dir*, segment at *threshold*, compute metrics.

    Returns dict with keys: case, threshold, dice, precision, recall, hd95,
    cldice, dist_avg_cl, hd95_cl, soft_dice, roc_auc, pr_auc, n_frames_used.
    """
    import torch

    meta = XRayMetaLoader().load(data_dir)
    case_name = data_dir.name

    if meta.label_3d_info is None or meta.label_3d_info.data is None:
        return {"case": case_name, "error": "no GT label"}

    label_info = meta.label_3d_info
    gt_label = label_info.data.astype(bool)
    aabb_mask = label_info.aabb.astype(bool)
    spacing = tuple(np.diag(meta.centering_affine)[:3])

    # Select frames for reconstruction
    print(f"[{case_name}] Selecting frames ...", flush=True)
    indices, alphas = preprocess_indices_alphas(
        meta, phase_min=phase_min, phase_max=phase_max,
        use_all_frames=use_all_frames,
    )

    try:
        volume = fdk_reconstruct(data_dir, meta, indices, alphas, use_filter=use_filter)
    except Exception as e:
        traceback.print_exc()
        return {"case": case_name, "error": f"FDK failed: {e}"}

    n_frames = len(indices)

    # Compute metrics (fast: dice, cldice, roc-auc, pr-auc only)
    computer = SegmentationMetricsComputer(
        gt=gt_label,
        aabb_roi=aabb_mask,
        spacing=spacing,
    )

    pred = ((volume > threshold) & aabb_mask)
    pred_t = torch.from_numpy(pred)

    # dice (fast CPU)
    seg_metrics = {"dice": float(_fast_dice(pred, gt_label))}

    # cldice
    if pred.any() and gt_label.any():
        from monai.losses.cldice import SoftclDiceLoss
        _soft_cldice_fn = SoftclDiceLoss(iter_=3, smooth=1.0)
        pred_bc = pred_t.unsqueeze(0).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt_label).to(device=pred_t.device, dtype=torch.float32)
        gt_bc = gt_t.unsqueeze(0).unsqueeze(0)
        pred_oh = torch.cat([1.0 - pred_bc, pred_bc], dim=1)
        gt_oh = torch.cat([1.0 - gt_bc, gt_bc], dim=1)
        seg_metrics["cldice"] = float((1.0 - _soft_cldice_fn(gt_oh, pred_oh)).item())
    else:
        seg_metrics["cldice"] = 1.0 if (not pred.any() and not gt_label.any()) else 0.0

    # density-based (roc-auc, pr-auc)
    density_metrics = computer.compute_density(torch.from_numpy(volume.astype(np.float32)))

    entry: dict = {
        "case": case_name,
        "threshold": threshold,
        "n_frames_used": n_frames,
    }
    entry.update(seg_metrics)
    entry.update(density_metrics)
    return entry


# ---------------------------------------------------------------------------
# Process all cases under a directory
# ---------------------------------------------------------------------------

def evaluate_all(
    data_dir: Path,
    threshold: float,
    output_csv: str | Path,
    phase_min: float = 0.0,
    phase_max: float = 0.5,
    use_all_frames: bool = False,
    use_filter: bool = True,
    workers: int = 1,
    max_cases: int | None = None,
) -> None:
    """Evaluate FDK + metrics for all subdirectories in *data_dir*."""
    cases = sorted([
        p for p in data_dir.iterdir()
        if p.is_dir() and (p / "rotate_dsa.json").exists()
    ])

    if max_cases is not None:
        cases = cases[:max_cases]

    print(f"Found {len(cases)} cases under {data_dir}")

    results: list[dict] = []
    errors: list[str] = []

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            fut_map = {
                pool.submit(
                    evaluate_case, c, threshold, phase_min, phase_max,
                    use_all_frames, use_filter,
                ): c for c in cases
            }
            for fut in as_completed(fut_map):
                c = fut_map[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {"case": c.name, "error": str(e)}
                    traceback.print_exc()

                if "error" in result:
                    errors.append(f"{c.name}: {result['error']}")
                    print(f"[FAIL] {c.name}: {result['error']}")
                else:
                    results.append(result)
                    print(f"[OK]   {c.name}: dice={result.get('dice', 'N/A'):.4f}")
    else:
        for c in cases:
            try:
                result = evaluate_case(
                    c, threshold, phase_min, phase_max,
                    use_all_frames, use_filter,
                )
            except Exception as e:
                result = {"case": c.name, "error": str(e)}
                traceback.print_exc()

            if "error" in result:
                errors.append(f"{c.name}: {result['error']}")
                print(f"[FAIL] {c.name}: {result['error']}")
            else:
                results.append(result)
                print(f"[OK]   {c.name}: dice={result.get('dice', 'N/A'):.4f}")

    # Write CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # Write error log
    if errors:
        err_path = Path(str(output_csv).replace(".csv", "_errors.txt"))
        err_path.write_text("\n".join(errors))

    # Summary
    print(f"\n{'='*60}")
    print(f"Total cases: {len(cases)}")
    print(f"Success: {len(results)}")
    print(f"Failed:  {len(errors)}")
    if results:
        dices = [r["dice"] for r in results]
        print(f"\nDice stats over {len(results)} cases:")
        print(f"  mean: {np.mean(dices):.4f}")
        print(f"  std:  {np.std(dices):.4f}")
        print(f"  min:  {np.min(dices):.4f}")
        print(f"  max:  {np.max(dices):.4f}")
    print(f"Results saved to: {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FDK reconstruction + 3D segmentation metrics",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- sweep ---
    sweep_p = sub.add_parser("sweep", help="Sweep threshold on one static case")
    sweep_p.add_argument("--data", type=Path, required=True)
    sweep_p.add_argument("--output", type=Path, default="outputs/fdk_metric/threshold_sweep.csv")
    sweep_p.add_argument("--n-thresholds", type=int, default=100)
    sweep_p.add_argument("--phase-min", type=float, default=0.0)
    sweep_p.add_argument("--phase-max", type=float, default=0.5)
    sweep_p.add_argument("--use-all-frames", action="store_true",
                         help="Use all frames regardless of phase")
    sweep_p.add_argument("--no-filter", action="store_true",
                         help="Disable Ram-Lak filter (use adjoint instead of FBP)")

    # --- run ---
    run_p = sub.add_parser("run", help="Run FDK + metrics on multiple cases")
    run_p.add_argument("--data-dir", type=Path, required=True)
    run_p.add_argument("--threshold", type=float, required=True)
    run_p.add_argument("--output", type=Path, required=True)
    run_p.add_argument("--phase-min", type=float, default=0.0)
    run_p.add_argument("--phase-max", type=float, default=0.5)
    run_p.add_argument("--use-all-frames", action="store_true")
    run_p.add_argument("--no-filter", action="store_true")
    run_p.add_argument("--workers", type=int, default=1)
    run_p.add_argument("--max-cases", type=int, default=None)

    # --- all ---
    all_p = sub.add_parser("all", help="Sweep threshold from static, then evaluate on all flow data")
    all_p.add_argument("--static-case", type=Path, required=True)
    all_p.add_argument("--flow-dir", type=Path, required=True)
    all_p.add_argument("--output-dir", type=Path, default="outputs/fdk_metric")
    all_p.add_argument("--n-thresholds", type=int, default=100)
    all_p.add_argument("--phase-min", type=float, default=0.0)
    all_p.add_argument("--phase-max", type=float, default=0.5)
    all_p.add_argument("--use-all-frames", action="store_true")
    all_p.add_argument("--no-filter", action="store_true")
    all_p.add_argument("--workers", type=int, default=1)
    all_p.add_argument("--max-cases", type=int, default=None)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    use_filter = not getattr(args, "no_filter", False)

    if args.cmd == "sweep":
        sweep_threshold(
            data_dir=args.data,
            output_csv=args.output,
            n_thresholds=args.n_thresholds,
            phase_min=args.phase_min,
            phase_max=args.phase_max,
            use_all_frames=args.use_all_frames,
            use_filter=use_filter,
        )

    elif args.cmd == "run":
        evaluate_all(
            data_dir=args.data_dir,
            threshold=args.threshold,
            output_csv=args.output,
            phase_min=args.phase_min,
            phase_max=args.phase_max,
            use_all_frames=args.use_all_frames,
            use_filter=use_filter,
            workers=args.workers,
            max_cases=args.max_cases,
        )

    elif args.cmd == "all":
        # Step 1: sweep on static case
        sweep_csv = args.output_dir / "threshold_sweep.csv"
        print(f"\n{'='*60}")
        print(f"Step 1: Sweeping threshold on {args.static_case}")
        print(f"{'='*60}")
        sweep_results = sweep_threshold(
            data_dir=args.static_case,
            output_csv=sweep_csv,
            n_thresholds=args.n_thresholds,
            phase_min=args.phase_min,
            phase_max=args.phase_max,
            use_all_frames=args.use_all_frames,
            use_filter=use_filter,
        )
        best_threshold = max(sweep_results, key=lambda r: r["dice"])["threshold"]
        print(f"\nBest threshold: {best_threshold:.6f}")

        # Step 2: evaluate on all flow data
        result_csv = args.output_dir / "fdk_metrics.csv"
        print(f"\n{'='*60}")
        print(f"Step 2: Evaluating on {args.flow_dir} with threshold={best_threshold:.6f}")
        print(f"{'='*60}")
        evaluate_all(
            data_dir=args.flow_dir,
            threshold=best_threshold,
            output_csv=result_csv,
            phase_min=args.phase_min,
            phase_max=args.phase_max,
            use_all_frames=args.use_all_frames,
            use_filter=use_filter,
            workers=args.workers,
            max_cases=args.max_cases,
        )

        print(f"\nAll done! Results:")
        print(f"  Threshold sweep: {sweep_csv}")
        print(f"  Evaluation:      {result_csv}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
