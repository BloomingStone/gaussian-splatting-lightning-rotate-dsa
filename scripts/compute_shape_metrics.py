"""统一 3D 评估脚本：对已导出的密度体积一次性输出全部分割/形状/拓扑指标。

针对冠状动脉这类稀疏细管结构，Dice 对全局刚体偏移过度敏感（形状正确但
位置偏 1mm 就会暴跌）。本脚本把两类指标合并到一个入口：

  * 分割 + 密度指标（复用 internal/metrics/metric_3d_utils.SegmentationMetricsComputer）：
      dice / precision / recall / hd95 / cldice / dist_avg_cl / hd95_cl
      soft_dice / roc_auc / pr_auc
  * 形状指标（本脚本实现）：
      NSD_tau —— 带容差的表面 Dice，边界体素在 tau mm 内就算命中，对轻微错位宽容；
      ASSD    —— 平均对称表面距离，比 HD95 更鲁棒的"整体贴合"度量。
  * 中心线拓扑（本脚本实现）：
      骨架图的分支数 / 分叉点数 / 分支长度分布(Wasserstein)，
      对全局刚体偏移完全不敏感，专门衡量"血管树结构对不对"。

主参考指标（sweep 排序建议）：NSD1(τ=1mm)、ASSD、dist_avg_cl、cldice、
n_br / br_len_mean（拓扑）。
诊断/文献指标：dice、precision/recall、roc_auc、pr_auc、W1_len。

用法：
    pixi run python scripts/compute_shape_metrics.py \
        --gt data/example_grids/coronary_label.nii.gz \
        --vols spike:outputs/dev_grid_views/stage1-spike/vol3d/density_phase0.000.nii.gz \
               scale-1.0:outputs/dev_grid_views/hashgrid-adaptive-smooth/hs-depth-scale-1.0/vol3d/density_phase0.000.nii.gz \
        --thr 0.0344 --tau 1.0 2.0
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from scipy.spatial import KDTree
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.morphology import skeletonize

from internal.metrics.metric_3d_utils import SegmentationMetricsComputer


def compute_aabb_mask(label: np.ndarray) -> np.ndarray:
    """Axis-aligned bounding box boolean mask of *label* (replicates meta.py)."""
    mask = label > 0
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    coords = np.argwhere(mask)
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    aabb = np.zeros_like(mask, dtype=bool)
    aabb[mn[0]:mx[0] + 1, mn[1]:mx[1] + 1, mn[2]:mx[2] + 1] = True
    return aabb


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Surface (boundary) voxels of a binary mask."""
    return mask & ~binary_erosion(mask)


def normalized_surface_dice(pred: np.ndarray, gt: np.ndarray,
                            spacing: tuple[float, float, float],
                            tau_mm: float) -> float:
    """NSD: |S_p ∩ B_g(tau)| + |S_g ∩ B_p(tau)| over |S_p| + |S_g|."""
    sp = surface_voxels(pred)
    sg = surface_voxels(gt)
    n_sp, n_sg = int(sp.sum()), int(sg.sum())
    if n_sp == 0 or n_sg == 0:
        return 0.0
    # anisotropic footprint: tau in voxels per axis
    tau_vox = np.ceil(tau_mm / np.abs(np.asarray(spacing))).astype(int)
    footprint = np.ones((2 * tau_vox[0] + 1, 2 * tau_vox[1] + 1, 2 * tau_vox[2] + 1), dtype=bool)
    bg_tau = binary_dilation(sg, structure=footprint)
    bp_tau = binary_dilation(sp, structure=footprint)
    hit_p = int((sp & bg_tau).sum())
    hit_g = int((sg & bp_tau).sum())
    return (hit_p + hit_g) / (n_sp + n_sg)


def average_symmetric_surface_distance(pred: np.ndarray, gt: np.ndarray,
                                       spacing) -> float:
    """ASSD in mm using surface voxel KD-trees."""
    sp = np.argwhere(surface_voxels(pred)).astype(np.float64) * np.asarray(spacing)
    sg = np.argwhere(surface_voxels(gt)).astype(np.float64) * np.asarray(spacing)
    if len(sp) == 0 or len(sg) == 0:
        return float("inf")
    tree_p, tree_g = KDTree(sp), KDTree(sg)
    d_p2g = tree_g.query(sp, k=1)[0]
    d_g2p = tree_p.query(sg, k=1)[0]
    return float((d_p2g.mean() + d_g2p.mean()) / 2.0)


def skeleton_graph_stats(skel: np.ndarray, spacing) -> dict:
    """Decompose a *skeleton* volume into branches and report topological stats.

    Returns counts (endpoints, junctions, branches) and per-branch lengths (mm).
    """
    idx = np.argwhere(skel)  # (N, 3) voxel coords
    n = len(idx)
    if n == 0:
        return {"n_endpoints": 0, "n_junctions": 0, "n_branches": 0,
                "branch_len_mean": 0.0, "branch_lens": np.array([])}

    coord_set = {tuple(c) for c in idx}
    # 26-neighbour connectivity
    offsets = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1)
                        for c in (-1, 0, 1) if not (a == 0 and b == 0 and c == 0)])
    neigh_count = {}
    for c in coord_set:
        cnt = 0
        for o in offsets:
            if (c[0] + o[0], c[1] + o[1], c[2] + o[2]) in coord_set:
                cnt += 1
        neigh_count[c] = cnt

    endpoints = {c for c, k in neigh_count.items() if k == 1}
    junctions = {c for c, k in neigh_count.items() if k >= 3}

    # walk branches: start from every endpoint/junction, follow degree-2 nodes
    spacing_arr = np.abs(np.asarray(spacing, dtype=np.float64))
    visited = set()
    branch_lens = []

    def walk(start):
        """Walk one branch from *start* to the next endpoint/junction."""
        prev = start
        cur = start
        steps = 0
        while True:
            # find next unvisited neighbour
            nxt = None
            for o in offsets:
                nb = (cur[0] + o[0], cur[1] + o[1], cur[2] + o[2])
                if nb in coord_set and nb != prev:
                    nxt = nb
                    break
            if nxt is None:
                break
            prev, cur = cur, nxt
            steps += 1
            if cur in endpoints or cur in junctions:
                break
        return steps

    branch_lens = []
    for s in endpoints | junctions:
        for o in offsets:
            nb = (s[0] + o[0], s[1] + o[1], s[2] + o[2])
            if nb in coord_set and nb not in visited:
                # only start walking from a seed if the seed is an endpoint or junction
                pass
        # simpler: for each endpoint/junction, walk along each incident degree-2 chain
        prev = s
        for o in offsets:
            nb = (s[0] + o[0], s[1] + o[1], s[2] + o[2])
            if nb not in coord_set:
                continue
            if nb in endpoints or nb in junctions:
                continue  # direct junction-to-junction edge, count separately below
            # walk the degree-2 chain
            cur = nb
            prev = s
            steps = 1
            while True:
                nxt = None
                for oo in offsets:
                    nn = (cur[0] + oo[0], cur[1] + oo[1], cur[2] + oo[2])
                    if nn in coord_set and nn != prev:
                        nxt = nn
                        break
                if nxt is None:
                    break
                prev, cur = cur, nxt
                steps += 1
                if cur in endpoints or cur in junctions:
                    break
            branch_lens.append(steps)
            visited.add(cur)

    # handle direct junction-junction / junction-endpoint edges (degree-2 path length 0)
    # (rare; skip for robustness)

    lens_mm = np.array(branch_lens, dtype=np.float64)
    # convert voxel steps to mm using average spacing along the path is approximate;
    # use L1 weighted by per-axis spacing: approximate each step ~ mean spacing
    step_mm = float(spacing_arr.mean())
    lens_mm = lens_mm * step_mm

    return {
        "n_endpoints": len(endpoints),
        "n_junctions": len(junctions),
        "n_branches": len(branch_lens),
        "branch_len_mean": float(lens_mm.mean()) if len(lens_mm) else 0.0,
        "branch_lens": lens_mm,
    }


def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-1 distance between two 1-D samples (sorted L1 mean)."""
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    a = np.sort(a)
    b = np.sort(b)
    # interpolate onto the same grid via quantiles (equal-length sampling)
    n = max(len(a), len(b))
    q = np.linspace(0, 1, n)
    qa = np.quantile(a, q)
    qb = np.quantile(b, q)
    return float(np.abs(qa - qb).mean())


def _fmt(v, w, prec=3):
    """Format a metric cell; handles inf."""
    if isinstance(v, float) and np.isinf(v):
        return f"{'inf':>{w}}"
    return f"{v:>{w}.{prec}f}"


def _clean(x):
    """JSON-friendly float; inf/nan -> None."""
    if isinstance(x, float) and (np.isinf(x) or np.isnan(x)):
        return None
    return float(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", type=str, required=True)
    p.add_argument("--vols", nargs="+", required=True, help="name:path pairs")
    p.add_argument("--thr", type=float, nargs="+", default=[0.0344],
                   help="binarisation thresholds (default 0.0344)")
    p.add_argument("--tau", type=float, nargs="+", default=[1.0, 2.0],
                   help="NSD tolerance in mm")
    p.add_argument("--json", type=str, default=None,
                   help="optional path to dump all metrics as JSON")
    args = p.parse_args()

    gt_img = nib.load(args.gt)
    gt = (np.asarray(gt_img.get_fdata()) > 0).astype(np.uint8)
    spacing = tuple(np.abs(np.diag(gt_img.affine)[:3]))
    aabb = compute_aabb_mask(gt)

    # crop to the label bounding box (aabb is the filled bbox) to cut the
    # volume ~10x for the expensive soft-clDice / skeleton ops.  Metrics are
    # invariant to the zero background outside the bbox.
    coords = np.argwhere(aabb)
    mn, mx = coords.min(axis=0), coords.max(axis=0)
    bbox_slice = tuple(slice(mn[i], mx[i] + 1) for i in range(3))
    gt_c = gt[bbox_slice]

    specs = [s.split(":", 1) for s in args.vols]

    # GT skeleton computed once, reused for topology stats.
    gt_skel = skeletonize(gt_c, method="lee")
    gt_stats = skeleton_graph_stats(gt_skel, spacing)

    # Unified metric computer (segmentation + density), reusing the internal
    # MONAI/torch logic in metric_3d_utils so this script is the single eval
    # entry point and numbers stay consistent with training-time 3D metrics
    # (dice/cldice/dist_avg_cl/roc_auc/...).
    computer = SegmentationMetricsComputer(
        gt_c, np.ones_like(gt_c, dtype=bool), spacing
    )

    print(f"GT: n_endpoints={gt_stats['n_endpoints']} n_junctions={gt_stats['n_junctions']} "
          f"n_branches={gt_stats['n_branches']} branch_len_mean={gt_stats['branch_len_mean']:.2f}mm")
    print(f"spacing={tuple(round(float(s), 4) for s in spacing)}  bbox={tuple(mx - mn + 1)}")
    print()

    all_results = {}
    for thr in args.thr:
        print(f"===== threshold {thr} =====")
        header = (f"{'name':<16} " + " ".join(f"{'NSD'+str(t).rstrip('0').rstrip('.'):>6}" for t in args.tau)
                  + f" {'ASSD':>6} {'dist_cl':>8} {'cldice':>7} {'dice':>6} {'prec':>6} {'rec':>6} {'hd95':>6}"
                  + f" {'roc_auc':>7} {'pr_auc':>7} {'n_jct':>6} {'n_br':>6} {'br_len':>7} {'W1_len':>7}")
        print(header)
        thr_results = []
        for name, path in specs:
            vol = np.asarray(nib.load(path).get_fdata(), dtype=np.float32)
            pred = ((vol > thr) & aabb).astype(np.uint8)
            pred_c = pred[bbox_slice]
            vol_c = vol[bbox_slice]

            # segmentation + density metrics (reuse internal SegmentationMetricsComputer)
            seg = computer.compute(torch.from_numpy(pred_c.astype(bool)))
            dens = computer.compute_density(torch.from_numpy(vol_c))

            # shape metrics (NSD / ASSD)
            nsds = [normalized_surface_dice(pred_c, gt_c, spacing, t) for t in args.tau]
            assd = average_symmetric_surface_distance(pred_c, gt_c, spacing)

            # topology metrics (prediction skeleton)
            pred_sk = skeletonize(pred_c, method="lee")
            p_stats = skeleton_graph_stats(pred_sk, spacing)
            w1_len = wasserstein1(p_stats["branch_lens"], gt_stats["branch_lens"])

            # record all metrics (JSON-friendly)
            m = {
                "name": name,
                "dice": _clean(seg['dice']),
                "precision": _clean(seg['precision']),
                "recall": _clean(seg['recall']),
                "hd95": _clean(seg['hd95']),
                "cldice": _clean(seg['cldice']),
                "dist_avg_cl": _clean(seg['dist_avg_cl']),
                "soft_dice": _clean(dens['soft_dice']),
                "roc_auc": _clean(dens['roc_auc']),
                "pr_auc": _clean(dens['pr_auc']),
                "assd": _clean(assd),
                "n_junctions": int(p_stats['n_junctions']),
                "n_branches": int(p_stats['n_branches']),
                "branch_len_mean": float(p_stats['branch_len_mean']),
                "w1_len": _clean(w1_len),
            }
            for i, t in enumerate(args.tau):
                m[f"NSD{str(t).rstrip('0').rstrip('.')}"] = float(nsds[i])
            thr_results.append(m)

            cells = [f"{name:<16}"]
            cells += [f"{x:>6.3f}" for x in nsds]
            cells += [_fmt(assd, 6, 2), _fmt(seg['dist_avg_cl'], 8, 3), f"{seg['cldice']:>7.3f}"]
            cells += [f"{seg['dice']:>6.3f}", f"{seg['precision']:>6.3f}", f"{seg['recall']:>6.3f}",
                      _fmt(seg['hd95'], 6, 2)]
            cells += [f"{dens['roc_auc']:>7.3f}", f"{dens['pr_auc']:>7.3f}"]
            cells += [f"{p_stats['n_junctions']:>6}", f"{p_stats['n_branches']:>6}",
                      f"{p_stats['branch_len_mean']:>7.2f}", f"{w1_len:>7.2f}"]
            print("".join(cells))
        all_results[str(thr)] = thr_results
        print()

    if args.json:
        import json
        payload = {
            "gt": {
                "n_endpoints": int(gt_stats['n_endpoints']),
                "n_junctions": int(gt_stats['n_junctions']),
                "n_branches": int(gt_stats['n_branches']),
                "branch_len_mean": float(gt_stats['branch_len_mean']),
            },
            "spacing": [float(s) for s in spacing],
            "results": all_results,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"saved -> {args.json}")


if __name__ == "__main__":
    main()
