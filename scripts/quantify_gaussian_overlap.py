"""量化高斯点云的重叠率、各向异性和 3D 毛刺。

背景：重建结果里冠脉区域像是由多个高斯点互相重叠堆叠出来的，高密度区域是
"点数量多"而非"椭球位置/形状贴合结构"，并伸出"毛刺"。本脚本给出量化指标：

高斯几何层面（ckpt）：
  1. covered_ratio        中心被其他高斯 1σ 椭球覆盖的高斯比例（越高=越堆叠）
  2. covered_ratio_2sig   2σ 覆盖比例
  3. mean_cover_count     每个高斯平均被几个其他高斯覆盖（1σ）
  4. d_nn_sigma_median    最近邻距离 / 自身最大 scale 的中位数（<1=点堆叠）
  5. anisotropy_median    scale max/min 比值中位数（针状毛刺 → 大）

3D 体积层面（可选 --vol）：
  6. spike_volume_ratio   二值化体积做 opening 后的体积损失比例（毛刺占比）

用法：
    pixi run python scripts/quantify_gaussian_overlap.py \
        --ckpt outputs/.../checkpoints/epoch=143-step=20000.ckpt \
        [--vol outputs/.../vol3d/density_phase0.000.nii.gz] [--threshold 0.0344]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy import ndimage

MU_WATER = 0.002  # mm^-1


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """q: [N,4] wxyz -> R: [N,3,3]"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.empty(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def load_gaussians(ckpt_path: str, device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    means = sd["gaussian_model.gaussians.means"].float()
    scales = F.softplus(sd["gaussian_model.gaussians.scales"].float(), threshold=10.0)
    rot = F.normalize(sd["gaussian_model.gaussians.rotations"].float(), dim=-1)
    density = MU_WATER * F.softplus(sd["gaussian_model.gaussians.density"].float(), threshold=10.0)
    return means, scales, rot, density


def compute_overlap(means, scales, rot, k: int = 64):
    """Coverage stats via KNN + Mahalanobis distance into each neighbour's ellipsoid."""
    n = means.shape[0]
    means_np = means.cpu().numpy()
    tree = cKDTree(means_np)
    kk = min(k, n - 1)
    dist, idx = tree.query(means_np, k=kk + 1)  # [N, kk+1], first is self
    idx = idx[:, 1:]  # [N, kk] exclude self
    d_nn = dist[:, 1]  # nearest neighbour distance

    # sigma_max = max scale per gaussian
    sigma_max = scales.max(dim=-1).values.cpu().numpy()
    d_nn_ratio = d_nn / (sigma_max + 1e-12)

    # Mahalanobis distance of each point's centre into each neighbour's ellipsoid
    R = quat_to_matrix(rot)  # [N,3,3]
    s = scales  # [N,3]

    # neighbour indices [N, kk]
    nb_idx = torch.from_numpy(idx).to(means.device)  # [N, kk]
    nb_means = means[nb_idx]  # [N, kk, 3]
    diff = means[:, None, :] - nb_means  # [N, kk, 3] (mu_i - mu_j)
    nb_R = R[nb_idx]  # [N, kk, 3, 3]
    nb_s = s[nb_idx]  # [N, kk, 3]

    # y = R_j^T (mu_i - mu_j) = diff @ R_j
    y = torch.einsum("nkd,nkde->nke", diff, nb_R)  # [N, kk, 3]
    d2 = ((y / nb_s) ** 2).sum(-1)  # [N, kk]  Mahalanobis squared into j's ellipsoid

    covered_1 = (d2 < 1.0).sum(-1).float()  # per point, how many neighbours cover it (1σ)
    covered_2 = (d2 < 4.0).sum(-1).float()  # 2σ

    covered_ratio_1 = float((covered_1 > 0).float().mean())
    covered_ratio_2 = float((covered_2 > 0).float().mean())
    mean_cover = float(covered_1.mean())

    # anisotropy = max/min scale per gaussian
    s_min = scales.min(dim=-1).values
    anisotropy = (scales.max(dim=-1).values / (s_min + 1e-12)).cpu().numpy()

    return {
        "n_gaussians": int(n),
        "covered_ratio_1sig": covered_ratio_1,
        "covered_ratio_2sig": covered_ratio_2,
        "mean_cover_count": mean_cover,
        "d_nn_sigma_median": float(np.median(d_nn_ratio)),
        "d_nn_sigma_mean": float(np.mean(d_nn_ratio)),
        "d_nn_sigma_lt1_ratio": float((d_nn_ratio < 1.0).mean()),
        "anisotropy_median": float(np.median(anisotropy)),
        "anisotropy_mean": float(np.mean(anisotropy)),
        "anisotropy_gt5_ratio": float((anisotropy > 5.0).mean()),
    }


def compute_volume_spike(vol_path: str, threshold: float, open_radius: int = 2):
    import nibabel as nib
    img = nib.load(vol_path)
    vol = np.asarray(img.get_fdata(), dtype=np.float32)
    mask = vol > threshold
    if mask.sum() == 0:
        return {"volume_threshold": threshold, "spike_volume_ratio": float("nan")}
    opened = ndimage.binary_opening(mask, structure=ndimage.generate_binary_structure(3, 1),
                                    iterations=open_radius)
    orig_v = int(mask.sum())
    opened_v = int(opened.sum())
    return {
        "volume_threshold": float(threshold),
        "volume_voxels": orig_v,
        "opened_voxels": opened_v,
        "spike_volume_ratio": float((orig_v - opened_v) / orig_v),
        "open_radius": open_radius,
    }


def main(args):
    means, scales, rot, density = load_gaussians(args.ckpt, device="cuda")
    print(f"n_gaussians = {means.shape[0]}")
    print(f"scale: min {scales.min().item():.4f} max {scales.max().item():.4f} mean {scales.mean().item():.4f}")
    print(f"anisotropy(=max/min scale): median {np.median((scales.max(-1).values/(scales.min(-1).values+1e-12)).cpu().numpy()):.2f}")

    overlap = compute_overlap(means, scales, rot, k=args.k)
    print("\n=== 高斯重叠指标 ===")
    for k_, v in overlap.items():
        print(f"  {k_}: {v:.4f}" if isinstance(v, float) else f"  {k_}: {v}")

    result = {"overlap": overlap}

    if args.vol:
        spike = compute_volume_spike(args.vol, args.threshold, args.open_radius)
        print("\n=== 3D 毛刺指标 ===")
        for k_, v in spike.items():
            print(f"  {k_}: {v}")
        result["volume_spike"] = spike

    if args.out:
        import json
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--vol", type=str, default=None)
    p.add_argument("--threshold", type=float, default=0.0344)
    p.add_argument("--open_radius", type=int, default=2)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--out", type=str, default=None)
    main(p.parse_args())
