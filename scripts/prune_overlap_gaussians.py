"""训练后冗余高斯剪枝（后处理诊断）。

贪心策略：按 density 降序遍历，删除"中心被已保留高斯 kσ 椭球覆盖"的冗余点，
使保留集的高斯中心互不 kσ 覆盖。输出剪枝后的 ckpt（所有 shape[0]==N 的 tensor
同步 mask），供后续重新渲染/评估。

用法：
    pixi run python scripts/prune_overlap_gaussians.py \
        --ckpt <in.ckpt> --out_ckpt <out.ckpt> [--sigma 2.0] [--batch 512]
"""
import argparse

import torch
import torch.nn.functional as F

MU_WATER = 0.002


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
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


def greedy_prune(means, scales, rot, density, sigma: float = 2.0, batch: int = 512):
    """Greedy coverage pruning. Returns keep indices (int64)."""
    device = means.device
    n = means.shape[0]
    density1 = density.squeeze(-1)
    order = torch.argsort(density1, descending=True)

    R = quat_to_matrix(rot)
    s = scales

    kept = []  # list of kept global indices (density-desc order)

    thr2 = sigma * sigma
    for start in range(0, n, batch):
        b_idx = order[start:start + batch]  # [B]
        b_means = means[b_idx]
        if kept:
            k_idx = torch.tensor(kept, device=device)  # [K]
            k_means = means[k_idx]
            k_R = R[k_idx]
            k_s = s[k_idx]
            # diff[b, k] = mu_b - mu_k ; y = R_k^T diff ; d2 = ||y/s_k||^2
            diff = b_means[:, None, :] - k_means[None, :, :]  # [B, K, 3]
            y = torch.einsum("bkd,kde->bke", diff, k_R)  # [B, K, 3]
            d2 = ((y / k_s) ** 2).sum(-1)  # [B, K]
            covered = (d2 < thr2).any(dim=-1)  # [B]
            keep = ~covered
            kept.extend(b_idx[keep].tolist())
        else:
            kept.extend(b_idx.tolist())

    keep_idx = torch.tensor(kept, device=device)
    return keep_idx


def main(args):
    ckpt = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    sd = ckpt["state_dict"]
    device = "cuda"

    means = sd["gaussian_model.gaussians.means"].float()
    scales = F.softplus(sd["gaussian_model.gaussians.scales"].float(), threshold=10.0)
    rot = F.normalize(sd["gaussian_model.gaussians.rotations"].float(), dim=-1)
    density = MU_WATER * F.softplus(sd["gaussian_model.gaussians.density"].float(), threshold=10.0)

    n = means.shape[0]
    keep = greedy_prune(means, scales, rot, density, sigma=args.sigma, batch=args.batch)
    n_keep = keep.shape[0]
    print(f"prune: {n} -> {n_keep} gaussians (removed {n - n_keep}, {100.0 * (n - n_keep) / n:.1f}%)")

    # apply mask to every tensor whose first dim == n
    for k, v in list(sd.items()):
        if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n:
            sd[k] = v[keep.cpu()]

    out_ckpt = args.out_ckpt
    ckpt["state_dict"] = sd
    torch.save(ckpt, out_ckpt)
    print(f"saved pruned ckpt -> {out_ckpt}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_ckpt", type=str, required=True)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--batch", type=int, default=512)
    main(p.parse_args())
