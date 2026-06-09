# import os
# import numpy as np
# import nibabel as nib
# from skimage.metrics import structural_similarity as ssim
# from scipy.spatial import KDTree

# # ===============================
# # IO
# # ===============================
# def load_nii(path):
#     """
#     读取 nii.gz 文件
#     """
#     nii = nib.load(path)
#     data = nii.get_fdata().astype(np.float32)
#     return data
# def extract_case_id(pred_name):
#     """
#     从预测文件名中提取 case id
#     例: 10033813_lca-views-2.nii.gz -> 10033813_lca
#     """
#     return pred_name.split("-views-")[0]

# def normalize(volume):
#     """
#     归一化到 [0, 1]，用于 SSIM
#     """
#     v_min, v_max = volume.min(), volume.max()
#     if v_max > v_min:
#         volume = (volume - v_min) / (v_max - v_min)
#     return volume

# # ===============================
# # Metrics
# # ===============================
# def dice_score(pred, gt, eps=1e-6):
#     """
#     Dice Similarity Coefficient
#     pred, gt: binary volume
#     """
#     intersection = np.sum(pred * gt)
#     return (2.0 * intersection + eps) / (np.sum(pred) + np.sum(gt) + eps)

# def completeness_ratio(pred, gt, eps=1e-6):
#     """
#     Completeness Ratio (CR)
#     pred, gt: binary volume
#     """
#     intersection = np.sum(pred * gt)
#     return (intersection + eps) / (np.sum(gt) + eps)

# def chamfer_distance(pred, gt):
#     """
#     Chamfer Distance (CD) between two binary volumes
#     pred, gt: binary volumes (0/1)
#     """
#     # 非零 voxel 转成点云坐标
#     pred_pts = np.array(np.nonzero(pred)).T
#     gt_pts   = np.array(np.nonzero(gt)).T

#     if len(pred_pts) == 0 or len(gt_pts) == 0:
#         return np.nan  # 避免空体素

#     tree_pred = KDTree(pred_pts)
#     tree_gt   = KDTree(gt_pts)

#     d_pred_to_gt, _ = tree_gt.query(pred_pts)
#     d_gt_to_pred, _ = tree_pred.query(gt_pts)

#     cd = np.mean(d_pred_to_gt**2) + np.mean(d_gt_to_pred**2)
#     return cd

# def ssim_slice_wise(pred, gt, axis=2):
#     """
#     对 3D volume 按 slice 计算 2D SSIM，然后求平均
#     axis: 0 / 1 / 2
#     """
#     assert pred.shape == gt.shape

#     pred = normalize(pred)
#     gt   = normalize(gt)

#     ssim_vals = []

#     for i in range(pred.shape[axis]):
#         if axis == 0:
#             p, g = pred[i, :, :], gt[i, :, :]
#         elif axis == 1:
#             p, g = pred[:, i, :], gt[:, i, :]
#         else:
#             p, g = pred[:, :, i], gt[:, :, i]

#         if np.std(g) < 1e-6:
#             continue

#         val = ssim(p, g, data_range=1.0)
#         ssim_vals.append(val)

#     return float(np.mean(ssim_vals)) if len(ssim_vals) > 0 else 0.0

# # ===============================
# # Evaluation
# # ===============================
# def evaluate_3d_reconstruction(
#     pred_dir,
#     gt_dir,
#     threshold=None,
#     axis=2
# ):
#     pred_files = sorted(
#         f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")
#     )

#     ssim_all = []
#     dice_all = []
#     cr_all   = []
#     cd_all   = []

#     for pred_fname in pred_files:
#         pred_path = os.path.join(pred_dir, pred_fname)

#         # ---------- match GT ----------
#         case_id = extract_case_id(pred_fname)
#         gt_fname = f"{case_id}.nii.gz"
#         gt_path = os.path.join(gt_dir, gt_fname)

#         if not os.path.exists(gt_path):
#             print(f"[Skip] GT not found for {pred_fname}")
#             continue

#         pred = load_nii(pred_path)
#         gt   = load_nii(gt_path)

#         # ---------- SSIM ----------
#         ssim_val = ssim_slice_wise(pred, gt, axis=axis)
#         ssim_all.append(ssim_val)

#         # ---------- Dice + CR + CD ----------
#         if threshold is not None:
#             pred_bin = (pred > threshold).astype(np.float32)
#             gt_bin   = (gt > threshold).astype(np.float32)

#             dice_val = dice_score(pred_bin, gt_bin)
#             cr_val   = completeness_ratio(pred_bin, gt_bin)
#             cd_val   = chamfer_distance(pred_bin, gt_bin)

#             dice_all.append(dice_val)
#             cr_all.append(cr_val)
#             cd_all.append(cd_val)

#             print(
#                 f"{pred_fname}: "
#                 f"SSIM={ssim_val:.4f}, "
#                 f"DSC={dice_val:.4f}, "
#                 f"CR={cr_val:.4f}, "
#                 f"CD={cd_val:.4f}"
#             )
#         else:
#             print(f"{pred_fname}: SSIM={ssim_val:.4f}")

#     print("\n========== Final Results ==========")
#     print(f"Mean SSIM: {np.mean(ssim_all):.4f}")
#     if threshold is not None:
#         print(f"Mean DSC : {np.mean(dice_all):.4f}")
#         print(f"Mean CR  : {np.mean(cr_all):.4f}")
#         print(f"Mean CD  : {np.nanmean(cd_all):.4f}")

# # ===============================
# # Main
# # ===============================
# if __name__ == "__main__":

#     pred_dir = "/media/I/xcw/3DGR-CAR-main/3dgs-car/gaussian_3dgs_result_2view_imagecas_lca"   # 预测结果目录
#     gt_dir   = "/media/I/xcw/3DGR-CAR-main/all_data/ImageCAS_lca_gt_volume_test"             # GT 目录

#     evaluate_3d_reconstruction(
#         pred_dir=pred_dir,
#         gt_dir=gt_dir,
#         threshold=0.01,   # 根据你的数据调整
#         axis=2
#     )

import os
import numpy as np
import nibabel as nib
from scipy.spatial import KDTree
from scipy.spatial.distance import directed_hausdorff


# skeletonize 兼容处理
try:
    from skimage.morphology import skeletonize
    HAS_SKELETON = True
except Exception:
    HAS_SKELETON = False


# =========================================================
# IO
# =========================================================
def load_nii(path):
    """
    读取 nii / nii.gz 文件
    """
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)
    return data


def normalize(volume):
    """
    归一化到 [0, 1]
    """
    v_min, v_max = volume.min(), volume.max()
    if v_max > v_min:
        volume = (volume - v_min) / (v_max - v_min)
    return volume


def trimmed_case_mean(values):
    """
    对 case 级结果去掉最高值和最低值后再求平均
    - 忽略 nan
    - 若有效值 <= 2，则直接返回均值
    """
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return np.nan
    if len(arr) <= 2:
        return float(np.mean(arr))

    arr_sorted = np.sort(arr)
    arr_trimmed = arr_sorted[1:-1]

    if len(arr_trimmed) == 0:
        return float(np.mean(arr))

    return float(np.mean(arr_trimmed))


# =========================================================
# Voxel-level metrics
# =========================================================
def dice_score(pred, gt, eps=1e-6):
    intersection = np.sum(pred * gt)
    return (2.0 * intersection + eps) / (np.sum(pred) + np.sum(gt) + eps)


def iou_score(pred, gt, eps=1e-6):
    intersection = np.sum(pred * gt)
    union = np.sum((pred + gt) > 0)
    return (intersection + eps) / (union + eps)


def precision_score(pred, gt, eps=1e-6):
    tp = np.sum(pred * gt)
    fp = np.sum((pred == 1) & (gt == 0))
    return (tp + eps) / (tp + fp + eps)


def recall_score(pred, gt, eps=1e-6):
    tp = np.sum(pred * gt)
    fn = np.sum((pred == 0) & (gt == 1))
    return (tp + eps) / (tp + fn + eps)


def completeness_ratio(pred, gt, eps=1e-6):
    intersection = np.sum(pred * gt)
    return (intersection + eps) / (np.sum(gt) + eps)


def volume_ratio(pred, gt, eps=1e-6):
    return (np.sum(pred) + eps) / (np.sum(gt) + eps)


def foreground_voxel_ratio(volume):
    return float(np.sum(volume > 0) / volume.size)


def ssim_3d_patchwise_strict(pred, gt, patch_size=7, stride=3, min_fg_voxels=20, data_range=1.0):
    """
    严格版 3D patch SSIM
    越高越好 ↑
    """
    assert pred.shape == gt.shape, "pred 和 gt shape 必须一致"

    pred = normalize(pred)
    gt = normalize(gt)

    D, H, W = pred.shape
    ps = patch_size

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_vals = []

    for z in range(0, D - ps + 1, stride):
        for y in range(0, H - ps + 1, stride):
            for x in range(0, W - ps + 1, stride):
                p_patch = pred[z:z + ps, y:y + ps, x:x + ps]
                g_patch = gt[z:z + ps, y:y + ps, x:x + ps]

                if np.sum(g_patch > 0) < min_fg_voxels:
                    continue

                mu_x = np.mean(p_patch)
                mu_y = np.mean(g_patch)
                sigma_x2 = np.var(p_patch)
                sigma_y2 = np.var(g_patch)
                sigma_xy = np.mean((p_patch - mu_x) * (g_patch - mu_y))

                num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
                den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)

                val = num / den if den != 0 else 0.0
                ssim_vals.append(val)

    if len(ssim_vals) == 0:
        return 0.0

    return float(np.mean(ssim_vals))


# =========================================================
# Point-cloud / geometry-level metrics
# =========================================================
def chamfer_distance(pred, gt, trim_ratio=0.95):
    pred_pts = np.array(np.nonzero(pred)).T
    gt_pts = np.array(np.nonzero(gt)).T
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan
    tree_pred = KDTree(pred_pts)
    tree_gt = KDTree(gt_pts)
    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)
    if trim_ratio < 1.0:
        keep_p = max(1, int(len(d_pred_to_gt) * trim_ratio))
        keep_g = max(1, int(len(d_gt_to_pred) * trim_ratio))
        d_pred_to_gt = np.sort(d_pred_to_gt)[:keep_p]
        d_gt_to_pred = np.sort(d_gt_to_pred)[:keep_g]
    return np.mean(d_pred_to_gt ** 2) + np.mean(d_gt_to_pred ** 2)


def hausdorff_distance(pred, gt):
    pred_pts = np.array(np.nonzero(pred)).T
    gt_pts = np.array(np.nonzero(gt)).T

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan

    hd1 = directed_hausdorff(pred_pts, gt_pts)[0]
    hd2 = directed_hausdorff(gt_pts, pred_pts)[0]
    return max(hd1, hd2)


def point_cloud_precision_recall_f1(pred, gt, threshold=1.0):
    pred_pts = np.array(np.nonzero(pred)).T
    gt_pts = np.array(np.nonzero(gt)).T

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan, np.nan, np.nan

    tree_gt = KDTree(gt_pts)
    tree_pred = KDTree(pred_pts)

    d_pred, _ = tree_gt.query(pred_pts)
    d_gt, _ = tree_pred.query(gt_pts)

    precision = np.mean(d_pred <= threshold)
    recall = np.mean(d_gt <= threshold)
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)

    return precision, recall, f1


def skeleton_chamfer_distance(pred, gt):
    if not HAS_SKELETON:
        return np.nan

    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return np.nan

    pred_skel = skeletonize(pred > 0)
    gt_skel = skeletonize(gt > 0)

    pred_pts = np.array(np.nonzero(pred_skel)).T
    gt_pts = np.array(np.nonzero(gt_skel)).T

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan

    tree_pred = KDTree(pred_pts)
    tree_gt = KDTree(gt_pts)

    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)

    return np.mean(d_pred_to_gt ** 2) + np.mean(d_gt_to_pred ** 2)


# =========================================================
# Structure-level metrics
# =========================================================
def cr_skeleton(pred, gt, eps=1e-6):
    if not HAS_SKELETON:
        return np.nan

    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    if np.sum(gt) == 0:
        return np.nan

    gt_skel = skeletonize(gt > 0)
    skel_total = np.sum(gt_skel)

    if skel_total == 0:
        return np.nan

    skel_intersection = np.sum(pred * gt_skel)
    return (skel_intersection + eps) / (skel_total + eps)


# =========================================================
# Evaluation
# =========================================================
from pathlib import Path
from cyclopts import App
app = App()

@app.default()
def evaluate_3d_reconstruction(
    pred_dir: Path,
    gt_dir: Path,
    patch_size: int=7,
    stride: int=3,
    min_fg_voxels: int=20,
    point_threshold: float=1.0
):
    r"""
    nushell scripts:
    > ls outputs/3DGR-CAR_summary/ | each {|nviews| 
        ls $nviews.name | each { |dataset|
            ls $dataset.name | each { 
                |side| pixi run python scripts/evaluation.py $side.name ./data | save -f $"outputs/results/($nviews.name|path basename)_($dataset.name| path basename)_($side.name|path basename)"
            }
        }
    }
    """
    
    pred_files = sorted(
        Path(root) / f
        for root, _, files in os.walk(pred_dir, followlinks=True)
        for f in files if f.endswith(".nii.gz")
    )
    print(f"Found {len(pred_files)} prediction files in {pred_dir}")

    gt_files = sorted(
        Path(root) / f
        for root, _, files in os.walk(gt_dir, followlinks=True)
        for f in files if f.endswith(".nii.gz")
    )
    print(f"Found {len(gt_files)} GT files in {gt_dir}")

    gt_files_map = {f.stem: f for f in gt_files}

    voxel_metrics = {
        "ssim3d": [],
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
        "cr": [],
        "vol_ratio": [],
        "fg_pred": [],
        "fg_gt": [],
    }

    geo_metrics = {
        "chamfer": [],
        "hausdorff": [],
        "pc_precision": [],
        "pc_recall": [],
        "pc_f1": [],
        "skel_chamfer": [],
    }

    struct_metrics = {
        "cr_skel": [],
    }

    for pred_path in pred_files:
        gt_path = gt_files_map.get(pred_path.stem)

        if not gt_path or not os.path.exists(gt_path):
            print(f"[Skip] GT not found for {pred_path.name}")
            continue

        pred = load_nii(pred_path)
        gt = load_nii(gt_path)

        pred_bin = (pred > 0).astype(np.uint8)
        gt_bin = (gt > 0).astype(np.uint8)

        ssim_val = ssim_3d_patchwise_strict(
            pred_bin, gt_bin,
            patch_size=patch_size,
            stride=stride,
            min_fg_voxels=min_fg_voxels,
            data_range=1.0
        )
        dice_val = dice_score(pred_bin, gt_bin)
        iou_val = iou_score(pred_bin, gt_bin)
        precision_val = precision_score(pred_bin, gt_bin)
        recall_val = recall_score(pred_bin, gt_bin)
        cr_val = completeness_ratio(pred_bin, gt_bin)
        vr_val = volume_ratio(pred_bin, gt_bin)
        fg_pred = foreground_voxel_ratio(pred_bin)
        fg_gt = foreground_voxel_ratio(gt_bin)

        voxel_metrics["ssim3d"].append(ssim_val)
        voxel_metrics["dice"].append(dice_val)
        voxel_metrics["iou"].append(iou_val)
        voxel_metrics["precision"].append(precision_val)
        voxel_metrics["recall"].append(recall_val)
        voxel_metrics["cr"].append(cr_val)
        voxel_metrics["vol_ratio"].append(vr_val)
        voxel_metrics["fg_pred"].append(fg_pred)
        voxel_metrics["fg_gt"].append(fg_gt)

        cd_val = chamfer_distance(pred_bin, gt_bin)
        hd_val = hausdorff_distance(pred_bin, gt_bin)
        pc_p, pc_r, pc_f1 = point_cloud_precision_recall_f1(pred_bin, gt_bin, threshold=point_threshold)
        skel_cd = skeleton_chamfer_distance(pred_bin, gt_bin)

        geo_metrics["chamfer"].append(cd_val)
        geo_metrics["hausdorff"].append(hd_val)
        geo_metrics["pc_precision"].append(pc_p)
        geo_metrics["pc_recall"].append(pc_r)
        geo_metrics["pc_f1"].append(pc_f1)
        geo_metrics["skel_chamfer"].append(skel_cd)

        cr_skel_val = cr_skeleton(pred_bin, gt_bin)
        struct_metrics["cr_skel"].append(cr_skel_val)

        print(f"\n[{pred_path.name}]")
        print("  [Voxel-level]")
        print(
            f"    SSIM3D↑={ssim_val:.4f}, DSC↑={dice_val:.4f}, IOU↑={iou_val:.4f}, "
            f"Precision↑={precision_val:.4f}, Recall↑={recall_val:.4f}, "
            f"CR↑={cr_val:.4f}, VolRatio↔1.0={vr_val:.4f}, "
            f"FG_pred={fg_pred:.6f}, FG_gt={fg_gt:.6f}"
        )

        print("  [Point-cloud / Geometry-level]")
        print(
            f"    Chamfer↓={cd_val:.4f}, Hausdorff↓={hd_val:.4f}, "
            f"PC_Precision↑={pc_p:.4f}, PC_Recall↑={pc_r:.4f}, PC_F1↑={pc_f1:.4f}, "
            f"SkeletonChamfer↓={skel_cd:.4f}"
        )

        print("  [Structure-level]")
        print(f"    CR_skel↑={cr_skel_val:.4f}")

    print("\n========== Final Results ==========")

    print("\n[Voxel-level]")
    print(f"  Trimmed Mean SSIM3D↑   : {trimmed_case_mean(voxel_metrics['ssim3d']):.4f}")
    print(f"  Trimmed Mean DSC↑      : {trimmed_case_mean(voxel_metrics['dice']):.4f}")
    print(f"  Trimmed Mean IOU↑      : {trimmed_case_mean(voxel_metrics['iou']):.4f}")
    print(f"  Trimmed Mean Precision↑: {trimmed_case_mean(voxel_metrics['precision']):.4f}")
    print(f"  Trimmed Mean Recall↑   : {trimmed_case_mean(voxel_metrics['recall']):.4f}")
    print(f"  Trimmed Mean CR↑       : {trimmed_case_mean(voxel_metrics['cr']):.4f}")
    print(f"  Trimmed Mean VolRatio↔1.0: {trimmed_case_mean(voxel_metrics['vol_ratio']):.4f}")
    print(f"  Trimmed Mean FG_pred   : {trimmed_case_mean(voxel_metrics['fg_pred']):.6f}")
    print(f"  Trimmed Mean FG_gt     : {trimmed_case_mean(voxel_metrics['fg_gt']):.6f}")

    print("\n[Point-cloud / Geometry-level]")
    print(f"  Trimmed Mean Chamfer↓        : {trimmed_case_mean(geo_metrics['chamfer']):.4f}")
    print(f"  Trimmed Mean Hausdorff↓      : {trimmed_case_mean(geo_metrics['hausdorff']):.4f}")
    print(f"  Trimmed Mean PC Precision↑   : {trimmed_case_mean(geo_metrics['pc_precision']):.4f}")
    print(f"  Trimmed Mean PC Recall↑      : {trimmed_case_mean(geo_metrics['pc_recall']):.4f}")
    print(f"  Trimmed Mean PC F1↑          : {trimmed_case_mean(geo_metrics['pc_f1']):.4f}")
    print(f"  Trimmed Mean SkeletonChamfer↓: {trimmed_case_mean(geo_metrics['skel_chamfer']):.4f}")

    print("\n[Structure-level]")
    print(f"  Trimmed Mean CR_skel↑ : {trimmed_case_mean(struct_metrics['cr_skel']):.4f}")

if __name__ == "__main__":
    app()