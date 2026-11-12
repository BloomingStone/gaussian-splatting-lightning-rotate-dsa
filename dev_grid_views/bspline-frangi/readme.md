# 实验：bspline-frangi（简化形变场 + Frangi 冠脉加权）

## 目标
在 stage1-spike 基础上，用低频 B-spline 形变场（简化形变）+ Frangi 冠脉加权训练，
迫使模型在简化形变场下仍优先重建冠脉，期望缓解"训练帧好、换视角差"的过拟合。

## 动机（用户分析）
- 训练帧已足够好，但换视角显著变差 = 过拟合；简化形变场方向正确。
- 之前 bspline 导致冠脉模糊，疑因：①背景主导 L1+SSIM loss；②形变场 lr 不够。
- 本实验：bspline 低频场 + Frangi 加权（冠脉优先）+ 提高形变场 lr。

## 方法
### 形变场
- `BSplineDeformConfig`（低频 B-spline 位移场 u(x, phase)，d_xyz=u，d_rot=curl(F)，
  d_scale=0），`grid_resolution=(8,8,8)`，`enable_flow=false`。
- 从 `exp/bspline` 分支引入 `bspline_deform.py`，`__init__.py` 导出。

### Frangi 训练加权
- 训练集用 `FrangiImagesDatasetBuilder`（Frangi vesselness → `frangi_soft_mask`）。
- metric 新增 `w_frangi`：L1 与 SSIM 损失按 `w = 1 + w_frangi * frangi_mask` 加权
  （背景权重 1，冠脉 2 倍），SSIM 用 per-pixel map 加权。
- 修正 `ssim.py` 的 `size_average=False` 返回 per-pixel map（原为标量）。

### 训练配置
- 形变场 lr：先 1e-2（失败），后 2e-3（温和提高）。

## 结果
### 第一次尝试（w=frangi_mask 完全加权 + lr=1e-2）：失败
| 指标 | spike | bspline-frangi |
|---|---|---|
| v_psnr | 28.53 | 13.36 |
| edge_contrast | 0.778 | 0.442 |
| ghost_score | 0.054 | 0.059 |
| 全图 PSNR | 28.35 | 12.78 |

- 完全 Frangi 加权把背景权重压到 0 → 背景无约束，全图 PSNR 暴跌；冠脉也欠重建。
- lr=1e-2 过高（xyz_var 4.04）。

### 第二次尝试（w=1+w_frangi*frangi + lr=2e-3）
（训练中）

## 对照
- stage1-spike（hashgrid + d_scaling_min=-0.5，无 Frangi 加权）。
