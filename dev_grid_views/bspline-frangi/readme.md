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

### 第二次尝试（w=1+w_frangi*frangi + lr=2e-3）：更差
| 指标 | spike | bspline-frangi |
|---|---|---|
| v_psnr | 28.53 | 8.31 |
| edge_contrast | 0.778 | 0.124 |
| v_ssim | 0.856 | 0.064 |
| 全图 PSNR | 28.35 | 5.84 |
| xyz_var | ~1.5 | 0.025 |

- 加权改为 w=1+frangi（背景权重 1），但结果更差：xyz_var 0.025 → bspline 形变场
  几乎没学到位移（lr=2e-3 过小），整个重建失败。
- lr 极敏感：1e-2 → xyz_var 4.04（过大）；2e-3 → xyz_var 0.025（过小）。

## 结论
- **bspline 低频形变场（8³ 控制点）在当前数据上不可行**：对 lr 极敏感，
  且低频场难以表达冠脉精细非刚体运动。
- **Frangi 加权无法救回 bspline**：完全加权（背景权重 0）背景失去约束；
  相对加权（背景权重 1）形变场学不到运动。
- 用户假设"bspline 失败因背景主导 loss + lr 不够"**未得到验证**——两者调整后
  依然远差于 stage1-spike（PSNR 5.7~12.8 vs 28.35）。
- 与之前 exp/bspline 结论一致（3D dice 0.346 vs hashgrid 0.614）：简化形变场
  方向对重建质量有害。重影根因是单视角稀疏角运动欠定，而非形变场复杂度。

---

## sweep：bspline 控制点数量（16/32/64/128）× Frangi 加权

### 伪影指标（grid 10 视角均值）
| job | ghost↓ | edge_contrast↑ | v_ssim↑ | v_psnr↑ | 全图 psnr↑ |
|---|---|---|---|---|---|
| spike | 0.054 | 0.778 | 0.856 | 28.53 | 30.35 |
| bs16 | 0.009 | 0.171 | 0.064 | 8.30 | 5.71 |
| bs32 | 0.012 | 0.173 | 0.082 | 8.77 | 6.33 |
| bs64 | 0.017 | 0.133 | 0.078 | 8.35 | 5.44 |
| bs128 | 0.022 | 0.177 | 0.088 | 8.33 | 5.91 |
| bs16-f | 0.015 | 0.162 | 0.244 | 6.46 | 5.86 |
| bs32-f | 0.011 | 0.088 | 0.051 | 8.26 | 5.57 |
| bs64-f | 0.010 | 0.157 | 0.046 | 8.24 | 5.86 |
| bs128-f | 0.012 | 0.097 | 0.039 | 8.12 | 5.63 |
| frangi-only | ❌ 训练发散（hashgrid xyz_var 爆炸 8740 → OOM） | | | | |

### 结论
- **bspline 形变场（16~128 控制点）在 lr=5e-3 下全部失败**（v_psnr ~8.3，全图 psnr 5.4~6.3，
  远差于 spike 28.5/30.35）；ghost_score 极低是"血管没重建"的假象。
- **控制点数量无关**：16→128 结果几乎不变（xyz_var 0.05~0.12，位移仍过小）。
- **Frangi 加权无帮助**（bs*-f 普遍 ≤ 对应 bs*）。
- **frangi-only（hashgrid）反而发散**：Frangi 加权让 hashgrid 形变场 xyz_var 爆炸到 8740 → OOM。
- 最终结论：简化形变场（bspline）方向不可行，Frangi 加权对 hashgrid 也有害（发散）。
  重影根因是单视角稀疏角运动欠定，需转向多视角/flow 监督或 P3 循环一致性。
