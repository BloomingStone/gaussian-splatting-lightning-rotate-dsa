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
- **frangi-only（hashgrid）曾被认为发散**，但见下方「根因排查」——是标准 SSIM 的
  autocast 数值 bug 导致，与 Frangi 加权无关。

---

## 根因排查：frangi-only 为何发散（重要修正）

### 现象
frangi-only（hashgrid + w_frangi=1.0 + fused_ssim=false）训练发散：
xyz_var 爆炸（8740 / 594000）→ 高斯数量异常增长到 80504 → 投影全黑 → OOM。

### 排查过程（依次排除）
1. hashgrid + fused_ssim=false + **w_frangi=0** → 仍发散（xyz_var→3390→OOM）→ 排除 Frangi 加权。
2. stage1-spike config + **fused_ssim=false** → 仍发散（xyz_var 3.26→83→1.77e+4）→ 定位 fused_ssim=false。
3. fp16 下标准 SSIM per-pixel map 溢出到 [-11.7, 13.7]（应 [-1,1]）→ 数值不稳定。
4. 仅 `pred.float()` cast 不够——autocast 仍影响 F.conv2d 窗口统计。
5. **决定性 CUDA 诊断**：autocast(fp16) 内标准 SSIM ssim=1.1426(>1)、梯度 max=7.0；
   `torch.autocast(enabled=False)` + fp32 后 ssim=0.9977、梯度 max=5.48e-05（差 13 万倍）。

### 根因
16-mixed 训练下，标准 SSIM（`internal/metrics/ssim.py`，F.conv2d 窗口统计 img*img / mu²）
在 autocast 上下文溢出，per-pixel SSIM > 1、梯度异常放大 ~1e5 倍，使 hashgrid 无界
xyz_warp 发散。fused_ssim（fp16 CUDA kernel）本身稳定，故此前 fused_ssim=true 的所有
实验（stage1-spike 等）不受影响。

### 修复（commit 794eabc）
- `internal/metrics/metric.py`：非 fused 分支改为
  `with torch.autocast(device_type=..., enabled=False): ssim(pred.float(), gt.float())`
- `internal/metrics/rotate_xray_metrics_with_flow.py`：加权 per-pixel ssim_map 同样禁用 autocast。

### frangi-only 重新训练结果（frangi-only-v2，修复后不发散）
训练稳定（xyz_var ~1.6-2.3，val/psnr 28.40），frame100 13 视角指标 vs stage1-spike：

| metric | stage1-spike | frangi-only | 变化 |
|---|---|---|---|
| psnr | 32.96 | 32.57 | -0.39 |
| v_psnr | 30.55 | 30.85 | +0.30 ✓ |
| v_ssim | 0.881 | 0.896 | +0.015 ✓ |
| edge_contrast | 0.816 | 0.843 | +0.027 ✓ |
| edge_sharpness | 0.747 | 0.774 | +0.027 ✓ |
| ghost_score | 0.051 | 0.050 | -0.001 ✓ |
| spike_score | 0.558 | 0.558 | 0.000 |

**修正结论：Frangi 加权对 hashgrid 是轻微正面的**（血管区域 v_psnr/v_ssim、清晰度
edge_contrast/sharpness 均改善，重影略降，代价是全图 PSNR 略降 0.39）。之前"frangi-only
发散 / Frangi 对 hashgrid 有害"的结论是 SSIM autocast bug 的假象，予以推翻。

---

## 最终结论（bspline-frangi 实验整体）
- **bspline 低频形变场（16~128 控制点）在当前数据上不可行**：对 lr 极敏感，
  低频场难以表达冠脉精细非刚体运动，所有控制点数量 + Frangi 组合均远差于 hashgrid。
- **Frangi 加权对 hashgrid 轻微正面**（见上表），但不是解决重影的关键手段。
- 重影根因仍是单视角稀疏角运动欠定，需转向多视角/flow 监督或 P3 循环一致性。
