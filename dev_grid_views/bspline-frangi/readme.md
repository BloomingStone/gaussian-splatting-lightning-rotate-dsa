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

## sweep：bspline 控制点数量（16/32/64/128）× Frangi 加权（SSIM 修复后重跑）

### 伪影指标（frame100 13 视角均值：10 grid + 3 train）
| job | ghost↓ | edge_contrast↑ | v_ssim↑ | v_psnr↑ | 全图 psnr↑ |
|---|---|---|---|---|---|
| spike | 0.051 | 0.816 | 0.881 | 30.55 | 32.96 |
| frangi-only | 0.050 | 0.843 | 0.896 | 30.85 | 32.58 |
| bs16 | 0.054 | 0.565 | 0.484 | 21.31 | 25.53 |
| bs32 | 0.101 | 0.391 | 0.515 | 20.67 | 24.21 |
| bs64 | 0.100 | 0.466 | 0.521 | 21.49 | 23.95 |
| bs128 | 0.116 | 0.475 | 0.537 | 22.52 | 24.99 |
| bs16-f | 0.059 | 0.547 | 0.470 | 19.99 | 25.05 |
| bs32-f | 0.085 | 0.452 | 0.510 | 20.13 | 24.07 |
| bs64-f | 0.106 | 0.453 | 0.532 | 21.59 | 24.69 |
| bs128-f | 0.106 | 0.535 | 0.562 | 23.26 | 24.94 |

### 结论（重跑后，对比旧结论的修正）
- **SSIM 修复后 bspline 不再崩溃**：全图 psnr 从 5.4~6.3 → 23.9~25.5，v_psnr 从 ~8 → 20~23。
  旧 sweep 的"全部失败"同样是 SSIM autocast bug 的假象。
- **但仍远低于 hashgrid 基线**：v_psnr 21~23 vs 30.5、v_ssim 0.47~0.56 vs 0.88、
  edge_contrast 0.39~0.57 vs 0.82。bspline 低频场表达冠脉精细非刚体运动能力不足是真实瓶颈。
- **控制点数量：bs16 最优**。bs32 是明显拐点（ghost 0.054→0.101 翻倍、ec 0.565→0.391、
  esharp 0.445→0.316 大跌）；bs64/128 略有回升（v_psnr/v_ssim 微升）但仍不如 bs16。
- **Frangi 加权效果不一致**：bs16 反作用（v_psnr 21.31→19.99、v_ssim 0.484→0.470）；
  bs32 中性；bs64/128 略正作用（v_ssim/ec 微升）。但都改变不了 bspline 远差于 hashgrid 的结论。
- 用户观察确认：bs32 往上效果变差（拐点在 bs32），bs16 处 frangi 起反作用。

---

## 平滑正则消融：bs32 为何差 → w_field_smooth 修复

### 动机
用户质疑 bs16 → bs32 差异过大是否为实现 bug。数值验证确认 bspline 插值/Jacobian
精确（仿射保持性 u-err ~1e-7）。假设：32³ 控制点比高斯点还多，多出的自由度拟合
噪声（高频振荡），导致 ghost 翻倍、清晰度大跌。

### w_field_smooth 扫描（frame100 13 视角均值）
| job | ghost↓ | edge_contrast↑ | v_ssim↑ | v_psnr↑ | 全图 psnr↑ | esharp↑ |
|---|---|---|---|---|---|---|
| bs32（无平滑） | 0.101 | 0.391 | 0.515 | 20.67 | 24.21 | 0.316 |
| bs32-smooth-0.005 | 0.083 | 0.383 | 0.517 | 20.25 | 24.53 | 0.308 |
| bs32-smooth-0.02 | 0.089 | 0.408 | 0.505 | 20.68 | 25.46 | 0.326 |
| bs32-smooth-0.1 | 0.066 | 0.609 | 0.538 | 21.87 | 26.92 | 0.465 |
| **bs32-smooth-0.2** | **0.066** | **0.641** | **0.588** | **22.59** | **27.80** | **0.524** |
| bs32-smooth-0.5 | 0.060 | 0.566 | 0.497 | 21.20 | 27.32 | 0.447 |
| bs32-smooth-1.0 | 0.062 | 0.534 | 0.506 | 21.58 | 27.95 | 0.417 |

### bs32 + 平滑 0.2 + Frangi 结合
| job | ghost↓ | ec↑ | v_ssim↑ | v_psnr↑ | psnr↑ | esharp↑ |
|---|---|---|---|---|---|---|
| bs32-smooth-0.2 | 0.066 | 0.641 | 0.588 | 22.59 | 27.80 | 0.524 |
| **bs32-smooth-0.2-frangi** | **0.060** | **0.676** | **0.611** | **23.99** | **28.38** | 0.518 |

### 结论
- **"高频自由度拟合噪声"假设成立**：w_field_smooth 单调改善 bs32（0.005→0.1→0.2），
  最优 0.2（v_psnr 20.67→22.59、ec 0.391→0.641、ghost 0.101→0.066），0.5/1.0 过平滑
  开始回退（血管质量下降）。
- **bs32+平滑 反而超过 bs16**（psnr 27.80 vs 25.53、v_psnr 22.59 vs 21.31）：说明 32³
  表达力其实更强，之前"bs16 最优"只是 bs32 缺正则。
- **Frangi 与平滑正则协同**：bs32-smooth-0.2-frangi 全面最优（psnr 28.38、v_psnr 23.99、
  v_ssim 0.611、ec 0.676、ghost 0.060）。之前在 bs16 处 frangi 反作用，是因为 bs16 形变场
  本身干净、frangi 只加了噪声；而在平滑正则压制高频噪声后，frangi 加权才起正作用。
- **但仍低于 hashgrid 基线**（psnr 32.96、v_psnr 30.55）：bspline 表达上限仍不如 hashgrid。

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
- **bspline 低频形变场（16~128 控制点）质量仍低于 hashgrid**：修复 SSIM bug 后不再崩溃，
  加平滑正则 + Frangi 后最优可达 psnr 28.38 / v_psnr 23.99，但仍低于 hashgrid 基线
  （psnr 32.96 / v_psnr 30.55）。bspline 表达冠脉精细非刚体运动的能力上限不足是真实瓶颈。
- **控制点数量结论修正**：bs16 并非最优——bs32 的坏表现是"控制点过多但无正则导致
  高频自由度拟合噪声"，加 w_field_smooth=0.2 后 bs32 全面超过 bs16。最优配置为
  bs32 + w_field_smooth=0.2 + w_frangi=1.0。
- **Frangi 加权作用依赖形变场状态**：形变场干净（有平滑正则）时 frangi 正作用；
  bs16（无正则但低频天然干净）时 frangi 反作用；对 hashgrid 轻微正面。
- 重影根因仍是单视角稀疏角运动欠定，需转向多视角/flow 监督或 P3 循环一致性。
