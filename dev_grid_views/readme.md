# dev_grid_views — 伪影消减实验

围绕 grid 多视角验证集上的三类伪影（stage1 针状、r2g 重影、共性模糊）做针对性实验。

## 实验组织规范

- 每个实验一个独立分支 `dev/grid_views/<实验名>`；**同一时刻只做一个实验**，每次试验前提交修改。
- 实验方法与结论记录在 `dev_grid_views/<实验名>/readme.md`。
- 每次实验产出两张图，放在 `dev_grid_views/<实验名>/`：
  - `frame_100_grid_compare_<实验名>.png`（frame 100 多视角对比，含红/黄框 + PSNR + 角度）
  - `artifact_metrics_compare_<实验名>.png`（各视角伪影指标柱状图）
- 训练/验证结果统一放 `outputs/dev_grid_views/<实验名>/`。
- 配置文件放 `configs/dev_grid_views/<实验名>/`。

## 两方法差异清单

| 维度 | stage1 (GS-contrast-flow) | r2g (R2-Gaussian) |
|------|---------------------------|-------------------|
| 初始化 | 随机点云 10K（AABB 均匀） | ODL FDK 100K |
| 数据域 | J 暗场 (0~1) | I = -ln(J) × scene_scale × proj_scale |
| 形变模型 | HashGrid（tcnn hashgrid xyz + 分离 time/phase 分支） | 纯 MLP（傅里叶 PE：xyz 10 + phase 6） |
| 形变输入 | time + phase（分离，t_phase_combined=false） | 仅 phase |
| 形变输出 | d_xyz/d_scale/d_rot/d_density（stage1 冻结 density） | d_xyz/d_scale/d_rot |
| 损失 | L1 + SSIM(1.0) + 正则 | L1 + 0.25·D-SSIM + 0.05·3D-TV |
| 密度控制 | 动态百分位 0.98 + density_reset(2000) | 固定梯度阈值 5e-5 |
| 优化 | AST + density ramp(2000) | 标准 Adam |
| 高斯点数 | ~100K | ~111K |

## 三大症状与成因假设

1. **stage1 针状伪影**：HashGrid 高频编码 / density_reset 周期性重置 / 高斯尺度失控（缺尺度下限约束）。
2. **r2g 重影**：FDK 初始化携带心跳运动伪影；纯 MLP 形变容量不足；仅 phase 输入无法区分同相位不同时间点的造影剂状态。
3. **共性模糊**：形变场未能充分融合帧间信息，导致重建为多帧平均而模糊。

## 实验计划

| # | 实验名 | 目标 | 关键改动 |
|---|--------|------|---------|
| 1 | `r2g-random-init` | 消减 r2g 重影 | r2g 改用随机点云初始化（对齐 stage1），其余不变 |
| 2 | `stage1-spike` | 消减 stage1 针状伪影 | 定位成因（density_reset / 尺度约束 / hashgrid 分辨率） |
| 3 | `deform-blur` | 消减共性模糊 | 形变场容量与正则调整 |

## 关键指标口径

- `spike_score`（背景高频能量比，>1.5 = 针状伪影）
- `v_ssim`（血管区域 SSIM，低 = 血管错位/重影）
- `ghost_score`（边缘展宽比 pred/gt，>1 = 重影/模糊）
- `edge_sharpness`（GT 血管边缘梯度比，低 = 错位/模糊）
- 通用口径：`psnr` / `ssim`(=1-ssim_loss) / `lpips`
