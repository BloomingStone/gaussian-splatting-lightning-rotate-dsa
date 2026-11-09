# 实验 2：stage1-spike

## 目标
消减 stage1 在 grid 多视角上的针状/散点伪影（view 3/4/9 严重）。

## 成因分析（已确认）

加载 stage1 ckpt，逐相位分析形变场输出：

| phase | d_scaling 范围 | d_scaling<-0.9 点数 | new_scale<0.003 |
|-------|---------------|--------------------|----------------|
| 0.0 | [-1.0, 1.0] | 5418（~50%） | 17 |
| 0.667 | [-1.0, 1.0] | 5092 | 18 |
| 0.833 | [-1.0, 1.0] | 5567 | 42 |
| 静态 | — | — | scale_min=0.0083 |

**结论**：针状伪影来自形变场 `scaling_warp` 饱和——约 50% 的高斯在任意相位下 `d_scaling` 接近 -1（scale 被压缩 >90%），少数点压缩到 <0.003（静态最小 0.0083），投影成针状线。

**根因**：`scaling_warp` 用 `tanh` 输出 `d_scaling ∈ [-1,1]`，无下界约束、无 scale/d_scaling 正则；形变场学会通过"压缩 scale 到接近 0"来关闭某些高斯，但 scale 无法真正归零（Softplus 下界），于是留下极细的高密度点 → 针状。

## 消减方案（候选）
1. **限制 d_scaling 下界**：`d_scaling ∈ [-0.5, 1]`（scale 最多压缩一半），改 `scaling_warp` 的 tanh 为缩放平移。
2. **scale 下限正则**：penalize 形变后 scale 过小。
3. **d_scaling 正则**：penalize `|d_scaling|` 接近 1。

## 方法（本实验）
方案 1：将 `HashGridDeformModel.scaling_warp` 的 `Tanh()` 改为 `0.75*Tanh()+0.25`（d_scaling ∈ [-0.5, 1]），其余不变。

## 结果
（待训练完成后填写）
