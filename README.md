# Gaussian Splatting

## 存储格式：VTK PolyData (.vtp)

GS 模型的持久化使用 [PyVista PolyData](https://docs.pyvista.org/) 格式（`.vtp` 文件），替代了旧的 PLY 格式。

### PolyData 结构约定

```
pyvista.PolyData
├── points       : (N, 3) float32  — Gaussian 中心位置 (means / xyz)
├── point_data   : dict[str, np.ndarray]  — 各 GS 属性，按模型类型不同
└── field_data   : dict[str, np.ndarray]  — 全局元信息
```

**field_data 通用字段：**

| Key | 类型 | 说明 |
|-----|------|------|
| `model_type` | `str` | 模型类型标识，如 `"VanillaGaussian"`、`"XrayCoronaryGaussian"` |

---

### VanillaGaussian (标准球谐模型)

**point_data：**

| Key | Shape | 说明 |
|-----|-------|------|
| `shs_dc` | `(N, 1, 3)` | 球谐 DC 分量（原始值，未激活） |
| `shs_rest` | `(N, rest_dim, 3)` | 球谐高阶分量，rest_dim = (sh_degree+1)²−1 |
| `opacities` | `(N, 1)` | 不透明度（原始值，sigmoid 前） |
| `scales` | `(N, 3)` | 各向异性缩放（原始值，exp 前） |
| `rotations` | `(N, 4)` | 四元数旋转（原始值） |

**field_data：**

| Key | 类型 | 说明 |
|-----|------|------|
| `active_sh_degree` | `int32` | 当前激活的球谐阶数 |

---

### XrayCoronaryGaussian (X 射线冠脉模型)

不使用球谐和 opacity，用 density 替代。

**point_data：**

| Key | Shape | 说明 |
|-----|-------|------|
| `density` | `(N, 1)` | 密度（原始值，softplus 前） |
| `scales` | `(N, 3)` | 各向异性缩放（原始值，softplus 前） |
| `rotations` | `(N, 4)` | 四元数旋转（原始值） |

**可选 point_data（由 `save_motion` 配置控制，默认保存）：**

| Key | Shape | 说明 |
|-----|-------|------|
| `d_motion_mean` | `(N, 7)` | 运动统计量 E[motion]，各列为 d_xyz(3) + d_scale(3) + d_angle(1) |
| `d_motion_2_mean` | `(N, 7)` | 运动统计量 E[motion²] |

加载时若 motion 数据缺失，自动初始化为零。

---

### Python API

```python
# 保存
model.save_to_vtp("path/to/point_cloud.vtp")

# 加载（需要先实例化对应模型）
import pyvista as pv
polydata = pv.read("path/to/point_cloud.vtp")
model.setup_from_polydata(polydata)

# 导出为 PolyData 对象（不写文件）
pd = model.to_polydata()
```

### 自定义模型

子类需实现两个抽象方法：

```python
class MyGaussianModel(GaussianModel):
    def to_polydata(self) -> pv.PolyData:
        """将自身属性导出为 PolyData"""
        ...

    def setup_from_polydata(self, polydata: pv.PolyData, *args, **kwargs):
        """从 PolyData 恢复属性"""
        ...
```
