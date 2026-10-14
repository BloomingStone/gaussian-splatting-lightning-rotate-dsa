# X-ray / DSA Gaussian Splatting

基于 **3D Gaussian Splatting** 的旋转 X 射线（rotational X-ray / DSA）造影序列重建框架。
训练与推理由 PyTorch Lightning 驱动，环境由 [pixi](https://pixi.sh/) 管理。

支持的典型任务：

| 任务 | 说明 | 代表配置 |
|---|---|---|
| 静态 3D 重建 | 从一圈投影重建静态三维密度体 | `configs/static/static.yaml` |
| 形变 / 4D 重建 | 高斯位置或密度随时间形变，建模心动周期 | `configs/gen_4d_output_all/motion.yaml` |
| 造影剂流建模 | 显式建模碘造影剂在血管内的流动 | `configs/gen_4d_output_all/flow-mlp.yaml` |
| 4D GS | 用傅里叶分量记录随时间变化的密度 | `configs/4D_dsa_generated/rotate_xray_4D.yaml` |
| DICOM 直读 | 跳过 PNG/TIFF 中间产物，直接读多帧旋转 DSA `.dcm` | `configs/dicom/dicom_chest.yaml` |

## 目录

- [1. 环境安装](#1-环境安装)
- [2. 快速开始](#2-快速开始)
- [3. 命令与参数](#3-命令与参数)
- [4. 数据目录结构](#4-数据目录结构)
- [5. Config 配置结构](#5-config-配置结构)
- [6. 主要 Config 示例与对应实验](#6-主要-config-示例与对应实验)
- [7. 输出产物](#7-输出产物)
- [8. 批量实验与结果汇总](#8-批量实验与结果汇总)
- [9. 测试](#9-测试)

---

## 1. 环境安装

此项目依赖 [pixi](https://pixi.sh/) 配置python环境。系统环境为(仅供参考)

```txt
OS: Ubuntu 22.04.3 LTS (Jammy Jellyfish) x86_64
Kernel: Linux 6.8.0-136-generic
GPU: NVIDIA GeForce RTX 3090
cuda-runtime:
    NVIDIA-SMI 580.173.02, Driver
    Version: 580.173.02
    CUDA Version: 13.0
cuda-toolkit(nvcc): release 12.0, V12.0.76
```

解压项目后，然后进入到项目目录中，使用 `pixi install --frozen` 从 `pixi.lock` 安装环境。该方法确保环境的一致性，避免因依赖版本变化引起的问题。

如果安装失败，可能是由于某些依赖包在当前环境中无法安装。可以尝试以下步骤：

+ 使用 `pixi install` 重新安装，这将允许pixi根据当前环境自动调整依赖版本。
+ 如果仍然无法安装，检查错误日志，确定是哪个包导致安装失败，并修改 `pyproject.toml`
  中 `[tool.pixi.dependencies]` 里对应包的版本

安装完成后，可以使用 `pixi shell` 进入环境，或者使用 `pixi run <command>` 直接运行命令，例如 `pixi run python XXX.py` 可以运行python脚本。

> 本项目使用自定义 CUDA 光栅化扩展（`xray-gaussian-rasterization-voxelization`、
> `diff_gaussian_rasterization`、`fused_ssim` 等），首次 import 会触发 CUDA 编译，
> 耗时较长且需与当前 CUDA 版本匹配。

---

## 2. 快速开始

下面两条命令都在本仓库自带数据上验证过（RTX 3090，各约 1 分钟）。

### 2.1 DICOM 直读

直接读取 `data/dicom/RXA_chest.dcm`（187 帧、634×862、uint16），无需任何预处理：

```bash
pixi run gs-fit --config configs/dicom/dicom_chest.yaml \
  --data.path data/dicom \
  --max_steps 1000 \
  -n dicom_chest_1k \
  --trainer.check_val_every_n_epoch 1
```

### 2.2 X-ray PNG 序列

使用 `rotate_dsa.json` + `rotate_dsa/*.png` 形式的序列：

```bash
pixi run gs-fit --config configs/gen_4d_output_all/static.yaml \
  --data.path data/gen_4d_output_all/static/asoca-diseased__Diseased_02__LCA \
  -n my_recon \
  --overwrite_output
```

### 2.3 其他任务的等价命令

> 2.1 / 2.2 已实测通过；下面几条只保证数据路径与配置在仓库中存在，未逐一实跑。

```bash
# 形变（4D）重建
pixi run gs-fit --config configs/gen_4d_output_all/motion.yaml \
  --data.path data/gen_4d_output_all/motion/Diseased_02__LCA -n motion_recon

# 造影剂流建模
pixi run gs-fit --config configs/gen_4d_output_all/flow-mlp.yaml \
  --data.path data/gen_4d_output_all/flow/asoca-diseased__Diseased_02__LCA -n flow_recon

# 猪心数据（TIFF + 相位）
pixi run gs-fit --config configs/4D_dsa/pigdata_3d_deform.yaml --data.path data/pigdata -n pig_recon
```

注意各数据目录的命名前缀并不统一：`gen_4d_output_all/static` 与 `flow` 下的 case 形如
`asoca-diseased__Diseased_02__LCA`，而 `motion` 下是 `Diseased_02__LCA`。用 `ls` 确认实际名称。

指定使用的 GPU：

```bash
CUDA_VISIBLE_DEVICES=3 pixi run gs-fit --config ... --data.path ...
```

---

## 3. 命令与参数

### 3.1 入口与子命令

统一入口是 `main.py`，它有 4 个 LightningCLI 子命令：

```bash
python main.py <fit|validate|test|predict> [选项]
```

| 子命令 | 作用 |
|---|---|
| `fit` | 训练 |
| `validate` | 加载已有 checkpoint，在 `val` 划分上评估 |
| `test` | 加载已有 checkpoint，在 `test` 划分上评估 |
| `predict` | 仅推理 |

等价的 pixi task 与 console script（省去 `python main.py`）：

| pixi task | 等价命令 | console script |
|---|---|---|
| `pixi run gs-fit` | `python main.py fit` | `gs-fit` |
| `pixi run gs-val` | `python main.py validate` | `gs-val` |
| `pixi run gs-test` | `python main.py test` | `gs-test` |
| `pixi run gs-predict` | `python main.py predict` | `gs-predict` |

评估已有 checkpoint：

```bash
# 显式指定 checkpoint
pixi run gs-test --config configs/dicom/dicom_chest.yaml \
  --data.path data/dicom \
  --ckpt_path outputs/dicom_chest_1k/checkpoints/epoch=7-step=1000.ckpt

# 也可以省略 --ckpt_path，程序会自动在 outputs/<name> 下查找
pixi run gs-val --config configs/dicom/dicom_chest.yaml \
  --data.path data/dicom -n dicom_chest_1k

# 从 checkpoint 续训（last = 自动找最新）
pixi run gs-fit --config configs/dicom/dicom_chest.yaml \
  --data.path data/dicom -n dicom_chest_1k --ckpt_path last
```

> 本仓库的 checkpoint 会把 numpy 标量写进 hyper-parameters，而 torch ≥ 2.6 收紧了
> `torch.load(weights_only=...)` 的默认值，restricted 加载器会直接拒绝它们
> （`UnpicklingError: Unsupported global: numpy._core.multiarray.scalar`）。
> `internal/utils/patches/fix_lightning_ckpt_weights_only.py` 已把 CLI 的参数解析与
> 实际加载都切到宽松加载器，所以上面的命令开箱可用；若要恢复受限加载（例如只加载
> 可信来源之外的权重），显式传 `--weights_only true`。

> 需要交互式查看渲染结果时用 `gs-viewer`，见 [3.2 Web 查看器](#32-web-查看器)。

### 3.2 Web 查看器

交互式查看训练结果（`internal/web_viewer/new_viewer.py`）：

```bash
# 启动服务（默认监听 0.0.0.0:8080）
pixi run gs-viewer outputs/dicom_chest_1k --port 8080 --show_cameras

# 直接指定 checkpoint、背景色、相机文件
pixi run gs-viewer outputs/dicom_chest_1k/checkpoints/epoch=7-step=1000.ckpt \
  --port 8080 --background_color black \
  --cameras_json outputs/dicom_chest_1k/cameras.json

# 多模型叠加查看，并保存截图到指定目录
pixi run gs-viewer outputs/run_a outputs/run_b --output_dir /tmp/shots
```

浏览器打开 `http://<服务器地址>:8080`。

**在服务器上训练、在本机查看（SSH 端口转发）：**

```bash
# 在本机执行；把服务器的 8080 转发到本机 8080
ssh -N -L 8080:127.0.0.1:8080 <user>@<server>
```

然后在本机浏览器打开 `http://127.0.0.1:8080`。
若本机 8080 已被占用，改成本机其他端口即可，例如 `-L 8888:127.0.0.1:8080`。

> 如果不做端口转发，也可以直接在服务器上装 X11 转发（`ssh -X`）后用本机浏览器访问，
> 但性能不如端口转发。

界面提供的控制项：

| 分组 | 内容 |
|---|---|
| Render | 输出类型（如 `gray` / `coronary`）、Max Clamp、Color Map、Scaling |
| Resolution | 静止 / 移动时的最大分辨率与 JPEG 质量 |
| Time | 时间滑块（形变 / 4D 模型，`--time_max` 可调上界） |
| Training cameras | 在 `cameras.json` 记录的训练相机间跳转（便于与 GT 逐帧对照） |
| Scene | 背景色、是否显示相机视锥、场景 up 方向 |
| Save frame（按钮） | 把当前视角渲染成 PNG，写入 `--output_dir`（默认 `./outputs`） |

`cameras.json` 会从模型目录自动查找，用于绘制相机视锥与时间范围；找不到时其余功能不受影响。

**训练过程中实时查看**（`fit --viewer`）：

```bash
pixi run gs-fit --config configs/dicom/dicom_chest.yaml --data.path data/dicom --viewer
```

训练进程内会起一个 viser 服务（`0.0.0.0:8080`），按上面的端口转发即可边训练边看。
浏览器可自由环绕，并用 `Training cameras → Jump to` 跳到某个训练视角与 GT 逐帧对照。

实现方式：浏览器把相机位姿放进请求队列，训练循环在每个 train / val batch 取出并渲染回传
（`internal/web_viewer/training_viewer.py`）。**没有浏览器连接时开销为 0**
（请求队列用 `get_nowait()` 排空）。需要注意：

* 额外渲染发生在训练主线程，且仅在有人操作时触发（每 batch 约 10 µs）；
* 多卡时只在 rank 0 生效；
* **训练结束后服务不会退出**：`KeepRunningIfWebViewerEnabled` 会让它继续响应浏览器
  （终端提示 `Training finished! Web viewer is still running. Press Ctrl+C to exit.`），
  不再需要时按 `Ctrl+C` 结束；若随后想换个模型再看，用上面的 `gs-viewer`；
* 浏览器长时间卡住时单次请求会在 15s 后超时并返回空帧，不会影响训练。

### 3.3 顶层参数

以下参数由 `internal/cli.py` 在 LightningCLI 之上追加。

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--config` | path | — | 加载 yaml 配置（可与命令行参数混用，命令行优先级更高） |
| `--data.path` | str | 必填 | 数据目录 |
| `--ckpt_path` | str | — | checkpoint 路径；`validate`/`test`/续训用，也支持 `last`（自动找最新） |
| `--weights_only` | bool | `false` | 传给 `torch.load` 的 `weights_only`；默认已由补丁设为 `false`，显式传 `true` 可恢复受限加载 |
| `--max_steps` | int | `30000` | 覆盖 `trainer.max_steps` |
| `--max_epochs` | int | — | 覆盖 `trainer.max_epochs` |
| `-n`, `--name` | str | 自动 | 输出目录名，输出到 `outputs/<name>`；不指定时自动取 `data.path` 的最后 3 段 |
| `-v`, `--version` | str | — | 版本子目录，输出到 `outputs/<name>/<version>` |
| `--output` | path | `<repo>/outputs` | 输出根目录 |
| `--logger` | str | `tensorboard` | 日志后端 |
| `--project` | str | `Gaussian-Splatting` | WandB 项目名 |
| `--viewer` | flag | off | 训练时启动 Web 查看器（监听 `0.0.0.0:8080`，见 3.2） |
| `--val_train` | flag | off | 用训练集做验证 |
| `--pbar_rate` | int | — | 进度条刷新间隔 |
| `-f`, `--float32_matmul_precision` | `medium\|high\|highest` | — | 矩阵乘精度，提速用 |
| `--overwrite_output` | flag | off | 允许覆盖已存在的输出目录（默认存在即报错） |

`trainer.*` 下所有 Lightning Trainer 参数都可直接覆写，例如
`--trainer.check_val_every_n_epoch 1`、`--trainer.precision 16-mixed`。

> 注意 `--trainer.val_check_interval` 不能大于一个 epoch 的 batch 数
> （默认划分下为训练帧数，如 149），否则 Lightning 会直接报错。

### 3.4 参数覆写语法

配置里的**任意**字段都能在命令行上用 jsonargparse 的点路径覆写，命令行优先级高于 yaml：

```bash
pixi run gs-fit --config configs/dicom/dicom_chest.yaml --data.path data/dicom \
  --data.parser.init_args.meta_loader.init_args.normalization percentile \
  --data.parser.init_args.meta_loader.init_args.gamma_target 0.5 \
  --data.parser.init_args.cloud_parser.init_args.num_points 50000 \
  --data.parser.init_args.cameras_builder.init_args.use_projection_matrix true \
  --model.metric.init_args.w_ssim_loss 2.0
```

> 列表 / 字典值需加引号：`--trainer.callbacks.0.init_args.saver.init_args.save_iterations "[1000,5000]"`。

---

## 4. 数据目录结构

### 4.1 X-ray 序列格式（`configs/static`、`configs/gen_4d_output_all` 等）

```
data/<case>/
├── rotate_dsa.json          # 元数据（必需）
├── rotate_dsa/              # 投影图像序列（必需）
│   ├── 000.png 001.png ...  # 按字典序与帧序号一一对应
├── label/                   # 可选：逐帧掩膜（同名 *.png，非零为有效像素）
├── depth_map.npz            # 可选：逐帧深度图，作为 extra_data["depth"]
├── coronary_label.nii.gz    # 可选：3D 血管标签，供 3D 指标 / 标签点云初始化
└── central_line.npz         # 可选：血管中心线点云，供中心线点云初始化
```

`rotate_dsa.json` 关键字段：

| 字段 | 说明 |
|---|---|
| `coronary_type` | `LCA` / `RCA`，决定用哪个 `*_centering_affine` |
| `volume_size` | `[sx, sy, sz]`，体积尺寸（mm） |
| `lca_centering_affine` / `rca_centering_affine` | `4×4` 体积仿射（体素 → 世界坐标） |
| `c_arm_geometry` | `sdd` / `sod` / `height` / `width` / `delx` / `dely` / `x0` / `y0` / `orientation_type` |
| `rotate_parameters` | `total_frame` / `alpha_start` / `beta_start` / `angular_velocity` / `fps` / `convention` |
| `frames` | 每帧 `{frame, time_s, phase, alpha_degree, beta_degree}` |
| `postprocess_meta` | 可选的归一化 / 噪声参数 |

若使用 TIFF 数据集（如 `data/pigdata`），则放置 `rotate_dsa.tiff` 并在配置中改用
`TiffDatasetBuilder`（见 `configs/4D_dsa/*.yaml`）。

### 4.2 DICOM 格式（`configs/dicom/dicom_chest.yaml`）

```
data/dicom/
└── RXA_chest.dcm            # 多帧旋转 DSA（NumberOfFrames > 1）
```

几何、每帧角度、时间、相位与像素全部从 DICOM 头读取，无需 JSON。用到的标签：

| 标签 | 含义 | 缺失时行为 |
|---|---|---|
| `(0018,1110)` DistanceSourceToDetector | SDD | **必需** |
| `(0018,1111)` DistanceSourceToPatient | SOD | **必需** |
| `(0028,0010)` / `(0028,0011)` Rows / Columns | 探测器尺寸 | **必需** |
| `(0018,1164)` ImagerPixelSpacing | 像素间距（探测面） | 回退 `(0028,0030)` PixelSpacing |
| `(0028,0008)` NumberOfFrames | 帧数 | 默认 1 |
| `(0018,0040)` CineRate | 帧率 | 回退 `(0008,2144)`，再回退 `1/FrameTime`，最后 30 |
| `(0018,1520)` / `(0018,1521)` PositionerPrimary/SecondaryAngleIncrement | 每帧 α/β 角 | 回退起始角 `(0018,1510/1511)`，再回退 0 |
| `(0018,1065)` FrameTimeVector | 每帧间隔（ms），累积为时间 | 退化 / 缺失时用 `i / fps` |
| `(0071,1010)` 私有相位数组 | 心动相位 | 全 0 |
| `(0019,1003)` / `(0021,1005)` 私有投影矩阵 | 厂商 3×4 投影矩阵 | 忽略，走理想圆模型 |

要求 `SamplesPerPixel == 1`、`BitsAllocated == 16`（非压缩 MONOCHROME）。

---

## 5. Config 配置结构

配置即 LightningCLI 的 yaml，顶层三大段：`trainer` / `data` / `model`，外加项目级字段。

```yaml
project: my_exp            # 可选，实验名（记入日志）
max_steps: 20000           # 顶层便捷字段，等价于 trainer.max_steps

trainer:                   # ① Lightning Trainer
  check_val_every_n_epoch: 10
  precision: 16-mixed
  callbacks:               # 回调：保存场景 / 验证 / 保存图片 / 进度条
    - class_path: internal.callbacks.scene_saver.SceneSaver
      init_args:
        save_iterations: [5000, 10000, 20000]
        saver:
          class_path: internal.savers.x_ray_saver_static.XRaySaver_Static
          init_args:
            save_ckpt: true
            save_vtp: true
            save_volume: true
    - class_path: internal.callbacks.save_image.SaveImage
      init_args:
        save_val_output: true
        max_save_val_output: 10

data:                      # ② 数据
  parser:                  # 由 DataParserBuilder 组装，见 5.2
    class_path: internal.dataparsers.dicom_dataparser.parser.DicomDataParserBuilder
    init_args: { ... }

model:                     # ③ 模型
  renderer: { class_path: ... }   # 渲染器
  metric:   { class_path: ... }   # 损失 / 指标
  gaussian: { class_path: ... }   # 高斯参数化
  density:  { class_path: ... }   # 致密化 / 剪枝
```

`data` 段除 `parser` 外还有 `DataModule` 的字段：

| 字段 | 默认 | 说明 |
|---|---|---|
| `path` | — | 数据目录（通常用 `--data.path` 传） |
| `num_workers` | `2` | 可用 `{train, val, test}` 分别指定 |
| `max_cache_num` | `-1` | 缓存图像数，`-1` 表示全部缓存 |
| `async_caching` | `false` | 异步预取缓存 |
| `shuffle_seed` | `42` | 训练集乱序种子 |
| `val_on_train` | `false` | 用训练集做验证 |

`model` 段字段（`GaussianSplatting.__init__`）：`gaussian`、`renderer`、`metric`、
`density`、`background_color`、`output_path`、`save_val_metrics`、`web_viewer`、
`initialize_from`、`drop_optimizer_states`。

> **写法约定**：需要实例化的类写成 `{class_path: <完整模块路径>, init_args: {...}}`；
> 嵌套 dataclass 字段可省略 `class_path`，只写 `init_args`（如 `dataset_config`）。

### 5.1 图像与掩膜

* `ImagesDatasetBuilder` 读取 `image_dir_name/*.png`（默认 `rotate_dsa`）与可选的
  `label/*.png` 掩膜；掩膜为**非零即有效**，并会转成 soft weight。
* `FrangiImagesDatasetBuilder` 在其上额外计算 Frangi 血管增强图，供
  `RotateXrayMetricsWithMasks` / `RotateXrayMetricsWeightPatch` 使用。
* `DicomDatasetBuilder` 直接从内存中的 DICOM 解码结果提供 `(C, H, W)` GT 帧，
  **不读取任何图像文件**。

### 5.2 `data.parser`：五个可插拔组件

`DataParserBuilder.build()` 把下面五者组装成 `DataParser`，产出
`{splits, datasets, point_cloud}`：

| 组件 | 职责 | 常用实现 |
|---|---|---|
| `meta_loader` | 读元数据 | `XRayMetaLoader`（JSON）、`DicomMetaLoader`（DICOM） |
| `cloud_parser` | 生成初始高斯位置 | `RandomCloudParser`、`UniformCloudParser`、`BallRandomCloudParser`、`LabelCloudParser`、`CentralLineCloudParser`、`FdkCloudParser` |
| `spliter` | 划分 train/val/test | `ReconstructionSpliter`（全用）、`UniformIntervalSpliter`、`RenderNewViewsSpliter`、`PhaseStratifiedSpliter`、`FileSpliter` |
| `cameras_builder` | 构建相机 | `RotateXRayCamerasBuilder`、`DicomCamerasBuilder` |
| `dataset_builder` | 产出逐帧样本 | `ImagesDatasetBuilder`、`FrangiImagesDatasetBuilder`、`TiffDatasetBuilder`、`DicomDatasetBuilder` |

`DataParser` 还有 `filter_visible_points`（默认 `true`）：剔除在所有相机视锥外的初始点。

### 5.3 DICOM 解析器参数

**`DicomMetaLoader`**：

| 参数 | 默认 | 说明 |
|---|---|---|
| `dcm_filename` | `RXA_chest.dcm` | 文件名或 glob；缺失时回退取目录下第一个 `*.dcm` |
| `orientation_type` | `AP` | `AP` / `PA`；DICOM 不存此项故需配置 |
| `normalization` | `minmax` | `minmax` 全域归一化；`percentile` 先按分位裁剪再归一化 |
| `percentile` | `(0.01, 0.99)` | `normalization=percentile` 时的裁剪窗口 |
| `gamma` | `none` | 显式 gamma 指数 |
| `gamma_target` | `none` | 未设 `gamma` 时，自动解出使全局中位数映射到该值的 gamma |
| `roi` | `none` | `(x0, y0, w, h)` 像素裁剪；同时修正相机内参，保证投影精确 |
| `flip_u` | `true` | 镜像探测器 `u` 轴（DICOM 显示坐标与投影物理坐标差一次左右镜像） |
| `volume_size` | `none` | 重建体积尺寸（mm）；缺省由等中心 FOV 推导为 `(fov_x, fov_x, fov_z)` |

**`DicomCamerasBuilder`**：`use_projection_matrix`（默认 `false`）。`false` 走理想圆
`α/β + SOD/SDD` 模型；`true` 时若私有标签带厂商 3×4 投影矩阵，则逐帧从矩阵解出精确几何
（可表达非理想圆、探测器滚转、主点漂移等）。

**`DicomDatasetBuilder`**：`dataset_config.channels`（默认 `3`）、
`dataset_config.image_uint8`（默认 `false`）。
`channels` 默认 3 是因为 `SaveImage` 回调会把 GT 与渲染器输出的 3 通道 RGB 沿宽度拼接，
通道数不一致会导致验证图保存失败；设为 `1` 可得到真正的单通道目标（代价是验证图无法保存）。

---

## 6. 主要 Config 示例与对应实验

`configs/` 按实验主题分目录：

| 配置 | 实验 / 用途 | renderer / metric / gaussian |
|---|---|---|
| **`dicom/dicom_chest.yaml`** | DICOM 直读静态重建（`DicomDataParserBuilder`） | `XrayRenderer` / `StaticXrayMetrics` / `XrayCoronaryGaussian` |
| `static/static.yaml` | xray PNG 序列静态重建基线 | `XrayRenderer` / `StaticXrayMetrics` / `XrayCoronaryGaussian` |
| `gen_4d_output_all/static.yaml` | ASOCA 生成数据集的静态重建 | 同上 |
| `gen_4d_output_all/motion.yaml` | 形变（4D）重建，deform 输入 `(x, t)` | `CoronaryDeformableXrayRenderer` / `RotateXrayMetrics` / `XrayCoronaryGaussian` |
| `gen_4d_output_all/motion-t-input.yaml` | 时间作为显式输入的形变重建 | `CoronaryDeformableXrayRenderer` |
| `gen_4d_output_all/flow-mlp.yaml` | 造影剂流建模（MLP deform + flow） | `DeformableXrayRendererWithFlow` / `XrayCoronaryGaussianWithFlow` |
| `gen_4d_output_all/flow-dynGS.yaml` | 动态 GS（傅里叶密度频率） | `Xray4DRender` / `Xray4DMetrics` / `Xray4DGaussian` |
| `4D_dsa_generated/rotate_xray_4D.yaml` | 4D GS：密度随时间变化的傅里叶分量 | `Xray4DRender` / `Xray4DMetrics` / `Xray4DGaussian` |
| `4D_dsa_generated/rotate_xray_with_flow.yaml` | 形变 + 流 | `DeformableXrayRendererWithFlow` / `XrayCoronaryGaussianWithFlow` |
| `mid_term/mid_term.yaml`、`mid_term_flow.yaml` | 中期形变 / 流实验 | 同 motion / flow-mlp |
| `rotate_xray_3dgr/frangi_mask.yaml` | Frangi 血管增强掩膜监督 | `FrangiImagesDatasetBuilder` / `RotateXrayMetricsWithMasks` |
| `rotate_xray_3dgr/patch_metric.yaml` | 局部 patch 加权指标 | `FrangiImagesDatasetBuilder` / `RotateXrayMetricsWeightPatch` |
| `rotate_xray_3dgr/rotate_xray_3dgr_deformable_xray_render.yaml` | 形变 xray 渲染主配置 | `CoronaryDeformableXrayRenderer` / `RotateXrayMetrics` |
| `rotate_xray_3dgr/rotate_xray_3dgr.yaml` | 3DGR 基线（自定义 renderer 组） | `DeformableRenderer` |
| `compares/init-points-*.yaml` | 点云初始化消融：点数（5K/20K/40K）、初始化策略（A/B/baseline）、密度阈值（`fixedth4/6/8`、`gp90/95/995`） | `DeformableXrayRendererWithFlow` |
| `4D_dsa/pigdata_*.yaml` | 猪心数据（TIFF + 相位，`TiffDatasetBuilder`） | `CoronaryDeformableXrayRenderer` / `Xray4DRender` |
| `experiments/*.yaml` | Hydra 批量实验定义（配合 experiment_manager） | — |

### 6.1 DICOM 配置示例（完整）

`configs/dicom/dicom_chest.yaml` 关键部分：

```yaml
max_steps: 20000

data:
  parser:
    class_path: internal.dataparsers.dicom_dataparser.parser.DicomDataParserBuilder
    init_args:
      meta_loader:
        class_path: internal.dataparsers.dicom_dataparser.meta.DicomMetaLoader
        init_args:
          dcm_filename: RXA_chest.dcm
          orientation_type: AP
          normalization: minmax
          flip_u: true
      cloud_parser:
        class_path: internal.dataparsers.xray_dataparser.cloud_parsers.RandomCloudParser
        init_args: { num_points: 10_000 }
      spliter:
        class_path: internal.dataparsers.xray_dataparser.splitters.UniformIntervalSpliter
        init_args: { train_ratio: 0.8 }
      cameras_builder:
        class_path: internal.dataparsers.dicom_dataparser.cameras_builder.DicomCamerasBuilder
        init_args: { use_projection_matrix: false }
      dataset_builder:
        class_path: internal.dataparsers.dicom_dataparser.datasets.DicomDatasetBuilder

model:
  renderer: { class_path: internal.renderers.xray_renderer.XrayRenderer }
  metric:
    class_path: internal.metrics.rotate_xray_metrics_static.StaticXrayMetrics
    init_args: { fused_ssim: true, w_gray_loss: 1.0, w_ssim_loss: 1.0 }
  gaussian: { class_path: internal.models.xray_coronary_gaussian.XrayCoronaryGaussian }
  density:
    class_path: internal.density_controllers.rotate_xray_density_controller.RotateXrayDensityController
    init_args: { densification_interval: 300, densify_grad_percentile: 0.98, cull_density_threshold: 5e-5 }
```

DICOM 解析器复用了 `xray_dataparser` 的 `cloud_parser` / `spliter` —— 因为 `DicomMeta`
继承自 `XRayMeta`，两者接口一致，组件可直接互换。

### 6.2 切换实验的两种方式

1. **换配置**：`--config configs/.../xxx.yaml`（配置里已含对应的
   renderer / metric / gaussian / density）。
2. **命令行覆写**：保留同一份配置，只改需要变的字段。例如把静态重建设成 4D：

```bash
pixi run gs-fit --config configs/dicom/dicom_chest.yaml --data.path data/dicom \
  --model.renderer.class_path internal.renderers.xray_4d_renderer.Xray4DRender \
  --model.metric.class_path internal.metrics.Xray4DMetrics
```

---

## 7. 输出产物

输出根目录为 `outputs/<name>[/<version>]`：

```
outputs/<name>/
├── checkpoints/            # epoch=..-step=..ckpt
├── lightning_logs/         # TensorBoard 事件、config.yaml、hparams.yaml
├── metrics/                # val-step=<n>.csv / test-step=<n>.csv（逐帧 + MEAN 行）
├── val/                    # 验证可视化 epoch=..-step=../<image_name>.jpg
│                           #   每张图 = GT | 各渲染输出 横向拼接
├── volumes/                # volume__epoch=..-step=...nii.gz（save_volume: true）
├── point_cloud/            # iteration_<n>.vtp（高斯点云）
├── splits.json             # train/val/test 帧索引
├── cameras.json            # 相机参数
└── input.vtp               # 初始点云
```

* `save_volume: true` 会用光栅化把高斯渲染成体积网格，affine 取自数据元信息
  （世界范围即 `volume_size`，以等中心为原点），可直接用 3D Slicer / `nibabel` 打开。
* `save_label_threshold: <float>` 可额外导出二值 mask。

查看结果：

```bash
# 体积形状 / 仿射
pixi run python -c "import nibabel as nib; i=nib.load('outputs/<name>/volumes/volume__epoch=7-step=1000.nii.gz'); print(i.shape, i.affine)"

# 验证对比图（每张 = GT | 渲染输出 横向拼接），直接用图片查看器打开
ls outputs/<name>/val/epoch=*/*.jpg

# 指标曲线（逐帧 + MEAN 行）
column -s, -t outputs/<name>/metrics/val-step=1000.csv
```

如需把体积切成三视图预览，可用 `matplotlib` + `nibabel` 自行渲染
（`dataobj` 为 `float32`，建议先按 40~99.5 百分位做窗宽窗位）。

---

## 8. 批量实验与结果汇总

多 case / 多超参的批量实验由 `scripts/experiment_manager` 负责，配置放在
`configs/experiments/`（Hydra 格式，含 `study_name`、`train:`、`sweeps:` 三段）。

```bash
# 干运行：只展开 sweep，不真正训练
python -m scripts.experiment_manager --config-name smoke train.dry_run=true

# 运行 Q1 扫参（形变网络的 t_multires × x_multires × 容量）
python -m scripts.experiment_manager --config-name q1 train.results_root=/path/to/out

# 仅对有 checkpoint 的 case 重跑 test 与指标
python -m scripts.experiment_manager --config-name q1 train.eval_only=true

# 汇总为 CSV 与图表
python scripts/summarize_experiments.py --results-root /path/to/out --study Q1
```

`configs/experiments/` 中已有的 study：

| study | 内容 |
|---|---|
| `smoke.yaml` | 最小冒烟（单个 HashGrid 配置） |
| `q1.yaml` | 形变网络容量：`HashGridDeformConfig` / `MLPDeformConfig` 的 `t_multires`、`x_multires`、`max_resolution` / `D` |
| `q2.yaml` | 稀疏视角下的划分策略：`random-shuffle` vs `random-start` |
| `q3.yaml` | 损失项组合：基线 / 全变分 TV / Frangi / TV+Frangi |
| `q4.yaml` | 形变网络复扫（sweep 名同 Q1） |
| `q6.yaml`、`q6_1.yaml` | 运动与流的正则项：`motion_var`、`motion_mean`、`coronary_props`、`coronary_props_entropy`、`motion_corr`、`density_corr` |

详见 [`scripts/experiment_manager/README.md`](scripts/experiment_manager/README.md)。

---

## 9. 测试

```bash
# 全部测试
pixi run python -m pytest tests -q

# 仅 dataparser 相关（DICOM 部分 30 项全通过）
pixi run python -m pytest tests/test_dataparsers -q
pixi run python -m pytest tests/test_dataparsers/test_dicom_dataparser -q
```

关于数据依赖，两类行为不同：

* `tests/test_dataparsers/test_dicom_dataparser` 在 `data/dicom/RXA_chest.dcm` 缺失时会
  自动 `skip`。
* `tests/test_dataparsers/test_xray_dataparser`、`tests/test_metrics` 等依赖
  `data/test_data`、`data/test_data_motion`、`data/pigdata/coronary_label.nii.gz` 的用例，
  在这些数据未提供时会**直接报错**（`Meta file not found` / `No such file`）。
  跑全量 `pytest tests` 时出现这批失败属预期现象，与代码改动无关。