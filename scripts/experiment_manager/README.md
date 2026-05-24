# Experiment Manager

批量实验编排与结果分析工具包。用于自动化管理大规模 Gaussian Splatting 重建实验的超参数搜索、并行执行、指标统计与可视化。

## 目录结构

```
scripts/experiment_manager/
├── __init__.py          # 公开 API
├── __main__.py          # CLI 入口（Hydra 实验编排）
├── _types.py            # 共享数据类型（6 个 dataclass）
├── _logging.py          # 线程安全日志
├── _utils.py            # 纯工具函数
├── config.py            # ① 配置加载
├── filesystem.py        # ② 实验文件夹创建
├── runner.py            # ③ 并行实验
├── metrics.py           # ④ 单 case 结果统计
├── stats.py             # ⑤ 统计汇总到 CSV
├── visualize.py         # ⑥ 可视化生成
└── summary.py           # 编排层
```

## 架构

```
_types ──→ _utils ──→ {config, filesystem, metrics, stats}
                          │              │
                       runner ────────────┘
                          │
               visualize ← stats
                          │
                      summary ──→ __main__
```

单向无环依赖，模块按职责分离。

| 模块 | 职责 |
|------|------|
| `config.py` | 数据扫描、sweep 展开（grid / single-variable）、参数去重 |
| `filesystem.py` | checkpoint 检查、alias 符号链接、`run_summary.json` 写入 |
| `runner.py` | ProcessRegistry 进程管理、单/多 GPU 并行调度 |
| `metrics.py` | 2D 指标 CSV 读取、进程内 3D 评估 |
| `stats.py` | run_meta 解析、异常值检测、跨 run 聚合、CSV 输出 |
| `visualize.py` | 柱状图、热力图（2D/3D 网格） |
| `summary.py` | study_summary 生成 + 调用 stats + visualize |

## 快速开始

### 1. 运行实验

```bash
# 基础用法
python -m scripts.experiment_manager --config-name q1

# 指定输出目录
python -m scripts.experiment_manager --config-name q1 train.results_root=/path/to/output

# 干运行（不实际执行训练）
python -m scripts.experiment_manager --config-name smoke train.dry_run=true

# 仅评估已有 checkpoint
python -m scripts.experiment_manager --config-name q1 train.eval_only=true
```

### 2. 汇总结果

```bash
python scripts/summarize_experiments.py --results-root /path/to/output --study my_study

# 指定指标和绘图参数
python scripts/summarize_experiments.py \
  --results-root /path/to/output \
  --study my_study \
  --plot-metrics 3d_dice 3d_hd95 2d_psnr \
  --plots-per-row 4
```

## 配置文件

配置使用 [Hydra](https://hydra.cc/) 管理，位于 `configs/experiments/`。

### 基础配置 (`base.yaml`)

```yaml
train:
  data_root: /path/to/dataset          # 数据集根目录
  base_config: configs/.../model.yaml  # 模型基础配置
  results_root: /path/to/output        # 结果输出根目录
  runner: pixi run python              # Python 启动器
  gpu: "0"                             # 默认 GPU
  parallel_mode: multi_gpu             # serial | multi_gpu
  gpus: ["0", "1", "2", "3"]          # 多 GPU 列表
  run_test: true
  run_3d_eval: true
  skip_existing: true                  # 跳过已完成 case
  retries: 0                           # 失败重试次数
  dry_run: false                       # 干运行模式
  case_filter:
    include_pattern: ""                # 按名称筛选 case
    exclude_pattern: ""
  static_overrides: {}                 # 所有 run 共享的固定参数覆写
```

### 实验配置 (`q1.yaml` 等)

```yaml
defaults:
  - base

study_name: my_study

sweeps:
  - name: hashgrid_resolution
    mode: grid                         # grid | single-variable
    fixed:
      model.renderer.init_args.deform_model_config.class_path: "...HashGridDeform"
    grid:
      model.renderer.init_args.deform_model_config.init_args.t_multires: [6, 7, 8]
      model.renderer.init_args.deform_model_config.init_args.x_multires: [4, 5, 6]
```

## 命令行参数

所有 `train.*` 路径均可在命令行覆写：

| 参数 | 类型 | 说明 |
|------|------|------|
| `train.data_root` | str | 数据集根目录 |
| `train.base_config` | str | 模型基础配置路径 |
| `train.results_root` | str | 结果输出根目录 |
| `train.gpu` | str | 单 GPU 模式下的 GPU ID |
| `train.gpus` | list | 多 GPU 模式下的 GPU ID 列表 |
| `train.parallel_mode` | str | `serial` 或 `multi_gpu` |
| `train.dry_run` | bool | 干运行（跳过实际训练） |
| `train.eval_only` | bool | 仅评估模式 |
| `train.skip_existing` | bool | 跳过已有 checkpoint 的 case |
| `train.retries` | int | 失败重试次数 |
| `train.run_test` | bool | 是否运行 test |
| `train.run_3d_eval` | bool | 是否运行 3D 评估 |
| `train.seed` | int | 随机种子 |
| `train.summary_require_exact_sweep_match` | bool | 严格校验 sweep 与结果一致性 |
| `train.case_filter.include_pattern` | str | 包含 case 的匹配模式 |
| `train.case_filter.exclude_pattern` | str | 排除 case 的匹配模式 |
| `train.static_overrides` | dict | 所有 run 共享的固定参数覆写 |

## 输出结构

```
{results_root}/{study_name}/
├── study_summary.json           # 整个 study 的汇总
├── {run_name}/
│   ├── run_spec.json            # run 的超参数
│   ├── run_summary.json         # run 的汇总指标
│   ├── run_summary.csv          # run 的汇总表格
│   └── cases/
│       └── {case_name}/
│           ├── run_status.json  # case 执行状态
│           ├── manager_run.log  # 执行日志
│           ├── checkpoints/     # 模型 checkpoint
│           ├── metrics/         # 2D 指标 CSV
│           └── metrics_3d/      # 3D 评估结果
└── manager_logs/                # 管理器日志
```

## 公开 API

```python
from scripts.experiment_manager import (
    main,              # Hydra 实验编排入口
    summarize_main,    # argparse 汇总入口
    Logger,            # 线程安全日志
    RunSpec,           # 运行规格
    CaseTask,          # case 任务
    CaseResult,        # case 结果
    ScheduledTask,     # 调度任务
    RunState,          # 运行状态
    RunMeta,           # run 元信息（model, sweep_var 等）
)
```

## 向后兼容

原有的两个入口脚本保留为薄包装：

```bash
# 与原来完全相同
python -m scripts.experiment_manager --config-name q1
python scripts/summarize_experiments.py --study my_study
```

## 依赖

- Python ≥ 3.10
- `hydra-core` / `omegaconf` — 配置管理
- `matplotlib` — 可视化（仅 `visualize.py`）
- `scripts/metrics_3D/` — 3D 评估指标（可选，`metrics.py` 动态导入）
