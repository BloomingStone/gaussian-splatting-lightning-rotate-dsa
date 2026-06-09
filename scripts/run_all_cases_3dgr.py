#!/usr/bin/env python3
"""
批量运行 3DGR-CAR 训练任务 (Python 版)

用法:
    ./scripts/run_all_cases_3dgr.py <config_file> <data_root> <experiment_name>
                                   [--splits <path>] [--phase {train,val,test}]
                                   [--num-views <N>] [--dry-run] [--overwrite-output]

示例:
    # 全部 case
    ./scripts/run_all_cases_3dgr.py configs/3DGRCAR/3DGR-CAR.yaml data/asoca 3DGR-CAR

    # 只跑 test 分组的 case
    ./scripts/run_all_cases_3dgr.py configs/3DGRCAR/3DGR-CAR.yaml data/asoca 3DGR-CAR \
        --splits scripts/splits/asoca_lca.json --phase test

    # 指定 num_views 并 dry-run 预览
    ./scripts/run_all_cases_3dgr.py configs/3DGRCAR/3DGR-CAR.yaml data/asoca 3DGR-CAR \
        --splits scripts/splits/asoca_lca.json --phase test --num-views 8 --dry-run
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cyclopts import App

app = App(
    name="run_all_cases_3dgr",
    help="批量运行 3DGR-CAR 训练任务",
)

# phase 参数 -> JSON 文件中的 key 映射
_PHASE_MAP: dict[str, str] = {
    "train": "train",
    "val": "validation",
    "test": "test",
}


@app.default
def main(
    config_file: Path,
    data_root: Path,
    experiment_name: str,
    *,
    splits: Path | None = None,
    phase: str = "test",
    num_views: int | None = None,
    dry_run: bool = False,
    overwrite_output: bool = False,
    logger: str | None = "wandb",
):
    r"""批量运行 3DGR-CAR 训练任务.
    
    example:
    # 跑全部 case (无 splits 过滤)
    python scripts/run_all_cases_3dgr.py configs/3DGRCAR/3DGR-CAR.yaml data/asoca 3DGR-CAR

    # 只跑 asoca_lca.json 中 test 分组的 20 个 case
    python scripts/run_all_cases_3dgr.py configs/3DGRCAR/3DGR-CAR.yaml data/asoca 3DGR-CAR \
        --splits scripts/splits/asoca_lca.json --phase test

    # 指定 num_views + dry-run 预览
    python scripts/run_all_cases_3dgr.py configs/3DGRCAR/3DGR-CAR.yaml data/asoca 3DGR-CAR \
        --splits scripts/splits/asoca_lca.json --phase test --num-views 8 --dry-run

    # imagecas 数据集 + validation 分组
    python scripts/run_all_cases_3dgr.py configs/3DGRCAR/3DGR-CAR.yaml data/imagecas 3DGR-CAR-imagecas \
        --splits scripts/splits/imagecas_lca.json --phase val

    Parameters
    ----------
    config_file:
        配置文件路径 (e.g. configs/3DGRCAR/3DGR-CAR.yaml)
    data_root:
        数据集根目录 (e.g. data/asoca), 同时作为 --data.path
    experiment_name:
        实验名称, 用于 -n (output name) 和 -v (output version) 参数, 输出路径将是 output/experiment_name/case_name
    splits:
        splits JSON 文件路径, 筛选需要执行的 case
    phase:
        splits 中的阶段: train / val / test (默认: test)
    num_views:
        覆盖 spliter.num_views (可选)
    dry_run:
        仅打印命令, 不实际执行
    overwrite_output:
        追加 --overwrite_output 参数
    logger:
        Logger 类型 (默认: wandb, 设为空字符串则不传)
    """
    # ── 验证输入 ────────────────────────────────────────────────
    if not config_file.is_file():
        print(f"错误: 配置文件不存在: {config_file}", file=sys.stderr)
        sys.exit(1)

    if not data_root.is_dir():
        print(f"错误: 数据目录不存在: {data_root}", file=sys.stderr)
        sys.exit(1)

    projs_dir = data_root / "projs"
    if not projs_dir.is_dir():
        print(f"错误: 找不到 projs 目录: {projs_dir}", file=sys.stderr)
        sys.exit(1)

    # ── 校验 phase ────────────────────────────────────────────
    phase_key = _PHASE_MAP.get(phase)
    if phase_key is None:
        print(f"错误: 无效的 phase '{phase}', 可选: {list(_PHASE_MAP.keys())}", file=sys.stderr)
        sys.exit(1)

    # ── 加载 splits (可选) ────────────────────────────────────
    allowed_cases: set[str] | None = None
    if splits is not None:
        if not splits.is_file():
            print(f"错误: splits 文件不存在: {splits}", file=sys.stderr)
            sys.exit(1)

        with open(splits, encoding="utf-8") as f:
            splits_data: dict[str, list[str]] = json.load(f)

        if phase_key not in splits_data:
            print(
                f"错误: splits 文件中找不到 phase '{phase_key}' (phase='{phase}'), "
                f"可用的 keys: {list(splits_data.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

        allowed_cases = set(splits_data[phase_key])
        print(f"splits 文件: {splits}  |  phase={phase} (key='{phase_key}')  |  "
              f"匹配 {len(allowed_cases)} 个 case")

    # ── 搜索 .pt 文件 ─────────────────────────────────────────
    pt_files = sorted(projs_dir.glob("*.pt"))

    if not pt_files:
        print(f"警告: 在 {projs_dir}/ 下未找到任何 .pt 文件", file=sys.stderr)
        sys.exit(1)

    # 如果提供了 splits，过滤
    if allowed_cases is not None:
        filtered: list[Path] = []
        for pt in pt_files:
            case_name = pt.stem  # 去掉 .pt 后缀
            if case_name in allowed_cases:
                filtered.append(pt)
        pt_files = filtered
        print(f"splits 过滤后: {len(pt_files)} 个 case\n")

    # ── 打印概览 ──────────────────────────────────────────────
    print("=" * 50)
    print(f"配置文件:     {config_file}")
    print(f"数据目录:     {data_root}")
    print(f"实验名称:     {experiment_name}")
    if num_views is not None:
        print(f"num_views:    {num_views}")
    print(f"总 case 数:   {len(pt_files)}")
    print("=" * 50)

    # ── 逐个执行 ──────────────────────────────────────────────
    for idx, pt_file in enumerate(pt_files, start=1):
        case_name = pt_file.stem
        total = len(pt_files)

        print(f"\n--- [{idx}/{total}] 处理 case: {case_name} ---")

        # 构建命令
        cmd_parts = [
            sys.executable or "python",
            "main.py",
            "fit",
            "--config", str(config_file),
            "--data.path", str(data_root),
            "-n", experiment_name,
            "-v", case_name,
            "--data.parser.init_args.meta_loader.case_name", case_name,
        ]

        if logger is not None:
            cmd_parts.extend(["--logger", logger])

        if overwrite_output:
            cmd_parts.append("--overwrite_output")

        if num_views is not None:
            cmd_parts.extend([
                "--data.parser.init_args.spliter.num_views", str(num_views),
                "--tags+", f"{num_views}-views",
            ])
        
        if splits is not None:
            cmd_parts.extend([
                "--tags+", f"splits-{splits.stem}",
                "--tags+", f"phase-{phase}",
            ])

        failed_cases = []
        if dry_run:
            print("[DRY-RUN] 将执行:")
            print("  " + " ".join(cmd_parts))
        else:
            print(f"  运行: python main.py fit ... -v {case_name}")
            try:
                result = subprocess.run(cmd_parts, cwd=Path(__file__).resolve().parent.parent)
                print(f"  完成: {case_name} (退出码: {result.returncode})")
            except Exception as e:
                print(f"  错误: {case_name} (异常: {e})")
                failed_cases.append(case_name)

    print(f"\n{'=' * 50}")
    print(f"全部完成! 共处理 {len(pt_files)} 个 case")
    if failed_cases:
        print(f"失败的 case: {failed_cases}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    app()
