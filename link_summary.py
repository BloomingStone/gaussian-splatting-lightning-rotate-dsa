#!/usr/bin/env python3
"""
将 outputs/3DGR-CAR_<dataset>_<views>_<side>/<case>/volumes/ 下 step 最大的 volume 文件
软链接到 outputs/3DGR-CAR_summary/<views>/<dataset>/<case>.nii.gz
"""
import re
from pathlib import Path

SUMMARY_DIR = Path("outputs/3DGR-CAR_summary")

# 匹配: outputs/3DGR-CAR_<dataset>_<views>_<side>
DIR_RE = re.compile(
    r"^outputs/3DGR-CAR_(?P<dataset>[^_]+)_(?P<views>\d+-views)_(?P<side>[^/]+)$"
)

# 匹配 volume__epoch=<num>-step=<num>.nii.gz 并提取 step
VOL_RE = re.compile(r"^volume__epoch=\d+-step=(?P<step>\d+)\.nii\.gz$")


def find_best_volume(case_dir: Path) -> Path | None:
    """找到 case 目录下 step 最大的 volume 文件"""
    vol_dir = case_dir / "volumes"
    if not vol_dir.is_dir():
        return None

    best_file = None
    best_step = -1
    for f in vol_dir.iterdir():
        m = VOL_RE.match(f.name)
        if m:
            step = int(m.group("step"))
            if step > best_step:
                best_step = step
                best_file = f
    return best_file


def main():
    exp_dirs = sorted(Path("outputs").glob("3DGR-CAR_*"))
    if not exp_dirs:
        print("⚠️  未找到任何 outputs/3DGR-CAR_* 目录")
        return

    created = 0
    skipped = 0

    for exp_dir in exp_dirs:
        if exp_dir.name == "3DGR-CAR_summary":
            continue

        m = DIR_RE.match(str(exp_dir))
        if not m:
            continue

        dataset = m.group("dataset")
        views = m.group("views")
        side= m.group("side")

        # 遍历该目录下每个 case
        case_dirs = sorted(exp_dir.iterdir())
        for case_dir in case_dirs:
            if not case_dir.is_dir():
                continue

            src = find_best_volume(case_dir)
            if src is None:
                print(f"⏭️  {exp_dir.name}/{case_dir.name}: 无 volume 文件")
                skipped += 1
                continue

            dst = SUMMARY_DIR / views / dataset / side / f"{case_dir.name}.nii.gz"
            dst.parent.mkdir(parents=True, exist_ok=True)

            if dst.exists() or dst.is_symlink():
                dst.unlink()

            dst.symlink_to(src.resolve())
            print(f"🔗 {dst} → {src.name}")
            created += 1

    print(f"\n✅ 完成: 创建 {created} 个软链接, 跳过 {skipped} 个")


if __name__ == "__main__":
    main()
