"""Analyze init-points experiment results focusing on 3D metrics."""

import csv
import os
from collections import defaultdict

EXPERIMENTS = {
    "5K": "outputs/debug/init-points-5K",
    "20K": "outputs/debug/init-points-20K",
    "40K": "outputs/debug/init-points-40K",
}

# All 30 diseased cases (LCA and RCA pairs)
DISEASED_IDS = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20]
CASES = []
for did in DISEASED_IDS:
    CASES.append(f"asoca-diseased__Diseased_{did:02d}__LCA")
    CASES.append(f"asoca-diseased__Diseased_{did:02d}__RCA")
# Remove non-existent ones (Diseased_13 doesn't have both)
CASES = [c for c in CASES if os.path.exists(
    f"/media/data3/sj/Code/GS-dev-contrast-flow/data/gen_4d_output_all/flow/{c}"
)]

# 3D metrics focus
KEY_METRICS = [
    "loss", "psnr",
    # 3D Dice metrics
    "metric3D/thd-0.0344/dice",
    "metric3D/thd-0.0344/hd95",
    "metric3D/thd-0.0344/cldice",
    "metric3D/thd-99.50%/dice",
    "metric3D/thd-99.50%/hd95",
    "metric3D/thd-99.50%/cldice",
    # Density-based 3D metrics
    "metric3D/density/soft_dice",
    "metric3D/density/roc_auc",
    "metric3D/density/pr_auc",
]

STEP_FILES = ["val-step=20000.csv", "val-step=19200.csv",
              "val-step=14400.csv", "val-step=9600.csv", "val-step=4800.csv"]

BASE_DIR = "/media/data3/sj/Code/GS-dev-contrast-flow"


def read_metrics_csv(filepath):
    if not os.path.exists(filepath):
        return None, None
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None, None
    return rows[0].keys(), rows


def find_best_step(sub_path, case):
    """Find the best metrics step file for a given case."""
    metrics_dir = os.path.join(sub_path, case, "metrics")
    if not os.path.exists(metrics_dir):
        return None, None, None

    for step_file in STEP_FILES:
        metrics_file = os.path.join(metrics_dir, step_file)
        fieldnames, rows = read_metrics_csv(metrics_file)
        if rows is None:
            continue
        # Check if it has at least some 3D metrics
        has_3d = any("metric3D" in f for f in fieldnames)
        if has_3d:
            return fieldnames, rows, step_file
        if fieldnames is not None:
            # Keep as fallback
            fallback_fn, fallback_rows, fallback_sf = fieldnames, rows, step_file

    # Return fallback (no 3D metrics found)
    return None, None, None


def compute_mean(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) > 1:
        std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
    else:
        std = None
    return mean, std


def main():
    all_data = {}  # exp_name -> {case -> {metric -> mean_value}}
    missing_3d_cases = defaultdict(list)

    for exp_name, exp_relpath in EXPERIMENTS.items():
        exp_path = os.path.join(BASE_DIR, exp_relpath)
        subs = [d for d in os.listdir(exp_path) if d.startswith("flow_init-points")]
        if not subs:
            print(f"WARNING: No subfolder for {exp_name}")
            continue
        sub = subs[0]
        sub_path = os.path.join(exp_path, sub)

        exp_data = {}
        for case in CASES:
            fn, rows, step = find_best_step(sub_path, case)
            if rows is None:
                missing_3d_cases[exp_name].append(case)
                continue

            case_means = {}
            for metric in KEY_METRICS:
                if metric not in fn:
                    continue
                values = [float(r[metric]) for r in rows if r[metric]]
                if values:
                    case_means[metric] = sum(values) / len(values)
            exp_data[case] = case_means

        all_data[exp_name] = exp_data
        print(f"{exp_name}: {len(exp_data)}/{len(CASES)} cases with 3D metrics")
        if missing_3d_cases[exp_name]:
            print(f"  Missing 3D: {missing_3d_cases[exp_name]}")

    # ─────────────────────────────────────────
    # 1. Per-experiment aggregate (ALL cases)
    # ─────────────────────────────────────────
    print("\n" + "=" * 140)
    print(f"{'3D Metric':45s} {'5K':>22s} {'20K':>22s} {'40K':>22s}")
    print("=" * 140)

    for metric in KEY_METRICS:
        row = f"{metric:45s}"
        for exp_name in EXPERIMENTS:
            exp_data = all_data.get(exp_name, {})
            values = [exp_data[c][metric] for c in CASES
                      if c in exp_data and metric in exp_data[c]]
            mean, std = compute_mean(values)
            if mean is not None:
                n = len(values)
                if std is not None:
                    row += f"  {mean:.4f}±{std:.4f} (n={n:2d})".ljust(24)
                else:
                    row += f"  {mean:.4f} (n={n:2d})".ljust(24)
            else:
                row += f"  {'N/A':>18s}".ljust(24)
        print(row)

    print("=" * 140)

    # ─────────────────────────────────────────
    # 2. Per-case comparison table (3D Dice only)
    # ─────────────────────────────────────────
    print("\n\n" + "=" * 140)
    print("PER-CASE COMPARISON — metric3D/thd-0.0344/dice (higher = better)")
    print("=" * 140)
    print(f"{'Case':40s} {'5K':>22s} {'20K':>22s} {'40K':>22s} {'Best':>10s}")
    print("-" * 140)

    wins = defaultdict(int)  # exp_name -> win count
    for case in CASES:
        case_short = case.replace("asoca-diseased__", "")
        row = f"{case_short:40s}"
        vals = {}
        for exp_name in EXPERIMENTS:
            exp_data = all_data.get(exp_name, {})
            if case in exp_data and "metric3D/thd-0.0344/dice" in exp_data[case]:
                v = exp_data[case]["metric3D/thd-0.0344/dice"]
                vals[exp_name] = v
                row += f"  {v:.4f}".ljust(24)
            else:
                row += f"  {'N/A':>18s}".ljust(24)

        if vals:
            best_exp = max(vals, key=vals.get)
            best_val = max(vals.values())
            wins[best_exp] += 1
            row += f"  {best_exp:>4s} ({best_val:.4f})".ljust(12)
        else:
            row += f"  {'N/A':>10s}".ljust(12)
        print(row)

    print("-" * 140)
    win_row = f"{'🏆 Wins':40s}"
    for exp_name in EXPERIMENTS:
        win_row += f"  {wins[exp_name]:>2d}/{len(CASES)} ({wins[exp_name]/len(CASES)*100:.0f}%)".ljust(24)
    print(win_row)
    print("=" * 140)

    # ─────────────────────────────────────────
    # 3. Per-case: HD95 (lower = better)
    # ─────────────────────────────────────────
    print("\n\n" + "=" * 140)
    print("PER-CASE COMPARISON — metric3D/thd-0.0344/hd95 (lower = better)")
    print("=" * 140)
    print(f"{'Case':40s} {'5K':>22s} {'20K':>22s} {'40K':>22s} {'Best':>10s}")
    print("-" * 140)

    wins_hd95 = defaultdict(int)
    for case in CASES:
        case_short = case.replace("asoca-diseased__", "")
        row = f"{case_short:40s}"
        vals = {}
        for exp_name in EXPERIMENTS:
            exp_data = all_data.get(exp_name, {})
            if case in exp_data and "metric3D/thd-0.0344/hd95" in exp_data[case]:
                v = exp_data[case]["metric3D/thd-0.0344/hd95"]
                vals[exp_name] = v
                row += f"  {v:.2f}".ljust(24)
            else:
                row += f"  {'N/A':>18s}".ljust(24)

        if vals:
            best_exp = min(vals, key=vals.get)
            best_val = min(vals.values())
            wins_hd95[best_exp] += 1
            row += f"  {best_exp:>4s} ({best_val:.2f})".ljust(12)
        else:
            row += f"  {'N/A':>10s}".ljust(12)
        print(row)

    print("-" * 140)
    win_row = f"{'🏆 Wins (HD95 lower)':40s}"
    for exp_name in EXPERIMENTS:
        win_row += f"  {wins_hd95[exp_name]:>2d}/{len(CASES)} ({wins_hd95[exp_name]/len(CASES)*100:.0f}%)".ljust(24)
    print(win_row)
    print("=" * 140)

    # ─────────────────────────────────────────
    # 4. Soft Dice (density-based)
    # ─────────────────────────────────────────
    print("\n\n" + "=" * 140)
    print("PER-CASE COMPARISON — metric3D/density/soft_dice (higher = better)")
    print("=" * 140)
    print(f"{'Case':40s} {'5K':>22s} {'20K':>22s} {'40K':>22s} {'Best':>10s}")
    print("-" * 140)

    wins_sd = defaultdict(int)
    for case in CASES:
        case_short = case.replace("asoca-diseased__", "")
        row = f"{case_short:40s}"
        vals = {}
        for exp_name in EXPERIMENTS:
            exp_data = all_data.get(exp_name, {})
            if case in exp_data and "metric3D/density/soft_dice" in exp_data[case]:
                v = exp_data[case]["metric3D/density/soft_dice"]
                vals[exp_name] = v
                row += f"  {v:.4f}".ljust(24)
            else:
                row += f"  {'N/A':>18s}".ljust(24)

        if vals:
            best_exp = max(vals, key=vals.get)
            best_val = max(vals.values())
            wins_sd[best_exp] += 1
            row += f"  {best_exp:>4s} ({best_val:.4f})".ljust(12)
        else:
            row += f"  {'N/A':>10s}".ljust(12)
        print(row)

    print("-" * 140)
    win_row = f"{'🏆 Wins (Soft Dice)':40s}"
    for exp_name in EXPERIMENTS:
        win_row += f"  {wins_sd[exp_name]:>2d}/{len(CASES)} ({wins_sd[exp_name]/len(CASES)*100:.0f}%)".ljust(24)
    print(win_row)
    print("=" * 140)

    # ─────────────────────────────────────────
    # 5. Export per-case metrics to CSV
    # ─────────────────────────────────────────
    csv_path = os.path.join(BASE_DIR, "outputs/debug/init-points-3d-comparison.csv")
    print(f"\n\nExporting per-case metrics to: {csv_path}")

    # Build metric display names
    METRIC_LABELS = {
        "loss": "loss",
        "psnr": "psnr",
        "metric3D/thd-0.0344/dice": "dice_0344",
        "metric3D/thd-0.0344/hd95": "hd95_0344",
        "metric3D/thd-0.0344/cldice": "cldice_0344",
        "metric3D/thd-99.50%/dice": "dice_995",
        "metric3D/thd-99.50%/hd95": "hd95_995",
        "metric3D/thd-99.50%/cldice": "cldice_995",
        "metric3D/density/soft_dice": "soft_dice",
        "metric3D/density/roc_auc": "roc_auc",
        "metric3D/density/pr_auc": "pr_auc",
    }

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header: case, then each metric for each experiment
        header = ["case"]
        for exp_name in EXPERIMENTS:
            for metric in KEY_METRICS:
                header.append(f"{exp_name}_{METRIC_LABELS.get(metric, metric)}")
        writer.writerow(header)

        # Per-case data rows
        all_values = {exp: {m: [] for m in KEY_METRICS} for exp in EXPERIMENTS}
        for case in CASES:
            case_short = case.replace("asoca-diseased__", "")
            row = [case_short]
            for exp_name in EXPERIMENTS:
                exp_data = all_data.get(exp_name, {})
                for metric in KEY_METRICS:
                    if case in exp_data and metric in exp_data[case]:
                        v = exp_data[case][metric]
                        row.append(f"{v:.6f}")
                        all_values[exp_name][metric].append(v)
                    else:
                        row.append("")
            writer.writerow(row)

        # Blank separator row
        writer.writerow([])

        # Mean row
        mean_row = ["mean"]
        for exp_name in EXPERIMENTS:
            for metric in KEY_METRICS:
                vals = all_values[exp_name][metric]
                if vals:
                    m = sum(vals) / len(vals)
                    mean_row.append(f"{m:.6f}")
                else:
                    mean_row.append("")
        writer.writerow(mean_row)

        # Std row
        std_row = ["std"]
        for exp_name in EXPERIMENTS:
            for metric in KEY_METRICS:
                vals = all_values[exp_name][metric]
                if len(vals) > 1:
                    m = sum(vals) / len(vals)
                    s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                    std_row.append(f"{s:.6f}")
                else:
                    std_row.append("")
        writer.writerow(std_row)

    print(f"  ✅ Saved {len(CASES)} cases × {len(EXPERIMENTS)} experiments × {len(KEY_METRICS)} metrics")
    print(f"  ✅ Mean and Std rows appended at bottom")

    # ─────────────────────────────────────────
    # 6. Summary ranking
    # ─────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("RANKING SUMMARY (win count across all 3D metrics)")
    print("=" * 80)
    
    all_3d_metrics = [m for m in KEY_METRICS if "metric3D" in m]
    for metric in all_3d_metrics:
        better = "higher" if "dice" in metric or "auc" in metric or "cldice" in metric else "lower"
        win_counts = defaultdict(int)
        for case in CASES:
            vals = {}
            for exp_name in EXPERIMENTS:
                exp_data = all_data.get(exp_name, {})
                if case in exp_data and metric in exp_data[case]:
                    vals[exp_name] = exp_data[case][metric]
            if len(vals) >= 2:
                if better == "higher":
                    best = max(vals, key=vals.get)
                else:
                    best = min(vals, key=vals.get)
                win_counts[best] += 1

        short_name = metric.replace("metric3D/", "")
        total = sum(win_counts.values())
        if total > 0:
            print(f"  {short_name:35s} | 5K={win_counts.get('5K',0):2d}/{total:2d} "
                  f"20K={win_counts.get('20K',0):2d}/{total:2d} "
                  f"40K={win_counts.get('40K',0):2d}/{total:2d} "
                  f"({better})")


if __name__ == "__main__":
    main()
