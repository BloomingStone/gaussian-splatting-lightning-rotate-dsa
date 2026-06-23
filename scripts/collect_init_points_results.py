"""Collect and compare init-points experiment results."""

import csv
import os
import re
from collections import defaultdict

EXPERIMENTS = {
    "baseline (10K/0.98)": "outputs/debug/init-points-baseline",
    "A (20K/0.95)": "outputs/debug/init-points-A",
    "B (5K/0.995)": "outputs/debug/init-points-B",
    # Batch 1
    "5K (5K/0.98)": "outputs/debug/init-points-5K",
    "20K (20K/0.98)": "outputs/debug/init-points-20K",
    "40K (40K/0.98)": "outputs/debug/init-points-40K",
    "gp90 (10K/0.90)": "outputs/debug/init-points-gp90",
    # Batch 2
    "gp95 (10K/0.95)": "outputs/debug/init-points-gp95",
    "gp995 (10K/0.995)": "outputs/debug/init-points-gp995",
    "fixedth6 (10K/fixed=6.0)": "outputs/debug/init-points-fixedth6",
    # Batch 3 - fixed threshold sweep
    "fixedth4 (10K/fixed=4.0)": "outputs/debug/init-points-fixedth4",
    "fixedth8 (10K/fixed=8.0)": "outputs/debug/init-points-fixedth8",
}

CASES = [
    "asoca-diseased__Diseased_02__LCA",
    "asoca-diseased__Diseased_04__LCA",
    "asoca-diseased__Diseased_05__LCA",
    "asoca-diseased__Diseased_06__LCA",
    "asoca-diseased__Diseased_07__LCA",
]

KEY_METRICS = [
    "loss",
    "psnr",
    "lpips",
    "metric3D/thd-0.0344/dice",
    "metric3D/thd-0.0344/hd95",
    "metric3D/thd-0.0344/cldice",
    "metric3D/thd-99.50%/dice",
    "metric3D/thd-99.50%/hd95",
    "metric3D/thd-99.50%/cldice",
    "metric3D/density/soft_dice",
    "metric3D/density/roc_auc",
    "metric3D/density/pr_auc",
]


def read_metrics_csv(filepath):
    """Read metrics CSV, return list of dicts."""
    if not os.path.exists(filepath):
        return None, None
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None, None
    return rows[0].keys(), rows


STEP_FILES = ["val-step=20000.csv", "val-step=19200.csv", "val-step=14400.csv", "val-step=9600.csv", "val-step=4800.csv"]


def main():
    all_data = {}  # experiment_name -> {case_name -> {metric_name -> mean_value}}
    
    base_dir = "/media/F/sj/Code/GS-contrast-flow"
    
    for exp_name, exp_relpath in EXPERIMENTS.items():
        exp_path = os.path.join(base_dir, exp_relpath)
        
        # Find the subfolder (flow_init-points-*)
        subs = [d for d in os.listdir(exp_path) if d.startswith("flow_init-points")]
        if not subs:
            print(f"WARNING: No subfolder found for {exp_name}")
            continue
        sub = subs[0]
        sub_path = os.path.join(exp_path, sub)
        
        exp_data = {}
        for case in CASES:
            metrics_dir = os.path.join(sub_path, case, "metrics")
            
            # Try each step file in order, use the first one that has all KEY_METRICS
            best_metrics = None
            best_step = None
            for step_file in STEP_FILES:
                metrics_file = os.path.join(metrics_dir, step_file)
                fieldnames, rows = read_metrics_csv(metrics_file)
                if rows is None:
                    continue
                # Check if all KEY_METRICS are present (skip 3D metrics check for non-3D)
                has_all_3d = all(m in fieldnames for m in KEY_METRICS if "metric3D" in m)
                if has_all_3d:
                    best_metrics = (fieldnames, rows)
                    best_step = step_file
                    break
                elif best_metrics is None:
                    # Keep first available file as fallback
                    best_metrics = (fieldnames, rows)
                    best_step = step_file
            
            if best_metrics is None:
                print(f"WARNING: No metrics file for {exp_name} / {case}")
                continue
            
            fieldnames, rows = best_metrics
            has_3d = any("metric3D" in f for f in fieldnames)
            if not has_3d:
                print(f"  {exp_name}/{case}: using {best_step} (no 3D metrics in this step)")
            
            # Compute mean across images for each metric
            case_means = {}
            for metric in KEY_METRICS:
                if metric not in fieldnames:
                    print(f"  {exp_name}/{case}: missing metric '{metric}' (using {best_step})")
                    continue
                values = [float(row[metric]) for row in rows if row[metric]]
                if values:
                    case_means[metric] = sum(values) / len(values)
            
            exp_data[case] = case_means
        
        all_data[exp_name] = exp_data
    
    # Compute across-case statistics
    print("=" * 120)
    print(f"{'Metric':40s} {'Baseline (10K/0.98)':30s} {'A (20K/0.95)':30s} {'B (5K/0.995)':30s}")
    print("=" * 120)
    
    for metric in KEY_METRICS:
        row = f"{metric:40s}"
        for exp_name in EXPERIMENTS:
            exp_data = all_data.get(exp_name, {})
            values = []
            cases_with_data = 0
            for case in CASES:
                if case in exp_data and metric in exp_data[case]:
                    values.append(exp_data[case][metric])
                    cases_with_data += 1
            
            if values:
                mean = sum(values) / len(values)
                if len(values) > 1:
                    std = (sum((v - mean)**2 for v in values) / (len(values) - 1))**0.5
                    row += f" {mean:.4f}±{std:.4f} ({cases_with_data}c)  ".ljust(30)
                else:
                    row += f" {mean:.4f} ({cases_with_data}c)      ".ljust(30)
            else:
                row += f" {'N/A':15s}           ".ljust(30)
        print(row)
    
    print("=" * 120)
    
    # Also save to CSV
    output_path = os.path.join(base_dir, "outputs/debug/init-points-comparison.csv")
    with open(output_path, "w") as f:
        writer = csv.writer(f)
        header = ["metric"] + list(EXPERIMENTS.keys())
        writer.writerow(header)
        for metric in KEY_METRICS:
            row_data = [metric]
            for exp_name in EXPERIMENTS:
                exp_data = all_data.get(exp_name, {})
                values = []
                for case in CASES:
                    if case in exp_data and metric in exp_data[case]:
                        values.append(exp_data[case][metric])
                if values:
                    mean = sum(values) / len(values)
                    if len(values) > 1:
                        std = (sum((v - mean)**2 for v in values) / (len(values) - 1))**0.5
                        row_data.append(f"{mean:.6f}±{std:.6f}")
                    else:
                        row_data.append(f"{mean:.6f}")
                else:
                    row_data.append("N/A")
            writer.writerow(row_data)
    
    print(f"\nSaved comparison to {output_path}")


if __name__ == "__main__":
    main()
