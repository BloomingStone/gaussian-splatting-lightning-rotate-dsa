#!/usr/bin/env python3
"""Aggregate TensorBoard scalar logs and summarize final metrics.

Features:
1. Recursively find TensorBoard event files under a root directory.
2. Group files by experiment name (derived from relative path depth).
3. Read scalar metrics from each experiment.
4. Resample by step interval and merge experiments into a new TensorBoard log.
5. Print and save final-value statistics for each metric.

Image data and other non-scalar summaries are intentionally skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, DefaultDict, Dict, List, Tuple, cast

from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


ScalarPoint = Tuple[int, float]
MetricSeries = Dict[str, List[ScalarPoint]]
ExperimentMetrics = Dict[str, MetricSeries]


@dataclass
class FinalMetricStats:
	metric: str
	count: int
	mean: float
	std: float
	min_value: float
	min_experiment: str
	max_value: float
	max_experiment: str
	outliers: List[Tuple[str, float]]


def normalize_keywords(raw_values: List[str] | None) -> List[str]:
	"""Normalize keyword args: split by comma, trim spaces, lowercase."""
	if not raw_values:
		return []
	keywords: List[str] = []
	for raw in raw_values:
		for part in raw.split(","):
			kw = part.strip().lower()
			if kw:
				keywords.append(kw)
	return keywords


def metric_selected(metric_name: str, include_keywords: List[str], exclude_keywords: List[str]) -> bool:
	"""Return True if metric passes include/exclude substring filters."""
	name = metric_name.lower()
	if include_keywords and not any(kw in name for kw in include_keywords):
		return False
	if exclude_keywords and any(kw in name for kw in exclude_keywords):
		return False
	return True


def find_event_files(root: Path) -> List[Path]:
	"""Recursively locate TensorBoard event files."""
	return sorted(root.rglob("events.out.tfevents.*"))


def experiment_name_from_path(file_path: Path, root: Path, exp_depth: int) -> str:
	"""Build experiment name from relative path components.

	Example:
	  root = ASOCA_recon_2
	  file = ASOCA_recon_2/LCA/Diseased_02__LCA/lightning_logs/version_0/events...
	  exp_depth = 2  -> LCA/Diseased_02__LCA
	"""
	rel_parent_parts = file_path.parent.relative_to(root).parts
	if not rel_parent_parts:
		return "root"
	depth = min(exp_depth, len(rel_parent_parts))
	return "/".join(rel_parent_parts[:depth])


def read_scalars_from_event_file(
	file_path: Path,
	include_keywords: List[str],
	exclude_keywords: List[str],
) -> MetricSeries:
	"""Stream-read scalar tags from one event file for better performance."""
	out: DefaultDict[str, List[ScalarPoint]] = defaultdict(list)
	loader = event_file_loader.RawEventFileLoader(str(file_path))

	event_cls = cast(Any, event_pb2.Event)
	for rec in loader.Load():
		try:
			event = event_cls.FromString(rec)
		except Exception:
			# Skip malformed records and continue with remaining events.
			continue

		if not event.HasField("summary"):
			continue

		step = int(event.step)
		for value in event.summary.value:
			if value.HasField("simple_value") and metric_selected(
				value.tag, include_keywords, exclude_keywords
			):
				out[value.tag].append((step, float(value.simple_value)))

	# Ensure deterministic order and unique steps (latest wins within a file).
	merged_out: MetricSeries = {}
	for tag, points in out.items():
		by_step: Dict[int, float] = {}
		for step, val in points:
			by_step[step] = val
		merged_out[tag] = sorted(by_step.items(), key=lambda x: x[0])

	return merged_out


def merge_metric_series(series_list: List[List[ScalarPoint]]) -> List[ScalarPoint]:
	"""Merge multiple series for same metric and keep latest value per step."""
	by_step: Dict[int, float] = {}
	for series in series_list:
		for step, value in series:
			by_step[step] = value
	return sorted(by_step.items(), key=lambda x: x[0])


def aggregate_by_interval(points: List[ScalarPoint], interval: int) -> List[ScalarPoint]:
	"""Downsample points by binning steps and averaging values in each bin."""
	if interval <= 1:
		return points

	bins: DefaultDict[int, List[float]] = defaultdict(list)
	for step, value in points:
		bin_step = (step // interval) * interval
		bins[bin_step].append(value)

	aggregated = [(step, mean(values)) for step, values in bins.items()]
	return sorted(aggregated, key=lambda x: x[0])


def build_experiment_metrics(
	event_files: List[Path],
	root: Path,
	exp_depth: int,
	include_keywords: List[str],
	exclude_keywords: List[str],
) -> ExperimentMetrics:
	"""Collect and merge scalar metrics per experiment."""
	raw: Dict[str, Dict[str, List[List[ScalarPoint]]]] = defaultdict(lambda: defaultdict(list))

	total = len(event_files)
	for idx, file_path in enumerate(event_files, start=1):
		if idx == 1 or idx % 5 == 0 or idx == total:
			print(f"Reading event file {idx}/{total}: {file_path}")
		exp_name = experiment_name_from_path(file_path, root, exp_depth)
		metric_map = read_scalars_from_event_file(file_path, include_keywords, exclude_keywords)
		for metric, points in metric_map.items():
			raw[exp_name][metric].append(points)

	merged: ExperimentMetrics = {}
	for exp_name, metric_group in raw.items():
		merged[exp_name] = {}
		for metric, multiple_series in metric_group.items():
			merged[exp_name][metric] = merge_metric_series(multiple_series)
	return merged


def write_scalar_event(writer: EventFileWriter, tag: str, step: int, value: float) -> None:
	"""Write one scalar datapoint to TensorBoard event file."""
	summary = summary_pb2.Summary()
	summary_any = cast(Any, summary)
	scalar = summary_any.value.add()
	scalar.tag = tag
	scalar.simple_value = float(value)

	event = event_pb2.Event()
	event_any = cast(Any, event)
	event_any.step = int(step)
	event_any.summary.CopyFrom(summary)
	writer.add_event(event)


def write_merged_log(experiment_metrics: ExperimentMetrics, out_dir: Path, interval: int) -> None:
	"""Create a merged TensorBoard log with cross-experiment statistics per step."""
	out_dir.mkdir(parents=True, exist_ok=True)
	writer = EventFileWriter(str(out_dir))

	metric_step_values: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

	for _exp_name, metric_map in experiment_metrics.items():
		for metric, points in metric_map.items():
			for step, value in aggregate_by_interval(points, interval):
				metric_step_values[metric][step].append(value)

	for metric, step_map in metric_step_values.items():
		for step in sorted(step_map.keys()):
			values = [v for v in step_map[step] if math.isfinite(v)]
			if not values:
				continue
			write_scalar_event(writer, f"merged/{metric}/mean", step, mean(values))
			std_value = pstdev(values) if len(values) > 1 else 0.0
			write_scalar_event(writer, f"merged/{metric}/std", step, std_value)
			write_scalar_event(writer, f"merged/{metric}/min", step, min(values))
			write_scalar_event(writer, f"merged/{metric}/max", step, max(values))

	writer.flush()
	writer.close()


def detect_outliers_iqr(exp_values: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
	"""Detect outliers using IQR rule (1.5 * IQR)."""
	if len(exp_values) < 4:
		return []

	sorted_values = sorted(v for _, v in exp_values)

	def percentile(values: List[float], p: float) -> float:
		if not values:
			return math.nan
		idx = (len(values) - 1) * p
		lower = math.floor(idx)
		upper = math.ceil(idx)
		if lower == upper:
			return values[lower]
		frac = idx - lower
		return values[lower] * (1 - frac) + values[upper] * frac

	q1 = percentile(sorted_values, 0.25)
	q3 = percentile(sorted_values, 0.75)
	iqr = q3 - q1

	if iqr == 0:
		return []

	lower_bound = q1 - 1.5 * iqr
	upper_bound = q3 + 1.5 * iqr
	return [(exp, val) for exp, val in exp_values if val < lower_bound or val > upper_bound]


def compute_final_metric_stats(experiment_metrics: ExperimentMetrics) -> List[FinalMetricStats]:
	"""Compute statistics on final value of each metric across experiments."""
	per_metric_values: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

	for exp_name, metric_map in experiment_metrics.items():
		for metric, points in metric_map.items():
			if points:
				final_step, final_value = points[-1]
				_ = final_step
				if math.isfinite(final_value):
					per_metric_values[metric].append((exp_name, final_value))

	stats_list: List[FinalMetricStats] = []
	for metric, exp_values in sorted(per_metric_values.items()):
		values = [v for _, v in exp_values]
		if not values:
			continue

		min_exp, min_val = min(exp_values, key=lambda x: x[1])
		max_exp, max_val = max(exp_values, key=lambda x: x[1])
		stats_list.append(
			FinalMetricStats(
				metric=metric,
				count=len(values),
				mean=mean(values),
				std=pstdev(values) if len(values) > 1 else 0.0,
				min_value=min_val,
				min_experiment=min_exp,
				max_value=max_val,
				max_experiment=max_exp,
				outliers=detect_outliers_iqr(exp_values),
			)
		)

	return stats_list


def save_stats_json(stats: List[FinalMetricStats], path: Path) -> None:
	"""Save summary statistics to JSON file."""
	payload = []
	for s in stats:
		payload.append(
			{
				"metric": s.metric,
				"count": s.count,
				"mean": s.mean,
				"std": s.std,
				"min": {"value": s.min_value, "experiment": s.min_experiment},
				"max": {"value": s.max_value, "experiment": s.max_experiment},
				"outliers": [
					{"experiment": exp, "value": val} for exp, val in s.outliers
				],
			}
		)

	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def print_stats(stats: List[FinalMetricStats]) -> None:
	"""Print human-readable metric summary."""
	if not stats:
		print("No scalar metrics found.")
		return

	for s in stats:
		print("=" * 90)
		print(f"metric: {s.metric}")
		print(f"  count: {s.count}")
		print(f"  mean:  {s.mean:.6g}")
		print(f"  std:   {s.std:.6g}")
		print(f"  min:   {s.min_value:.6g} ({s.min_experiment})")
		print(f"  max:   {s.max_value:.6g} ({s.max_experiment})")
		if s.outliers:
			outlier_str = ", ".join([f"{exp}:{val:.6g}" for exp, val in s.outliers])
			print(f"  outliers(IQR): {outlier_str}")
		else:
			print("  outliers(IQR): none")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Recursively merge TensorBoard scalar logs and summarize final metrics."
	)
	parser.add_argument(
		"--root",
		type=Path,
		required=True,
		help="Root directory to recursively search events.out.tfevents.* files.",
	)
	parser.add_argument(
		"--interval",
		type=int,
		default=10,
		help="Step interval for aggregation bins (default: 10).",
	)
	parser.add_argument(
		"--exp-depth",
		type=int,
		default=2,
		help="Use first N relative path parts as experiment name (default: 2).",
	)
	parser.add_argument(
		"--metric-include",
		action="append",
		default=None,
		help=(
			"Only keep metrics containing these keywords (case-insensitive substring). "
			"Can be repeated, or pass comma-separated values, e.g. --metric-include val,loss"
		),
	)
	parser.add_argument(
		"--metric-exclude",
		action="append",
		default=None,
		help=(
			"Exclude metrics containing these keywords (case-insensitive substring). "
			"Can be repeated, or pass comma-separated values, e.g. --metric-exclude train,grad"
		),
	)
	parser.add_argument(
		"--output-log-dir",
		type=Path,
		default=Path("merged_tb_logs"),
		help="Output directory for merged TensorBoard event file.",
	)
	parser.add_argument(
		"--stats-json",
		type=Path,
		default=Path("merged_tb_logs/final_metric_stats.json"),
		help="Path to save final metric stats as JSON.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	root = args.root.resolve()

	if not root.exists() or not root.is_dir():
		raise FileNotFoundError(f"Invalid --root directory: {root}")

	if args.interval < 1:
		raise ValueError("--interval must be >= 1")
	if args.exp_depth < 1:
		raise ValueError("--exp-depth must be >= 1")

	include_keywords = normalize_keywords(args.metric_include)
	exclude_keywords = normalize_keywords(args.metric_exclude)
	if include_keywords:
		print(f"Metric include keywords: {include_keywords}")
	if exclude_keywords:
		print(f"Metric exclude keywords: {exclude_keywords}")

	event_files = find_event_files(root)
	if not event_files:
		print(f"No TensorBoard event files found under: {root}")
		return

	print(f"Found {len(event_files)} event files.")
	experiment_metrics = build_experiment_metrics(
		event_files,
		root,
		args.exp_depth,
		include_keywords,
		exclude_keywords,
	)
	print(f"Parsed experiments: {len(experiment_metrics)}")

	write_merged_log(experiment_metrics, args.output_log_dir, args.interval)
	print(f"Merged TensorBoard log written to: {args.output_log_dir.resolve()}")

	stats = compute_final_metric_stats(experiment_metrics)
	print_stats(stats)
	save_stats_json(stats, args.stats_json)
	print(f"Final metric stats JSON written to: {args.stats_json.resolve()}")


if __name__ == "__main__":
	os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
	main()
