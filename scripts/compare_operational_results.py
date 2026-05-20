#!/usr/bin/env python3
import argparse
import ast
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")

import numpy as np
import pandas as pd


REQUIRED_DETECTION_COLUMNS = {
    "round",
    "threshold",
    "f1_score",
    "precision",
    "recall",
    "benign_fpr",
}


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_") or "comparison"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate plots. Install matplotlib in the active "
            "environment or rerun with --no_plots to generate only CSV/Markdown outputs."
        ) from exc
    return plt


def export_figure(fig, output_stem: Path) -> list[Path]:
    plt = get_pyplot()
    saved_paths = []
    for ext in ("png", "pdf", "svg"):
        out_path = output_stem.with_suffix(f".{ext}")
        fig.savefig(out_path, bbox_inches="tight", dpi=220)
        saved_paths.append(out_path)
    plt.close(fig)
    return saved_paths


def discover_results_csv(results_dir: Path, basename: str) -> Path | None:
    direct = results_dir / f"{basename}.csv"
    if direct.exists():
        return direct
    candidates = sorted(results_dir.glob(f"{basename}_*.csv"))
    return candidates[0] if candidates else None


def filter_round_range(
    df: pd.DataFrame, round_min: int | None, round_max: int | None
) -> pd.DataFrame:
    if df.empty or "round" not in df.columns:
        return df
    filtered = df.copy()
    filtered["round"] = pd.to_numeric(filtered["round"], errors="coerce")
    if round_min is not None:
        filtered = filtered[filtered["round"] >= int(round_min)]
    if round_max is not None:
        filtered = filtered[filtered["round"] <= int(round_max)]
    return filtered.sort_values("round").reset_index(drop=True)


def _validate_detection_columns(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    if "f1" in df.columns and "f1_score" not in df.columns:
        df = df.rename(columns={"f1": "f1_score"})

    missing = sorted(REQUIRED_DETECTION_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"{csv_path} is missing required detection columns: {missing}. "
            "Expected at least round, threshold, f1_score, precision, recall and benign_fpr."
        )

    numeric_cols = [
        "round",
        "threshold",
        "f1_score",
        "precision",
        "recall",
        "benign_fpr",
        "k",
        "fpr_target",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_detection_results(
    results_dir: Path,
    *,
    threshold_mode: str,
    fpr_target: float,
    k: int,
    granularity: str,
    round_min: int | None,
    round_max: int | None,
) -> pd.DataFrame:
    csv_path = discover_results_csv(results_dir, "f1_scores")
    if csv_path is None:
        raise FileNotFoundError(
            f"No operational f1_scores CSV found in {results_dir}. "
            "Generate the evaluation CSV first and then rerun the comparison."
        )

    df = pd.read_csv(csv_path)
    df = _validate_detection_columns(df, csv_path)
    if "threshold_mode" in df.columns:
        df = df[df["threshold_mode"].astype(str).str.lower() == threshold_mode.lower()]
    if "k" in df.columns:
        df = df[df["k"].astype(int) == int(k)]
    if "granularity" in df.columns:
        df = df[df["granularity"].astype(str).str.lower() == granularity.lower()]
    if threshold_mode == "fpr_target" and "fpr_target" in df.columns:
        df = df[np.isclose(df["fpr_target"].astype(float), float(fpr_target), atol=1e-9)]
    df = filter_round_range(df, round_min, round_max)
    if df.empty:
        raise ValueError(
            f"No detection rows found in {csv_path} after filtering by "
            f"threshold_mode={threshold_mode}, fpr_target={fpr_target}, k={k}, "
            f"granularity={granularity}, round_min={round_min}, round_max={round_max}."
        )
    return df


def validate_round_budget(
    legacy_detection: pd.DataFrame,
    proposed_detection: pd.DataFrame,
    *,
    legacy_label: str,
    proposed_label: str,
    expected_rounds: int | None,
    require_same_rounds: bool,
) -> None:
    legacy_rounds = int(legacy_detection["round"].nunique())
    proposed_rounds = int(proposed_detection["round"].nunique())
    if expected_rounds is not None:
        if legacy_rounds != int(expected_rounds):
            raise ValueError(
                f"{legacy_label} has {legacy_rounds} detection rounds after filtering, "
                f"but --expected_rounds={expected_rounds} was requested."
            )
        if proposed_rounds != int(expected_rounds):
            raise ValueError(
                f"{proposed_label} has {proposed_rounds} detection rounds after filtering, "
                f"but --expected_rounds={expected_rounds} was requested."
            )
    if require_same_rounds and legacy_rounds != proposed_rounds:
        raise ValueError(
            f"Round-budget mismatch after filtering: {legacy_label} has {legacy_rounds} "
            f"rounds and {proposed_label} has {proposed_rounds} rounds."
        )


def build_preflight_summary(
    legacy_results_dir: Path,
    proposed_results_dir: Path,
    legacy_detection: pd.DataFrame,
    proposed_detection: pd.DataFrame,
    *,
    threshold_mode: str,
    fpr_target: float,
    k: int,
    granularity: str,
    round_min: int | None,
    round_max: int | None,
    expected_rounds: int | None,
) -> pd.DataFrame:
    rows = []
    for role, results_dir, df in [
        ("original", legacy_results_dir, legacy_detection),
        ("improved", proposed_results_dir, proposed_detection),
    ]:
        csv_path = discover_results_csv(results_dir, "f1_scores")
        rows.append(
            {
                "role": role,
                "results_dir": str(results_dir),
                "f1_csv": str(csv_path) if csv_path else "",
                "rows_after_filter": int(len(df)),
                "num_rounds_after_filter": int(df["round"].nunique()),
                "min_round": int(df["round"].min()),
                "max_round": int(df["round"].max()),
                "threshold_mode": threshold_mode,
                "fpr_target": float(fpr_target),
                "k": int(k),
                "granularity": granularity,
                "round_min_filter": round_min,
                "round_max_filter": round_max,
                "expected_rounds": expected_rounds,
            }
        )
    return pd.DataFrame(rows)


def select_best_row(df: pd.DataFrame, metric: str) -> pd.Series:
    ordered = df.sort_values(
        [metric, "f1_score", "precision", "recall"],
        ascending=[False, False, False, False],
        kind="stable",
    )
    return ordered.iloc[0]


def load_inference_benchmark(results_dir: Path) -> pd.DataFrame:
    csv_path = discover_results_csv(results_dir, "inference_benchmark")
    if csv_path is None:
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if "benchmark_batch_size" not in df.columns:
        return pd.DataFrame()
    return df.sort_values(["round", "benchmark_batch_size"]).reset_index(drop=True)


def parse_duration_to_seconds(raw: str) -> float:
    raw = str(raw).strip()
    if not raw:
        return float("nan")
    total = 0.0
    match_h = re.search(r"(\d+)\s*h", raw)
    match_m = re.search(r"(\d+)\s*m", raw)
    match_s = re.search(r"(\d+)\s*s", raw)
    if match_h:
        total += int(match_h.group(1)) * 3600.0
    if match_m:
        total += int(match_m.group(1)) * 60.0
    if match_s:
        total += int(match_s.group(1))
    return total if total > 0 else float("nan")


def parse_baseline_training_log(log_path: Path) -> pd.DataFrame:
    client_rows = []
    current_round = None
    current_client = None

    train_start_pattern = re.compile(
        r"^Round (?P<round>\d+): Training Client (?P<client>\d+) with lr (?P<lr>[-+eE0-9.]+)$"
    )

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            match = train_start_pattern.match(line)
            if match:
                current_round = int(match.group("round"))
                current_client = int(match.group("client"))
                continue
            if line.startswith("{") and "train_runtime" in line:
                try:
                    parsed = ast.literal_eval(line)
                except (SyntaxError, ValueError):
                    continue
                if current_round is None or current_client is None:
                    continue
                client_rows.append(
                    {
                        "round": int(current_round),
                        "client_id": int(current_client),
                        "train_runtime_s": float(parsed.get("train_runtime", math.nan)),
                        "train_loss": float(parsed.get("train_loss", math.nan)),
                        "train_steps_per_second": float(
                            parsed.get("train_steps_per_second", math.nan)
                        ),
                    }
                )

    if not client_rows:
        return pd.DataFrame()

    client_df = pd.DataFrame(client_rows)
    round_df = (
        client_df.groupby("round", as_index=False)
        .agg(
            num_clients=("client_id", "count"),
            mean_client_runtime_s=("train_runtime_s", "mean"),
            round_wall_time_s=("train_runtime_s", "sum"),
            mean_train_loss=("train_loss", "mean"),
            mean_train_steps_per_second=("train_steps_per_second", "mean"),
        )
        .sort_values("round")
        .reset_index(drop=True)
    )
    round_df["cumulative_wall_time_s"] = round_df["round_wall_time_s"].cumsum()
    return round_df


def parse_proposed_training_log(log_path: Path) -> pd.DataFrame:
    client_rows = []
    round_completion = {}
    current_round = None
    current_client = None

    train_start_pattern = re.compile(
        r"^--- Starting training for Client (?P<client>\d+) in Round (?P<round>\d+) ---$"
    )
    round_complete_pattern = re.compile(
        r"^Round (?P<round>\d+) completed in (?P<round_time>.+?) \| Total: (?P<total_time>.+)$"
    )

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            match = train_start_pattern.match(line)
            if match:
                current_round = int(match.group("round"))
                current_client = int(match.group("client"))
                continue

            match = round_complete_pattern.match(line)
            if match:
                round_num = int(match.group("round"))
                round_completion[round_num] = {
                    "round_wall_time_s": parse_duration_to_seconds(match.group("round_time")),
                    "cumulative_wall_time_s": parse_duration_to_seconds(match.group("total_time")),
                }
                continue

            if line.startswith("{") and "train_runtime" in line:
                try:
                    parsed = ast.literal_eval(line)
                except (SyntaxError, ValueError):
                    continue
                if current_round is None or current_client is None:
                    continue
                client_rows.append(
                    {
                        "round": int(current_round),
                        "client_id": int(current_client),
                        "train_runtime_s": float(parsed.get("train_runtime", math.nan)),
                        "train_loss": float(parsed.get("train_loss", math.nan)),
                        "train_steps_per_second": float(
                            parsed.get("train_steps_per_second", math.nan)
                        ),
                    }
                )

    if not client_rows:
        return pd.DataFrame()

    client_df = pd.DataFrame(client_rows)
    round_df = (
        client_df.groupby("round", as_index=False)
        .agg(
            num_clients=("client_id", "count"),
            mean_client_runtime_s=("train_runtime_s", "mean"),
            client_runtime_sum_s=("train_runtime_s", "sum"),
            mean_train_loss=("train_loss", "mean"),
            mean_train_steps_per_second=("train_steps_per_second", "mean"),
        )
        .sort_values("round")
        .reset_index(drop=True)
    )
    round_df["round_wall_time_s"] = round_df["round"].map(
        lambda value: round_completion.get(int(value), {}).get("round_wall_time_s", math.nan)
    )
    round_df["cumulative_wall_time_s"] = round_df["round"].map(
        lambda value: round_completion.get(int(value), {}).get("cumulative_wall_time_s", math.nan)
    )
    return round_df


def load_training_rounds(method: str, results_dir: Path, log_path: Path | None) -> pd.DataFrame:
    if method == "legacy":
        cached = results_dir / "training_metrics.csv"
        if cached.exists():
            df = pd.read_csv(cached)
            rename_map = {}
            if "mean_train_runtime" in df.columns:
                rename_map["mean_train_runtime"] = "mean_client_runtime_s"
            if "num_clients_trained" in df.columns:
                rename_map["num_clients_trained"] = "num_clients"
            df = df.rename(columns=rename_map)
            if "round_wall_time_s" not in df.columns and {
                "mean_client_runtime_s",
                "num_clients",
            }.issubset(df.columns):
                df["round_wall_time_s"] = (
                    df["mean_client_runtime_s"].astype(float)
                    * df["num_clients"].astype(float)
                )
            if "cumulative_wall_time_s" not in df.columns and "round_wall_time_s" in df.columns:
                df["cumulative_wall_time_s"] = df["round_wall_time_s"].cumsum()
            return df.sort_values("round").reset_index(drop=True)
        if log_path is None:
            return pd.DataFrame()
        return parse_baseline_training_log(log_path)

    if log_path is None:
        return pd.DataFrame()
    return parse_proposed_training_log(log_path)


def summarize_training(df: pd.DataFrame, method_label: str) -> dict:
    if df.empty:
        return {
            "method": method_label,
            "num_rounds": 0,
            "mean_round_wall_time_s": math.nan,
            "total_wall_time_s": math.nan,
            "mean_client_runtime_s": math.nan,
        }

    total_wall = (
        float(df["cumulative_wall_time_s"].dropna().iloc[-1])
        if "cumulative_wall_time_s" in df.columns and df["cumulative_wall_time_s"].notna().any()
        else float(df["round_wall_time_s"].sum())
        if "round_wall_time_s" in df.columns
        else math.nan
    )
    return {
        "method": method_label,
        "num_rounds": int(df["round"].nunique()),
        "mean_round_wall_time_s": float(df["round_wall_time_s"].mean())
        if "round_wall_time_s" in df.columns
        else math.nan,
        "total_wall_time_s": total_wall,
        "mean_client_runtime_s": float(df["mean_client_runtime_s"].mean())
        if "mean_client_runtime_s" in df.columns
        else math.nan,
    }


def load_communication_rounds(
    comm_path: Path, round_min: int | None, round_max: int | None
) -> pd.DataFrame:
    if not comm_path.exists():
        return pd.DataFrame()
    return filter_round_range(
        normalize_communication_columns(pd.read_csv(comm_path)),
        round_min,
        round_max,
    )


def summarize_communication(df: pd.DataFrame, method_label: str) -> dict:
    if df.empty:
        return {
            "method": method_label,
            "num_rounds": 0,
            "mean_mb_per_round": math.nan,
            "total_mb": math.nan,
        }
    total_col = "mb_total_estimated" if "mb_total_estimated" in df.columns else None
    return {
        "method": method_label,
        "num_rounds": int(len(df)),
        "mean_mb_per_round": float(df[total_col].mean()) if total_col else math.nan,
        "total_mb": float(df[total_col].sum()) if total_col else math.nan,
    }


def normalize_communication_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "mb_total_estimated" not in df.columns:
        if "bytes_total" in df.columns:
            df["mb_total_estimated"] = df["bytes_total"].astype(float) / (1024 ** 2)
        elif "bytes_total_estimated" in df.columns:
            df["mb_total_estimated"] = df["bytes_total_estimated"].astype(float) / (1024 ** 2)
    return df


def choose_benchmark_rows(
    benchmark_df: pd.DataFrame, detection_df: pd.DataFrame, round_policy: str, fixed_round: int | None
) -> tuple[pd.DataFrame, pd.Series]:
    best_precision_row = select_best_row(detection_df, "precision")
    best_f1_row = select_best_row(detection_df, "f1_score")

    if benchmark_df.empty:
        return benchmark_df, best_precision_row

    if round_policy == "precision":
        selected_round = int(best_precision_row["round"])
        reference_row = best_precision_row
    elif round_policy == "f1":
        selected_round = int(best_f1_row["round"])
        reference_row = best_f1_row
    else:
        if fixed_round is None:
            raise ValueError("fixed_round must be provided when round_policy='fixed'.")
        selected_round = int(fixed_round)
        reference_row = detection_df[detection_df["round"].astype(int) == selected_round]
        if reference_row.empty:
            raise ValueError(f"Round {selected_round} not found in detection results.")
        reference_row = reference_row.iloc[0]

    bench_subset = benchmark_df[benchmark_df["round"].astype(int) == selected_round].copy()
    return bench_subset.sort_values("benchmark_batch_size"), reference_row


def build_detection_round_comparison(
    baseline_df: pd.DataFrame, proposed_df: pd.DataFrame
) -> pd.DataFrame:
    left = baseline_df[
        ["round", "threshold", "f1_score", "precision", "recall", "benign_fpr"]
    ].rename(
        columns={
            "threshold": "legacy_threshold",
            "f1_score": "legacy_f1_score",
            "precision": "legacy_precision",
            "recall": "legacy_recall",
            "benign_fpr": "legacy_benign_fpr",
        }
    )
    right = proposed_df[
        ["round", "threshold", "f1_score", "precision", "recall", "benign_fpr"]
    ].rename(
        columns={
            "threshold": "proposed_threshold",
            "f1_score": "proposed_f1_score",
            "precision": "proposed_precision",
            "recall": "proposed_recall",
            "benign_fpr": "proposed_benign_fpr",
        }
    )
    return left.merge(right, on="round", how="outer").sort_values("round").reset_index(drop=True)


def build_best_metric_delta_summary(
    legacy_best_f1: pd.Series,
    proposed_best_f1: pd.Series,
    legacy_label: str,
    proposed_label: str,
) -> pd.DataFrame:
    rows = []
    for metric in ["f1_score", "precision", "recall", "benign_fpr"]:
        legacy_value = float(legacy_best_f1[metric])
        proposed_value = float(proposed_best_f1[metric])
        delta = proposed_value - legacy_value
        relative_pct = (
            float(delta / legacy_value * 100.0)
            if legacy_value not in (0.0, -0.0) and np.isfinite(legacy_value)
            else float("nan")
        )
        rows.append(
            {
                "metric": metric,
                "original_label": legacy_label,
                "improved_label": proposed_label,
                "original_best_f1_round_value": legacy_value,
                "improved_best_f1_round_value": proposed_value,
                "absolute_delta_improved_minus_original": delta,
                "relative_delta_percent": relative_pct,
                "higher_is_better": metric != "benign_fpr",
            }
        )
    return pd.DataFrame(rows)


def maybe_plot_detection_compare(
    df: pd.DataFrame,
    plot_dir: Path,
    legacy_label: str,
    proposed_label: str,
) -> list[Path]:
    if df.empty:
        return []
    plt = get_pyplot()
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    specs = [
        ("precision", "Precisao"),
        ("recall", "Recall"),
        ("f1_score", "F1"),
        ("benign_fpr", "Benign FPR"),
    ]
    for axis, (metric, title) in zip(axes, specs):
        axis.plot(df["round"], df[f"legacy_{metric}"], marker="o", linewidth=2, label=legacy_label)
        axis.plot(df["round"], df[f"proposed_{metric}"], marker="o", linewidth=2, label=proposed_label)
        axis.set_ylabel(title)
        axis.set_title(f"{title} por rodada com o mesmo criterio operacional")
        axis.grid(alpha=0.3, linestyle="--")
        axis.legend()
    axes[-1].set_xlabel("Rodada")
    return export_figure(fig, plot_dir / "01_detection_by_round_compare")


def maybe_plot_best_metrics_compare(
    delta_df: pd.DataFrame,
    plot_dir: Path,
    legacy_label: str,
    proposed_label: str,
) -> list[Path]:
    if delta_df.empty:
        return []

    plt = get_pyplot()
    label_map = {
        "f1_score": "F1",
        "precision": "Precision",
        "recall": "Recall",
        "benign_fpr": "Benign FPR",
    }
    metrics = [metric for metric in label_map if metric in set(delta_df["metric"])]
    subset = delta_df.set_index("metric").loc[metrics].reset_index()

    x = np.arange(len(subset))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        x - width / 2,
        subset["original_best_f1_round_value"],
        width,
        label=legacy_label,
        color="#4c78a8",
        alpha=0.88,
    )
    ax.bar(
        x + width / 2,
        subset["improved_best_f1_round_value"],
        width,
        label=proposed_label,
        color="#54a24b",
        alpha=0.88,
    )
    ax.set_title("Melhores metricas de deteccao no mesmo criterio operacional")
    ax.set_ylabel("Valor")
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[m] for m in subset["metric"]])
    ax.set_ylim(0, max(1.05, float(subset[[
        "original_best_f1_round_value",
        "improved_best_f1_round_value",
    ]].max().max()) * 1.12))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend()
    return export_figure(fig, plot_dir / "00_best_detection_metrics_compare")


def maybe_plot_inference_compare(df: pd.DataFrame, plot_dir: Path) -> list[Path]:
    if df.empty:
        return []

    plt = get_pyplot()
    method_order = list(dict.fromkeys(df["method_label"].astype(str).tolist()))
    palette = ["#4c78a8", "#54a24b", "#e45756", "#f58518"]
    color_map = {
        method: palette[i % len(palette)] for i, method in enumerate(method_order)
    }
    labels = [
        f"{row['method_label']}|b{int(row['benchmark_batch_size'])}" for _, row in df.iterrows()
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    specs = [
        ("mean_e2e_ms_per_sample", "Media E2E (ms/amostra)"),
        ("p95_e2e_ms_per_sample", "P95 E2E (ms/amostra)"),
        ("p99_e2e_ms_per_sample", "P99 E2E (ms/amostra)"),
        ("throughput_samples_per_second", "Throughput (amostras/s)"),
    ]
    for axis, (column, title) in zip(axes.flat, specs):
        axis.bar(x, df[column], color=[color_map[str(method)] for method in df["method_label"]])
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.3, linestyle="--")
    return export_figure(fig, plot_dir / "02_inference_compare")


def summarize_inference(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    keep_cols = [
        "method_label",
        "reference_round",
        "benchmark_batch_size",
        "timed_samples",
        "mean_e2e_ms_per_sample",
        "median_e2e_ms_per_sample",
        "p95_e2e_ms_per_sample",
        "p99_e2e_ms_per_sample",
        "mean_forward_ms_per_sample",
        "median_forward_ms_per_sample",
        "p95_forward_ms_per_sample",
        "p99_forward_ms_per_sample",
        "throughput_samples_per_second",
        "peak_cuda_memory_mb",
    ]
    out = df[keep_cols].copy()
    out = out.sort_values(["method_label", "benchmark_batch_size"]).reset_index(drop=True)
    return out


def maybe_plot_training_compare(df: pd.DataFrame, plot_dir: Path) -> list[Path]:
    if df.empty:
        return []
    plt = get_pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for method_label, subset in df.groupby("method_label", sort=False):
        subset = subset.sort_values("round")
        axes[0].plot(
            subset["round"],
            subset["round_wall_time_s"] / 60.0,
            marker="o",
            linewidth=2,
            label=method_label,
        )
        if "cumulative_wall_time_s" in subset.columns and subset["cumulative_wall_time_s"].notna().any():
            cumulative = subset["cumulative_wall_time_s"] / 60.0
        else:
            cumulative = subset["round_wall_time_s"].cumsum() / 60.0
        axes[1].plot(
            subset["round"],
            cumulative,
            marker="o",
            linewidth=2,
            label=method_label,
        )
    axes[0].set_title("Tempo de treino por rodada")
    axes[0].set_ylabel("Minutos")
    axes[0].grid(alpha=0.3, linestyle="--")
    axes[0].legend()
    axes[1].set_title("Tempo acumulado de treino")
    axes[1].set_ylabel("Minutos acumulados")
    axes[1].set_xlabel("Rodada")
    axes[1].grid(alpha=0.3, linestyle="--")
    axes[1].legend()
    return export_figure(fig, plot_dir / "03_training_time_compare")


def maybe_plot_communication_compare(df: pd.DataFrame, plot_dir: Path) -> list[Path]:
    if df.empty:
        return []
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 5))
    for method_label, subset in df.groupby("method_label", sort=False):
        subset = subset.sort_values("round")
        ax.plot(subset["round"], subset["mb_total_estimated"], marker="o", linewidth=2, label=method_label)
    ax.set_title("Comunicacao por rodada")
    ax.set_xlabel("Rodada")
    ax.set_ylabel("MB totais estimados")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    return export_figure(fig, plot_dir / "04_communication_compare")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Builds a fair comparison pipeline between the legacy FL-LLM-AD baseline "
            "and the FL_NETWORK_FLOW_LLM WiFi pipeline."
        )
    )
    parser.add_argument(
        "--legacy_root",
        type=Path,
        default=None,
        help=(
            "Root of the legacy baseline project. If omitted, the script tries to find "
            "a sibling FL-LLM-AD checkout next to this repository."
        ),
    )
    parser.add_argument(
        "--proposed_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root of the FL_NETWORK_FLOW_LLM project.",
    )
    parser.add_argument(
        "--legacy_sim",
        default="wifi_tuesday_1152k_fl_llm_ad_12r",
        help="Legacy simulation name.",
    )
    parser.add_argument(
        "--proposed_sim",
        default="WiFi_CICIDS2018_Tuesday_balanced_seqcls_1152k_12r_200s",
        help="Proposed simulation name.",
    )
    parser.add_argument(
        "--legacy_results_dir",
        type=Path,
        default=None,
        help="Optional directory containing the original-code f1_scores*.csv results.",
    )
    parser.add_argument(
        "--proposed_results_dir",
        type=Path,
        default=None,
        help="Optional directory containing the improved-code f1_scores*.csv results.",
    )
    parser.add_argument(
        "--legacy_label",
        default="Codigo original",
        help="Label used for the original article code in plots and summaries.",
    )
    parser.add_argument(
        "--proposed_label",
        default="Codigo melhorado",
        help="Label used for the improved WiFi code in plots and summaries.",
    )
    parser.add_argument(
        "--legacy_log",
        type=Path,
        default=None,
        help="Optional legacy training log. Defaults to auto-discovery by simulation name.",
    )
    parser.add_argument(
        "--proposed_log",
        type=Path,
        default=None,
        help="Optional proposed training log. Defaults to run_wifi_balanced_12r.log or run_wifi_balanced.log heuristics.",
    )
    parser.add_argument(
        "--skip_training_logs",
        action="store_true",
        help="Skip training-log parsing and compare only CSV result artifacts.",
    )
    parser.add_argument("--threshold_mode", default="fpr_target", help="Operational threshold mode.")
    parser.add_argument("--fpr_target", type=float, default=0.10, help="Operational benign FPR target.")
    parser.add_argument("--k", type=int, default=1, help="Top-k to compare.")
    parser.add_argument("--granularity", default="flow", help="Granularity to compare.")
    parser.add_argument(
        "--round_min",
        type=int,
        default=None,
        help="Optional minimum round to keep after metric filters. Use 1 to exclude a PT/round-0 baseline.",
    )
    parser.add_argument(
        "--round_max",
        type=int,
        default=None,
        help="Optional maximum round to keep after metric filters.",
    )
    parser.add_argument(
        "--expected_rounds",
        type=int,
        default=None,
        help="Fail if either method does not have exactly this many detection rounds after filtering.",
    )
    parser.add_argument(
        "--require_same_rounds",
        action="store_true",
        help="Fail if the two methods do not have the same number of detection rounds after filtering.",
    )
    parser.add_argument(
        "--benchmark_round_policy",
        choices=["precision", "f1", "fixed"],
        default="precision",
        help="How to choose the checkpoint round for latency comparison.",
    )
    parser.add_argument("--fixed_round", type=int, default=None, help="Required when benchmark_round_policy=fixed.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to results/comparisons/<legacy>__vs__<proposed>/ inside the proposed repo.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Generate CSV/Markdown summaries only, without importing matplotlib or saving figures.",
    )
    return parser


def discover_legacy_log(legacy_root: Path, sim_name: str) -> Path | None:
    for candidate in sorted(legacy_root.glob("*.log")):
        if sim_name in candidate.stem:
            return candidate
    return None


def discover_proposed_log(proposed_root: Path, sim_name: str) -> Path | None:
    heuristics = []
    if "12r" in sim_name:
        heuristics.append(proposed_root / "run_wifi_balanced_12r.log")
    if "30r" in sim_name:
        heuristics.append(proposed_root / "run_wifi_balanced.log")
    heuristics.extend(sorted(proposed_root.glob("run_*.log")))
    for candidate in heuristics:
        if candidate.exists():
            return candidate
    return None


def discover_legacy_root(proposed_root: Path) -> Path:
    candidates = [
        proposed_root.parent / "FL-LLM-AD",
        proposed_root.parent
        / "Fine-Tuning Eficiente de Modelos de Linguagem Para Detectar Anomalias em Logs Privados usando Aprendizado Federado"
        / "FL-LLM-AD",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not auto-discover the legacy FL-LLM-AD repository. "
        "Pass --legacy_root /path/to/FL-LLM-AD explicitly."
    )


def main() -> None:
    args = build_parser().parse_args()
    proposed_root = args.proposed_root.resolve()
    legacy_root = (
        args.legacy_root.resolve()
        if args.legacy_root is not None
        else discover_legacy_root(proposed_root).resolve()
    )
    legacy_results_dir = (
        args.legacy_results_dir.resolve()
        if args.legacy_results_dir is not None
        else legacy_root / "results" / args.legacy_sim
    )
    proposed_results_dir = (
        args.proposed_results_dir.resolve()
        if args.proposed_results_dir is not None
        else proposed_root / "results" / args.proposed_sim
    )
    legacy_label = str(args.legacy_label)
    proposed_label = str(args.proposed_label)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else proposed_root
        / "results"
        / "comparisons"
        / f"{slugify(args.legacy_sim)}__vs__{slugify(args.proposed_sim)}"
    )
    plot_dir = ensure_dir(output_dir / "plots")
    ensure_dir(output_dir)

    legacy_detection = load_detection_results(
        legacy_results_dir,
        threshold_mode=args.threshold_mode,
        fpr_target=args.fpr_target,
        k=args.k,
        granularity=args.granularity,
        round_min=args.round_min,
        round_max=args.round_max,
    )
    proposed_detection = load_detection_results(
        proposed_results_dir,
        threshold_mode=args.threshold_mode,
        fpr_target=args.fpr_target,
        k=args.k,
        granularity=args.granularity,
        round_min=args.round_min,
        round_max=args.round_max,
    )
    validate_round_budget(
        legacy_detection,
        proposed_detection,
        legacy_label=legacy_label,
        proposed_label=proposed_label,
        expected_rounds=args.expected_rounds,
        require_same_rounds=args.require_same_rounds,
    )

    preflight_summary = build_preflight_summary(
        legacy_results_dir,
        proposed_results_dir,
        legacy_detection,
        proposed_detection,
        threshold_mode=args.threshold_mode,
        fpr_target=args.fpr_target,
        k=args.k,
        granularity=args.granularity,
        round_min=args.round_min,
        round_max=args.round_max,
        expected_rounds=args.expected_rounds,
    )
    preflight_summary.to_csv(output_dir / "comparison_inputs_summary.csv", index=False)

    detection_round_compare = build_detection_round_comparison(
        legacy_detection, proposed_detection
    )
    detection_round_compare.to_csv(output_dir / "detection_round_by_round.csv", index=False)

    legacy_best_precision = select_best_row(legacy_detection, "precision")
    legacy_best_f1 = select_best_row(legacy_detection, "f1_score")
    proposed_best_precision = select_best_row(proposed_detection, "precision")
    proposed_best_f1 = select_best_row(proposed_detection, "f1_score")

    best_summary_rows = []
    for label, row_kind, row in [
        (legacy_label, "best_precision", legacy_best_precision),
        (legacy_label, "best_f1", legacy_best_f1),
        (proposed_label, "best_precision", proposed_best_precision),
        (proposed_label, "best_f1", proposed_best_f1),
    ]:
        best_summary_rows.append(
            {
                "method": label,
                "selection": row_kind,
                "round": int(row["round"]),
                "threshold": float(row["threshold"]),
                "f1_score": float(row["f1_score"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "benign_fpr": float(row["benign_fpr"]),
            }
        )
    pd.DataFrame(best_summary_rows).to_csv(output_dir / "detection_best_summary.csv", index=False)

    delta_summary = build_best_metric_delta_summary(
        legacy_best_f1,
        proposed_best_f1,
        legacy_label,
        proposed_label,
    )
    delta_summary.to_csv(output_dir / "detection_best_f1_delta_summary.csv", index=False)

    legacy_benchmark_df = load_inference_benchmark(legacy_results_dir)
    proposed_benchmark_df = load_inference_benchmark(proposed_results_dir)
    legacy_benchmark_rows, legacy_reference_row = choose_benchmark_rows(
        legacy_benchmark_df,
        legacy_detection,
        args.benchmark_round_policy,
        args.fixed_round,
    )
    proposed_benchmark_rows, proposed_reference_row = choose_benchmark_rows(
        proposed_benchmark_df,
        proposed_detection,
        args.benchmark_round_policy,
        args.fixed_round,
    )

    benchmark_compare = pd.concat(
        [
            legacy_benchmark_rows.assign(
                method_label=legacy_label,
                reference_round=int(legacy_reference_row["round"]),
            ),
            proposed_benchmark_rows.assign(
                method_label=proposed_label,
                reference_round=int(proposed_reference_row["round"]),
            ),
        ],
        ignore_index=True,
    )
    if not benchmark_compare.empty:
        benchmark_compare.to_csv(output_dir / "inference_comparison.csv", index=False)
    inference_summary = summarize_inference(benchmark_compare)
    if not inference_summary.empty:
        inference_summary.to_csv(output_dir / "inference_summary.csv", index=False)

    if args.skip_training_logs:
        legacy_log = None
        proposed_log = None
    else:
        legacy_log = args.legacy_log.resolve() if args.legacy_log else discover_legacy_log(legacy_root, args.legacy_sim)
        proposed_log = args.proposed_log.resolve() if args.proposed_log else discover_proposed_log(proposed_root, args.proposed_sim)

    legacy_training = filter_round_range(
        load_training_rounds("legacy", legacy_results_dir, legacy_log),
        args.round_min,
        args.round_max,
    )
    proposed_training = filter_round_range(
        load_training_rounds("proposed", proposed_results_dir, proposed_log),
        args.round_min,
        args.round_max,
    )
    training_compare = pd.concat(
        [
            legacy_training.assign(method_label=legacy_label),
            proposed_training.assign(method_label=proposed_label),
        ],
        ignore_index=True,
    )
    if not training_compare.empty:
        training_compare.to_csv(output_dir / "training_round_comparison.csv", index=False)

    training_summary = pd.DataFrame(
        [
            summarize_training(legacy_training, legacy_label),
            summarize_training(proposed_training, proposed_label),
        ]
    )
    training_summary.to_csv(output_dir / "training_summary.csv", index=False)

    legacy_comm_path = legacy_results_dir / "communication_metrics.csv"
    proposed_comm_path = proposed_results_dir / "communication_metrics.csv"
    legacy_comm_df = load_communication_rounds(
        legacy_comm_path,
        args.round_min,
        args.round_max,
    )
    proposed_comm_df = load_communication_rounds(
        proposed_comm_path,
        args.round_min,
        args.round_max,
    )
    comm_rows = []
    if not legacy_comm_df.empty:
        comm_rows.append(legacy_comm_df.assign(method_label=legacy_label))
    if not proposed_comm_df.empty:
        comm_rows.append(proposed_comm_df.assign(method_label=proposed_label))
    communication_compare = pd.concat(comm_rows, ignore_index=True) if comm_rows else pd.DataFrame()
    if not communication_compare.empty:
        communication_compare.to_csv(output_dir / "communication_round_comparison.csv", index=False)
    communication_summary = pd.DataFrame(
        [
            summarize_communication(legacy_comm_df, legacy_label),
            summarize_communication(proposed_comm_df, proposed_label),
        ]
    )
    communication_summary.to_csv(output_dir / "communication_summary.csv", index=False)

    generated_plots = []
    if not args.no_plots:
        generated_plots += maybe_plot_best_metrics_compare(
            delta_summary,
            plot_dir,
            legacy_label,
            proposed_label,
        )
        generated_plots += maybe_plot_detection_compare(
            detection_round_compare,
            plot_dir,
            legacy_label,
            proposed_label,
        )
        generated_plots += maybe_plot_inference_compare(benchmark_compare, plot_dir)
        generated_plots += maybe_plot_training_compare(training_compare, plot_dir)
        generated_plots += maybe_plot_communication_compare(communication_compare, plot_dir)

    summary_md = output_dir / "comparison_summary.md"
    legacy_rounds = int(legacy_detection["round"].nunique())
    proposed_rounds = int(proposed_detection["round"].nunique())
    same_round_budget = legacy_rounds == proposed_rounds
    delta_rows = {
        row["metric"]: row for _, row in delta_summary.iterrows()
    }
    lines = [
        "# Comparacao Operacional",
        "",
        f"- Resultados do codigo original: `{legacy_results_dir}`",
        f"- Resultados do codigo melhorado: `{proposed_results_dir}`",
        f"- Modo de threshold comparado: `{args.threshold_mode}`",
        f"- Meta operacional comum: `fpr_target = {args.fpr_target}`",
        f"- `k` comparado: `{args.k}`",
        f"- Granularidade: `{args.granularity}`",
        f"- Filtro minimo de rodada: `{args.round_min}`",
        f"- Filtro maximo de rodada: `{args.round_max}`",
        f"- Rodadas esperadas por metodo: `{args.expected_rounds}`",
        f"- Rodadas comparadas no codigo original: `{legacy_rounds}`",
        f"- Rodadas comparadas no codigo melhorado: `{proposed_rounds}`",
        "",
        "## Conclusao principal",
        "",
    ]
    if same_round_budget:
        lines.append(
            "- Os dois metodos foram comparados com o mesmo numero de rodadas apos os filtros aplicados."
        )
    else:
        lines.append(
            "- Atencao: os dois metodos nao possuem o mesmo numero de rodadas apos os filtros aplicados."
        )
    if {"f1_score", "recall", "benign_fpr"}.issubset(delta_rows):
        f1_delta = float(delta_rows["f1_score"]["absolute_delta_improved_minus_original"])
        recall_delta = float(delta_rows["recall"]["absolute_delta_improved_minus_original"])
        fpr_delta = float(delta_rows["benign_fpr"]["absolute_delta_improved_minus_original"])
        lines.extend(
            [
                (
                    f"- No melhor checkpoint por F1, o codigo melhorado altera o F1 em "
                    f"`{f1_delta:+.4f}`, o recall em `{recall_delta:+.4f}` e a FPR benigna "
                    f"em `{fpr_delta:+.4f}` em relacao ao codigo original."
                ),
                "- Valores positivos em F1/precision/recall favorecem o codigo melhorado; para Benign FPR, valores negativos indicam menos falsos positivos.",
            ]
        )
    lines.extend(
        [
            "",
            "## Melhor rodada por precisao no mesmo criterio operacional",
            "",
            (
                f"- {legacy_label}: rodada `{int(legacy_best_precision['round'])}`, "
                f"precision `{float(legacy_best_precision['precision']):.4f}`, "
                f"recall `{float(legacy_best_precision['recall']):.4f}`, "
                f"F1 `{float(legacy_best_precision['f1_score']):.4f}`, "
                f"Benign FPR `{float(legacy_best_precision['benign_fpr']):.4f}`"
            ),
            (
                f"- {proposed_label}: rodada `{int(proposed_best_precision['round'])}`, "
                f"precision `{float(proposed_best_precision['precision']):.4f}`, "
                f"recall `{float(proposed_best_precision['recall']):.4f}`, "
                f"F1 `{float(proposed_best_precision['f1_score']):.4f}`, "
                f"Benign FPR `{float(proposed_best_precision['benign_fpr']):.4f}`"
            ),
            "",
            "## Melhor rodada por F1 no mesmo criterio operacional",
            "",
            (
                f"- {legacy_label}: rodada `{int(legacy_best_f1['round'])}`, "
                f"precision `{float(legacy_best_f1['precision']):.4f}`, "
                f"recall `{float(legacy_best_f1['recall']):.4f}`, "
                f"F1 `{float(legacy_best_f1['f1_score']):.4f}`, "
                f"Benign FPR `{float(legacy_best_f1['benign_fpr']):.4f}`"
            ),
            (
                f"- {proposed_label}: rodada `{int(proposed_best_f1['round'])}`, "
                f"precision `{float(proposed_best_f1['precision']):.4f}`, "
                f"recall `{float(proposed_best_f1['recall']):.4f}`, "
                f"F1 `{float(proposed_best_f1['f1_score']):.4f}`, "
                f"Benign FPR `{float(proposed_best_f1['benign_fpr']):.4f}`"
            ),
            "",
            "## Observacao metodologica",
            "",
            "- A comparacao usa o mesmo criterio operacional (`fpr_target`) e nao o mesmo valor numerico bruto de threshold.",
            "- Isso torna a comparacao de precisao e recall muito mais justa entre score spaces diferentes.",
            "",
            "## Quase tempo real / latencia operacional",
            "",
        ]
    )
    if inference_summary.empty:
        lines.extend(
            [
                "- Nenhum `inference_benchmark*.csv` foi encontrado ainda.",
                "- Para comparar latencia e throughput, rode os benchmarks nos dois metodos e execute novamente este comparador.",
                "",
            ]
        )
    else:
        for _, row in inference_summary.iterrows():
            lines.append(
                (
                    f"- {row['method_label']} | rodada `{int(row['reference_round'])}` | "
                    f"batch `{int(row['benchmark_batch_size'])}` | "
                    f"latencia media E2E `{float(row['mean_e2e_ms_per_sample']):.3f} ms/amostra` | "
                    f"p95 `{float(row['p95_e2e_ms_per_sample']):.3f} ms` | "
                    f"p99 `{float(row['p99_e2e_ms_per_sample']):.3f} ms` | "
                    f"throughput `{float(row['throughput_samples_per_second']):.3f} amostras/s`"
                )
            )
        lines.extend(
            [
                "",
                "- `batch = 1` representa o cenario mais proximo de deteccao online por fluxo.",
                "- `batch > 1` representa micro-batching, util para IDS quase em tempo real com maior throughput.",
                "",
            ]
        )

    lines.extend(
        [
            "## Arquivos gerados",
            "",
            f"- CSVs e sumarios: `{output_dir}`",
            f"- Graficos: `{plot_dir}`" if not args.no_plots else "- Graficos: nao gerados porque `--no_plots` foi usado.",
        ]
    )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Comparison output dir: {output_dir}")
    print(f"Generated plots: {len(generated_plots)}")
    for path in generated_plots:
        print(f"  - {path}")
    print(f"Summary markdown: {summary_md}")


if __name__ == "__main__":
    main()
