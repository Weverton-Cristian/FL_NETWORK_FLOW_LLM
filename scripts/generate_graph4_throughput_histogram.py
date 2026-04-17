from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import PercentFormatter


@dataclass(frozen=True)
class BinSpec:
    label: str
    lower_exclusive: float | None
    upper_inclusive: float | None


BIN_SPECS: tuple[BinSpec, ...] = (
    BinSpec("0", None, 0.0),
    BinSpec("0-1", 0.0, 1.0),
    BinSpec("1-10", 1.0, 10.0),
    BinSpec("10-100", 10.0, 100.0),
    BinSpec("100-1k", 100.0, 1_000.0),
    BinSpec("1k-10k", 1_000.0, 10_000.0),
    BinSpec("10k-100k", 10_000.0, 100_000.0),
    BinSpec("100k-1M", 100_000.0, 1_000_000.0),
    BinSpec(">1M", 1_000_000.0, None),
)

PALETTE = ["#0f766e", "#e76f51", "#3a86ff", "#6a4c93", "#bc4749", "#495057"]
NEUTRAL_COLOR = "#98a2b3"
GRID_COLOR = "#d9dee7"
BASE_FONT_SIZE = 11
FIG_WIDTH = 12.6
FIG_HEIGHT = 9.6
LINE_WIDTH = 2.2
MARKER_SIZE = 5.4
TARGET_FPR = 0.10


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()
    return slug or "plot"


def style_axis(ax) -> None:
    ax.grid(True, alpha=0.8, linestyle="--", linewidth=0.7, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b0b8c2")
    ax.spines["bottom"].set_color("#b0b8c2")
    ax.set_axisbelow(True)


def add_subtitle(ax, text: str, y: float = 1.08) -> None:
    ax.text(
        0.0,
        y,
        text,
        transform=ax.transAxes,
        fontsize=BASE_FONT_SIZE - 1,
        color="#5b6470",
    )


def find_results_csv(results_dir: Path, basename: str) -> Path | None:
    exact = results_dir / f"{basename}.csv"
    if exact.exists():
        return exact
    matches = sorted(results_dir.glob(f"{basename}_*.csv"))
    return matches[0] if matches else None


def prefer_granularity(df: pd.DataFrame, preferred: str) -> pd.DataFrame:
    if "granularity" not in df.columns:
        return df.copy()
    subset = df[df["granularity"].astype(str) == preferred].copy()
    return subset if not subset.empty else df.copy()


def assign_bin_labels(values: pd.Series) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    labels = pd.Series(index=numeric.index, dtype="object")

    for spec in BIN_SPECS:
        if spec.lower_exclusive is None and spec.upper_inclusive is not None:
            mask = numeric <= spec.upper_inclusive
        elif spec.upper_inclusive is None and spec.lower_exclusive is not None:
            mask = numeric > spec.lower_exclusive
        else:
            mask = (numeric > float(spec.lower_exclusive)) & (
                numeric <= float(spec.upper_inclusive)
            )
        labels.loc[mask] = spec.label

    ordered = [spec.label for spec in BIN_SPECS]
    labels = labels.fillna(ordered[-1])
    return pd.Categorical(labels, categories=ordered, ordered=True)


def save_figure(
    fig,
    *,
    output_dir: Path,
    stem: str,
    save_dpi: int = 300,
    save_svg: bool = True,
    save_pdf: bool = True,
) -> dict[str, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=save_dpi, bbox_inches="tight")
    svg_path = None
    pdf_path = None
    if save_svg:
        svg_path = output_dir / f"{stem}.svg"
        fig.savefig(svg_path, bbox_inches="tight")
    if save_pdf:
        pdf_path = output_dir / f"{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
    return {"png": png_path, "svg": svg_path, "pdf": pdf_path}


def load_default_simulation_from_config(config_path: Path) -> tuple[str, str]:
    fallback = "WiFi_CICIDS2018_Tuesday_balanced_seqcls_1152k_12r_200s"
    if not config_path.exists():
        return fallback, fallback

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except Exception:
        return fallback, fallback

    simulation_name = str(config.get("simulation_name", fallback))
    simulation_label = str(config.get("description", simulation_name))
    return simulation_name, simulation_label


def load_f1_dataframe(results_dir: Path, preferred_granularity: str) -> pd.DataFrame:
    f1_path = find_results_csv(results_dir, "f1_scores")
    if f1_path is None:
        raise FileNotFoundError(
            f"Nao encontrei f1_scores.csv ou f1_scores_*.csv em {results_dir}"
        )
    df = pd.read_csv(f1_path)
    if df.empty:
        raise ValueError(f"{f1_path} esta vazio.")
    return prefer_granularity(df, preferred_granularity)


def summarize_train_distribution(
    train_df: pd.DataFrame, throughput_column: str
) -> pd.DataFrame:
    df = train_df.copy()
    df["bin_label"] = assign_bin_labels(df[throughput_column])

    if "Label" not in df.columns:
        counts = (
            df.groupby("bin_label", observed=False)
            .size()
            .reindex([spec.label for spec in BIN_SPECS], fill_value=0)
        )
        return pd.DataFrame(
            {
                "bin_label": [spec.label for spec in BIN_SPECS],
                "train_total_logs": counts.astype(int).values,
                "train_benign_logs": counts.astype(int).values,
                "train_anomaly_logs": np.zeros(len(BIN_SPECS), dtype=int),
            }
        )

    grouped = (
        df.groupby("bin_label", observed=False)
        .agg(
            train_total_logs=("Label", "size"),
            train_benign_logs=("Label", lambda s: int((s.astype(int) == 0).sum())),
            train_anomaly_logs=("Label", lambda s: int((s.astype(int) == 1).sum())),
        )
        .reset_index()
    )
    return grouped


def summarize_test_distribution(
    detail_df: pd.DataFrame, throughput_column: str
) -> pd.DataFrame:
    df = detail_df.copy()
    df["bin_label"] = assign_bin_labels(df[throughput_column])

    grouped = (
        df.groupby("bin_label", observed=False)
        .agg(
            test_total_logs=("Label", "size"),
            test_anomaly_logs=("Label", lambda s: int((s.astype(int) == 1).sum())),
            test_benign_logs=("Label", lambda s: int((s.astype(int) == 0).sum())),
            flagged_as_anomaly=("pred", lambda s: int(s.astype(int).sum())),
            detected_true_anomalies=("detected_true_anomaly", lambda s: int(s.sum())),
            false_positive_alerts=("false_positive", lambda s: int(s.sum())),
        )
        .reset_index()
    )

    grouped["detection_rate"] = grouped.apply(
        lambda row: float(row["detected_true_anomalies"]) / float(row["test_anomaly_logs"])
        if float(row["test_anomaly_logs"]) > 0
        else 0.0,
        axis=1,
    )
    grouped["benign_fpr_in_bin"] = grouped.apply(
        lambda row: float(row["false_positive_alerts"]) / float(row["test_benign_logs"])
        if float(row["test_benign_logs"]) > 0
        else 0.0,
        axis=1,
    )
    grouped["alert_precision_in_bin"] = grouped.apply(
        lambda row: float(row["detected_true_anomalies"]) / float(row["flagged_as_anomaly"])
        if float(row["flagged_as_anomaly"]) > 0
        else 0.0,
        axis=1,
    )
    grouped["anomaly_share_in_test_bin"] = grouped.apply(
        lambda row: float(row["test_anomaly_logs"]) / float(row["test_total_logs"])
        if float(row["test_total_logs"]) > 0
        else 0.0,
        axis=1,
    )
    return grouped


def build_graph4_summary(
    *,
    project_root: Path,
    simulation_name: str,
    preferred_granularity: str,
    throughput_column: str,
    throughput_bins_for_cache: int,
    analysis_round: str,
    analysis_k: str,
    config_path: Path,
    force_recompute: bool,
) -> tuple[pd.DataFrame, pd.Series, Path]:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    plot_root = project_root / "results" / "plot"
    if str(plot_root) not in sys.path:
        sys.path.insert(0, str(plot_root))

    from throughput_detection_helpers import build_throughput_detection_data

    results_dir = project_root / "results" / simulation_name
    flow_f1_df = load_f1_dataframe(results_dir, preferred_granularity)
    payload = build_throughput_detection_data(
        project_root=project_root,
        simulation_name=simulation_name,
        flow_f1_df=flow_f1_df,
        throughput_column=throughput_column,
        throughput_bins=throughput_bins_for_cache,
        analysis_round=analysis_round,
        analysis_k=analysis_k,
        config_path=config_path,
        force_recompute=force_recompute,
    )

    train_path = project_root / "data" / "wifi" / "processed" / "train.csv"
    train_columns = pd.read_csv(train_path, nrows=0).columns.tolist()
    train_usecols = [throughput_column]
    if "Label" in train_columns:
        train_usecols.append("Label")
    train_df = pd.read_csv(train_path, usecols=train_usecols)
    train_summary = summarize_train_distribution(train_df, throughput_column)
    test_summary = summarize_test_distribution(payload["detail_df"], throughput_column)

    summary_df = train_summary.merge(test_summary, on="bin_label", how="left").fillna(0)
    summary_df["throughput_metric"] = throughput_column
    summary_df["analysis_round"] = int(payload["target_row"]["round"])
    summary_df["analysis_k"] = int(payload["target_row"]["k"])
    summary_df["threshold"] = float(payload["target_row"]["threshold"])
    summary_df["threshold_mode"] = str(payload["target_row"].get("threshold_mode", ""))

    cache_dir = (
        project_root
        / "results"
        / "plot"
        / "generated"
        / slugify(simulation_name)
        / "cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cache_dir / (
        f"graph4_throughput_distribution_round_{int(payload['target_row']['round'])}"
        f"_k_{int(payload['target_row']['k'])}_{slugify(throughput_column)}_log_bins_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)
    return summary_df, payload["target_row"], summary_path


def plot_graph4(
    *,
    summary_df: pd.DataFrame,
    target_row: pd.Series,
    output_dir: Path,
    simulation_label: str,
    throughput_column: str,
    show_plot: bool = False,
) -> dict[str, Path | None]:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#b0b8c2",
            "axes.linewidth": 0.9,
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE + 2,
            "axes.labelsize": BASE_FONT_SIZE,
            "legend.fontsize": BASE_FONT_SIZE - 1,
            "xtick.labelsize": BASE_FONT_SIZE - 1,
            "ytick.labelsize": BASE_FONT_SIZE - 1,
        }
    )

    x = np.arange(len(summary_df))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        sharex=True,
        gridspec_kw={"height_ratios": [1.18, 1.0]},
    )

    zero_bin_idx = 0
    axes[0].axvspan(zero_bin_idx - 0.5, zero_bin_idx + 0.5, color="#f8edeb", alpha=0.65)
    axes[1].axvspan(zero_bin_idx - 0.5, zero_bin_idx + 0.5, color="#f8edeb", alpha=0.65)

    train_has_anomaly = bool(
        "train_anomaly_logs" in summary_df.columns
        and (summary_df["train_anomaly_logs"].astype(float) > 0).any()
    )

    if train_has_anomaly:
        width = 0.2
        axes[0].bar(
            x - 1.5 * width,
            summary_df["train_benign_logs"],
            width=width,
            color=PALETTE[0],
            label="Treino benigno",
        )
        axes[0].bar(
            x - 0.5 * width,
            summary_df["train_anomaly_logs"],
            width=width,
            color="#ef476f",
            label="Treino anomalo",
        )
        axes[0].bar(
            x + 0.5 * width,
            summary_df["test_benign_logs"],
            width=width,
            color=NEUTRAL_COLOR,
            label="Teste benigno",
        )
        axes[0].bar(
            x + 1.5 * width,
            summary_df["test_anomaly_logs"],
            width=width,
            color=PALETTE[1],
            label="Teste anomalo",
        )
    else:
        width = 0.26
        axes[0].bar(
            x - width,
            summary_df["train_benign_logs"],
            width=width,
            color=PALETTE[0],
            label="Treino benigno",
        )
        axes[0].bar(
            x,
            summary_df["test_benign_logs"],
            width=width,
            color=NEUTRAL_COLOR,
            label="Teste benigno",
        )
        axes[0].bar(
            x + width,
            summary_df["test_anomaly_logs"],
            width=width,
            color=PALETTE[1],
            label="Teste anomalo",
        )
    axes[0].set_yscale("symlog", linthresh=10)
    axes[0].set_ylabel("Quantidade de logs (symlog)")
    axes[0].set_title("Distribuicao real do throughput por faixa fixa", loc="left", pad=8)
    add_subtitle(
        axes[0],
        (
            f"{simulation_label} | checkpoint R{int(target_row['round'])}, "
            f"K={int(target_row['k'])}, threshold={float(target_row['threshold']):.4f} | "
            "bins log fixos"
        ),
    )
    style_axis(axes[0])
    axes[0].legend(
        frameon=False,
        ncol=4 if train_has_anomaly else 3,
        loc="upper right",
    )
    axes[0].text(
        zero_bin_idx,
        axes[0].get_ylim()[1] * 0.92,
        "throughput zero",
        ha="center",
        va="top",
        fontsize=9,
        color="#7f5539",
    )

    axes[1].plot(
        x,
        summary_df["detection_rate"],
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        color=PALETTE[2],
        label="Taxa de deteccao",
    )
    axes[1].plot(
        x,
        summary_df["benign_fpr_in_bin"],
        marker="s",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE - 0.2,
        color="#d62828",
        label="Benign FPR por faixa",
    )
    axes[1].plot(
        x,
        summary_df["alert_precision_in_bin"],
        marker="^",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE - 0.2,
        linestyle="--",
        color=PALETTE[3],
        label="Precisao dos alertas",
    )
    axes[1].axhline(TARGET_FPR, color=NEUTRAL_COLOR, linestyle=":", linewidth=1.4)
    axes[1].text(
        len(summary_df) - 0.2,
        TARGET_FPR,
        "meta 10%",
        color="#5b6470",
        va="bottom",
        ha="right",
        fontsize=9,
    )
    axes[1].set_ylabel("Razao")
    axes[1].set_xlabel(f"Faixas fixas de throughput ({throughput_column})")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary_df["bin_label"], rotation=0, ha="center")
    style_axis(axes[1])
    axes[1].legend(frameon=False, ncol=3, loc="upper center")

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    paths = save_figure(
        fig,
        output_dir=output_dir,
        stem="03b_throughput_distribution_histogram",
    )
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_project_root = script_dir.parent
    default_config_path = default_project_root / "configs" / "config_wifi.yaml"
    default_simulation, default_label = load_default_simulation_from_config(
        default_config_path
    )

    parser = argparse.ArgumentParser(
        description="Gera o grafico 4 de distribuicao de throughput por bins fixos."
    )
    parser.add_argument("--project-root", type=Path, default=default_project_root)
    parser.add_argument("--simulation-name", default=default_simulation)
    parser.add_argument("--simulation-label", default=default_label)
    parser.add_argument("--preferred-granularity", default="flow")
    parser.add_argument("--throughput-column", default="Flow Byts/s")
    parser.add_argument("--throughput-bins-for-cache", type=int, default=8)
    parser.add_argument("--analysis-round", default="best")
    parser.add_argument("--analysis-k", default="best")
    parser.add_argument("--config-path", type=Path, default=default_config_path)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config_path.resolve()
    simulation_label = args.simulation_label or args.simulation_name

    summary_df, target_row, summary_path = build_graph4_summary(
        project_root=project_root,
        simulation_name=args.simulation_name,
        preferred_granularity=args.preferred_granularity,
        throughput_column=args.throughput_column,
        throughput_bins_for_cache=args.throughput_bins_for_cache,
        analysis_round=args.analysis_round,
        analysis_k=args.analysis_k,
        config_path=config_path,
        force_recompute=args.force_recompute,
    )

    output_dir = (
        project_root / "results" / "plot" / "generated" / slugify(args.simulation_name)
    )
    paths = plot_graph4(
        summary_df=summary_df,
        target_row=target_row,
        output_dir=output_dir,
        simulation_label=simulation_label,
        throughput_column=args.throughput_column,
    )

    print(
        {
            "png": str(paths["png"]),
            "svg": str(paths["svg"]) if paths["svg"] else None,
            "pdf": str(paths["pdf"]) if paths["pdf"] else None,
            "summary_csv": str(summary_path),
            "analysis_round": int(target_row["round"]),
            "analysis_k": int(target_row["k"]),
            "threshold": float(target_row["threshold"]),
        }
    )


if __name__ == "__main__":
    main()
