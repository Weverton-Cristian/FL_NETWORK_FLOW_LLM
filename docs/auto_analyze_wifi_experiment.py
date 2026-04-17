from __future__ import annotations

import argparse
import csv
import html
import json
import math
import time
from pathlib import Path
from typing import Iterable

from generate_wifi_report import (
    DOCS_DIR,
    ROOT,
    _environment_snapshot,
    _fmt_float,
    _html_table,
    convert_html_to_pdf,
    html_to_text,
    write_simple_pdf,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _slugify(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch.lower())
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def _count_labels(path: Path) -> dict[str, int]:
    counts = {"total": 0, "benign": 0, "anomaly": 0}
    if not path.exists():
        return counts
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts["total"] += 1
            label = str(row.get("Label", "")).strip()
            if label == "0":
                counts["benign"] += 1
            elif label == "1":
                counts["anomaly"] += 1
    return counts


def _find_results_csv(results_dir: Path, basename: str) -> Path | None:
    exact = results_dir / f"{basename}.csv"
    if exact.exists():
        return exact

    matches = sorted(results_dir.glob(f"{basename}_*.csv"))
    if matches:
        return matches[0]
    return None


def _wait_for_outputs(results_dir: Path, poll_seconds: int) -> tuple[Path, Path | None, Path]:
    while True:
        f1_path = _find_results_csv(results_dir, "f1_scores")
        temporal_path = _find_results_csv(results_dir, "temporal_metrics")
        comm_path = results_dir / "communication_metrics.csv"

        if f1_path is not None and comm_path.exists():
            return f1_path, temporal_path, comm_path

        print(
            f"[wait] Results not complete yet in {results_dir}. "
            f"Polling again in {poll_seconds}s..."
        )
        time.sleep(poll_seconds)


def _load_rows(results_dir: Path, wait: bool, poll_seconds: int) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    if wait:
        f1_path, temporal_path, comm_path = _wait_for_outputs(results_dir, poll_seconds)
    else:
        f1_path = _find_results_csv(results_dir, "f1_scores")
        temporal_path = _find_results_csv(results_dir, "temporal_metrics")
        comm_path = results_dir / "communication_metrics.csv"

    if f1_path is None or not comm_path.exists():
        raise FileNotFoundError(
            f"Required result files were not found in {results_dir}. "
            "Expected at least communication_metrics.csv and an f1_scores*.csv file."
        )

    f1_rows = _read_csv(f1_path)
    temporal_rows = _read_csv(temporal_path) if temporal_path and temporal_path.exists() else []
    comm_rows = _read_csv(comm_path)
    return f1_rows, temporal_rows, comm_rows


def _filter_flow_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return rows
    if "granularity" not in rows[0]:
        return rows
    flow_rows = [row for row in rows if row.get("granularity", "flow") == "flow"]
    return flow_rows or rows


def _best_overall(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(rows, key=lambda row: float(row["f1_score"]))


def _best_by_k(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, str]] = {}
    for row in rows:
        k = str(row["k"])
        if k not in grouped or float(row["f1_score"]) > float(grouped[k]["f1_score"]):
            grouped[k] = row
    return [grouped[k] for k in sorted(grouped, key=lambda item: int(item))]


def _communication_summary(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {}
    total_rounds = len(rows)
    mean_selected = sum(float(row["num_selected_clients"]) for row in rows) / total_rounds
    mean_bytes = sum(float(row["bytes_total"]) for row in rows) / total_rounds
    mean_params = sum(float(row["params_total"]) for row in rows) / total_rounds
    return {
        "rounds": total_rounds,
        "mean_selected_clients": mean_selected,
        "mean_bytes_total": mean_bytes,
        "mean_params_total": mean_params,
    }


def _find_baseline_best(compare_dir: Path | None) -> dict[str, str] | None:
    if compare_dir is None:
        return None
    f1_path = _find_results_csv(compare_dir, "f1_scores")
    if f1_path is None or not f1_path.exists():
        return None
    rows = _filter_flow_rows(_read_csv(f1_path))
    k1_rows = [row for row in rows if str(row["k"]) == "1"]
    if not k1_rows:
        return None
    return _best_overall(k1_rows)


def _observation_lines(best: dict[str, str], rounds: int) -> list[str]:
    precision = float(best["precision"])
    recall = float(best["recall"])
    benign_fpr = float(best["benign_fpr"])
    best_round = int(best["round"])

    lines = []
    if recall >= 0.95 and benign_fpr >= 0.20:
        lines.append(
            "O detector está muito sensível: encontra quase todos os anômalos, mas ainda produz muitos falsos positivos benignos."
        )
    elif recall >= 0.95:
        lines.append(
            "O detector apresenta alta sensibilidade operacional, com revocação muito alta."
        )
    elif precision >= 0.75:
        lines.append(
            "O detector ficou relativamente conservador, priorizando precisão acima de sensibilidade extrema."
        )

    if best_round == rounds:
        lines.append(
            "A melhor rodada apareceu no fim do treino; isso sugere que ainda havia espaço para ganho com mais rodadas."
        )
    elif best_round <= max(2, rounds // 3):
        lines.append(
            "A melhor rodada apareceu cedo; isso sugere saturação rápida e possível retorno decrescente ao prolongar o treino."
        )
    else:
        lines.append(
            "A melhor rodada apareceu no meio do treino, indicando aprendizado contínuo sem saturação imediata."
        )

    return lines


def _summary_payload(
    simulation_name: str,
    results_dir: Path,
    f1_rows: list[dict[str, str]],
    temporal_rows: list[dict[str, str]],
    comm_rows: list[dict[str, str]],
    compare_best_k1: dict[str, str] | None,
) -> dict:
    flow_rows = _filter_flow_rows(f1_rows)
    best_rows = _best_by_k(flow_rows)
    best_overall = _best_overall(flow_rows)
    comm_summary = _communication_summary(comm_rows)

    processed_dir = ROOT / "data" / "wifi" / "processed"
    train_counts = _count_labels(processed_dir / "train.csv")
    test_counts = _count_labels(processed_dir / "test.csv")
    calibration_counts = _count_labels(processed_dir / "calibration.csv")

    payload = {
        "simulation_name": simulation_name,
        "results_dir": str(results_dir),
        "best_overall": best_overall,
        "best_by_k": best_rows,
        "communication_summary": comm_summary,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "calibration_counts": calibration_counts,
        "temporal_rows": len(temporal_rows),
        "observations": _observation_lines(best_overall, int(comm_summary.get("rounds", 0) or 0)),
    }

    if compare_best_k1 is not None:
        payload["baseline_k1"] = compare_best_k1
        payload["delta_vs_baseline_k1"] = {
            "f1_score": float(best_overall["f1_score"]) - float(compare_best_k1["f1_score"]),
            "precision": float(best_overall["precision"]) - float(compare_best_k1["precision"]),
            "recall": float(best_overall["recall"]) - float(compare_best_k1["recall"]),
            "benign_fpr": float(best_overall["benign_fpr"]) - float(compare_best_k1["benign_fpr"]),
        }

    return payload


def _markdown_report(
    payload: dict,
    f1_rows: list[dict[str, str]],
    temporal_rows: list[dict[str, str]],
    comm_rows: list[dict[str, str]],
) -> str:
    best = payload["best_overall"]
    lines = [
        f"# Analise Automatizada: {payload['simulation_name']}",
        "",
        f"Resultados em: `{payload['results_dir']}`",
        "",
        "## Melhor Resultado",
        f"- Rodada: {best['round']}",
        f"- K: {best['k']}",
        f"- Modo de threshold: {best.get('threshold_mode', 'N/A')}",
        f"- Threshold: {_fmt_float(best['threshold'], 4)}",
        f"- F1: {_fmt_float(best['f1_score'])}",
        f"- Precisao: {_fmt_float(best['precision'])}",
        f"- Recall: {_fmt_float(best['recall'])}",
        f"- Benign FPR: {_fmt_float(best['benign_fpr'])}",
        "",
        "## Tamanho dos Splits",
        f"- Train benigno: {payload['train_counts']['total']}",
        f"- Teste total: {payload['test_counts']['total']}",
        f"- Teste benigno: {payload['test_counts']['benign']}",
        f"- Teste anomalo: {payload['test_counts']['anomaly']}",
        f"- Calibracao benigna: {payload['calibration_counts']['total']}",
        "",
        "## Observacoes",
    ]

    for item in payload["observations"]:
        lines.append(f"- {item}")

    if "delta_vs_baseline_k1" in payload:
        delta = payload["delta_vs_baseline_k1"]
        lines.extend(
            [
                "",
                "## Comparacao com Baseline",
                f"- Delta F1: {_fmt_float(delta['f1_score'])}",
                f"- Delta Precisao: {_fmt_float(delta['precision'])}",
                f"- Delta Recall: {_fmt_float(delta['recall'])}",
                f"- Delta Benign FPR: {_fmt_float(delta['benign_fpr'])}",
            ]
        )

    lines.extend(
        [
            "",
            "## Rodadas por K",
        ]
    )
    for row in payload["best_by_k"]:
        lines.append(
            f"- K={row['k']}: melhor rodada {row['round']} com F1={_fmt_float(row['f1_score'])}, "
            f"Precision={_fmt_float(row['precision'])}, Recall={_fmt_float(row['recall'])}, "
            f"BenignFPR={_fmt_float(row['benign_fpr'])}"
        )

    lines.extend(
        [
            "",
            "## Arquivos Detectados",
            f"- F1 rows: {len(f1_rows)}",
            f"- Temporal rows: {len(temporal_rows)}",
            f"- Communication rows: {len(comm_rows)}",
        ]
    )

    return "\n".join(lines) + "\n"


def _html_report(payload: dict, f1_rows: list[dict[str, str]], temporal_rows: list[dict[str, str]], comm_rows: list[dict[str, str]]) -> str:
    best = payload["best_overall"]
    env = _environment_snapshot()

    best_table = _html_table(
        [
            {
                "round": best["round"],
                "k": best["k"],
                "threshold_mode": best.get("threshold_mode", "N/A"),
                "threshold": _fmt_float(best["threshold"], 4),
                "f1_score": _fmt_float(best["f1_score"]),
                "precision": _fmt_float(best["precision"]),
                "recall": _fmt_float(best["recall"]),
                "benign_fpr": _fmt_float(best["benign_fpr"]),
            }
        ],
        [
            ("round", "Rodada"),
            ("k", "K"),
            ("threshold_mode", "Threshold"),
            ("threshold", "Limiar"),
            ("f1_score", "F1"),
            ("precision", "Precisao"),
            ("recall", "Recall"),
            ("benign_fpr", "Benign FPR"),
        ],
    )

    best_by_k_table = _html_table(
        [
            {
                "k": row["k"],
                "round": row["round"],
                "f1_score": _fmt_float(row["f1_score"]),
                "precision": _fmt_float(row["precision"]),
                "recall": _fmt_float(row["recall"]),
                "benign_fpr": _fmt_float(row["benign_fpr"]),
            }
            for row in payload["best_by_k"]
        ],
        [
            ("k", "K"),
            ("round", "Melhor Rodada"),
            ("f1_score", "F1"),
            ("precision", "Precisao"),
            ("recall", "Recall"),
            ("benign_fpr", "Benign FPR"),
        ],
    )

    comm_table = _html_table(
        [
            {
                "round": row["round"],
                "num_selected_clients": row["num_selected_clients"],
                "bytes_total": row["bytes_total"],
                "params_total": row["params_total"],
                "aggregation_method": row.get("aggregation_method", ""),
            }
            for row in comm_rows
        ],
        [
            ("round", "Rodada"),
            ("num_selected_clients", "Clientes"),
            ("bytes_total", "Bytes Totais"),
            ("params_total", "Parametros Totais"),
            ("aggregation_method", "Agregacao"),
        ],
    )

    temporal_table = ""
    if temporal_rows:
        temporal_table = _html_table(
            [
                {
                    "round": row["round"],
                    "k": row["k"],
                    "detection_coverage": _fmt_float(row["detection_coverage"]),
                    "benign_fpr": _fmt_float(row["benign_fpr"]),
                    "mean_ttd_seconds": _fmt_float(row["mean_ttd_seconds"]),
                }
                for row in temporal_rows
            ],
            [
                ("round", "Rodada"),
                ("k", "K"),
                ("detection_coverage", "Cobertura"),
                ("benign_fpr", "Benign FPR"),
                ("mean_ttd_seconds", "TTD Medio (s)"),
            ],
        )

    delta_block = ""
    if "delta_vs_baseline_k1" in payload:
        delta = payload["delta_vs_baseline_k1"]
        delta_block = (
            "<h2>Comparacao com Baseline</h2>"
            f"<p>Delta F1: <strong>{_fmt_float(delta['f1_score'])}</strong>; "
            f"Delta Precisao: <strong>{_fmt_float(delta['precision'])}</strong>; "
            f"Delta Recall: <strong>{_fmt_float(delta['recall'])}</strong>; "
            f"Delta Benign FPR: <strong>{_fmt_float(delta['benign_fpr'])}</strong>.</p>"
        )

    observations = "".join(f"<li>{html.escape(item)}</li>" for item in payload["observations"])

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>Analise Automatizada - {html.escape(payload['simulation_name'])}</title>
  <style>
    @page {{ size: A4; margin: 1.4cm; }}
    body {{ font-family: "Liberation Serif", "DejaVu Serif", serif; color: #111; line-height: 1.45; font-size: 11pt; }}
    h1, h2, h3 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 0.7em 0 1em 0; font-size: 10pt; }}
    th, td {{ border: 1px solid #cbd2d9; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #eaf2f8; text-align: left; }}
    code {{ background: #f0f4f8; padding: 0.12em 0.25em; border-radius: 3px; }}
    pre {{ background: #f8fafc; border: 1px solid #d9e2ec; padding: 10px; white-space: pre-wrap; }}
    .note {{ background: #fffbea; border-left: 4px solid #f0b429; padding: 10px 12px; margin: 0.8em 0; }}
  </style>
</head>
<body>
  <h1>Analise Automatizada do Experimento</h1>
  <p><strong>Experimento:</strong> <code>{html.escape(payload['simulation_name'])}</code></p>
  <p><strong>Diretorio de resultados:</strong> <code>{html.escape(payload['results_dir'])}</code></p>

  <h2>Melhor Resultado</h2>
  {best_table}

  <h2>Melhor Rodada por K</h2>
  {best_by_k_table}

  <h2>Splits Utilizados</h2>
  {_html_table(
      [
          {"split": "train", "total": payload["train_counts"]["total"], "benign": payload["train_counts"]["benign"], "anomaly": payload["train_counts"]["anomaly"]},
          {"split": "test", "total": payload["test_counts"]["total"], "benign": payload["test_counts"]["benign"], "anomaly": payload["test_counts"]["anomaly"]},
          {"split": "calibration", "total": payload["calibration_counts"]["total"], "benign": payload["calibration_counts"]["benign"], "anomaly": payload["calibration_counts"]["anomaly"]},
      ],
      [("split", "Split"), ("total", "Total"), ("benign", "Benignos"), ("anomaly", "Anomalos")],
  )}

  <h2>Observacoes</h2>
  <ul>{observations}</ul>

  {delta_block}

  <h2>Metricas Temporais</h2>
  {temporal_table or '<p>Nenhuma metrica temporal encontrada.</p>'}

  <h2>Comunicacao Federada</h2>
  {comm_table}

  <div class="note">
    <strong>Snapshot do ambiente.</strong><br />
    Host: <code>{html.escape(env['hostname'])}</code><br />
    GPU/driver: <code>{html.escape(env['nvidia'].splitlines()[0] if env['nvidia'] else 'N/A')}</code>
  </div>
</body>
</html>
"""


def build_analysis(simulation_name: str, compare_simulation: str | None, wait: bool, poll_seconds: int) -> dict[str, Path]:
    results_dir = ROOT / "results" / simulation_name
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    compare_dir = ROOT / "results" / compare_simulation if compare_simulation else None
    f1_rows, temporal_rows, comm_rows = _load_rows(results_dir, wait=wait, poll_seconds=poll_seconds)
    compare_best_k1 = _find_baseline_best(compare_dir)

    payload = _summary_payload(
        simulation_name=simulation_name,
        results_dir=results_dir,
        f1_rows=f1_rows,
        temporal_rows=temporal_rows,
        comm_rows=comm_rows,
        compare_best_k1=compare_best_k1,
    )

    slug = _slugify(simulation_name)
    json_path = DOCS_DIR / f"{slug}_analysis_summary.json"
    md_path = DOCS_DIR / f"{slug}_analysis_summary.md"
    html_path = DOCS_DIR / f"{slug}_analysis_summary.html"
    pdf_path = DOCS_DIR / f"{slug}_analysis_summary.pdf"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        _markdown_report(payload, f1_rows, temporal_rows, comm_rows), encoding="utf-8"
    )
    html_content = _html_report(payload, f1_rows, temporal_rows, comm_rows)
    html_path.write_text(html_content, encoding="utf-8")

    try:
        convert_html_to_pdf(html_path, pdf_path)
    except Exception:
        write_simple_pdf(html_to_text(html_content), pdf_path)

    return {
        "json": json_path,
        "markdown": md_path,
        "html": html_path,
        "pdf": pdf_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatically analyze a WiFi experiment and generate a report."
    )
    parser.add_argument("--simulation-name", required=True)
    parser.add_argument("--compare-simulation", default=None)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()

    outputs = build_analysis(
        simulation_name=args.simulation_name,
        compare_simulation=args.compare_simulation,
        wait=args.wait,
        poll_seconds=max(10, int(args.poll_seconds)),
    )

    print("Analysis generated successfully:")
    for key, path in outputs.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
