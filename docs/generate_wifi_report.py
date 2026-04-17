from __future__ import annotations

import csv
import html
import json
import math
import os
import subprocess
import textwrap
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
RESULTS_DIR = ROOT / "results" / "WiFi_CICIDS2018_Tuesday"
CONFIG_PATH = ROOT / "configs" / "config_wifi.yaml"
F1_PATH = RESULTS_DIR / "f1_scores.csv"
TEMPORAL_PATH = RESULTS_DIR / "temporal_metrics.csv"
COMM_PATH = RESULTS_DIR / "communication_metrics.csv"
CLIENT_META_PATH = RESULTS_DIR / "client_data" / "client_data_metadata.json"

HTML_PATH = DOCS_DIR / "WiFi_CICIDS2018_Technical_Report.html"
PDF_PATH = DOCS_DIR / "WiFi_CICIDS2018_Technical_Report.pdf"


TRAINING_OBSERVATIONS = [
    {
        "round": 1,
        "client_0_loss": 1.9170,
        "client_1_loss": 1.9170,
        "round_time_s": 25,
        "comment": "Parallel training initialized and both GPUs were engaged.",
    },
    {
        "round": 2,
        "client_0_loss": 0.8371,
        "client_1_loss": 0.8640,
        "round_time_s": 22,
        "comment": "Fast loss drop indicates that the model quickly learned the benign-flow distribution.",
    },
    {
        "round": 3,
        "client_0_loss": 0.6368,
        "client_1_loss": 0.6999,
        "round_time_s": 21,
        "comment": "Training entered a more stable region with diminishing gains.",
    },
    {
        "round": 4,
        "client_0_loss": 0.6569,
        "client_1_loss": 0.6474,
        "round_time_s": 21,
        "comment": "Small oscillation, still within the same stability band.",
    },
    {
        "round": 5,
        "client_0_loss": 0.5925,
        "client_1_loss": 0.6257,
        "round_time_s": 22,
        "comment": "Lowest observed training losses, but evaluation did not improve materially beyond round 2.",
    },
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run(cmd: list[str]) -> str:
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
        )
    except Exception as exc:
        return f"Command failed: {' '.join(cmd)}\n{exc}"
    out = completed.stdout.strip()
    err = completed.stderr.strip()
    if completed.returncode != 0 and err:
        return f"{out}\n{err}".strip()
    return out or err


def _fmt_float(value: str | float, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if math.isnan(number):
        return "NaN"
    return f"{number:.{digits}f}"


def _html_table(rows: Iterable[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    rows = list(rows)
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body_lines: list[str] = []
    for row in rows:
        cells = []
        for key, _label in columns:
            value = row.get(key, "")
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body_lines.append("<tr>" + "".join(cells) + "</tr>")
    body = "\n".join(body_lines)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _pre_block(text: str) -> str:
    return f"<pre>{html.escape(text.strip())}</pre>"


def _load_results() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, int]]:
    f1_rows = _read_csv(F1_PATH)
    temporal_rows = _read_csv(TEMPORAL_PATH)
    comm_rows = _read_csv(COMM_PATH)
    client_meta = json.loads(CLIENT_META_PATH.read_text(encoding="utf-8"))
    client_meta = {str(k): int(v) for k, v in client_meta.items()}
    return f1_rows, temporal_rows, comm_rows, client_meta


def _best_rows_by_k(f1_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, str]] = {}
    for row in f1_rows:
        k = row["k"]
        current = grouped.get(k)
        if current is None or float(row["f1_score"]) > float(current["f1_score"]):
            grouped[k] = row
    out: list[dict[str, object]] = []
    for k in sorted(grouped, key=lambda item: int(item)):
        row = grouped[k]
        out.append(
            {
                "k": row["k"],
                "round": row["round"],
                "f1": _fmt_float(row["f1_score"]),
                "precision": _fmt_float(row["precision"]),
                "recall": _fmt_float(row["recall"]),
                "benign_fpr": _fmt_float(row["benign_fpr"]),
                "threshold": _fmt_float(row["threshold"], 2),
            }
        )
    return out


def _environment_snapshot() -> dict[str, str]:
    return {
        "hostname": _run(["hostname"]),
        "uname": _run(["uname", "-a"]),
        "lscpu": _run(["lscpu"]),
        "free": _run(["free", "-h"]),
        "nvidia": _run(["nvidia-smi"]),
    }


def build_html() -> str:
    f1_rows, temporal_rows, comm_rows, client_meta = _load_results()
    env = _environment_snapshot()
    config_text = CONFIG_PATH.read_text(encoding="utf-8")

    best_by_k = _best_rows_by_k(f1_rows)
    best_overall = max(f1_rows, key=lambda row: float(row["f1_score"]))

    raw_size = _run(["du", "-sh", str(ROOT / "data" / "wifi" / "raw")])
    processed_size = _run(["du", "-sh", str(ROOT / "data" / "wifi" / "processed")])
    results_size = _run(["du", "-sh", str(RESULTS_DIR)])

    training_rows = [
        {
            "round": item["round"],
            "client_0_loss": f"{item['client_0_loss']:.4f}",
            "client_1_loss": f"{item['client_1_loss']:.4f}",
            "round_time_s": item["round_time_s"],
            "comment": item["comment"],
        }
        for item in TRAINING_OBSERVATIONS
    ]

    file_inventory = """FL_NETWORK_FLOW_LLM/
├── configs/config_wifi.yaml
├── data/wifi/raw/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv
├── data/wifi/processed/train.csv
├── data/wifi/processed/test.csv
├── data/wifi/processed/tokenized/
├── results/WiFi_CICIDS2018_Tuesday/
│   ├── client_data/
│   ├── communication_metrics.csv
│   ├── f1_scores.csv
│   ├── temporal_metrics.csv
│   └── round_0 ... round_5/
├── src/data_processing/wifi_processor.py
├── src/federated_learning/client.py
├── src/federated_learning/server.py
└── src/evaluation/evaluator.py"""

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>WiFi CIC-IDS2018 Technical Report</title>
  <style>
    @page {{
      size: A4;
      margin: 1.5cm;
    }}
    body {{
      font-family: "Liberation Serif", "DejaVu Serif", serif;
      color: #111;
      line-height: 1.45;
      font-size: 11pt;
    }}
    h1, h2, h3 {{
      color: #102a43;
      margin-top: 1.2em;
      margin-bottom: 0.4em;
    }}
    h1 {{
      font-size: 22pt;
      border-bottom: 2px solid #243b53;
      padding-bottom: 8px;
    }}
    h2 {{
      font-size: 16pt;
      border-bottom: 1px solid #bcccdc;
      padding-bottom: 4px;
    }}
    h3 {{
      font-size: 13pt;
    }}
    p {{
      margin: 0.45em 0;
      text-align: justify;
    }}
    ul {{
      margin: 0.35em 0 0.6em 1.2em;
    }}
    li {{
      margin: 0.2em 0;
    }}
    code {{
      background: #f0f4f8;
      padding: 0.12em 0.25em;
      border-radius: 3px;
      font-family: "Liberation Mono", monospace;
      font-size: 10pt;
    }}
    pre {{
      background: #f8fafc;
      border: 1px solid #d9e2ec;
      padding: 10px;
      white-space: pre-wrap;
      font-family: "Liberation Mono", monospace;
      font-size: 9.5pt;
      line-height: 1.35;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 0.6em 0 1em 0;
      font-size: 10pt;
    }}
    th, td {{
      border: 1px solid #cbd2d9;
      padding: 6px 8px;
      vertical-align: top;
    }}
    th {{
      background: #eaf2f8;
      text-align: left;
    }}
    .note {{
      background: #fffbea;
      border-left: 4px solid #f0b429;
      padding: 10px 12px;
      margin: 0.8em 0;
    }}
    .ok {{
      background: #f0fff4;
      border-left: 4px solid #38a169;
      padding: 10px 12px;
      margin: 0.8em 0;
    }}
    .page-break {{
      page-break-before: always;
    }}
  </style>
</head>
<body>
  <h1>Technical Report: CIC-IDS2018 Tuesday Wi-Fi / Network-Flow Pipeline in FL_NETWORK_FLOW_LLM</h1>
  <p><strong>Project root:</strong> <code>{html.escape(str(ROOT))}</code></p>
  <p><strong>Experiment:</strong> <code>WiFi_CICIDS2018_Tuesday</code></p>
  <p><strong>Objective of this report:</strong> document, in a deep and complete manner, the dataset onboarding, source-code modifications, performance optimizations, execution workflow, final configuration, generated artifacts, and the results obtained in the final experiment.</p>

  <h2>1. Executive Summary</h2>
  <p>This work adapted the <code>FL_NETWORK_FLOW_LLM</code> repository so that it could ingest and process a large public CIC-IDS2018 CSV directly from <code>data/wifi/raw/</code>, transform it into an anomaly-detection-as-language-modeling dataset, distribute benign training shards across federated clients, train a LoRA-adapted SmolLM model, and evaluate the resulting checkpoints on a balanced held-out set.</p>
  <p>The refactor replaced the previous toy Wi-Fi logic, which was tied to two miniature CSV files and synthetic fallback data, with a robust single-CSV processor capable of detecting labels, preserving metadata such as <code>Timestamp</code> and <code>Src IP</code>, capping the effective experiment size without touching the raw file, and preparing reproducible train/test/tokenized artifacts.</p>
  <p>Additional engineering changes were applied to improve speed and server friendliness: conservative multi-GPU parallel training, DataLoader worker controls, evaluation in batches rather than one sample at a time, and delayed construction of <code>Content</code> so that textual conversion is performed only for the subset of flows effectively used in the run.</p>
  <div class="ok">
    <strong>Final experimental footprint.</strong><br />
    Raw dataset on disk: {html.escape(raw_size)}<br />
    Processed dataset on disk: {html.escape(processed_size)}<br />
    Results directory on disk: {html.escape(results_size)}<br />
    Final split sizes: train = 250,000 benign flows; test = 250,000 flows (125,000 benign + 125,000 anomalous).
  </div>

  <h2>2. Repository and Artifact Inventory</h2>
  <p>The following file structure is the effective layout used by the final pipeline:</p>
  {_pre_block(file_inventory)}

  <h2>3. Dataset Acquisition and Storage Strategy</h2>
  <p>The selected dataset is <code>Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv</code>, from CIC-IDS2018 processed traffic data for machine-learning algorithms. The operational decision was to store the untouched dataset in <code>data/wifi/raw/</code> and to ensure that all derived artifacts are created under <code>data/wifi/processed/</code>. This preserves reproducibility, keeps the raw source immutable, and makes it possible to reprocess the experiment at any time from the same original file.</p>
  <p>The download command used for the server, because the <code>aws</code> CLI was unavailable, was:</p>
  {_pre_block('wget -O data/wifi/raw/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv "https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"')}
  <p>The raw file size observed on the server was approximately 3.8 GB. The first-line schema validation confirmed 84 columns and the presence of the fields required by the new processor, including <code>Label</code>, <code>Timestamp</code>, <code>Src IP</code>, <code>Dst IP</code>, <code>Protocol</code>, <code>Flow Byts/s</code>, and <code>Flow Duration</code>.</p>

  <h2>4. Original Wi-Fi Implementation Before the Refactor</h2>
  <p>Before the modifications described in this report, the Wi-Fi path in the repository was not prepared for the CIC-IDS2018 Tuesday CSV. The pre-refactor processor expected two small files named <code>benign_wifi.csv</code> and <code>anomaly_wifi.csv</code> in <code>data/wifi/raw/</code>, assumed a very small tabular schema composed of <code>rssi</code>, <code>snr</code>, <code>latency</code>, and <code>throughput</code>, and could fabricate fake rows if the files were missing or empty.</p>
  <p>That old behavior was useful only as a placeholder. It was not suitable for the real CIC-IDS2018 schema, could not read a single large CSV, did not preserve operational metadata needed by the new evaluator, and did not provide a mechanism to cap experiment size cleanly without editing the raw dataset itself.</p>

  <h2>5. Detailed Code Changes</h2>
  <h3>5.1 <code>configs/config_wifi.yaml</code></h3>
  <p>A dedicated configuration file for the Wi-Fi/CIC-IDS2018 experiment was created. It defines the dataset identity, raw CSV name, content columns to expose in the text representation, the benign-only training strategy, caps used for the current experiment, federated-learning hyperparameters, and the performance-related options introduced later in the process.</p>
  {_pre_block(config_text)}
  <p>The most relevant configuration decisions are:</p>
  <ul>
    <li><code>raw_csv_file</code> points explicitly to the CIC-IDS2018 Tuesday CSV so that legacy toy files do not interfere.</li>
    <li><code>benign_train_cap</code>, <code>test_benign_cap</code>, and <code>test_anomaly_cap</code> make it possible to run a smaller controlled experiment without touching the raw file.</li>
    <li><code>use_parallel_training</code> and <code>max_parallel_gpus</code> control conservative multi-GPU training.</li>
    <li><code>dataloader_num_workers</code> and related flags tune CPU usage without overloading a shared server.</li>
    <li><code>eval_batch_size</code>, <code>eval_use_autocast</code>, and <code>eval_torch_dtype</code> accelerate the evaluation phase.</li>
  </ul>

  <h3>5.2 <code>src/data_processing/wifi_processor.py</code></h3>
  <p>The Wi-Fi processor was completely redesigned. The new version can read a single large CSV, detect the configured label column, normalize labels into binary anomaly targets, select relevant flow features for textual rendering, preserve metadata such as <code>Flow ID</code>, <code>Src IP</code>, <code>Dst IP</code>, and <code>Timestamp</code>, and export the final train/test splits and tokenized dataset in the format expected by the rest of the framework.</p>
  <p>The main functional improvements are listed below:</p>
  <ul>
    <li><strong>Single-file loading.</strong> The processor resolves <code>raw_csv_file</code> from the config and supports the CIC-IDS2018 CSV directly.</li>
    <li><strong>Backward compatibility.</strong> The legacy pair <code>benign_wifi.csv</code> + <code>anomaly_wifi.csv</code> is still accepted as a fallback mode.</li>
    <li><strong>Robust label normalization.</strong> Text labels such as <code>Benign</code> and numeric <code>0/1</code> values are normalized into a clean binary column <code>Label</code>, while the raw original label is preserved in <code>Label_raw</code>.</li>
    <li><strong>Flexible feature selection.</strong> The processor uses whichever of the configured preferred columns actually exist in the raw schema.</li>
    <li><strong>Benign-only training strategy.</strong> Only benign flows are written to <code>train.csv</code>; test data is composed of benign holdout plus anomalous flows.</li>
    <li><strong>Caps without raw-data mutation.</strong> The current run uses 250k benign training flows and a balanced 250k test set via configuration only.</li>
    <li><strong>Metadata preservation.</strong> The final CSVs keep fields that later enable flow-level and temporal evaluation.</li>
    <li><strong>Text generation after capping.</strong> This optimization is crucial: <code>Content</code> is now created only for the rows that survive the configured caps, instead of being created for millions of discarded rows.</li>
    <li><strong>Sanitization.</strong> IP and MAC-like patterns are masked inside <code>Content</code> while preserving numeric flow metrics.</li>
    <li><strong>Tokenizer preparation.</strong> The processed benign training split is turned into a Hugging Face <code>Dataset</code>/<code>DatasetDict</code> structure and saved under <code>processed/tokenized/</code>.</li>
  </ul>

  <h3>5.3 <code>src/federated_learning/server.py</code></h3>
  <p>The server already contained a sequential and a parallel path, but the experiment-specific configuration was not enabling the parallel branch. The final configuration now activates <code>use_parallel_training</code>, and the server code was refined so that the number of GPUs used can be capped explicitly via <code>max_parallel_gpus</code>.</p>
  <p>This change is important on a shared server because it allows controlled use of hardware: the experiment can use two GPUs when appropriate, while still respecting a project-level limit and avoiding accidental uncontrolled fan-out.</p>

  <h3>5.4 <code>src/federated_learning/client.py</code></h3>
  <p>The client-side training code was augmented with conservative DataLoader controls. Specifically, the training arguments now accept <code>dataloader_num_workers</code>, <code>dataloader_pin_memory</code>, and <code>dataloader_persistent_workers</code> when the installed <code>transformers</code> version supports them. For the final server-friendly configuration, this was set to four workers per process.</p>
  <p>The intent was not to saturate all 32 CPU threads on the machine, but to reduce starvation of the GPUs and data-delivery stalls while keeping the experiment respectful of a shared academic environment.</p>

  <h3>5.5 <code>src/evaluation/evaluator.py</code></h3>
  <p>The evaluator received the most important speed optimization. In the original evaluation logic, the code tokenized and scored one text sample at a time in a serial loop. For a test set with 250,000 flows, this would still be very slow and underutilize the GPU. The new evaluator uses batched tokenization and batched inference, with optional autocast in half precision. This change substantially reduces the runtime of the evaluation phase and makes GPU usage much more meaningful.</p>
  <p>The evaluator now also clears model objects and CUDA cache between rounds more explicitly, which helps keep memory usage stable while iterating over all checkpoints.</p>

  <h2>6. End-to-End Pipeline Description</h2>
  <h3>6.1 Entry point and orchestration</h3>
  <p>The process is orchestrated by <code>main.py</code>. The runtime first loads the YAML configuration, applies Hugging Face cache/offline environment options, instantiates the dataset-specific processor based on <code>dataset_name</code>, runs preprocessing/tokenization, launches federated training, and then invokes the selected evaluator.</p>
  <p>For this experiment, the control flow is:</p>
  <ul>
    <li><code>main.py</code> loads <code>configs/config_wifi.yaml</code>.</li>
    <li><code>WiFiProcessor</code> creates <code>train.csv</code>, <code>test.csv</code>, and <code>tokenized/</code>.</li>
    <li><code>FederatedServer</code> initializes the global model and shards the benign tokenized training set.</li>
    <li>Two federated clients are trained across five rounds.</li>
    <li><code>NewEvaluator</code> evaluates checkpoints from rounds 1 through 5 on the balanced test set.</li>
  </ul>

  <h3>6.2 Train/test generation for the final experiment</h3>
  <p>The final experiment uses the following logic:</p>
  <ul>
    <li>All benign rows are shuffled and split into a training portion and a benign holdout portion.</li>
    <li>The benign training set is capped to 250,000 rows.</li>
    <li>The benign holdout is capped to 125,000 rows.</li>
    <li>The anomaly subset is capped to 125,000 rows.</li>
    <li>The final test set is the shuffled union of the benign holdout and anomaly cap.</li>
  </ul>
  <p>This preserves the anomaly-detection philosophy of the repository: learning is performed on normal behavior only, and the model is later evaluated on a mixture of benign and malicious flows.</p>

  <h3>6.3 Flow-to-text representation</h3>
  <p>Each selected network flow is converted into a textual sequence in the <code>Content</code> field. The generated tokens correspond to semantically meaningful flow features such as protocol, duration, packet counts, byte rates, inter-arrival times, packet length statistics, and flag counters. The goal is to let the language model learn regularities over a textualized representation of network behavior rather than over raw numeric tensors only.</p>
  <p>The chosen features for the final run were:</p>
  <ul>
    <li><code>Protocol</code></li>
    <li><code>Flow Duration</code></li>
    <li><code>Tot Fwd Pkts</code></li>
    <li><code>Tot Bwd Pkts</code></li>
    <li><code>TotLen Fwd Pkts</code></li>
    <li><code>TotLen Bwd Pkts</code></li>
    <li><code>Flow Byts/s</code></li>
    <li><code>Flow Pkts/s</code></li>
    <li><code>Flow IAT Mean</code></li>
    <li><code>Fwd IAT Mean</code></li>
    <li><code>Bwd IAT Mean</code></li>
    <li><code>Pkt Len Mean</code></li>
    <li><code>Pkt Len Var</code></li>
    <li><code>SYN Flag Cnt</code></li>
    <li><code>ACK Flag Cnt</code></li>
    <li><code>PSH Flag Cnt</code></li>
  </ul>

  <h3>6.4 Tokenization</h3>
  <p>The processed benign training subset is loaded from <code>train.csv</code>, converted to a Hugging Face dataset, tokenized with the SmolLM tokenizer, and stored in <code>data/wifi/processed/tokenized/</code>. Only benign training data is tokenized for the federated learning stage, because the approach treats anomaly detection as modeling normality.</p>

  <h3>6.5 Client sharding</h3>
  <p>The tokenized benign dataset is then split IID across two clients. The resulting metadata is:</p>
  {_html_table(
      [{"client_id": k, "samples": v} for k, v in sorted(client_meta.items(), key=lambda item: int(item[0]))],
      [("client_id", "Client"), ("samples", "Samples")]
  )}
  <p>Each client therefore received exactly 125,000 benign training examples.</p>

  <h2>7. Execution Environment and Server Constraints</h2>
  <p>The experiment ran on a shared academic server, so performance tuning had to balance speed and etiquette. The hardware snapshot captured during the analysis phase is reproduced below.</p>
  <h3>7.1 Host information</h3>
  {_pre_block(env["hostname"] + "\\n" + env["uname"])}
  <h3>7.2 CPU and memory</h3>
  {_pre_block(env["lscpu"])}
  {_pre_block(env["free"])}
  <h3>7.3 GPU inventory</h3>
  {_pre_block(env["nvidia"])}
  <p>The machine provides an Intel Core i9-13900K with 32 logical CPUs, 123 GiB of RAM, and two NVIDIA RTX 4090 GPUs with 24 GiB each. Monitoring also showed that, prior to the speed-oriented changes, the code underutilized both CPU and GPU resources, often keeping most CPU threads idle while only partially engaging a single GPU.</p>

  <h2>8. Performance Engineering Decisions</h2>
  <p>The tuning strategy followed a conservative philosophy appropriate for a multi-user server:</p>
  <ul>
    <li>Use at most two GPUs because the machine physically has two, and because the experiment already uses exactly two clients.</li>
    <li>Limit DataLoader workers to four per process rather than aggressively using all CPUs.</li>
    <li>Keep <code>batch_size = 2</code> for training to avoid unexpected VRAM pressure.</li>
    <li>Use <code>eval_batch_size = 4</code> with half precision to accelerate evaluation while keeping memory risk low on 24 GiB cards.</li>
    <li>Delay expensive text conversion until after caps are applied to the raw data.</li>
  </ul>
  <div class="note">
    <strong>Important operational observation.</strong><br />
    The parallel training mode improved hardware utilization but did not reduce total training time for this relatively small experiment. The additional process launch and model-loading overhead outweighed the gain in raw compute. However, batched evaluation produced a much more meaningful speedup because the original evaluator was strongly serial.
  </div>

  <h2>9. Commands Used to Reproduce the Experiment</h2>
  <p>The following commands summarize the final operational workflow used on the server:</p>
  {_pre_block(
      "cd ~/Artigo-LANC-Thiago/FL_NETWORK_FLOW_LLM\\n"
      "source venv/bin/activate\\n"
      "mkdir -p data/wifi/raw data/wifi/processed\\n"
      "wget -O data/wifi/raw/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv \\\"https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv\\\"\\n"
      "python main.py --config configs/config_wifi.yaml"
  )}

  <h2>10. Training Observations</h2>
  <p>The training stage ran for five rounds, using two clients and LoRA rank 8 on SmolLM-135M. The observed per-round training logs from the final run are summarized below.</p>
  {_html_table(
      training_rows,
      [
          ("round", "Round"),
          ("client_0_loss", "Client 0 Loss"),
          ("client_1_loss", "Client 1 Loss"),
          ("round_time_s", "Round Time (s)"),
          ("comment", "Observation"),
      ],
  )}
  <p>Loss dropped sharply between rounds 1 and 2 and then stabilized. This is a useful signal that the model learned the benign-flow language pattern quickly. Nevertheless, training loss alone is not enough; the downstream anomaly-detection metrics must also improve in a balanced way.</p>

  <h2>11. Communication Metrics</h2>
  <p>The federated server saved communication-cost information for every round:</p>
  {_html_table(
      [
          {
              "round": row["round"],
              "num_selected_clients": row["num_selected_clients"],
              "bytes_total": row["bytes_total"],
              "bytes_mean_per_client": row["bytes_mean_per_client"],
              "params_total": row["params_total"],
          }
          for row in comm_rows
      ],
      [
          ("round", "Round"),
          ("num_selected_clients", "Selected Clients"),
          ("bytes_total", "Bytes Total"),
          ("bytes_mean_per_client", "Bytes Mean / Client"),
          ("params_total", "Params Total"),
      ],
  )}
  <p>Every round transmitted the same update volume: 3,686,400 bytes in total and 921,600 LoRA parameters across the two clients. This stability is expected because the model architecture and LoRA rank remain constant throughout the run.</p>

  <h2>12. Evaluation Results</h2>
  <h3>12.1 Best checkpoints by top-k setting</h3>
  {_html_table(
      best_by_k,
      [
          ("k", "K"),
          ("round", "Best Round"),
          ("f1", "F1"),
          ("precision", "Precision"),
          ("recall", "Recall"),
          ("benign_fpr", "Benign FPR"),
          ("threshold", "Threshold"),
      ],
  )}
  <p>The best overall result in the final experiment was obtained for <strong>K = {html.escape(best_overall["k"])}</strong> at <strong>round {html.escape(best_overall["round"])}</strong>, with <strong>F1 = {_fmt_float(best_overall["f1_score"])}</strong>.</p>

  <h3>12.2 Full F1 table</h3>
  {_html_table(
      [
          {
              "round": row["round"],
              "k": row["k"],
              "threshold": _fmt_float(row["threshold"], 2),
              "f1_score": _fmt_float(row["f1_score"]),
              "precision": _fmt_float(row["precision"]),
              "recall": _fmt_float(row["recall"]),
              "benign_fpr": _fmt_float(row["benign_fpr"]),
          }
          for row in f1_rows
      ],
      [
          ("round", "Round"),
          ("k", "K"),
          ("threshold", "Threshold"),
          ("f1_score", "F1"),
          ("precision", "Precision"),
          ("recall", "Recall"),
          ("benign_fpr", "Benign FPR"),
      ],
  )}

  <h3>12.3 Temporal metrics</h3>
  {_html_table(
      [
          {
              "round": row["round"],
              "k": row["k"],
              "mean_ttd_seconds": _fmt_float(row["mean_ttd_seconds"]),
              "median_ttd_seconds": _fmt_float(row["median_ttd_seconds"]),
              "detection_coverage": _fmt_float(row["detection_coverage"]),
              "benign_fpr": _fmt_float(row["benign_fpr"]),
              "num_attacked_devices": row["num_attacked_devices"],
          }
          for row in temporal_rows
      ],
      [
          ("round", "Round"),
          ("k", "K"),
          ("mean_ttd_seconds", "Mean TTD (s)"),
          ("median_ttd_seconds", "Median TTD (s)"),
          ("detection_coverage", "Coverage"),
          ("benign_fpr", "Benign FPR"),
          ("num_attacked_devices", "Attacked Devices"),
      ],
  )}
  <p>The temporal table shows zero time-to-detection and full coverage for all rounds because the selected thresholds cause nearly immediate detection once anomalous flows appear. However, this apparently perfect temporal responsiveness is offset by an unacceptably high benign false-positive rate, so it should not be interpreted as operational readiness.</p>

  <h2>13. Interpretation of the Final Results</h2>
  <p>The final experiment demonstrates that the pipeline is technically functional and that the model does learn a separation signal. F1 values around 0.69 on a balanced test set confirm that the system is not random. The best checkpoint emerged early, around round 2, which means this setup converges quickly under the current training budget.</p>
  <p>However, the model’s operational profile is not yet satisfactory. The most striking issue is the extremely high benign false-positive rate. For example, the best overall K=1 result at round 2 achieved F1≈0.6949 and recall≈0.9907, but also a benign FPR≈0.8604. In plain terms, the system catches almost all anomalous traffic, but it misclassifies a very large fraction of normal traffic as anomalous.</p>
  <p>This means the current setup is useful as a proof-of-pipeline and as an experimental baseline, but not yet as a production-quality anomaly detector. The most likely next improvements would be threshold calibration on a benign-only calibration set, alternative feature engineering for <code>Content</code>, larger or more representative training subsets, and more operational threshold selection such as target-FPR tuning.</p>

  <h2>14. Files Added or Modified During This Work</h2>
  <ul>
    <li><code>configs/config_wifi.yaml</code>: created and iteratively refined to control dataset loading, caps, federated settings, and performance parameters.</li>
    <li><code>src/data_processing/wifi_processor.py</code>: replaced placeholder toy logic with a full CIC-IDS2018 pipeline.</li>
    <li><code>src/federated_learning/server.py</code>: refined parallel GPU selection through <code>max_parallel_gpus</code>.</li>
    <li><code>src/federated_learning/client.py</code>: added DataLoader worker configuration hooks to the training arguments.</li>
    <li><code>src/evaluation/evaluator.py</code>: changed evaluation from serial single-sample inference to batched inference with autocast and explicit CUDA cleanup.</li>
    <li><code>docs/generate_wifi_report.py</code>: added to generate this report reproducibly.</li>
  </ul>

  <h2>15. Practical Guidance for Future Runs</h2>
  <ul>
    <li>If the objective is quick iteration, keep the current caps.</li>
    <li>If the objective is higher fidelity, raise <code>benign_train_cap</code> and possibly <code>test_*_cap</code>, but be aware that preprocessing, tokenization, and evaluation time will grow.</li>
    <li>If VRAM becomes a concern, reduce <code>eval_batch_size</code> from 4 to 2.</li>
    <li>If the server becomes busier with other students, reduce <code>dataloader_num_workers</code> from 4 to 2.</li>
    <li>For scientific evaluation quality, add a benign calibration split and use operational thresholding rather than only oracle <code>f1_max</code>.</li>
  </ul>

  <h2>16. How to Download the PDF</h2>
  <p>The generated PDF is written to:</p>
  {_pre_block(str(PDF_PATH))}
  <p>It can be copied to another machine with a command such as:</p>
  {_pre_block(f"scp avancos:{PDF_PATH} .")}

  <h2>17. Closing Statement</h2>
  <p>This documentation captures the complete Wi-Fi/CIC-IDS2018 onboarding and experiment lifecycle performed in the repository: dataset download and organization, code refactor, performance tuning, controlled federated training, batch evaluation, and interpretation of the resulting metrics. The repository is now in a materially more reproducible and usable state for future experiments on large network-flow datasets.</p>
</body>
</html>
"""
    return html_doc


def convert_html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    outdir = str(pdf_path.parent)
    runtime_dir = Path("/tmp") / f"libreoffice-runtime-{os.getuid()}"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.chmod(0o700)
    env = os.environ.copy()
    env["XDG_RUNTIME_DIR"] = str(runtime_dir)
    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        outdir,
        str(html_path),
    ]
    completed = subprocess.run(
        cmd, check=False, text=True, capture_output=True, env=env
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"PDF conversion failed.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    generated_pdf = html_path.with_suffix(".pdf")
    if generated_pdf != pdf_path and generated_pdf.exists():
        generated_pdf.replace(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Expected PDF not found after conversion: {pdf_path}")


class HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.in_pre = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"h1", "h2", "h3", "p", "pre", "table"}:
            self.parts.append("\n\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "tr":
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")
        if tag == "pre":
            self.in_pre = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre":
            self.in_pre = False
            self.parts.append("\n")
        elif tag in {"p", "h1", "h2", "h3", "table"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.in_pre:
            self.parts.append(data)
        else:
            normalized = " ".join(data.split())
            if normalized:
                self.parts.append(normalized)

    def get_text(self) -> str:
        text = "".join(self.parts)
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        return text.strip() + "\n"


def html_to_text(html_content: str) -> str:
    parser = HTMLTextExtractor()
    parser.feed(html_content)
    return parser.get_text()


def _escape_pdf_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def write_simple_pdf(text: str, pdf_path: Path) -> None:
    page_width = 595
    page_height = 842
    margin = 42
    font_size = 10
    leading = 13
    usable_height = page_height - (2 * margin)
    lines_per_page = int(usable_height // leading)
    wrap_width = 88

    paragraphs = [block.rstrip() for block in text.split("\n")]
    wrapped_lines: list[str] = []
    for paragraph in paragraphs:
        if not paragraph.strip():
            wrapped_lines.append("")
            continue
        if paragraph.startswith(" | "):
            wrapped_lines.append(paragraph.strip())
            continue
        if paragraph.startswith("- "):
            bullet = paragraph[2:].strip()
            wrapped = textwrap.wrap(
                bullet,
                width=wrap_width - 2,
                subsequent_indent="  ",
                break_long_words=False,
                break_on_hyphens=False,
            )
            if wrapped:
                wrapped_lines.append("- " + wrapped[0])
                wrapped_lines.extend(wrapped[1:])
            else:
                wrapped_lines.append("-")
            continue
        wrapped = textwrap.wrap(
            paragraph,
            width=wrap_width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped_lines.extend(wrapped if wrapped else [""])

    pages: list[list[str]] = []
    current: list[str] = []
    for line in wrapped_lines:
        current.append(line)
        if len(current) >= lines_per_page:
            pages.append(current)
            current = []
    if current:
        pages.append(current)

    objects: list[bytes] = []

    def add_object(data: str | bytes) -> int:
        if isinstance(data, str):
            data = data.encode("latin-1", errors="replace")
        objects.append(data)
        return len(objects)

    font_obj = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    page_ids: list[int] = []
    content_ids: list[int] = []

    for page_lines in pages:
        y = page_height - margin
        text_ops = ["BT", f"/F1 {font_size} Tf", f"{leading} TL", f"{margin} {y} Td"]
        first_line = True
        for line in page_lines:
            safe = _escape_pdf_text(line)
            if first_line:
                text_ops.append(f"({safe}) Tj")
                first_line = False
            else:
                text_ops.append("T*")
                text_ops.append(f"({safe}) Tj")
        text_ops.append("ET")
        stream = "\n".join(text_ops).encode("latin-1", errors="replace")
        content_obj = add_object(
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
        content_ids.append(content_obj)
        page_ids.append(0)

    pages_kids_placeholder = " ".join(f"{idx} 0 R" for idx in page_ids)
    pages_obj = add_object(
        f"<< /Type /Pages /Kids [{pages_kids_placeholder}] /Count {len(page_ids)} >>"
    )

    for i, content_obj in enumerate(content_ids):
        page_obj = add_object(
            f"<< /Type /Page /Parent {pages_obj} 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {content_obj} 0 R >>"
        )
        page_ids[i] = page_obj

    objects[pages_obj - 1] = (
        f"<< /Type /Pages /Kids [{' '.join(f'{idx} 0 R' for idx in page_ids)}] "
        f"/Count {len(page_ids)} >>"
    ).encode("latin-1")

    catalog_obj = add_object(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>")

    pdf = bytearray()
    pdf.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_obj} 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )

    pdf_path.write_bytes(pdf)


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    html_doc = build_html()
    HTML_PATH.write_text(html_doc, encoding="utf-8")
    try:
        convert_html_to_pdf(HTML_PATH, PDF_PATH)
    except Exception:
        text_doc = html_to_text(html_doc)
        write_simple_pdf(text_doc, PDF_PATH)
    print(f"HTML report generated at: {HTML_PATH}")
    print(f"PDF report generated at: {PDF_PATH}")


if __name__ == "__main__":
    main()
