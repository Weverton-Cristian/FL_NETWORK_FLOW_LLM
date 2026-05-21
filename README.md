# FL_NETWORK_FLOW_LLM

Federated LoRA tuning of small language models for network-flow detection.

This repository currently focuses on a WiFi / CIC-IDS2018 Tuesday experiment that converts network flows into text, trains a small language model in a federated setup, and evaluates anomaly detection performance with operational threshold calibration.

Historically, this project also includes older experiment tracks tied to the original FL-TFlow line of work. Those legacy configurations and results are still preserved for comparison.

## Current Status

The recommended experiment in the current version is:

- `configs/config_wifi.yaml`
- simulation name: `WiFi_CICIDS2018_Tuesday_balanced_seqcls_1152k_12r_200s`

This experiment uses:

- flow-to-text serialization
- balanced supervised binary classification
- federated training with LoRA
- `SmolLM-135M` as the backbone
- `FedAvg` aggregation
- operational threshold selection from a benign calibration split

It supersedes the previous `30`-round WiFi run as the main stable result for this methodology.

## What Changed In This Version

The repository now supports a new methodology for the WiFi/CIC-IDS2018 Tuesday setting:

- training task changed to `sequence_classification`
- split strategy changed to `balanced_binary`
- benign and anomaly classes are matched by the minority class
- the classifier predicts `benign` vs `anomaly` directly
- thresholding is based on anomaly probability and `fpr_target`
- evaluation was stabilized in `float32`
- the recommended run length was reduced from `30` to `12` rounds

This means the current main experiment is no longer the old benign-only, top-k-only detection setup. The flow-to-text representation remains, but the learning objective is now supervised binary classification.

## Main Methodology

### 1. Flow-to-Text Serialization

The raw network-flow CSV is converted into a textual representation using selected flow statistics, such as:

- `Protocol`
- `Flow Duration`
- `Tot Fwd Pkts`
- `Tot Bwd Pkts`
- `TotLen Fwd Pkts`
- `TotLen Bwd Pkts`
- `Flow Byts/s`
- `Flow Pkts/s`
- `Flow IAT Mean`
- `Fwd IAT Mean`
- `Bwd IAT Mean`
- `Pkt Len Mean`
- `Pkt Len Var`
- `SYN Flag Cnt`
- `ACK Flag Cnt`
- `PSH Flag Cnt`

The text field is stored as `Content` and becomes the input sequence for the tokenizer and model.

### 2. Balanced Supervised Split

The active WiFi configuration uses:

- `training_task: "sequence_classification"`
- `wifi_split_strategy: "balanced_binary"`
- `balanced_anchor: "minority"`
- `train_fraction_per_class: 0.8`

For the current CIC-IDS2018 Tuesday data, the processed experiment uses:

- `576,191` anomaly flows
- `576,191` matched benign flows
- `1,152,382` balanced total samples
- `921,904` train samples
- `230,478` test samples
- `35,000` additional benign calibration samples

### 3. Federated LoRA Training

The current WiFi run uses:

- `model_name: HuggingFaceTB/SmolLM-135M`
- `lora: true`
- `lora_rank: 8`
- `lora_alpha_multiplier: 2`
- `lora_dropout: 0.05`

Only the trainable LoRA parameters and classification head updates are exchanged between clients and server.

### 4. Operational Scoring

In the current methodology:

- the model outputs anomaly probabilities
- the score is the probability of the anomaly class
- the decision rule is `score >= threshold`
- the threshold is selected from benign calibration data using `fpr_target = 0.10`

### 5. Stability Fixes

The current version also incorporates an important evaluation fix:

- evaluation now uses `float32`
- evaluation autocast is disabled
- threshold selection includes protection against numerical collapse to zero

This was introduced because the older `30`-round WiFi run showed threshold collapse in late rounds under reduced-precision evaluation.

## Repository Layout

```text
FL_NETWORK_FLOW_LLM/
├── configs/
│   ├── config.yaml
│   ├── config_paper.yaml
│   ├── config_paper_basiline.yaml
│   └── config_wifi.yaml
├── data/
│   └── wifi/
│       ├── raw/
│       └── processed/
├── docs/
├── results/
│   ├── plot/
│   │   ├── generate_experiment_plots.ipynb
│   │   ├── throughput_detection_helpers.py
│   │   └── generated/
│   └── <simulation_name>/
├── scripts/
│   ├── fetch_cicids2018_tuesday.sh
│   └── generate_graph4_throughput_histogram.py
├── src/
│   ├── data_processing/
│   ├── evaluation/
│   ├── federated_learning/
│   ├── models/
│   └── utils/
├── main.py
└── requirements.txt
```

## Installation

### Requirements

- Python `3.10+`
- CUDA-compatible GPU recommended for training
- CPU-only execution is acceptable for evaluation and plots, though slower

### Setup

```bash
git clone https://github.com/Weverton-Cristian/FL_NETWORK_FLOW_LLM.git
cd FL_NETWORK_FLOW_LLM

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you already have the project locally and the `venv` exists, just activate it:

```bash
cd /path/to/FL_NETWORK_FLOW_LLM
source venv/bin/activate
```

## Data Layout

The current WiFi experiment expects the raw dataset here:

```text
data/wifi/raw/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv
```

The raw dataset is intentionally not versioned in this repository because the original CSV is several gigabytes in size and exceeds the practical limits of a standard GitHub repository. Reproducibility is preserved by documenting the exact upstream source, file name, placement, and execution steps needed to recreate the experiment locally.

### Dataset Reference

This repository uses the Tuesday processed-flow file from the CSE-CIC-IDS2018 dataset:

- dataset family: `CSE-CIC-IDS2018`
- expected file: `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv`
- local destination: `data/wifi/raw/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv`
- official AWS registry page: `https://registry.opendata.aws/cse-cic-ids2018/`
- direct file URL used by this repository:
  `https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv`

Recommended citation for the dataset:

- Sharafaldin, I., Habibi Lashkari, A., and Ghorbani, A. A. "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization." ICISSP 2018.

### Downloading the Dataset

Create the expected directory and download the CSV:

```bash
mkdir -p data/wifi/raw
wget -O data/wifi/raw/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv \
  "https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
```

If you prefer to use the helper script shipped with the repository:

```bash
bash scripts/fetch_cicids2018_tuesday.sh
```

### Verifying Local Placement

Before running the experiment, confirm that the file exists exactly where the WiFi configuration expects it:

```bash
ls -lh data/wifi/raw/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv
```

You can also verify that the current configuration points to the same file:

```bash
rg -n "raw_csv_file|dataset_name|data_base_path" configs/config_wifi.yaml
```

After preprocessing, the main files are:

- `data/wifi/processed/train.csv`
- `data/wifi/processed/test.csv`
- `data/wifi/processed/calibration.csv`
- `data/wifi/processed/tokenized/`

Additional dataset notes are documented in:

- `data/README.md`

## Recommended Experiment Configuration

The active WiFi configuration is in:
[configs/config_wifi.yaml](configs/config_wifi.yaml)

Key parameters:

| Parameter | Value |
|---|---:|
| `simulation_name` | `WiFi_CICIDS2018_Tuesday_balanced_seqcls_1152k_12r_200s` |
| `training_task` | `sequence_classification` |
| `wifi_split_strategy` | `balanced_binary` |
| `num_labels` | `2` |
| `anomaly_label_id` | `1` |
| `lora_rank` | `8` |
| `num_rounds` | `12` |
| `num_clients` | `50` |
| `client_frac` | `0.2` |
| `batch_size` | `2` |
| `max_steps` | `200` |
| `initial_lr` | `0.001` |
| `eval_use_autocast` | `false` |
| `eval_torch_dtype` | `float32` |
| `threshold_selection` | `fpr_target` |
| `fpr_target` | `0.10` |
| `hf_offline` | `true` |
| `hf_local_files_only` | `true` |

The checked-in configuration is optimized for controlled reruns on a machine that
already has the Hugging Face model cached. For a first run on a fresh clone, pass
temporary command-line overrides to allow downloading the model.

## Running the Current WiFi Experiment

### Standard Run

```bash
source venv/bin/activate
python main.py --config configs/config_wifi.yaml
```

### End-to-End Reproduction From Scratch

For a clean local reproduction of the current WiFi experiment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p data/wifi/raw
wget -O data/wifi/raw/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv \
  "https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
python main.py --config configs/config_wifi.yaml \
  --set hf_offline=false \
  --set hf_local_files_only=false
```

If the processed data already exists and you want to force regeneration from the
raw CSV, add:

```bash
--set force_reprocess_data=true
```

The `--set KEY=VALUE` option overrides top-level YAML fields without editing the
tracked configuration file.

### Recommended Offline Run

If you are running on a server with unstable internet, use offline mode:

```bash
source venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
python main.py --config configs/config_wifi.yaml | tee run_wifi_balanced_12r.log
```

### Recommended `tmux` Run

To keep the process alive if the SSH/VS Code connection drops:

```bash
tmux new -s wifi_12r_run
```

Then, inside `tmux`:

```bash
source venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
python main.py --config configs/config_wifi.yaml | tee run_wifi_balanced_12r.log
```

Detach without killing the process:

- `Ctrl+B`
- then `D`

Reattach later:

```bash
tmux attach -t wifi_12r_run
```

## Legacy Experiment Tracks

This repository still preserves older experiment configurations and results for comparison.

Examples:

- `configs/config_paper.yaml`
- `configs/config_paper_basiline.yaml`
- `results/WiFi_CICIDS2018_Tuesday_balanced_seqcls_1152k_30r_200s`
- `results/WiFi_CICIDS2018_Tuesday_700k_30r_200s_article`

These are useful for historical comparison, but they are not the recommended primary run for the current WiFi methodology.

## Generating Figures And Comparison Artifacts

Generated figures and result tables are intentionally written under `results/`,
which is ignored by Git because these artifacts can become large. The tracked
code needed to recreate them is kept under `scripts/`.

### Evaluate An Existing Experiment

If checkpoints already exist and you want to regenerate operational metrics
without retraining:

```bash
source venv/bin/activate
python scripts/evaluate_existing_experiment.py \
  --config configs/config_wifi.yaml \
  --simulation_name WiFi_CICIDS2018_Tuesday_balanced_seqcls_1152k_12r_200s \
  --threshold_selection fpr_target \
  --fpr_target 0.10
```

### Compare Against FL-LLM-AD

If a sibling `FL-LLM-AD` checkout exists next to this repository, the comparison
script can auto-discover it:

```bash
source venv/bin/activate
python scripts/compare_operational_results.py \
  --legacy_sim wifi_tuesday_1152k_fl_llm_ad_12r \
  --proposed_sim WiFi_CICIDS2018_Tuesday_balanced_seqcls_1152k_12r_200s \
  --threshold_mode fpr_target \
  --fpr_target 0.10 \
  --expected_rounds 12 \
  --require_same_rounds
```

If the legacy checkout is elsewhere, pass it explicitly:

```text
--legacy_root /path/to/FL-LLM-AD
```

Outputs are saved under:

```text
results/comparisons/<legacy_sim>__vs__<proposed_sim>/
```

### Throughput Histogram Only

```bash
source venv/bin/activate
python scripts/generate_graph4_throughput_histogram.py
```

Note:

- if cached throughput files do not exist yet, the script may still run model inference on the selected round
- this works without GPU, but may be slower on CPU

## Current Results Summary

### Main Stable WiFi Run: `12r`

Results from:

- `results/WiFi_CICIDS2018_Tuesday_balanced_seqcls_1152k_12r_200s/f1_scores_fpr_target.csv`
- `results/WiFi_CICIDS2018_Tuesday_balanced_seqcls_1152k_12r_200s/temporal_metrics_fpr_target.csv`
- `results/WiFi_CICIDS2018_Tuesday_balanced_seqcls_1152k_12r_200s/communication_metrics.csv`

Best checkpoint:

| Metric | Value |
|---|---:|
| Best round | `8` |
| F1 | `0.9528` |
| Precision | `0.9098` |
| Recall | `1.0000` |
| Benign FPR | `0.0991` |

Global run behavior:

- `12/12` rounds remained stable
- no round collapsed to `threshold = 0`
- final round remained operationally valid

### Previous WiFi Run: `30r`

The previous `30`-round WiFi run reached a strong peak, but became unstable in late rounds:

- best round: `5`
- best F1: `0.9523`
- late rounds collapsed due to threshold instability
- final round became invalid as a representative result

### Legacy Baseline

The older baseline run preserved in the repository achieved much lower recall and F1 than the current supervised WiFi methodology.

Best legacy baseline:

| Metric | Value |
|---|---:|
| F1 | `0.6273` |
| Precision | `0.8310` |
| Recall | `0.5038` |
| Benign FPR | `0.1024` |

### Practical Comparison

Compared with the legacy baseline, the current `12r` experiment improves:

- F1 by roughly `+0.326`
- precision by roughly `+0.078`
- recall by roughly `+0.496`

while keeping benign false-positive rate at approximately the same operational target.

## Additional Documentation

Detailed technical analysis of the current WiFi methodology and results:

- [docs/wifi_balanced_seqcls_12r_analysis.md](docs/wifi_balanced_seqcls_12r_analysis.md)

## Citation

If you use this repository, the WiFi/CIC-IDS2018 supervised federated
sequence-classification pipeline, or the comparison scripts in academic work,
please cite the corresponding LANC 2026 paper:

```bibtex
@inproceedings{cardoso2026lightweight,
  title={Lightweight Anomaly Detection in Enterprise Wi-Fi Networks Using LoRA-Adapted Small Language Models},
  author={Tiago Cardoso, Weverton Duarte, Allan Douglas Costa, Eduardo Cerqueira},
  booktitle={2026 Latin America Networking Conference (LANC 2026)},
  year={2026}
}
```

If you use or discuss the original FL-TFlow line of work preserved as historical
context in this repository, also cite the original publication when appropriate:

```bibtex
@inproceedings{cruz2025fltflow,
  title={FL-TFlow: Benign-Only Federated LoRA Tuning of SLMs for Edge Ransomware Detection},
  author={Wallace P. Cruz, Jose Pinto, Thiago Marques, Rafael Veiga, Hugo Santos, Lucas Bastos, Gabriel Talasso, Allan Costa, Denis Ros{\'a}rio, Eduardo Cerqueira},
  booktitle={SBRC 2025},
  year={2025}
}
```

When writing about the current WiFi experiment, make it explicit that this
repository version contains the newer balanced supervised sequence-classification
track beyond the original benign-only formulation.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- Federal University of Pará (UFPA)
- Federal University of South and Southeast of Pará (UNIFESPA)
- Federal Rural University of the Amazon (UFRA)
- State University of Pará (UEPA)
- University of Campinas (UNICAMP)
