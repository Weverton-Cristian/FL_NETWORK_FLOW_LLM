# Reproducibility Checklist

This file documents the minimal path to reproduce the current main WiFi
experiment from a fresh clone.

## 1. Environment

```bash
git clone https://github.com/Weverton-Cristian/FL_NETWORK_FLOW_LLM.git
cd FL_NETWORK_FLOW_LLM
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Training is GPU-oriented. CPU-only runs are practical for evaluation and plotting
but can be slow for full training.

## 2. Dataset

```bash
bash scripts/fetch_cicids2018_tuesday.sh
```

Equivalent AWS Open Data command:

```bash
mkdir -p data/wifi/raw
aws s3 cp --no-sign-request --region ca-central-1 \
  "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv" \
  data/wifi/raw/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv
```

Expected raw file:

```text
data/wifi/raw/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv
```

The raw CSV, processed splits, tokenized data, checkpoints, logs, and generated
plots are intentionally not committed to Git.

## 3. First Run On A Fresh Machine

The tracked WiFi config uses offline Hugging Face mode to make controlled reruns
deterministic after the model is cached. On a fresh machine, allow the first
model download with temporary CLI overrides:

```bash
python main.py --config configs/config_wifi.yaml \
  --set hf_offline=false \
  --set hf_local_files_only=false
```

To force rebuilding the processed dataset from the raw CSV:

```bash
python main.py --config configs/config_wifi.yaml \
  --set hf_offline=false \
  --set hf_local_files_only=false \
  --set force_reprocess_data=true
```

## 4. Controlled Offline Rerun

After the model is cached locally:

```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
python main.py --config configs/config_wifi.yaml
```

## 5. Evaluation Without Retraining

If checkpoints already exist under `results/<simulation_name>/`:

```bash
python scripts/evaluate_existing_experiment.py \
  --config configs/config_wifi.yaml \
  --simulation_name WiFi_CICIDS2018_Tuesday_balanced_seqcls_1152k_12r_200s \
  --threshold_selection fpr_target \
  --fpr_target 0.10
```

## 6. Comparison Against FL-LLM-AD

Place the legacy repository as a sibling checkout when possible:

```text
parent/
├── FL_NETWORK_FLOW_LLM/
└── FL-LLM-AD/
```

Then run:

```bash
python scripts/compare_operational_results.py \
  --legacy_sim wifi_tuesday_1152k_fl_llm_ad_12r \
  --proposed_sim WiFi_CICIDS2018_Tuesday_balanced_seqcls_1152k_12r_200s \
  --threshold_mode fpr_target \
  --fpr_target 0.10 \
  --expected_rounds 12 \
  --require_same_rounds
```

If `FL-LLM-AD` is elsewhere, add:

```text
--legacy_root /path/to/FL-LLM-AD
```

Generated comparison files are written under `results/comparisons/`.
