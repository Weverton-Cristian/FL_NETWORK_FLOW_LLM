import yaml
import argparse
import os
import copy
import random
import numpy as np
import torch
from src.data_processing.wifi_processor import WiFiProcessor
from src.data_processing.ransomlog_processor import RansomLogProcessor
from src.data_processing.hdfs_processor import HDFSProcessor
from src.data_processing.edge_ransomware_processor import EdgeRansomwareProcessor
from src.federated_learning.server import FederatedServer
from src.evaluation.evaluator_antigo import Evaluator as OldEvaluator
from src.evaluation.evaluator import Evaluator as NewEvaluator
from src.utils.hf import apply_hf_environment

def _merge_overrides(base: dict, overrides: dict) -> dict:
    """
    Shallow-merge overrides into a copy of base config.
    Intended for evaluation-only overrides (thresholding, temporal settings, etc.).
    """
    merged = copy.deepcopy(base)
    for k, v in (overrides or {}).items():
        merged[k] = v
    return merged


def _set_global_seed(config: dict) -> None:
    """
    Sets the main-process random seed for better run-to-run reproducibility.

    This does not make every CUDA kernel fully deterministic, but it does fix the
    project-level stochastic sources we control directly: Python, NumPy and torch.
    """
    seed = int(config.get("random_seed", 42))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Global random seed set to {seed}.")

def main(config_path, config_overrides=None):
    """
    Main function to orchestrate the federated learning pipeline.
    """
    # 1. Load Configuration
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    config = _merge_overrides(config, config_overrides)

    print("Configuration loaded successfully.")
    print(f"Starting simulation: {config.get('simulation_name', 'N/A')}")

    # Configure Hugging Face cache/offline behavior before any tokenizer/model loads.
    apply_hf_environment(config)
    _set_global_seed(config)

    # 2. Execute Data Processing Pipeline
    # This factory pattern dynamically selects the correct processor based on the config.
    print("\n--- Initializing Data Processing ---")
    if config['dataset_name'] == 'ransomlog':
        processor = RansomLogProcessor(config)
    elif config['dataset_name'] in ('edge_ransomware', 'edge_ransomware_new'):
        processor = EdgeRansomwareProcessor(config)
    elif config['dataset_name'] == 'hdfs':
        processor = HDFSProcessor(config)
    elif config['dataset_name'] == 'wifi':
        processor = WiFiProcessor(config)
    else:
        raise ValueError(f"Dataset '{config['dataset_name']}' not supported in the current implementation.")
    
    processor.run()

    # 3. Execute Federated Training
    print("\n--- Starting Federated Training ---")
    server = FederatedServer(config)
    server.run_federated_training()
    print("--- Federated Training Complete ---")

    # 4. Execute Evaluation
    evaluator_mode = config.get("evaluator_version", "old")

    if evaluator_mode in ["old", "both"]:
        print("\n--- Starting Evaluation (Antigo) ---")
        old_evaluator = OldEvaluator(config)
        old_evaluator.evaluate()
        print("--- Evaluation (Antigo) Complete ---")

    if evaluator_mode in ["new", "both"]:
        print("\n--- Starting Evaluation (Novo) ---")
        new_evaluator = NewEvaluator(config)
        new_evaluator.evaluate()
        print("--- Evaluation (Novo) Complete ---")

    # 5. Optional additional evaluation runs (e.g., operational fpr_target + temporal metrics)
    evaluation_runs = config.get("evaluation_runs", [])
    if evaluation_runs:
        print("\n--- Starting Additional Evaluations (Overrides) ---")
        for i, run_cfg in enumerate(evaluation_runs, start=1):
            name = run_cfg.get("name", f"eval_{i}")
            overrides = run_cfg.get("overrides", {})
            eval_config = _merge_overrides(config, overrides)
            print(f"\n--- Evaluation Override {i}/{len(evaluation_runs)}: {name} ---")
            evaluator = NewEvaluator(eval_config)
            evaluator.evaluate()
        print("--- Additional Evaluations Complete ---")

import multiprocessing as mp

def _parse_cli_overrides(raw_overrides: list[str] | None) -> dict:
    """
    Parses flat KEY=VALUE overrides using YAML semantics for values.

    Example:
      --set hf_offline=false --set calibration_num_samples=1000
    """
    parsed = {}
    for raw in raw_overrides or []:
        if "=" not in raw:
            raise ValueError(f"Invalid override '{raw}'. Expected KEY=VALUE.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{raw}'. Empty key.")
        parsed[key] = yaml.safe_load(value)
    return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Federated Learning for Anomaly Detection.")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to the YAML configuration file.')
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a top-level YAML value. Can be used multiple times.",
    )
    args = parser.parse_args()

    # Define o método de início de multiprocessamento como 'spawn'.
    # Isso é crucial para evitar erros de inicialização da CUDA em processos filhos.
    try:
        mp.set_start_method('spawn', force=True)
        print("Método de início de multiprocessamento configurado para 'spawn'.")
    except RuntimeError:
        # Pode já ter sido definido, o que não é um problema.
        pass
    
    # Ensure the script's working directory is the project root
    # This makes path handling in config.yaml consistent
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    cli_overrides = _parse_cli_overrides(args.overrides)
    main(args.config, cli_overrides)
