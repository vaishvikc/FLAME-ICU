#!/usr/bin/env python3
"""
Approach 1 - Stage 1: Local Evaluation
Each site evaluates RUSH models on their local 2023-2024 test data.
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'shared'))
sys.path.append(str(Path(__file__).parent.parent.parent / 'models'))

from temporal_split import TemporalDataSplitter
from model_io import ModelManager
from metrics import MetricsCalculator

def main(site_name, config_path):
    """
    Evaluate RUSH models on local site data.

    Args:
        site_name: Name of the current site
        config_path: Path to configuration file
    """

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"=== Approach 1 - Stage 1: Local Evaluation at {site_name} ===")

    # Initialize components
    splitter = TemporalDataSplitter(config)
    model_manager = ModelManager(config)
    metrics_calc = MetricsCalculator()

    # Load local test data (2023-2024)
    print("Loading local test data (2023-2024)...")
    test_data = splitter.get_testing_data(years=[2023, 2024])
    print(f"Test data shape: {test_data.shape}")

    # Load RUSH models
    model_dir = Path(config['federated']['model_sharing_path']) / "distributed_models" / site_name / "approach1_models"

    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        print("Please ensure models have been distributed to this site.")
        return None

    print(f"Loading RUSH models from: {model_dir}")

    # Load XGBoost model
    xgb_model_path = model_dir / "RUSH_xgboost_model.pkl"
    if not xgb_model_path.exists():
        print(f"ERROR: XGBoost model not found: {xgb_model_path}")
        return None

    xgb_model = model_manager.load_model(xgb_model_path, model_type='xgboost')

    # Load Neural Network model
    nn_model_path = model_dir / "RUSH_nn_model.pkl"
    if not nn_model_path.exists():
        print(f"ERROR: Neural Network model not found: {nn_model_path}")
        return None

    nn_model = model_manager.load_model(nn_model_path, model_type='neural_network')

    # Load RUSH training metrics for comparison
    metrics_path = model_dir / "RUSH_training_metrics.json"
    rush_metrics = {}
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            rush_metrics = json.load(f)

    # Evaluate XGBoost model on local test data
    print("\n--- Evaluating XGBoost Model ---")
    xgb_predictions = model_manager.predict_xgboost(xgb_model, test_data)
    xgb_local_metrics = metrics_calc.calculate_all_metrics(
        test_data['mortality_label'],
        xgb_predictions
    )

    print(f"XGBoost Performance at {site_name}:")
    for metric, value in xgb_local_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Compare with RUSH performance if available
    if 'xgboost_metrics' in rush_metrics:
        rush_auc = rush_metrics['xgboost_metrics']['auc']
        local_auc = xgb_local_metrics['auc']
        auc_diff = local_auc - rush_auc
        print(f"  AUC difference vs RUSH: {auc_diff:+.4f} ({auc_diff/rush_auc*100:+.1f}%)")

    # Evaluate Neural Network model on local test data
    print("\n--- Evaluating Neural Network Model ---")
    nn_predictions = model_manager.predict_neural_network(nn_model, test_data)
    nn_local_metrics = metrics_calc.calculate_all_metrics(
        test_data['mortality_label'],
        nn_predictions
    )

    print(f"Neural Network Performance at {site_name}:")
    for metric, value in nn_local_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Compare with RUSH performance if available
    if 'neural_network_metrics' in rush_metrics:
        rush_auc = rush_metrics['neural_network_metrics']['auc']
        local_auc = nn_local_metrics['auc']
        auc_diff = local_auc - rush_auc
        print(f"  AUC difference vs RUSH: {auc_diff:+.4f} ({auc_diff/rush_auc*100:+.1f}%)")

    # Save evaluation results
    output_dir = Path(config['outputs']['approach1'])
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_results = {
        'site_name': site_name,
        'evaluation_date': str(Path(__file__).stat().st_mtime),
        'test_data_years': [2023, 2024],
        'test_data_size': len(test_data),
        'xgboost_performance': xgb_local_metrics,
        'neural_network_performance': nn_local_metrics,
        'rush_comparison': {
            'xgboost_auc_diff': (xgb_local_metrics['auc'] - rush_metrics.get('xgboost_metrics', {}).get('auc', 0))
                if 'xgboost_metrics' in rush_metrics else None,
            'nn_auc_diff': (nn_local_metrics['auc'] - rush_metrics.get('neural_network_metrics', {}).get('auc', 0))
                if 'neural_network_metrics' in rush_metrics else None
        }
    }

    results_path = output_dir / f"{site_name}_approach1_evaluation.json"
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"\n--- Evaluation Complete ---")
    print(f"Results saved to: {results_path}")
    print(f"XGBoost AUC: {xgb_local_metrics['auc']:.4f}")
    print(f"Neural Network AUC: {nn_local_metrics['auc']:.4f}")

    # Summary for coordination
    print(f"\n--- Summary for {site_name} ---")
    print(f"Site: {site_name}")
    print(f"Test samples: {len(test_data)}")
    print(f"XGBoost AUC: {xgb_local_metrics['auc']:.4f}")
    print(f"Neural Network AUC: {nn_local_metrics['auc']:.4f}")

    return evaluation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RUSH models locally for Approach 1")
    parser.add_argument("--site_name", required=True, help="Name of current site")
    parser.add_argument("--config_path", default="../../../config_demo.json",
                        help="Path to configuration file")

    args = parser.parse_args()

    results = main(args.site_name, args.config_path)

    if results is None:
        sys.exit(1)