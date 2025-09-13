#!/usr/bin/env python3
"""
Approach 1 - Stage 1: Main Site Model Training
Train XGBoost and Neural Network models at RUSH site using 2018-2022 data.
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
    Train main site models (RUSH only).

    Args:
        site_name: Name of the site (should be "RUSH" for this approach)
        config_path: Path to configuration file
    """

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    if site_name != "RUSH":
        print(f"Warning: This script is intended for RUSH site only, not {site_name}")
        return

    print(f"=== Approach 1 - Stage 1: Training Models at {site_name} ===")

    # Initialize components
    splitter = TemporalDataSplitter(config)
    model_manager = ModelManager(config)
    metrics_calc = MetricsCalculator()

    # Load and split data temporally (2018-2022 for training, 2023-2024 for testing)
    print("Loading and splitting temporal data...")
    train_data = splitter.get_training_data(years=[2018, 2019, 2020, 2021, 2022])
    test_data = splitter.get_testing_data(years=[2023, 2024])

    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")

    # Train XGBoost model
    print("\n--- Training XGBoost Model ---")
    xgb_model = model_manager.train_xgboost(train_data, config['models']['xgboost'])

    # Evaluate XGBoost on local test data
    xgb_predictions = model_manager.predict_xgboost(xgb_model, test_data)
    xgb_metrics = metrics_calc.calculate_all_metrics(
        test_data['mortality_label'],
        xgb_predictions
    )

    print("XGBoost Local Performance:")
    for metric, value in xgb_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Train Neural Network model
    print("\n--- Training Neural Network Model ---")
    nn_model = model_manager.train_neural_network(train_data, config['models']['neural_network'])

    # Evaluate Neural Network on local test data
    nn_predictions = model_manager.predict_neural_network(nn_model, test_data)
    nn_metrics = metrics_calc.calculate_all_metrics(
        test_data['mortality_label'],
        nn_predictions
    )

    print("Neural Network Local Performance:")
    for metric, value in nn_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save models for distribution
    output_dir = Path(config['federated']['model_sharing_path']) / "approach1_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Saving Models to {output_dir} ---")

    # Save XGBoost model
    xgb_path = output_dir / f"{site_name}_xgboost_model.pkl"
    model_manager.save_model(xgb_model, xgb_path, model_type='xgboost')

    # Save Neural Network model
    nn_path = output_dir / f"{site_name}_nn_model.pkl"
    model_manager.save_model(nn_model, nn_path, model_type='neural_network')

    # Save performance metrics
    metrics_dict = {
        'site_name': site_name,
        'training_years': [2018, 2019, 2020, 2021, 2022],
        'testing_years': [2023, 2024],
        'xgboost_metrics': xgb_metrics,
        'neural_network_metrics': nn_metrics,
        'model_paths': {
            'xgboost': str(xgb_path),
            'neural_network': str(nn_path)
        }
    }

    metrics_path = output_dir / f"{site_name}_training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"Training complete! Models and metrics saved to {output_dir}")
    print(f"XGBoost AUC: {xgb_metrics['auc']:.4f}")
    print(f"Neural Network AUC: {nn_metrics['auc']:.4f}")

    return {
        'xgboost_model_path': str(xgb_path),
        'nn_model_path': str(nn_path),
        'metrics_path': str(metrics_path),
        'performance': {
            'xgboost_auc': xgb_metrics['auc'],
            'nn_auc': nn_metrics['auc']
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train main site models for Approach 1")
    parser.add_argument("--site_name", required=True, help="Site name (should be RUSH)")
    parser.add_argument("--config_path", default="../../../config_demo.json",
                        help="Path to configuration file")

    args = parser.parse_args()

    main(args.site_name, args.config_path)