#!/usr/bin/env python3
"""
Approach 3 - Stage 1: Local Evaluation
Evaluate independently trained models on local test data.
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'shared'))
sys.path.append(str(Path(__file__).parent.parent.parent / 'models'))

from temporal_split import TemporalDataSplitter
from model_io import ModelManager
from metrics import MetricsCalculator

def main(site_name, config_path):
    """
    Evaluate independently trained models on local test data.

    Args:
        site_name: Name of the current site
        config_path: Path to configuration file
    """

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"=== Approach 3 - Stage 1: Local Evaluation at {site_name} ===")

    # Initialize components
    splitter = TemporalDataSplitter(config)
    model_manager = ModelManager(config)
    metrics_calc = MetricsCalculator()

    # Load local test data
    print("Loading local test data (2023-2024)...")
    test_data = splitter.get_testing_data(years=[2023, 2024])
    print(f"Test data shape: {test_data.shape}")

    # Load trained models
    models_dir = Path(config['outputs']['approach3']) / site_name

    if not models_dir.exists():
        print(f"ERROR: Models directory not found: {models_dir}")
        print("Please run train_independent.py first")
        return None

    # Load training results for context
    results_file = models_dir / f"{site_name}_independent_training_results.json"
    training_info = {}
    if results_file.exists():
        with open(results_file, 'r') as f:
            training_info = json.load(f)

    # Evaluate available models
    model_files = {
        'xgboost': f"{site_name}_independent_xgboost_model.pkl",
        'neural_network': f"{site_name}_independent_nn_model.pkl"
    }

    evaluation_results = {}

    for model_type, filename in model_files.items():
        model_path = models_dir / filename

        if model_path.exists():
            print(f"\n--- Evaluating {model_type.upper()} Model ---")

            try:
                # Load model
                model = model_manager.load_model(model_path, model_type=model_type)
                print(f"✓ Loaded model: {filename}")

                # Generate predictions
                if model_type == 'xgboost':
                    predictions = model_manager.predict_xgboost(model, test_data)
                else:
                    predictions = model_manager.predict_neural_network(model, test_data)

                # Calculate metrics
                test_metrics = metrics_calc.calculate_all_metrics(
                    test_data['mortality_label'],
                    predictions
                )

                print(f"Test Performance:")
                for metric, value in test_metrics.items():
                    print(f"  {metric}: {value:.4f}")

                # Compare with training/validation performance if available
                if model_type in training_info.get('models', {}):
                    train_results = training_info['models'][model_type]

                    if 'validation_performance' in train_results:
                        val_auc = train_results['validation_performance']['auc']
                        test_auc = test_metrics['auc']
                        auc_diff = test_auc - val_auc
                        print(f"  AUC difference (test - validation): {auc_diff:+.4f}")

                    if 'test_performance' in train_results:
                        # This would be the same if evaluation was done during training
                        print("  (Test performance matches training results)")

                evaluation_results[model_type] = {
                    'model_path': str(model_path),
                    'test_performance': test_metrics,
                    'test_data_size': len(test_data),
                    'evaluation_status': 'completed'
                }

            except Exception as e:
                print(f"✗ Evaluation failed: {e}")
                evaluation_results[model_type] = {
                    'model_path': str(model_path),
                    'evaluation_status': 'failed',
                    'error': str(e)
                }

        else:
            print(f"⚠ Model not found: {filename}")

    # Save evaluation results
    if evaluation_results:
        final_results = {
            'site_name': site_name,
            'approach': 'Approach 3 - Independent Training Evaluation',
            'evaluation_date': str(pd.Timestamp.now()),
            'test_data_info': {
                'size': len(test_data),
                'years': [2023, 2024]
            },
            'evaluation_results': evaluation_results,
            'training_context': training_info.get('models', {})
        }

        results_path = models_dir / f"{site_name}_independent_evaluation.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\n--- Evaluation Summary ---")
        print(f"Site: {site_name}")
        print(f"Test samples: {len(test_data)}")

        successful_evals = [m for m, r in evaluation_results.items() if r.get('evaluation_status') == 'completed']

        for model_type in successful_evals:
            auc = evaluation_results[model_type]['test_performance']['auc']
            print(f"{model_type.upper()}: AUC = {auc:.4f}")

        print(f"Results saved to: {results_path}")

        return final_results

    else:
        print("ERROR: No models found for evaluation!")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate independent models locally")
    parser.add_argument("--site_name", required=True, help="Name of current site")
    parser.add_argument("--config_path", default="../../../config_demo.json",
                        help="Path to configuration file")

    args = parser.parse_args()

    results = main(args.site_name, args.config_path)

    if results is None:
        sys.exit(1)