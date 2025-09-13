#!/usr/bin/env python3
"""
Approach 2 - Stage 1: Fine-tune Models Locally
Fine-tune RUSH base models using local training data (50% split).
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

def main(site_name, config_path, model_type='all'):
    """
    Fine-tune base models using local training data.

    Args:
        site_name: Name of the current site
        config_path: Path to configuration file
        model_type: Type of model to fine-tune ('xgboost', 'nn', or 'all')
    """

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"=== Approach 2 - Stage 1: Fine-tuning at {site_name} ===")

    # Initialize components
    splitter = TemporalDataSplitter(config)
    model_manager = ModelManager(config)
    metrics_calc = MetricsCalculator()

    # Load local data
    print("Loading local data...")
    train_data_full = splitter.get_training_data(years=[2018, 2019, 2020, 2021, 2022])
    test_data = splitter.get_testing_data(years=[2023, 2024])

    # Split training data 50-50 for fine-tuning
    print("Splitting training data (50% for fine-tuning, 50% reserved)...")
    np.random.seed(42)  # For reproducibility
    train_indices = np.random.permutation(len(train_data_full))
    split_point = len(train_indices) // 2

    finetune_indices = train_indices[:split_point]
    reserved_indices = train_indices[split_point:]

    finetune_data = train_data_full.iloc[finetune_indices].reset_index(drop=True)
    reserved_data = train_data_full.iloc[reserved_indices].reset_index(drop=True)

    print(f"Original training data: {len(train_data_full)} samples")
    print(f"Fine-tuning data: {len(finetune_data)} samples")
    print(f"Reserved data: {len(reserved_data)} samples")
    print(f"Test data: {len(test_data)} samples")

    # Verify minimum data requirements
    min_samples = config.get('transfer_learning', {}).get('min_samples', 500)
    if len(finetune_data) < min_samples:
        print(f"WARNING: Fine-tuning data ({len(finetune_data)}) below minimum ({min_samples})")
        print("Results may be suboptimal. Consider using more data or different approach.")

    # Define paths
    base_models_dir = Path(config['outputs']['approach2']) / site_name / "base_models"
    output_dir = Path(config['outputs']['approach2']) / site_name / "fine_tuned_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define models to process
    models_to_process = []
    if model_type in ['all', 'xgboost']:
        models_to_process.append('xgboost')
    if model_type in ['all', 'nn']:
        models_to_process.append('neural_network')

    fine_tuned_results = {}

    for model_name in models_to_process:
        print(f"\n--- Fine-tuning {model_name.upper()} ---")

        # Load base model
        if model_name == 'xgboost':
            base_model_path = base_models_dir / "RUSH_xgboost_model.pkl"
        else:
            base_model_path = base_models_dir / "RUSH_nn_model.pkl"

        if not base_model_path.exists():
            print(f"ERROR: Base model not found: {base_model_path}")
            print(f"Please run receive_base_model.py first")
            continue

        try:
            base_model = model_manager.load_model(base_model_path, model_type=model_name)
            print(f"✓ Loaded base model from {base_model_path}")

            # Evaluate base model on test data (baseline)
            print("Evaluating base model performance...")
            if model_name == 'xgboost':
                base_predictions = model_manager.predict_xgboost(base_model, test_data)
            else:
                base_predictions = model_manager.predict_neural_network(base_model, test_data)

            base_metrics = metrics_calc.calculate_all_metrics(
                test_data['mortality_label'],
                base_predictions
            )

            print(f"Base model performance:")
            for metric, value in base_metrics.items():
                print(f"  {metric}: {value:.4f}")

            # Fine-tune model
            print(f"Fine-tuning {model_name} with {len(finetune_data)} samples...")

            transfer_config = config.get('transfer_learning', {})

            if model_name == 'xgboost':
                # Fine-tune XGBoost
                fine_tuned_model = model_manager.fine_tune_xgboost(
                    base_model=base_model,
                    train_data=finetune_data,
                    config={
                        'num_boost_rounds': transfer_config.get('num_boost_rounds', 50),
                        'learning_rate': transfer_config.get('learning_rate_multiplier', 0.1),
                        'early_stopping_rounds': transfer_config.get('early_stopping_patience', 10)
                    }
                )
            else:
                # Fine-tune Neural Network
                fine_tuned_model = model_manager.fine_tune_neural_network(
                    base_model=base_model,
                    train_data=finetune_data,
                    config={
                        'learning_rate_multiplier': transfer_config.get('learning_rate_multiplier', 0.1),
                        'max_epochs': transfer_config.get('max_epochs', 50),
                        'patience': transfer_config.get('early_stopping_patience', 10)
                    }
                )

            print(f"✓ Fine-tuning completed")

            # Evaluate fine-tuned model
            print("Evaluating fine-tuned model...")
            if model_name == 'xgboost':
                finetuned_predictions = model_manager.predict_xgboost(fine_tuned_model, test_data)
            else:
                finetuned_predictions = model_manager.predict_neural_network(fine_tuned_model, test_data)

            finetuned_metrics = metrics_calc.calculate_all_metrics(
                test_data['mortality_label'],
                finetuned_predictions
            )

            print(f"Fine-tuned model performance:")
            for metric, value in finetuned_metrics.items():
                print(f"  {metric}: {value:.4f}")

            # Compare performance
            auc_improvement = finetuned_metrics['auc'] - base_metrics['auc']
            print(f"\nPerformance comparison:")
            print(f"  AUC improvement: {auc_improvement:+.4f} ({auc_improvement/base_metrics['auc']*100:+.1f}%)")

            # Save fine-tuned model
            if model_name == 'xgboost':
                model_save_path = output_dir / f"{site_name}_transfer_xgboost_model.pkl"
            else:
                model_save_path = output_dir / f"{site_name}_transfer_nn_model.pkl"

            model_manager.save_model(fine_tuned_model, model_save_path, model_type=model_name)
            print(f"✓ Saved fine-tuned model: {model_save_path}")

            # Store results
            fine_tuned_results[model_name] = {
                'base_model_path': str(base_model_path),
                'fine_tuned_model_path': str(model_save_path),
                'finetune_data_size': len(finetune_data),
                'test_data_size': len(test_data),
                'base_performance': base_metrics,
                'fine_tuned_performance': finetuned_metrics,
                'auc_improvement': auc_improvement,
                'relative_improvement': auc_improvement / base_metrics['auc']
            }

        except Exception as e:
            print(f"ERROR fine-tuning {model_name}: {e}")
            continue

    # Save comprehensive results
    if fine_tuned_results:
        results_summary = {
            'site_name': site_name,
            'approach': 'Approach 2 - Transfer Learning',
            'fine_tuning_date': str(pd.Timestamp.now()),
            'data_split': {
                'full_training_size': len(train_data_full),
                'finetune_size': len(finetune_data),
                'reserved_size': len(reserved_data),
                'test_size': len(test_data),
                'split_ratio': 0.5
            },
            'models': fine_tuned_results
        }

        results_path = output_dir / f"{site_name}_transfer_learning_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)

        print(f"\n--- Fine-tuning Summary ---")
        print(f"Site: {site_name}")
        print(f"Models fine-tuned: {len(fine_tuned_results)}")

        for model_name, results in fine_tuned_results.items():
            auc_improvement = results['auc_improvement']
            print(f"{model_name.upper()}: {auc_improvement:+.4f} AUC improvement ({results['relative_improvement']*100:+.1f}%)")

        print(f"Results saved to: {results_path}")
        print(f"Models saved to: {output_dir}")

        return results_summary

    else:
        print("ERROR: No models were successfully fine-tuned!")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune base models locally")
    parser.add_argument("--site_name", required=True, help="Name of current site")
    parser.add_argument("--model_type", choices=['xgboost', 'nn', 'all'], default='all',
                        help="Type of model to fine-tune")
    parser.add_argument("--config_path", default="../../../config_demo.json",
                        help="Path to configuration file")

    args = parser.parse_args()

    results = main(args.site_name, args.config_path, args.model_type)

    if results is None:
        sys.exit(1)