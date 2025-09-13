#!/usr/bin/env python3
"""
Approach 3 - Stage 1: Independent Training
Train models from scratch using only local data at each site.
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
    Train independent models from scratch using only local data.

    Args:
        site_name: Name of the current site
        config_path: Path to configuration file
        model_type: Type of model to train ('xgboost', 'nn', or 'all')
    """

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"=== Approach 3 - Stage 1: Independent Training at {site_name} ===")

    # Initialize components
    splitter = TemporalDataSplitter(config)
    model_manager = ModelManager(config)
    metrics_calc = MetricsCalculator()

    # Load local data
    print("Loading local data...")
    train_data_full = splitter.get_training_data(years=[2018, 2019, 2020, 2021, 2022])
    test_data = splitter.get_testing_data(years=[2023, 2024])

    print(f"Training data shape: {train_data_full.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Check minimum data requirements
    min_samples = config.get('independent_training', {}).get('min_samples', 200)
    if len(train_data_full) < min_samples:
        print(f"ERROR: Training data ({len(train_data_full)}) below minimum ({min_samples})")
        print("Cannot proceed with independent training.")
        return None

    # Split training data for validation (80-20 split)
    np.random.seed(42)  # For reproducibility
    train_indices = np.random.permutation(len(train_data_full))
    split_point = int(len(train_indices) * 0.8)

    train_indices_subset = train_indices[:split_point]
    val_indices = train_indices[split_point:]

    train_data = train_data_full.iloc[train_indices_subset].reset_index(drop=True)
    val_data = train_data_full.iloc[val_indices].reset_index(drop=True)

    print(f"Training subset: {len(train_data)} samples")
    print(f"Validation: {len(val_data)} samples")

    # Define models to train
    models_to_train = []
    if model_type in ['all', 'xgboost']:
        models_to_train.append('xgboost')
    if model_type in ['all', 'nn']:
        models_to_train.append('neural_network')

    # Create output directory
    output_dir = Path(config['outputs']['approach3']) / site_name
    output_dir.mkdir(parents=True, exist_ok=True)

    training_results = {}

    for model_name in models_to_train:
        print(f"\n--- Training Independent {model_name.upper()} Model ---")

        try:
            # Get independent training config
            independent_config = config.get('independent_training', {})

            if model_name == 'xgboost':
                # Train XGBoost with hyperparameter optimization
                print("Starting hyperparameter optimization for XGBoost...")

                # Define hyperparameter search space
                param_space = {
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'n_estimators': [50, 100, 200]
                }

                best_model, best_params, optimization_history = model_manager.train_xgboost_with_optimization(
                    train_data=train_data,
                    val_data=val_data,
                    param_space=param_space,
                    cv_folds=independent_config.get('cross_validation_folds', 5),
                    early_stopping_rounds=independent_config.get('early_stopping_rounds', 10)
                )

                print(f"Best XGBoost parameters: {best_params}")

            else:  # neural_network
                # Train Neural Network with hyperparameter optimization
                print("Starting hyperparameter optimization for Neural Network...")

                # Define hyperparameter search space
                param_space = {
                    'hidden_layers': [[64], [64, 32], [128, 64, 32]],
                    'dropout': [0.2, 0.3, 0.5],
                    'learning_rate': [0.001, 0.01, 0.1],
                    'batch_size': [32, 64, 128]
                }

                best_model, best_params, optimization_history = model_manager.train_neural_network_with_optimization(
                    train_data=train_data,
                    val_data=val_data,
                    param_space=param_space,
                    max_epochs=independent_config.get('max_epochs', 100),
                    patience=independent_config.get('early_stopping_rounds', 10)
                )

                print(f"Best Neural Network parameters: {best_params}")

            # Evaluate on validation data
            print("Evaluating on validation data...")
            if model_name == 'xgboost':
                val_predictions = model_manager.predict_xgboost(best_model, val_data)
            else:
                val_predictions = model_manager.predict_neural_network(best_model, val_data)

            val_metrics = metrics_calc.calculate_all_metrics(
                val_data['mortality_label'],
                val_predictions
            )

            print("Validation Performance:")
            for metric, value in val_metrics.items():
                print(f"  {metric}: {value:.4f}")

            # Evaluate on test data
            print("Evaluating on test data...")
            if model_name == 'xgboost':
                test_predictions = model_manager.predict_xgboost(best_model, test_data)
            else:
                test_predictions = model_manager.predict_neural_network(best_model, test_data)

            test_metrics = metrics_calc.calculate_all_metrics(
                test_data['mortality_label'],
                test_predictions
            )

            print("Test Performance:")
            for metric, value in test_metrics.items():
                print(f"  {metric}: {value:.4f}")

            # Analyze feature importance (for XGBoost)
            feature_importance = None
            if model_name == 'xgboost' and hasattr(best_model, 'feature_importances_'):
                feature_names = list(train_data.columns[:-1])  # Exclude target column
                feature_importance = dict(zip(feature_names, best_model.feature_importances_))

                # Sort by importance
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

                print(f"\nTop 10 Most Important Features ({model_name}):")
                for i, (feature, importance) in enumerate(sorted_features[:10]):
                    print(f"  {i+1:2d}. {feature}: {importance:.4f}")

            # Save model
            if model_name == 'xgboost':
                model_path = output_dir / f"{site_name}_independent_xgboost_model.pkl"
            else:
                model_path = output_dir / f"{site_name}_independent_nn_model.pkl"

            model_manager.save_model(best_model, model_path, model_type=model_name)
            print(f"✓ Saved model: {model_path}")

            # Store training results
            training_results[model_name] = {
                'model_path': str(model_path),
                'best_params': best_params,
                'optimization_history': optimization_history,
                'data_sizes': {
                    'training': len(train_data),
                    'validation': len(val_data),
                    'test': len(test_data)
                },
                'validation_performance': val_metrics,
                'test_performance': test_metrics,
                'feature_importance': feature_importance,
                'training_status': 'completed'
            }

            print(f"✓ {model_name.upper()} training completed successfully")

        except Exception as e:
            print(f"✗ {model_name.upper()} training failed: {e}")
            training_results[model_name] = {
                'training_status': 'failed',
                'error': str(e)
            }

    # Save comprehensive training results
    if training_results:
        results_summary = {
            'site_name': site_name,
            'approach': 'Approach 3 - Independent Training',
            'training_date': str(pd.Timestamp.now()),
            'data_split_info': {
                'full_training_size': len(train_data_full),
                'training_size': len(train_data),
                'validation_size': len(val_data),
                'test_size': len(test_data),
                'train_val_split_ratio': 0.8
            },
            'training_config': config.get('independent_training', {}),
            'models': training_results
        }

        results_path = output_dir / f"{site_name}_independent_training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)

        print(f"\n--- Independent Training Summary ---")
        print(f"Site: {site_name}")

        successful_models = [m for m, r in training_results.items() if r.get('training_status') == 'completed']
        failed_models = [m for m, r in training_results.items() if r.get('training_status') == 'failed']

        print(f"Successfully trained: {len(successful_models)} models {successful_models}")
        if failed_models:
            print(f"Failed training: {len(failed_models)} models {failed_models}")

        for model_name, results in training_results.items():
            if results.get('training_status') == 'completed':
                test_auc = results['test_performance']['auc']
                print(f"{model_name.upper()}: Test AUC = {test_auc:.4f}")

        print(f"Results saved to: {results_path}")

        return results_summary

    else:
        print("ERROR: No models were successfully trained!")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train independent models from scratch")
    parser.add_argument("--site_name", required=True, help="Name of current site")
    parser.add_argument("--model_type", choices=['xgboost', 'nn', 'all'], default='all',
                        help="Type of model to train")
    parser.add_argument("--config_path", default="../../../config_demo.json",
                        help="Path to configuration file")

    args = parser.parse_args()

    results = main(args.site_name, args.config_path, args.model_type)

    if results is None:
        sys.exit(1)