#!/usr/bin/env python3
"""
Approach 2 - Stage 2: Ensemble Testing
Create and test ensemble models using transfer learning models.
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
from ensemble_utils import EnsembleConstructor

def main(current_site, config_path):
    """
    Create and test ensemble models using transfer learning models (leave-one-out).

    Args:
        current_site: Name of the current site (where testing is performed)
        config_path: Path to configuration file
    """

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"=== Approach 2 - Stage 2: Ensemble Testing at {current_site} ===")

    # Initialize components
    splitter = TemporalDataSplitter(config)
    model_manager = ModelManager(config)
    metrics_calc = MetricsCalculator()
    ensemble_constructor = EnsembleConstructor()

    # Load local test data
    print(f"Loading {current_site} test data (2023-2024)...")
    test_data = splitter.get_testing_data(years=[2023, 2024])
    print(f"Test data shape: {test_data.shape}")

    # Find all available transfer learning models (excluding current site - leave-one-out)
    shared_models_dir = Path(config['federated']['model_sharing_path']) / "approach2_models"

    if not shared_models_dir.exists():
        print(f"ERROR: Shared models directory not found: {shared_models_dir}")
        return None

    # Get models from other sites (leave-one-out approach)
    available_models = {
        'xgboost': [],
        'neural_network': []
    }

    site_local_performance = {}  # For accuracy-weighted ensemble

    for site_dir in shared_models_dir.iterdir():
        if site_dir.is_dir() and site_dir.name != current_site:  # Exclude current site (leave-one-out)

            # Check for XGBoost model
            xgb_model_path = site_dir / f"{site_dir.name}_transfer_xgboost_model.pkl"
            if xgb_model_path.exists():
                try:
                    model = model_manager.load_model(xgb_model_path, model_type='xgboost')
                    available_models['xgboost'].append({
                        'site': site_dir.name,
                        'model': model,
                        'path': xgb_model_path
                    })
                except Exception as e:
                    print(f"Warning: Could not load XGBoost model from {site_dir.name}: {e}")

            # Check for Neural Network model
            nn_model_path = site_dir / f"{site_dir.name}_transfer_nn_model.pkl"
            if nn_model_path.exists():
                try:
                    model = model_manager.load_model(nn_model_path, model_type='neural_network')
                    available_models['neural_network'].append({
                        'site': site_dir.name,
                        'model': model,
                        'path': nn_model_path
                    })
                except Exception as e:
                    print(f"Warning: Could not load Neural Network model from {site_dir.name}: {e}")

            # Load local performance for accuracy weighting
            results_file = site_dir / f"{site_dir.name}_transfer_learning_results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        site_results = json.load(f)
                        local_perf = {}
                        for model_type, results in site_results.get('models', {}).items():
                            local_perf[model_type] = results.get('fine_tuned_performance', {}).get('auc', 0.5)
                        site_local_performance[site_dir.name] = local_perf
                except Exception as e:
                    print(f"Warning: Could not load performance for {site_dir.name}: {e}")

    print(f"Available models for ensemble:")
    print(f"  XGBoost: {len(available_models['xgboost'])} models from sites {[m['site'] for m in available_models['xgboost']]}")
    print(f"  Neural Network: {len(available_models['neural_network'])} models from sites {[m['site'] for m in available_models['neural_network']]}")

    if len(available_models['xgboost']) < 2 and len(available_models['neural_network']) < 2:
        print("WARNING: Need at least 2 models per type for meaningful ensemble")

    # Test ensembles for each model type
    ensemble_results = {}

    for model_type in ['xgboost', 'neural_network']:
        if len(available_models[model_type]) < 2:
            print(f"\nSkipping {model_type} ensemble: only {len(available_models[model_type])} models available")
            continue

        print(f"\n--- Creating {model_type.upper()} Ensembles ---")

        # Generate predictions from all models
        predictions = []
        model_info = []
        weights_for_accuracy = []

        for model_data in available_models[model_type]:
            try:
                if model_type == 'xgboost':
                    pred = model_manager.predict_xgboost(model_data['model'], test_data)
                else:
                    pred = model_manager.predict_neural_network(model_data['model'], test_data)

                predictions.append(pred)
                model_info.append(model_data['site'])

                # Get weight for accuracy-weighted ensemble
                local_auc = site_local_performance.get(model_data['site'], {}).get(model_type, 0.5)
                weights_for_accuracy.append(local_auc)

                print(f"  ✓ {model_data['site']}: predictions generated (local AUC: {local_auc:.3f})")

            except Exception as e:
                print(f"  ✗ {model_data['site']}: prediction failed - {e}")

        if len(predictions) < 2:
            print(f"Insufficient predictions for {model_type} ensemble")
            continue

        predictions = np.array(predictions)
        weights_for_accuracy = np.array(weights_for_accuracy)

        # Create Simple Average Ensemble
        simple_avg_pred = ensemble_constructor.simple_average_ensemble(predictions)
        simple_avg_metrics = metrics_calc.calculate_all_metrics(
            test_data['mortality_label'],
            simple_avg_pred
        )

        print(f"  Simple Average Ensemble AUC: {simple_avg_metrics['auc']:.4f}")

        # Create Accuracy-Weighted Ensemble
        accuracy_weighted_pred = ensemble_constructor.accuracy_weighted_ensemble(
            predictions, weights_for_accuracy
        )
        accuracy_weighted_metrics = metrics_calc.calculate_all_metrics(
            test_data['mortality_label'],
            accuracy_weighted_pred
        )

        print(f"  Accuracy-Weighted Ensemble AUC: {accuracy_weighted_metrics['auc']:.4f}")

        # Find best individual model for comparison
        best_individual_auc = 0
        best_individual_site = None
        individual_results = []

        for i, pred in enumerate(predictions):
            ind_metrics = metrics_calc.calculate_all_metrics(
                test_data['mortality_label'],
                pred
            )
            individual_results.append({
                'site': model_info[i],
                'auc': ind_metrics['auc'],
                'metrics': ind_metrics
            })

            if ind_metrics['auc'] > best_individual_auc:
                best_individual_auc = ind_metrics['auc']
                best_individual_site = model_info[i]

        print(f"  Best Individual Model: {best_individual_site} (AUC: {best_individual_auc:.4f})")

        # Calculate improvements
        simple_improvement = simple_avg_metrics['auc'] - best_individual_auc
        weighted_improvement = accuracy_weighted_metrics['auc'] - best_individual_auc

        print(f"  Simple Average Improvement: {simple_improvement:+.4f} ({simple_improvement/best_individual_auc*100:+.1f}%)")
        print(f"  Accuracy-Weighted Improvement: {weighted_improvement:+.4f} ({weighted_improvement/best_individual_auc*100:+.1f}%)")

        # Store results
        ensemble_results[model_type] = {
            'participating_sites': model_info,
            'model_count': len(predictions),
            'individual_results': individual_results,
            'best_individual': {
                'site': best_individual_site,
                'auc': best_individual_auc
            },
            'simple_average_ensemble': {
                'performance': simple_avg_metrics,
                'improvement_over_best': simple_improvement,
                'relative_improvement': simple_improvement / best_individual_auc
            },
            'accuracy_weighted_ensemble': {
                'performance': accuracy_weighted_metrics,
                'weights': weights_for_accuracy.tolist(),
                'improvement_over_best': weighted_improvement,
                'relative_improvement': weighted_improvement / best_individual_auc
            }
        }

    # Compare across model types
    if len(ensemble_results) > 1:
        print(f"\n--- Cross-Model Type Comparison ---")
        for model_type, results in ensemble_results.items():
            simple_auc = results['simple_average_ensemble']['performance']['auc']
            weighted_auc = results['accuracy_weighted_ensemble']['performance']['auc']
            print(f"{model_type.upper()}: Simple={simple_auc:.3f}, Weighted={weighted_auc:.3f}")

    # Save comprehensive results
    output_dir = Path(config['outputs']['approach2']) / current_site
    output_dir.mkdir(parents=True, exist_ok=True)

    final_results = {
        'test_site': current_site,
        'test_date': str(pd.Timestamp.now()),
        'approach': 'Approach 2 - Transfer Learning Ensemble Testing',
        'test_data_size': len(test_data),
        'ensemble_strategy': 'leave-one-out (exclude current site models)',
        'available_sites': list(set([model['site'] for models in available_models.values() for model in models])),
        'ensemble_results': ensemble_results,
        'site_local_performance': site_local_performance
    }

    results_path = output_dir / f"{current_site}_approach2_ensemble_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n--- Ensemble Testing Summary ---")
    print(f"Test site: {current_site}")
    print(f"Ensembles created: {len(ensemble_results)}")

    for model_type, results in ensemble_results.items():
        best_method = 'simple_average' if results['simple_average_ensemble']['performance']['auc'] > results['accuracy_weighted_ensemble']['performance']['auc'] else 'accuracy_weighted'
        best_auc = results[f'{best_method}_ensemble']['performance']['auc']
        improvement = results[f'{best_method}_ensemble']['improvement_over_best']

        print(f"{model_type.upper()}: Best ensemble AUC = {best_auc:.4f} ({improvement:+.4f} vs best individual)")

    print(f"Results saved to: {results_path}")

    return final_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble testing for transfer learning models")
    parser.add_argument("--site_name", required=True, help="Name of current site")
    parser.add_argument("--config_path", default="../../../config_demo.json",
                        help="Path to configuration file")

    args = parser.parse_args()

    results = main(args.site_name, args.config_path)

    if results is None:
        sys.exit(1)