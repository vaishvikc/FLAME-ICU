#!/usr/bin/env python3
"""
Approach 3 - Stage 2: Ensemble Testing
Create and test ensemble models using independent models.
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
    """Create and test ensemble models using independent models (leave-one-out)."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"=== Approach 3 - Stage 2: Ensemble Testing at {current_site} ===")

    splitter = TemporalDataSplitter(config)
    model_manager = ModelManager(config)
    metrics_calc = MetricsCalculator()
    ensemble_constructor = EnsembleConstructor()

    # Load test data
    test_data = splitter.get_testing_data(years=[2023, 2024])

    # Find independent models (excluding current site)
    shared_models_dir = Path(config['federated']['model_sharing_path']) / "approach3_models"

    available_models = {'xgboost': [], 'neural_network': []}
    site_performance = {}

    for site_dir in shared_models_dir.iterdir():
        if site_dir.is_dir() and site_dir.name != current_site:

            # Load site's local performance for weighting
            results_file = site_dir / f"{site_dir.name}_independent_training_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    site_results = json.load(f)
                    local_perf = {}
                    for model_type, results in site_results.get('models', {}).items():
                        if 'test_performance' in results:
                            local_perf[model_type] = results['test_performance']['auc']
                    site_performance[site_dir.name] = local_perf

            # Load models
            for model_type, filename_template in [('xgboost', '_independent_xgboost_model.pkl'),
                                                  ('neural_network', '_independent_nn_model.pkl')]:
                model_path = site_dir / f"{site_dir.name}{filename_template}"
                if model_path.exists():
                    try:
                        model = model_manager.load_model(model_path, model_type=model_type)
                        available_models[model_type].append({
                            'site': site_dir.name,
                            'model': model,
                            'path': model_path
                        })
                    except Exception as e:
                        print(f"Warning: Could not load {model_type} from {site_dir.name}: {e}")

    ensemble_results = {}

    for model_type in ['xgboost', 'neural_network']:
        if len(available_models[model_type]) < 2:
            continue

        print(f"\n--- Creating {model_type.upper()} Ensembles ---")

        # Generate predictions
        predictions = []
        model_info = []
        weights = []

        for model_data in available_models[model_type]:
            if model_type == 'xgboost':
                pred = model_manager.predict_xgboost(model_data['model'], test_data)
            else:
                pred = model_manager.predict_neural_network(model_data['model'], test_data)

            predictions.append(pred)
            model_info.append(model_data['site'])

            # Weight for accuracy-weighted ensemble
            local_auc = site_performance.get(model_data['site'], {}).get(model_type, 0.5)
            weights.append(local_auc)

        predictions = np.array(predictions)
        weights = np.array(weights)

        # Simple average ensemble
        simple_pred = ensemble_constructor.simple_average_ensemble(predictions)
        simple_metrics = metrics_calc.calculate_all_metrics(test_data['mortality_label'], simple_pred)

        # Accuracy-weighted ensemble
        weighted_pred = ensemble_constructor.accuracy_weighted_ensemble(predictions, weights)
        weighted_metrics = metrics_calc.calculate_all_metrics(test_data['mortality_label'], weighted_pred)

        # Best individual model
        best_auc = 0
        for i, pred in enumerate(predictions):
            ind_metrics = metrics_calc.calculate_all_metrics(test_data['mortality_label'], pred)
            if ind_metrics['auc'] > best_auc:
                best_auc = ind_metrics['auc']

        ensemble_results[model_type] = {
            'participating_sites': model_info,
            'simple_average': {'performance': simple_metrics, 'improvement': simple_metrics['auc'] - best_auc},
            'accuracy_weighted': {'performance': weighted_metrics, 'improvement': weighted_metrics['auc'] - best_auc},
            'best_individual_auc': best_auc
        }

        print(f"  Simple Average AUC: {simple_metrics['auc']:.4f} ({simple_metrics['auc'] - best_auc:+.4f})")
        print(f"  Weighted Average AUC: {weighted_metrics['auc']:.4f} ({weighted_metrics['auc'] - best_auc:+.4f})")

    # Save results
    output_dir = Path(config['outputs']['approach3']) / current_site
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'test_site': current_site,
        'test_date': str(pd.Timestamp.now()),
        'approach': 'Approach 3 - Independent Models Ensemble Testing',
        'ensemble_results': ensemble_results
    }

    results_path = output_dir / f"{current_site}_approach3_ensemble_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nEnsemble testing complete. Results saved to: {results_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble testing for independent models")
    parser.add_argument("--site_name", required=True, help="Name of current site")
    parser.add_argument("--config_path", default="../../../config_demo.json")

    args = parser.parse_args()
    results = main(args.site_name, args.config_path)
    if results is None:
        sys.exit(1)