#!/usr/bin/env python3
"""
Approach 4 - Stage 2: Ensemble Testing
Test round robin models in ensemble combinations.
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
    """Test round robin models with other approaches in ensemble."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"=== Approach 4 - Stage 2: Ensemble Testing at {current_site} ===")

    splitter = TemporalDataSplitter(config)
    model_manager = ModelManager(config)
    metrics_calc = MetricsCalculator()
    ensemble_constructor = EnsembleConstructor()

    # Load test data
    test_data = splitter.get_testing_data(years=[2023, 2024])

    # Load round robin models
    rr_models_dir = Path(config['federated']['model_sharing_path']) / "approach4_models"

    # For this approach, we mainly evaluate the final round robin models
    # compared to individual models from other approaches
    model_files = {
        'xgboost': 'round_robin_final_xgboost_model.pkl',
        'neural_network': 'round_robin_final_neural_network_model.pkl'
    }

    ensemble_results = {}

    for model_type, filename in model_files.items():
        model_path = rr_models_dir / filename

        if model_path.exists():
            try:
                model = model_manager.load_model(model_path, model_type=model_type)

                if model_type == 'xgboost':
                    predictions = model_manager.predict_xgboost(model, test_data)
                else:
                    predictions = model_manager.predict_neural_network(model, test_data)

                metrics = metrics_calc.calculate_all_metrics(test_data['mortality_label'], predictions)

                ensemble_results[model_type] = {
                    'round_robin_performance': metrics,
                    'model_path': str(model_path)
                }

                print(f"Round Robin {model_type.upper()} AUC: {metrics['auc']:.4f}")

            except Exception as e:
                print(f"Error testing {model_type}: {e}")

    # Save results
    output_dir = Path(config['outputs']['approach4']) / current_site
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'test_site': current_site,
        'test_date': str(pd.Timestamp.now()),
        'approach': 'Approach 4 - Round Robin Ensemble Testing',
        'ensemble_results': ensemble_results
    }

    results_path = output_dir / f"{current_site}_approach4_ensemble_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble testing for round robin approach")
    parser.add_argument("--site_name", required=True)
    parser.add_argument("--config_path", default="../../../config_demo.json")
    args = parser.parse_args()
    main(args.site_name, args.config_path)