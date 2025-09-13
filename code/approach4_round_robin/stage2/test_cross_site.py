#!/usr/bin/env python3
"""
Approach 4 - Stage 2: Cross-Site Testing
Test final round robin models across all participating sites.
"""

import sys
import os
import argparse
import json
import pandas as pd
from pathlib import Path

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'shared'))
sys.path.append(str(Path(__file__).parent.parent.parent / 'models'))

from temporal_split import TemporalDataSplitter
from model_io import ModelManager
from metrics import MetricsCalculator

def main(current_site, config_path):
    """Test final round robin models on current site's data."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"=== Approach 4 - Stage 2: Cross-Site Testing at {current_site} ===")

    splitter = TemporalDataSplitter(config)
    model_manager = ModelManager(config)
    metrics_calc = MetricsCalculator()

    # Load test data
    test_data = splitter.get_testing_data(years=[2023, 2024])

    # Load final round robin models
    final_models_dir = Path(config['federated']['model_sharing_path']) / "approach4_models"

    model_files = {
        'xgboost': 'round_robin_final_xgboost_model.pkl',
        'neural_network': 'round_robin_final_neural_network_model.pkl'
    }

    testing_results = {}

    for model_type, filename in model_files.items():
        model_path = final_models_dir / filename

        if model_path.exists():
            try:
                print(f"\n--- Testing {model_type.upper()} Round Robin Model ---")

                model = model_manager.load_model(model_path, model_type=model_type)

                if model_type == 'xgboost':
                    predictions = model_manager.predict_xgboost(model, test_data)
                else:
                    predictions = model_manager.predict_neural_network(model, test_data)

                metrics = metrics_calc.calculate_all_metrics(test_data['mortality_label'], predictions)

                print(f"Round Robin {model_type.upper()} Performance:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")

                testing_results[model_type] = {
                    'model_path': str(model_path),
                    'performance': metrics,
                    'test_data_size': len(test_data)
                }

            except Exception as e:
                print(f"✗ Testing failed for {model_type}: {e}")

    # Save results
    output_dir = Path(config['outputs']['approach4']) / current_site
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'test_site': current_site,
        'test_date': str(pd.Timestamp.now()),
        'approach': 'Approach 4 - Round Robin Cross-Site Testing',
        'testing_results': testing_results
    }

    results_path = output_dir / f"{current_site}_approach4_cross_site_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nRound robin cross-site testing complete at {current_site}")
    print(f"Results saved to: {results_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-site testing for round robin models")
    parser.add_argument("--site_name", required=True)
    parser.add_argument("--config_path", default="../../../config_demo.json")
    args = parser.parse_args()
    main(args.site_name, args.config_path)