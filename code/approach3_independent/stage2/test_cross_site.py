#!/usr/bin/env python3
"""
Approach 3 - Stage 2: Cross-Site Testing
Test independent models across all participating sites.
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
    """Test all independent models on current site's data."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"=== Approach 3 - Stage 2: Cross-Site Testing at {current_site} ===")

    splitter = TemporalDataSplitter(config)
    model_manager = ModelManager(config)
    metrics_calc = MetricsCalculator()

    # Load test data
    test_data = splitter.get_testing_data(years=[2023, 2024])

    # Find all independent models (excluding current site)
    shared_models_dir = Path(config['federated']['model_sharing_path']) / "approach3_models"

    cross_site_results = {}

    for site_dir in shared_models_dir.iterdir():
        if site_dir.is_dir() and site_dir.name != current_site:
            site_results = {'source_site': site_dir.name, 'test_site': current_site, 'models_tested': {}}

            # Test XGBoost model
            xgb_path = site_dir / f"{site_dir.name}_independent_xgboost_model.pkl"
            if xgb_path.exists():
                try:
                    model = model_manager.load_model(xgb_path, model_type='xgboost')
                    predictions = model_manager.predict_xgboost(model, test_data)
                    metrics = metrics_calc.calculate_all_metrics(test_data['mortality_label'], predictions)
                    site_results['models_tested']['xgboost'] = {'performance': metrics}
                    print(f"✓ {site_dir.name} XGBoost AUC: {metrics['auc']:.4f}")
                except Exception as e:
                    print(f"✗ {site_dir.name} XGBoost failed: {e}")

            # Test NN model
            nn_path = site_dir / f"{site_dir.name}_independent_nn_model.pkl"
            if nn_path.exists():
                try:
                    model = model_manager.load_model(nn_path, model_type='neural_network')
                    predictions = model_manager.predict_neural_network(model, test_data)
                    metrics = metrics_calc.calculate_all_metrics(test_data['mortality_label'], predictions)
                    site_results['models_tested']['neural_network'] = {'performance': metrics}
                    print(f"✓ {site_dir.name} NN AUC: {metrics['auc']:.4f}")
                except Exception as e:
                    print(f"✗ {site_dir.name} NN failed: {e}")

            cross_site_results[site_dir.name] = site_results

    # Save results
    output_dir = Path(config['outputs']['approach3']) / current_site
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'test_site': current_site,
        'test_date': str(pd.Timestamp.now()),
        'approach': 'Approach 3 - Independent Models Cross-Site Testing',
        'cross_site_results': cross_site_results
    }

    results_path = output_dir / f"{current_site}_approach3_cross_site_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nCross-site testing complete. Results saved to: {results_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-site testing for independent models")
    parser.add_argument("--site_name", required=True, help="Name of current site")
    parser.add_argument("--config_path", default="../../../config_demo.json")

    args = parser.parse_args()
    results = main(args.site_name, args.config_path)
    if results is None:
        sys.exit(1)