#!/usr/bin/env python3
"""
Approach 2 - Stage 2: Cross-Site Testing
Test transfer learning models across all participating sites.
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

def main(current_site, config_path):
    """
    Test all available transfer learning models on current site's data.

    Args:
        current_site: Name of the current site (where testing is performed)
        config_path: Path to configuration file
    """

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"=== Approach 2 - Stage 2: Cross-Site Testing at {current_site} ===")

    # Initialize components
    splitter = TemporalDataSplitter(config)
    model_manager = ModelManager(config)
    metrics_calc = MetricsCalculator()

    # Load local test data
    print(f"Loading {current_site} test data (2023-2024)...")
    test_data = splitter.get_testing_data(years=[2023, 2024])
    print(f"Test data shape: {test_data.shape}")

    # Find all available transfer learning models
    shared_models_dir = Path(config['federated']['model_sharing_path']) / "approach2_models"

    if not shared_models_dir.exists():
        print(f"ERROR: Shared models directory not found: {shared_models_dir}")
        print("Please ensure Stage 1 transfer learning has been completed at all sites.")
        return None

    # Get list of sites with available models
    available_sites = []
    for site_dir in shared_models_dir.iterdir():
        if site_dir.is_dir() and site_dir.name != current_site:  # Exclude current site
            # Check if models exist
            xgb_model = site_dir / f"{site_dir.name}_transfer_xgboost_model.pkl"
            nn_model = site_dir / f"{site_dir.name}_transfer_nn_model.pkl"

            if xgb_model.exists() or nn_model.exists():
                available_sites.append(site_dir.name)

    if not available_sites:
        print(f"ERROR: No transfer learning models found from other sites")
        print(f"Checked directory: {shared_models_dir}")
        return None

    print(f"Found models from {len(available_sites)} sites: {available_sites}")

    # Test each model on local data
    cross_site_results = {}

    for source_site in available_sites:
        print(f"\n--- Testing Models from {source_site} ---")

        site_dir = shared_models_dir / source_site
        site_results = {
            'source_site': source_site,
            'test_site': current_site,
            'test_data_size': len(test_data),
            'models_tested': {}
        }

        # Test XGBoost model if available
        xgb_model_path = site_dir / f"{source_site}_transfer_xgboost_model.pkl"
        if xgb_model_path.exists():
            try:
                print(f"  Testing XGBoost model...")
                xgb_model = model_manager.load_model(xgb_model_path, model_type='xgboost')
                xgb_predictions = model_manager.predict_xgboost(xgb_model, test_data)
                xgb_metrics = metrics_calc.calculate_all_metrics(
                    test_data['mortality_label'],
                    xgb_predictions
                )

                site_results['models_tested']['xgboost'] = {
                    'model_path': str(xgb_model_path),
                    'performance': xgb_metrics,
                    'predictions_generated': len(xgb_predictions)
                }

                print(f"    XGBoost AUC: {xgb_metrics['auc']:.4f}")

            except Exception as e:
                print(f"  ✗ XGBoost model failed: {e}")
                site_results['models_tested']['xgboost'] = {
                    'model_path': str(xgb_model_path),
                    'error': str(e),
                    'performance': None
                }

        # Test Neural Network model if available
        nn_model_path = site_dir / f"{source_site}_transfer_nn_model.pkl"
        if nn_model_path.exists():
            try:
                print(f"  Testing Neural Network model...")
                nn_model = model_manager.load_model(nn_model_path, model_type='neural_network')
                nn_predictions = model_manager.predict_neural_network(nn_model, test_data)
                nn_metrics = metrics_calc.calculate_all_metrics(
                    test_data['mortality_label'],
                    nn_predictions
                )

                site_results['models_tested']['neural_network'] = {
                    'model_path': str(nn_model_path),
                    'performance': nn_metrics,
                    'predictions_generated': len(nn_predictions)
                }

                print(f"    Neural Network AUC: {nn_metrics['auc']:.4f}")

            except Exception as e:
                print(f"  ✗ Neural Network model failed: {e}")
                site_results['models_tested']['neural_network'] = {
                    'model_path': str(nn_model_path),
                    'error': str(e),
                    'performance': None
                }

        # Load source site's local performance for comparison
        results_file = site_dir / f"{source_site}_transfer_learning_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    source_results = json.load(f)
                    site_results['source_local_performance'] = {
                        model: results['fine_tuned_performance']
                        for model, results in source_results.get('models', {}).items()
                    }
            except Exception as e:
                print(f"  Warning: Could not load source performance: {e}")

        cross_site_results[source_site] = site_results

    # Analyze cross-site performance
    print(f"\n--- Cross-Site Performance Analysis ---")

    # Create performance matrix
    performance_data = []
    for source_site, results in cross_site_results.items():
        for model_type, model_results in results['models_tested'].items():
            if model_results.get('performance'):
                performance_data.append({
                    'source_site': source_site,
                    'test_site': current_site,
                    'model_type': model_type,
                    'auc': model_results['performance']['auc'],
                    'precision': model_results['performance']['precision'],
                    'recall': model_results['performance']['recall'],
                    'f1': model_results['performance']['f1']
                })

    if performance_data:
        perf_df = pd.DataFrame(performance_data)

        print(f"\nPerformance Summary on {current_site} data:")
        print(f"{'Source Site':<12} {'Model':<15} {'AUC':<6} {'F1':<6}")
        print("-" * 45)

        for _, row in perf_df.iterrows():
            print(f"{row['source_site']:<12} {row['model_type']:<15} {row['auc']:.3f} {row['f1']:.3f}")

        # Calculate summary statistics
        xgb_data = perf_df[perf_df['model_type'] == 'xgboost']
        nn_data = perf_df[perf_df['model_type'] == 'neural_network']

        if len(xgb_data) > 0:
            print(f"\nXGBoost Cross-Site Performance:")
            print(f"  Mean AUC: {xgb_data['auc'].mean():.4f} ± {xgb_data['auc'].std():.4f}")
            print(f"  Range: {xgb_data['auc'].min():.4f} - {xgb_data['auc'].max():.4f}")

        if len(nn_data) > 0:
            print(f"\nNeural Network Cross-Site Performance:")
            print(f"  Mean AUC: {nn_data['auc'].mean():.4f} ± {nn_data['auc'].std():.4f}")
            print(f"  Range: {nn_data['auc'].min():.4f} - {nn_data['auc'].max():.4f}")

    # Save results
    output_dir = Path(config['outputs']['approach2']) / current_site
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed cross-site results
    detailed_results = {
        'test_site': current_site,
        'test_date': str(pd.Timestamp.now()),
        'test_data_size': len(test_data),
        'approach': 'Approach 2 - Transfer Learning Cross-Site Testing',
        'source_sites_tested': available_sites,
        'cross_site_results': cross_site_results,
        'performance_summary': performance_data
    }

    results_path = output_dir / f"{current_site}_approach2_cross_site_results.json"
    with open(results_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    # Save performance matrix CSV
    if performance_data:
        perf_df.to_csv(output_dir / f"{current_site}_approach2_performance_matrix.csv", index=False)

    print(f"\n--- Cross-Site Testing Complete ---")
    print(f"Test site: {current_site}")
    print(f"Models tested: {len(available_sites)} sites × 2 model types = {len(performance_data)} models")
    print(f"Results saved to: {results_path}")

    return detailed_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-site testing for transfer learning models")
    parser.add_argument("--site_name", required=True, help="Name of current site (where testing is performed)")
    parser.add_argument("--config_path", default="../../../config_demo.json",
                        help="Path to configuration file")

    args = parser.parse_args()

    results = main(args.site_name, args.config_path)

    if results is None:
        sys.exit(1)