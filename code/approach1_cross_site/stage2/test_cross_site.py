#!/usr/bin/env python3
"""
Approach 1 - Stage 2: Cross-Site Testing
Comprehensive evaluation of Approach 1 models across all sites.
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

def main(config_path, output_summary=True):
    """
    Perform comprehensive cross-site testing for Approach 1.

    Args:
        config_path: Path to configuration file
        output_summary: Whether to generate summary report
    """

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("=== Approach 1 - Stage 2: Cross-Site Testing ===")

    # Initialize components
    splitter = TemporalDataSplitter(config)
    model_manager = ModelManager(config)
    metrics_calc = MetricsCalculator()

    # Get participating sites
    participating_sites = config['federated']['participating_sites']
    print(f"Analyzing results from {len(participating_sites)} sites: {participating_sites}")

    # Collect all evaluation results
    all_results = {}
    results_dir = Path(config['outputs']['approach1'])

    for site in participating_sites:
        results_file = results_dir / f"{site}_approach1_evaluation.json"

        if results_file.exists():
            with open(results_file, 'r') as f:
                site_results = json.load(f)
                all_results[site] = site_results
                print(f"✓ Loaded results for {site}")
        else:
            print(f"⚠ Missing results for {site}: {results_file}")

    if not all_results:
        print("ERROR: No evaluation results found!")
        print("Please ensure Stage 1 evaluation has been completed at all sites.")
        return None

    # Analyze cross-site performance
    print(f"\n--- Cross-Site Performance Analysis ---")

    # Create performance matrix
    performance_matrix = {
        'site': [],
        'test_samples': [],
        'xgboost_auc': [],
        'xgboost_precision': [],
        'xgboost_recall': [],
        'xgboost_f1': [],
        'nn_auc': [],
        'nn_precision': [],
        'nn_recall': [],
        'nn_f1': []
    }

    for site, results in all_results.items():
        performance_matrix['site'].append(site)
        performance_matrix['test_samples'].append(results['test_data_size'])

        # XGBoost metrics
        xgb_perf = results['xgboost_performance']
        performance_matrix['xgboost_auc'].append(xgb_perf['auc'])
        performance_matrix['xgboost_precision'].append(xgb_perf['precision'])
        performance_matrix['xgboost_recall'].append(xgb_perf['recall'])
        performance_matrix['xgboost_f1'].append(xgb_perf['f1'])

        # Neural Network metrics
        nn_perf = results['neural_network_performance']
        performance_matrix['nn_auc'].append(nn_perf['auc'])
        performance_matrix['nn_precision'].append(nn_perf['precision'])
        performance_matrix['nn_recall'].append(nn_perf['recall'])
        performance_matrix['nn_f1'].append(nn_perf['f1'])

    # Convert to DataFrame for analysis
    perf_df = pd.DataFrame(performance_matrix)

    # Calculate summary statistics
    print("\n--- XGBoost Cross-Site Performance ---")
    xgb_auc_mean = perf_df['xgboost_auc'].mean()
    xgb_auc_std = perf_df['xgboost_auc'].std()
    xgb_auc_min = perf_df['xgboost_auc'].min()
    xgb_auc_max = perf_df['xgboost_auc'].max()

    print(f"AUC - Mean: {xgb_auc_mean:.4f} ± {xgb_auc_std:.4f}")
    print(f"AUC - Range: {xgb_auc_min:.4f} to {xgb_auc_max:.4f}")
    print(f"AUC - CV: {xgb_auc_std/xgb_auc_mean*100:.1f}%")

    print("\n--- Neural Network Cross-Site Performance ---")
    nn_auc_mean = perf_df['nn_auc'].mean()
    nn_auc_std = perf_df['nn_auc'].std()
    nn_auc_min = perf_df['nn_auc'].min()
    nn_auc_max = perf_df['nn_auc'].max()

    print(f"AUC - Mean: {nn_auc_mean:.4f} ± {nn_auc_std:.4f}")
    print(f"AUC - Range: {nn_auc_min:.4f} to {nn_auc_max:.4f}")
    print(f"AUC - CV: {nn_auc_std/nn_auc_mean*100:.1f}%")

    # Site-specific analysis
    print("\n--- Site-Specific Performance ---")
    for _, row in perf_df.iterrows():
        print(f"{row['site']:>8}: XGBoost={row['xgboost_auc']:.3f}, NN={row['nn_auc']:.3f}, samples={row['test_samples']:>5}")

    # Generalization analysis
    print("\n--- Generalization Analysis ---")

    # Find RUSH performance (baseline)
    rush_results = all_results.get('RUSH', {})
    if rush_results:
        rush_xgb_auc = rush_results['xgboost_performance']['auc']
        rush_nn_auc = rush_results['neural_network_performance']['auc']

        print(f"RUSH baseline - XGBoost: {rush_xgb_auc:.4f}, NN: {rush_nn_auc:.4f}")

        # Calculate performance drops
        other_sites = [site for site in participating_sites if site != 'RUSH']
        if other_sites:
            other_xgb_aucs = [all_results[site]['xgboost_performance']['auc'] for site in other_sites if site in all_results]
            other_nn_aucs = [all_results[site]['neural_network_performance']['auc'] for site in other_sites if site in all_results]

            if other_xgb_aucs:
                xgb_drop = rush_xgb_auc - np.mean(other_xgb_aucs)
                print(f"Average XGBoost performance drop: {xgb_drop:.4f} ({xgb_drop/rush_xgb_auc*100:.1f}%)")

            if other_nn_aucs:
                nn_drop = rush_nn_auc - np.mean(other_nn_aucs)
                print(f"Average NN performance drop: {nn_drop:.4f} ({nn_drop/rush_nn_auc*100:.1f}%)")

    # Save comprehensive results
    output_dir = Path(config['outputs']['approach1'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save performance matrix
    perf_df.to_csv(output_dir / "approach1_cross_site_performance.csv", index=False)

    # Save summary analysis
    summary_results = {
        'approach': 'Approach 1 - Cross-Site Model Validation',
        'analysis_date': str(pd.Timestamp.now()),
        'participating_sites': participating_sites,
        'sites_with_data': list(all_results.keys()),
        'xgboost_summary': {
            'mean_auc': xgb_auc_mean,
            'std_auc': xgb_auc_std,
            'min_auc': xgb_auc_min,
            'max_auc': xgb_auc_max,
            'coefficient_of_variation': xgb_auc_std/xgb_auc_mean
        },
        'neural_network_summary': {
            'mean_auc': nn_auc_mean,
            'std_auc': nn_auc_std,
            'min_auc': nn_auc_min,
            'max_auc': nn_auc_max,
            'coefficient_of_variation': nn_auc_std/nn_auc_mean
        },
        'generalization_metrics': {
            'sites_analyzed': len(all_results),
            'performance_range_xgb': xgb_auc_max - xgb_auc_min,
            'performance_range_nn': nn_auc_max - nn_auc_min
        },
        'detailed_results': all_results
    }

    summary_path = output_dir / "approach1_cross_site_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2)

    if output_summary:
        print(f"\n--- Final Summary ---")
        print(f"Analysis complete for Approach 1")
        print(f"Sites analyzed: {len(all_results)}/{len(participating_sites)}")
        print(f"XGBoost performance: {xgb_auc_mean:.3f} ± {xgb_auc_std:.3f}")
        print(f"Neural Network performance: {nn_auc_mean:.3f} ± {nn_auc_std:.3f}")
        print(f"Results saved to: {output_dir}")

    return summary_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-site testing analysis for Approach 1")
    parser.add_argument("--config_path", default="../../../config_demo.json",
                        help="Path to configuration file")
    parser.add_argument("--no_summary", action="store_true",
                        help="Skip summary output")

    args = parser.parse_args()

    results = main(args.config_path, output_summary=not args.no_summary)

    if results is None:
        sys.exit(1)