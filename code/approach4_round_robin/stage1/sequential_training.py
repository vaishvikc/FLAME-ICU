#!/usr/bin/env python3
"""
Approach 4 - Stage 1: Sequential Training
Perform round robin training at each site in sequence.
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
    """Perform sequential training for round robin approach."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"=== Approach 4 - Stage 1: Sequential Training at {site_name} ===")

    # Initialize components
    splitter = TemporalDataSplitter(config)
    model_manager = ModelManager(config)
    metrics_calc = MetricsCalculator()

    # Load local training data
    train_data = splitter.get_training_data(years=[2018, 2019, 2020, 2021, 2022])
    test_data = splitter.get_testing_data(years=[2023, 2024])

    # Get site order and current position
    site_order = config['federated']['participating_sites']
    current_position = site_order.index(site_name)
    is_first_site = (current_position == 0)

    round_robin_dir = Path(config['federated']['model_sharing_path']) / "approach4_round_robin"

    models_to_train = []
    if model_type in ['all', 'xgboost']:
        models_to_train.append('xgboost')
    if model_type in ['all', 'nn']:
        models_to_train.append('neural_network')

    training_results = {}

    for model_name in models_to_train:
        print(f"\n--- Round Robin Training: {model_name.upper()} ---")

        if is_first_site:
            # First site: initialize model
            print(f"Initializing {model_name} model...")
            if model_name == 'xgboost':
                model = model_manager.train_xgboost(train_data, config['models']['xgboost'])
            else:
                model = model_manager.train_neural_network(train_data, config['models']['neural_network'])
        else:
            # Subsequent sites: load model from previous site
            prev_site = site_order[current_position - 1]
            model_filename = f"round_robin_{model_name}_round_{current_position}.pkl"
            model_path = round_robin_dir / model_filename

            if not model_path.exists():
                print(f"ERROR: Model from {prev_site} not found: {model_path}")
                continue

            print(f"Loading model from {prev_site}...")
            model = model_manager.load_model(model_path, model_type=model_name)

            # Continue training with local data
            print(f"Continuing training with {site_name} data...")
            round_robin_config = config.get('round_robin', {})
            epochs = round_robin_config.get('epochs_per_site', 15)

            if model_name == 'xgboost':
                model = model_manager.continue_xgboost_training(
                    model, train_data,
                    num_boost_rounds=epochs,
                    learning_rate_decay=round_robin_config.get('learning_rate_decay', 0.9)
                )
            else:
                model = model_manager.continue_neural_network_training(
                    model, train_data,
                    max_epochs=epochs,
                    learning_rate_decay=round_robin_config.get('learning_rate_decay', 0.9)
                )

        # Evaluate model locally
        if model_name == 'xgboost':
            predictions = model_manager.predict_xgboost(model, test_data)
        else:
            predictions = model_manager.predict_neural_network(model, test_data)

        metrics = metrics_calc.calculate_all_metrics(test_data['mortality_label'], predictions)

        print(f"Local performance after round {current_position + 1}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Save model for next site (or final model if last site)
        next_round = current_position + 1
        if current_position == len(site_order) - 1:
            # Last site: save final model
            final_filename = f"round_robin_final_{model_name}_model.pkl"
        else:
            # Not last site: save for next round
            final_filename = f"round_robin_{model_name}_round_{next_round}.pkl"

        model_save_path = round_robin_dir / final_filename
        model_manager.save_model(model, model_save_path, model_type=model_name)

        training_results[model_name] = {
            'model_path': str(model_save_path),
            'site_position': current_position,
            'round_number': current_position + 1,
            'is_final': (current_position == len(site_order) - 1),
            'local_performance': metrics,
            'training_data_size': len(train_data)
        }

        print(f"✓ Saved: {final_filename}")

    # Save training results
    output_dir = Path(config['outputs']['approach4']) / site_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results_summary = {
        'site_name': site_name,
        'approach': 'Approach 4 - Round Robin Training',
        'training_date': str(pd.Timestamp.now()),
        'site_position': current_position,
        'round_number': current_position + 1,
        'is_first_site': is_first_site,
        'is_last_site': (current_position == len(site_order) - 1),
        'training_results': training_results
    }

    results_path = output_dir / f"{site_name}_round_robin_training.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n--- Round Robin Training Complete at {site_name} ---")
    print(f"Position: {current_position + 1}/{len(site_order)}")
    print(f"Models processed: {len(training_results)}")
    print(f"Results saved to: {results_path}")

    return results_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential round robin training")
    parser.add_argument("--site_name", required=True, help="Name of current site")
    parser.add_argument("--model_type", choices=['xgboost', 'nn', 'all'], default='all')
    parser.add_argument("--config_path", default="../../../config_demo.json")

    args = parser.parse_args()
    results = main(args.site_name, args.config_path, args.model_type)
    if results is None:
        sys.exit(1)