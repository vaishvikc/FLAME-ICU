#!/usr/bin/env python3
"""
Approach 2 - Stage 1: Receive Base Model
Download and verify RUSH base models for transfer learning.
"""

import sys
import os
import argparse
import json
import shutil
from pathlib import Path

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'shared'))

from model_io import ModelManager

def main(site_name, config_path, model_type='all'):
    """
    Download and verify base models from RUSH for transfer learning.

    Args:
        site_name: Name of the current site
        config_path: Path to configuration file
        model_type: Type of model to download ('xgboost', 'nn', or 'all')
    """

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"=== Approach 2 - Stage 1: Receiving Base Models at {site_name} ===")

    # Define paths
    distributed_models_dir = Path(config['federated']['model_sharing_path']) / "distributed_models" / site_name / "approach1_models"
    local_models_dir = Path(config['outputs']['approach2']) / site_name / "base_models"

    print(f"Source (distributed): {distributed_models_dir}")
    print(f"Local destination: {local_models_dir}")

    # Create local directory
    local_models_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model manager
    model_manager = ModelManager(config)

    # Define models to download
    models_to_process = []
    if model_type in ['all', 'xgboost']:
        models_to_process.append(('xgboost', 'RUSH_xgboost_model.pkl'))
    if model_type in ['all', 'nn']:
        models_to_process.append(('neural_network', 'RUSH_nn_model.pkl'))

    downloaded_models = {}

    for model_name, filename in models_to_process:
        print(f"\n--- Processing {model_name.upper()} Model ---")

        # Source and destination paths
        src_path = distributed_models_dir / filename
        dst_path = local_models_dir / filename

        # Check if source exists
        if not src_path.exists():
            print(f"ERROR: Base model not found: {src_path}")
            print("Please ensure Approach 1 has been completed and models distributed.")
            continue

        # Copy model to local directory
        try:
            shutil.copy2(src_path, dst_path)
            print(f"✓ Copied {filename}")

            # Verify model can be loaded
            try:
                model = model_manager.load_model(dst_path, model_type=model_name)
                print(f"✓ Model verified and can be loaded")

                # Get model info
                if model_name == 'xgboost':
                    num_features = model.num_features() if hasattr(model, 'num_features') else 'unknown'
                    num_trees = model.num_boosted_rounds() if hasattr(model, 'num_boosted_rounds') else 'unknown'
                    print(f"  Features: {num_features}, Trees: {num_trees}")

                elif model_name == 'neural_network':
                    if hasattr(model, 'named_parameters'):
                        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        print(f"  Trainable parameters: {param_count:,}")

                downloaded_models[model_name] = {
                    'path': str(dst_path),
                    'verified': True,
                    'filename': filename
                }

            except Exception as e:
                print(f"✗ Model verification failed: {e}")
                downloaded_models[model_name] = {
                    'path': str(dst_path),
                    'verified': False,
                    'error': str(e)
                }

        except Exception as e:
            print(f"ERROR copying model: {e}")
            continue

    # Also copy metrics for reference
    metrics_src = distributed_models_dir / "RUSH_training_metrics.json"
    metrics_dst = local_models_dir / "RUSH_training_metrics.json"

    if metrics_src.exists():
        try:
            shutil.copy2(metrics_src, metrics_dst)
            print(f"✓ Copied training metrics")

            # Load and display baseline metrics
            with open(metrics_dst, 'r') as f:
                metrics = json.load(f)

            print(f"\n--- Baseline Performance (RUSH) ---")
            if 'xgboost_metrics' in metrics:
                xgb_auc = metrics['xgboost_metrics']['auc']
                print(f"XGBoost AUC: {xgb_auc:.4f}")

            if 'neural_network_metrics' in metrics:
                nn_auc = metrics['neural_network_metrics']['auc']
                print(f"Neural Network AUC: {nn_auc:.4f}")

        except Exception as e:
            print(f"Warning: Could not copy metrics file: {e}")

    # Save download record
    download_record = {
        'site_name': site_name,
        'download_date': str(pd.Timestamp.now()) if 'pd' in globals() else 'unknown',
        'source_directory': str(distributed_models_dir),
        'local_directory': str(local_models_dir),
        'requested_type': model_type,
        'downloaded_models': downloaded_models
    }

    record_path = local_models_dir / f"{site_name}_download_record.json"
    with open(record_path, 'w') as f:
        json.dump(download_record, f, indent=2)

    # Summary
    print(f"\n--- Download Summary ---")
    verified_count = sum(1 for model in downloaded_models.values() if model['verified'])
    total_count = len(downloaded_models)

    print(f"Models downloaded: {total_count}")
    print(f"Models verified: {verified_count}")
    print(f"Success rate: {verified_count/total_count*100:.0f}%" if total_count > 0 else "No models processed")
    print(f"Local models saved to: {local_models_dir}")

    if verified_count == 0:
        print("\nERROR: No models successfully downloaded and verified!")
        print("Cannot proceed with transfer learning.")
        return None

    if verified_count < total_count:
        print(f"\nWARNING: {total_count - verified_count} model(s) failed verification")

    print(f"\nNext step: Run fine_tune_locally.py to begin transfer learning")

    return download_record

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download base models for transfer learning")
    parser.add_argument("--site_name", required=True, help="Name of current site")
    parser.add_argument("--model_type", choices=['xgboost', 'nn', 'all'], default='all',
                        help="Type of model to download")
    parser.add_argument("--config_path", default="../../../config_demo.json",
                        help="Path to configuration file")

    args = parser.parse_args()

    # Add pandas import for timestamp
    import pandas as pd

    results = main(args.site_name, args.config_path, args.model_type)

    if results is None:
        sys.exit(1)