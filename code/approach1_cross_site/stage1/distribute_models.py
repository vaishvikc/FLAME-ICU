#!/usr/bin/env python3
"""
Approach 1 - Stage 1: Model Distribution
Distribute trained RUSH models to all federated sites.
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

def main(config_path, dry_run=False):
    """
    Distribute RUSH models to federated sites.

    Args:
        config_path: Path to configuration file
        dry_run: If True, only simulate the distribution process
    """

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("=== Approach 1 - Stage 1: Model Distribution ===")

    # Define model source and distribution paths
    model_source = Path(config['federated']['model_sharing_path']) / "approach1_models"
    distribution_base = Path(config['federated']['model_sharing_path']) / "distributed_models"

    print(f"Source directory: {model_source}")
    print(f"Distribution directory: {distribution_base}")

    # Check if source models exist
    required_files = [
        "RUSH_xgboost_model.pkl",
        "RUSH_nn_model.pkl",
        "RUSH_training_metrics.json"
    ]

    missing_files = []
    for file in required_files:
        if not (model_source / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"ERROR: Missing required files: {missing_files}")
        print("Please run train_main_models.py first at RUSH site.")
        return False

    # Get list of federated sites from config
    federated_sites = config['federated']['participating_sites']
    print(f"Distributing models to {len(federated_sites)} sites: {federated_sites}")

    # Create distribution structure
    if not dry_run:
        distribution_base.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for site in federated_sites:
        try:
            print(f"\n--- Distributing to {site} ---")

            # Create site-specific directory
            site_dir = distribution_base / site / "approach1_models"

            if dry_run:
                print(f"[DRY RUN] Would create directory: {site_dir}")
                print(f"[DRY RUN] Would copy {len(required_files)} files")
            else:
                site_dir.mkdir(parents=True, exist_ok=True)

                # Copy models and metrics to site directory
                for file in required_files:
                    src_path = model_source / file
                    dst_path = site_dir / file
                    shutil.copy2(src_path, dst_path)
                    print(f"  Copied: {file}")

                # Create site-specific readme
                readme_content = f"""# RUSH Models for {site}

## Files in this directory:
- RUSH_xgboost_model.pkl: Pre-trained XGBoost model
- RUSH_nn_model.pkl: Pre-trained Neural Network model
- RUSH_training_metrics.json: Training performance metrics

## Usage:
Run the evaluation script from your site directory:
```bash
python ../stage1/evaluate_locally.py --site_name "{site}"
```

## Training Information:
- Training data: RUSH 2018-2022
- Model architectures: XGBoost + Neural Network
- Intended for testing on: {site} 2023-2024 data
"""

                with open(site_dir / "README.md", 'w') as f:
                    f.write(readme_content)

                print(f"  Created README for {site}")

            success_count += 1

        except Exception as e:
            print(f"ERROR distributing to {site}: {e}")

    print(f"\n=== Distribution Summary ===")
    print(f"Successfully distributed to {success_count}/{len(federated_sites)} sites")

    if not dry_run:
        print(f"Models available at: {distribution_base}")
        print("\nNext steps:")
        print("1. Notify sites that models are available")
        print("2. Sites should run evaluate_locally.py with their site name")
        print("3. Collect performance results for analysis")

    return success_count == len(federated_sites)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distribute RUSH models to federated sites")
    parser.add_argument("--config_path", default="../../../config_demo.json",
                        help="Path to configuration file")
    parser.add_argument("--dry_run", action="store_true",
                        help="Simulate distribution without copying files")

    args = parser.parse_args()

    success = main(args.config_path, args.dry_run)

    if success:
        print("Distribution completed successfully!")
    else:
        print("Distribution completed with errors.")
        sys.exit(1)