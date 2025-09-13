#!/usr/bin/env python3
"""
Approach 4 - Stage 1: Upload Final Round Robin Model
Upload final round robin models after complete training sequence.
"""

import sys
import os
import argparse
import json
import shutil
from pathlib import Path

def main(config_path):
    """Upload final round robin models to shared location."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    print("=== Approach 4 - Stage 1: Uploading Final Round Robin Models ===")

    round_robin_dir = Path(config['federated']['model_sharing_path']) / "approach4_round_robin"
    final_models_dir = Path(config['federated']['model_sharing_path']) / "approach4_models"
    final_models_dir.mkdir(parents=True, exist_ok=True)

    # Find final models
    final_models = [
        "round_robin_final_xgboost_model.pkl",
        "round_robin_final_neural_network_model.pkl"
    ]

    uploaded_files = []

    for filename in final_models:
        src_path = round_robin_dir / filename
        dst_path = final_models_dir / filename

        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            uploaded_files.append(filename)
            print(f"✓ Uploaded: {filename}")
        else:
            print(f"⚠ Model not found: {filename}")

    # Create upload manifest
    manifest = {
        'approach': 'Approach 4 - Round Robin Training',
        'upload_date': str(pd.Timestamp.now()) if 'pd' in globals() else 'unknown',
        'final_models': uploaded_files,
        'total_rounds': len(config['federated']['participating_sites'])
    }

    with open(final_models_dir / "round_robin_upload_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nFinal models uploaded: {len(uploaded_files)}")
    print(f"Destination: {final_models_dir}")

    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload final round robin models")
    parser.add_argument("--config_path", default="../../../config_demo.json")
    args = parser.parse_args()

    import pandas as pd
    main(args.config_path)