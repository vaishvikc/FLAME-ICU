#!/usr/bin/env python3
"""
Approach 3 - Stage 1: Upload Independent Models
Upload independently trained models to shared location for Stage 2.
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

def main(site_name, config_path, verify_models=True):
    """Upload independent models to shared location."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"=== Approach 3 - Stage 1: Uploading Independent Models from {site_name} ===")

    local_dir = Path(config['outputs']['approach3']) / site_name
    shared_dir = Path(config['federated']['model_sharing_path']) / "approach3_models" / site_name
    shared_dir.mkdir(parents=True, exist_ok=True)

    if not local_dir.exists():
        print(f"ERROR: Local models directory not found: {local_dir}")
        return None

    model_manager = ModelManager(config) if verify_models else None

    # Files to upload
    files_to_upload = [
        f"{site_name}_independent_xgboost_model.pkl",
        f"{site_name}_independent_nn_model.pkl",
        f"{site_name}_independent_training_results.json"
    ]

    uploaded_files = []

    for filename in files_to_upload:
        local_path = local_dir / filename
        shared_path = shared_dir / filename

        if local_path.exists():
            # Verify model if requested
            if verify_models and filename.endswith('.pkl'):
                try:
                    model_type = 'xgboost' if 'xgboost' in filename else 'neural_network'
                    model = model_manager.load_model(local_path, model_type=model_type)
                    print(f"✓ Verified: {filename}")
                except Exception as e:
                    print(f"✗ Verification failed for {filename}: {e}")
                    continue

            # Upload file
            shutil.copy2(local_path, shared_path)
            uploaded_files.append(filename)
            print(f"✓ Uploaded: {filename}")

    # Create upload manifest
    manifest = {
        'site_name': site_name,
        'approach': 'Approach 3 - Independent Training',
        'upload_date': str(pd.Timestamp.now()) if 'pd' in globals() else 'unknown',
        'uploaded_files': uploaded_files
    }

    with open(shared_dir / f"{site_name}_upload_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nUpload Summary: {len(uploaded_files)} files uploaded to {shared_dir}")
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload independent models")
    parser.add_argument("--site_name", required=True, help="Name of current site")
    parser.add_argument("--config_path", default="../../../config_demo.json")
    parser.add_argument("--no_verify", action="store_true")

    args = parser.parse_args()

    import pandas as pd
    results = main(args.site_name, args.config_path, verify_models=not args.no_verify)
    if results is None:
        sys.exit(1)