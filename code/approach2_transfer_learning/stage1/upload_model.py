#!/usr/bin/env python3
"""
Approach 2 - Stage 1: Upload Fine-tuned Models
Upload fine-tuned models to shared location for Stage 2 analysis.
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
    """
    Upload fine-tuned models to shared location for Stage 2.

    Args:
        site_name: Name of the current site
        config_path: Path to configuration file
        verify_models: Whether to verify models before uploading
    """

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"=== Approach 2 - Stage 1: Uploading Models from {site_name} ===")

    # Define paths
    local_models_dir = Path(config['outputs']['approach2']) / site_name / "fine_tuned_models"
    shared_models_dir = Path(config['federated']['model_sharing_path']) / "approach2_models"

    print(f"Source (local): {local_models_dir}")
    print(f"Destination (shared): {shared_models_dir}")

    # Check if local models exist
    if not local_models_dir.exists():
        print(f"ERROR: Local models directory not found: {local_models_dir}")
        print("Please run fine_tune_locally.py first")
        return None

    # Create shared directory
    shared_models_dir.mkdir(parents=True, exist_ok=True)
    site_shared_dir = shared_models_dir / site_name
    site_shared_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model manager for verification
    if verify_models:
        model_manager = ModelManager(config)

    # Find models to upload
    model_files = {
        'xgboost': f"{site_name}_transfer_xgboost_model.pkl",
        'neural_network': f"{site_name}_transfer_nn_model.pkl"
    }

    results_file = f"{site_name}_transfer_learning_results.json"

    uploaded_files = []
    verification_results = {}

    # Upload models
    for model_type, filename in model_files.items():
        local_path = local_models_dir / filename
        shared_path = site_shared_dir / filename

        if local_path.exists():
            print(f"\n--- Uploading {model_type.upper()} Model ---")

            # Verify model before upload
            if verify_models:
                try:
                    model = model_manager.load_model(local_path, model_type=model_type)
                    print(f"✓ Model verification passed")

                    # Get model info
                    if model_type == 'xgboost':
                        num_features = model.num_features() if hasattr(model, 'num_features') else 'unknown'
                        num_trees = model.num_boosted_rounds() if hasattr(model, 'num_boosted_rounds') else 'unknown'
                        model_info = f"Features: {num_features}, Trees: {num_trees}"
                    else:
                        if hasattr(model, 'named_parameters'):
                            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                            model_info = f"Parameters: {param_count:,}"
                        else:
                            model_info = "Neural network model"

                    print(f"  {model_info}")

                    verification_results[model_type] = {
                        'verified': True,
                        'info': model_info,
                        'file_size': local_path.stat().st_size
                    }

                except Exception as e:
                    print(f"✗ Model verification failed: {e}")
                    print(f"Skipping upload of {filename}")
                    verification_results[model_type] = {
                        'verified': False,
                        'error': str(e)
                    }
                    continue
            else:
                verification_results[model_type] = {'verified': 'skipped'}

            # Upload model
            try:
                shutil.copy2(local_path, shared_path)
                print(f"✓ Uploaded: {filename}")
                uploaded_files.append(filename)

                # Set permissions (if on Unix-like system)
                try:
                    os.chmod(shared_path, 0o644)
                except:
                    pass  # Permissions not critical

            except Exception as e:
                print(f"✗ Upload failed: {e}")
                continue

        else:
            print(f"⚠ Model not found: {filename}")

    # Upload results file
    local_results_path = local_models_dir / results_file
    shared_results_path = site_shared_dir / results_file

    if local_results_path.exists():
        try:
            shutil.copy2(local_results_path, shared_results_path)
            print(f"✓ Uploaded results: {results_file}")
            uploaded_files.append(results_file)
        except Exception as e:
            print(f"✗ Results upload failed: {e}")
    else:
        print(f"⚠ Results file not found: {results_file}")

    # Create upload manifest
    upload_manifest = {
        'site_name': site_name,
        'upload_date': str(pd.Timestamp.now()) if 'pd' in globals() else 'unknown',
        'approach': 'Approach 2 - Transfer Learning',
        'source_directory': str(local_models_dir),
        'destination_directory': str(site_shared_dir),
        'uploaded_files': uploaded_files,
        'verification_results': verification_results,
        'upload_status': 'completed' if uploaded_files else 'failed'
    }

    manifest_path = site_shared_dir / f"{site_name}_upload_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(upload_manifest, f, indent=2)

    # Create README for the shared directory
    readme_content = f"""# Transfer Learning Models - {site_name}

## Uploaded Models
"""

    for filename in uploaded_files:
        if filename.endswith('.pkl'):
            model_name = filename.replace(f"{site_name}_transfer_", "").replace("_model.pkl", "")
            readme_content += f"- **{filename}**: Fine-tuned {model_name} model\n"
        elif filename.endswith('.json'):
            readme_content += f"- **{filename}**: Training results and performance metrics\n"

    readme_content += f"""
## Upload Information
- Site: {site_name}
- Approach: Transfer Learning (Approach 2)
- Files uploaded: {len(uploaded_files)}

## Usage in Stage 2
These models will be used for:
1. Cross-site testing on other sites' data
2. Ensemble construction with other transfer learning models
3. Performance comparison across federated approaches

## Verification
"""

    for model_type, result in verification_results.items():
        if result.get('verified') == True:
            readme_content += f"- {model_type.upper()}: ✓ Verified ({result['info']})\n"
        elif result.get('verified') == False:
            readme_content += f"- {model_type.upper()}: ✗ Failed verification\n"
        else:
            readme_content += f"- {model_type.upper()}: - Verification skipped\n"

    readme_path = site_shared_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"\n--- Upload Summary ---")
    print(f"Site: {site_name}")
    print(f"Files uploaded: {len(uploaded_files)}")
    print(f"Destination: {site_shared_dir}")

    if verification_results:
        verified_count = sum(1 for r in verification_results.values() if r.get('verified') == True)
        total_models = len([f for f in uploaded_files if f.endswith('.pkl')])
        print(f"Models verified: {verified_count}/{total_models}")

    if uploaded_files:
        print(f"✓ Upload completed successfully")
        print(f"Models are ready for Stage 2 analysis")
    else:
        print(f"✗ No files were uploaded")
        return None

    return upload_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload fine-tuned models for Stage 2")
    parser.add_argument("--site_name", required=True, help="Name of current site")
    parser.add_argument("--config_path", default="../../../config_demo.json",
                        help="Path to configuration file")
    parser.add_argument("--no_verify", action="store_true",
                        help="Skip model verification before upload")

    args = parser.parse_args()

    # Add pandas import for timestamp
    import pandas as pd

    results = main(args.site_name, args.config_path, verify_models=not args.no_verify)

    if results is None:
        sys.exit(1)