#!/usr/bin/env python3
"""
Approach 4 - Stage 1: Coordinate Round Robin Training
Manage the sequential training process across all participating sites.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'shared'))

def main(config_path):
    """Coordinate the round robin training process across all sites."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    print("=== Approach 4 - Stage 1: Round Robin Training Coordination ===")

    # Define training order
    site_order = config['federated']['participating_sites']
    print(f"Site training order: {' → '.join(site_order)}")

    round_robin_dir = Path(config['federated']['model_sharing_path']) / "approach4_round_robin"
    round_robin_dir.mkdir(parents=True, exist_ok=True)

    # Create coordination manifest
    coordination_manifest = {
        'approach': 'Approach 4 - Round Robin Federated Training',
        'site_order': site_order,
        'total_sites': len(site_order),
        'current_round': 0,
        'status': 'initialized',
        'start_time': str(time.time()),
        'round_history': []
    }

    manifest_path = round_robin_dir / "coordination_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(coordination_manifest, f, indent=2)

    print(f"Coordination manifest created: {manifest_path}")
    print(f"Ready to begin round robin training with {len(site_order)} sites")

    return coordination_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coordinate round robin training")
    parser.add_argument("--config_path", default="../../../config_demo.json")
    args = parser.parse_args()
    main(args.config_path)