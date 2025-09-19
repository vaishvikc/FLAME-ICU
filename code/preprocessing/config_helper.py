"""
Simple helper functions for FLAME-ICU project
"""

import os
import json
from pathlib import Path


def get_project_root():
    """Get the project root directory"""
    current_file = Path(__file__).resolve()
    # Go up from code/preprocessing/ to project root (2 levels up)
    return str(current_file.parent.parent.parent)


def ensure_dir(file_path):
    """Create directory for a file path if it doesn't exist"""
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    return directory


def get_output_path(stage, filename):
    """Get standardized output path for protected_outputs"""
    project_root = get_project_root()
    return os.path.join(project_root, 'protected_outputs', stage, filename)


def load_config():
    """Load configuration from clif_config.json"""
    project_root = get_project_root()
    
    # Try code/configs directory first (where config actually is)
    config_path = os.path.join(project_root, "code", "configs", "clif_config.json")
    
    if not os.path.exists(config_path):
        # Try project root/configs directory
        config_path = os.path.join(project_root, "configs", "clif_config.json")
    
    if not os.path.exists(config_path):
        # Fallback to project root
        config_path = os.path.join(project_root, "clif_config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        print(f"✅ Loaded configuration from {config_path}")
        return config
    else:
        raise FileNotFoundError(f"Configuration file not found. Tried: {config_path}")