#!/usr/bin/env python3
"""
Setup script for Git LFS to handle large model files in WUPERR federated learning.
"""

import os
import subprocess
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command, check=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def setup_git_lfs():
    """Set up Git LFS for handling large model files."""
    logger.info("Setting up Git LFS for WUPERR model files...")
    
    # Check if Git LFS is installed
    success, stdout, stderr = run_command("git lfs version", check=False)
    if not success:
        logger.error("Git LFS is not installed. Please install it first:")
        logger.error("  - macOS: brew install git-lfs")
        logger.error("  - Ubuntu: sudo apt-get install git-lfs")
        logger.error("  - Windows: Download from https://git-lfs.github.io/")
        return False
    
    logger.info(f"Git LFS version: {stdout.strip()}")
    
    # Initialize Git LFS
    success, stdout, stderr = run_command("git lfs install")
    if not success:
        logger.error(f"Failed to initialize Git LFS: {stderr}")
        return False
    
    logger.info("Git LFS initialized successfully")
    
    # Track model files
    file_patterns = [
        "*.pt",      # PyTorch model files
        "*.pth",     # PyTorch checkpoint files
        "*.pkl",     # Pickle files (scalers, importance weights)
        "*.h5",      # HDF5 files
        "*.json",    # Large JSON files (metadata)
        "*.parquet"  # Parquet data files
    ]
    
    for pattern in file_patterns:
        success, stdout, stderr = run_command(f"git lfs track '{pattern}'")
        if success:
            logger.info(f"Added {pattern} to Git LFS tracking")
        else:
            logger.warning(f"Failed to add {pattern} to Git LFS: {stderr}")
    
    # Add .gitattributes file
    success, stdout, stderr = run_command("git add .gitattributes")
    if success:
        logger.info("Added .gitattributes to git")
    else:
        logger.warning(f"Failed to add .gitattributes: {stderr}")
    
    return True

def create_gitignore():
    """Create or update .gitignore file."""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# Data files (except LFS tracked)
*.csv
data/sites/*.parquet
logs/
temp/
tmp/

# OS
.DS_Store
Thumbs.db

# WUPERR specific
data/sites/site_*_data.parquet
results/
model/checkpoints/
*.log
"""
    
    gitignore_path = ".gitignore"
    
    # Read existing .gitignore if it exists
    existing_content = ""
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            existing_content = f.read()
    
    # Add new content if not already present
    if "# WUPERR specific" not in existing_content:
        with open(gitignore_path, 'a') as f:
            f.write(gitignore_content)
        logger.info("Updated .gitignore with WUPERR-specific entries")
    else:
        logger.info(".gitignore already contains WUPERR entries")

def create_model_directory():
    """Create model directory structure."""
    directories = [
        "model",
        "model/checkpoints",
        "data/sites",
        "logs",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Create README files for each directory
    readme_files = {
        "model/README.md": """# Model Directory

This directory contains the WUPERR federated learning models:

- `lstm_wuperr_model.pt` - Current model weights
- `model_metadata.json` - Model metadata and training history
- `weight_importance.pkl` - Weight importance scores for WUPERR
- `site_contributions.json` - Site-by-site training contributions
- `checkpoints/` - Model checkpoints from each training round
""",
        "data/sites/README.md": """# Site Data Directory

This directory contains site-specific datasets for federated learning:

- `site_1_data.parquet` through `site_8_data.parquet` - Site datasets
- `site_metadata.json` - Metadata about each site
- `training_results.json` - Results from sequential training
- `federated_analysis.json` - Analysis of federated learning performance
""",
        "logs/README.md": """# Logs Directory

This directory contains training logs:

- Training logs from each site
- Error logs and debugging information
- Performance metrics over time
""",
        "results/README.md": """# Results Directory

This directory contains evaluation results:

- Performance comparisons
- Federated learning analysis
- Visualizations and plots
"""
    }
    
    for filepath, content in readme_files.items():
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(content)
        logger.info(f"Created README: {filepath}")

def main():
    """Main setup function."""
    logger.info("Starting WUPERR Git LFS setup...")
    
    # Check if we're in a git repository
    success, stdout, stderr = run_command("git rev-parse --git-dir", check=False)
    if not success:
        logger.error("Not in a git repository. Please run 'git init' first.")
        return False
    
    # Set up Git LFS
    if not setup_git_lfs():
        return False
    
    # Create .gitignore
    create_gitignore()
    
    # Create directory structure
    create_model_directory()
    
    logger.info("WUPERR Git LFS setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Run: python 07_simulate_multisite.py --setup_only")
    logger.info("2. Run: python 07_simulate_multisite.py --num_rounds 3")
    logger.info("3. Monitor progress in the logs/ directory")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)