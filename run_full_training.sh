#!/bin/bash

# ===============================================================================
# FLAME-ICU Complete Model Training Pipeline (Simplified)
# ===============================================================================
# This script runs the complete model training pipeline:
# 1. LSTM model training
# 2. XGBoost model training  
# 3. Neural Network model training
# All models are saved in site-specific directories under models_stage_1/
# ===============================================================================

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===============================================================================
# CONFIGURATION LOADING
# ===============================================================================

get_site_name() {
    echo "Loading site configuration..."
    
    # Load site name from config_demo.json
    if [ -f "$SCRIPT_DIR/config_demo.json" ]; then
        SITE_NAME=$(python3 -c "import json; print(json.load(open('$SCRIPT_DIR/config_demo.json'))['site'])")
        echo "Site name: $SITE_NAME"
    else
        echo "ERROR: Configuration file config_demo.json not found"
        exit 1
    fi
}

# ===============================================================================
# DEPENDENCY CHECKING
# ===============================================================================

check_dependencies() {
    echo "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "ERROR: Python3 not found. Please install Python 3.7+"
        exit 1
    fi
    
    # Check that preprocessing data exists
    local data_files=(
        "protected_outputs/preprocessing/by_event_wide_df.parquet"
        "protected_outputs/preprocessing/by_hourly_wide_df.parquet"
        "protected_outputs/intermediate/data/train_df.parquet"
        "protected_outputs/intermediate/data/test_df.parquet"
        "protected_outputs/intermediate/lstm/train_sequences.pkl"
        "protected_outputs/intermediate/lstm/test_sequences.pkl"
    )
    
    echo "Checking preprocessing outputs..."
    for file in "${data_files[@]}"; do
        if [ ! -f "$SCRIPT_DIR/$file" ]; then
            echo "ERROR: Required data file not found: $file"
            echo "Please run the preprocessing pipeline first: ./run_full_preprocessing.sh"
            exit 1
        fi
    done
    
    echo "All dependencies checked successfully"
}

# ===============================================================================
# MODEL TRAINING
# ===============================================================================

train_model() {
    local model_name=$1
    local model_dir=$2
    local description=$3
    
    echo ""
    echo "Training $description..."
    
    # Change to model directory and run training
    if source "$SCRIPT_DIR/flameICU/bin/activate" && \
       cd "$SCRIPT_DIR/$model_dir" && \
       python3 training.py; then
        
        echo "$description completed"
        return 0
    else
        echo "ERROR: $description failed"
        return 1
    fi
}

# ===============================================================================
# DIRECTORY SETUP
# ===============================================================================

create_output_directories() {
    echo "Setting up output directories..."
    
    # Create site-specific model directories
    local model_types=("lstm" "xgboost" "nn")
    
    for model in "${model_types[@]}"; do
        local dir="protected_outputs/models_stage_1/$SITE_NAME/$model"
        mkdir -p "$SCRIPT_DIR/$dir"
        mkdir -p "$SCRIPT_DIR/$dir/plots"
    done
    
    echo "Output directories created"
}

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

main() {
    echo "==============================================================================="
    echo "              FLAME-ICU COMPLETE MODEL TRAINING PIPELINE                      "
    echo "==============================================================================="
    echo ""
    
    # Get site name
    get_site_name
    
    # Create output directories
    create_output_directories
    
    # Check dependencies
    check_dependencies
    
    # Training Phase
    echo ""
    echo "==============================================================================="
    echo "MODEL TRAINING PHASE"
    echo "==============================================================================="
    
    # Train LSTM
    train_model "lstm" "code/models/lstm" "LSTM Model"
    
    # Train XGBoost
    train_model "xgboost" "code/models/xgboost" "XGBoost Model"
    
    # Train Neural Network
    train_model "nn" "code/models/nn" "Neural Network Model"
    
    echo ""
    echo "==============================================================================="
    echo "PIPELINE COMPLETED!"
    echo "==============================================================================="
    echo "All models saved in: protected_outputs/models_stage_1/$SITE_NAME/"
    echo ""
}

# ===============================================================================
# SCRIPT ENTRY POINT
# ===============================================================================

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi