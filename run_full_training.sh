#!/bin/bash

# ===============================================================================
# FLAME-ICU Complete Model Training Pipeline
# ===============================================================================
# This script runs the complete model training pipeline:
# 1. LSTM model training
# 2. XGBoost model training  
# 3. Neural Network model training
# All models are saved in site-specific directories under models_stage_1/
# ===============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
START_TIME=$(date +%s)

# Create logs directory
mkdir -p "$LOG_DIR"

# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

# Logging functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

print_header() {
    echo ""
    echo -e "${BLUE}===============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===============================================================================${NC}"
    log "HEADER: $1"
}

print_step() {
    echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} ${GREEN}$1${NC}"
    log "STEP: $1"
}

print_substep() {
    echo -e "  ${YELLOW}→${NC} $1"
    log "SUBSTEP: $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
    log "SUCCESS: $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1" >&2
    log "ERROR: $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
    log "WARNING: $1"
}

print_info() {
    echo -e "${PURPLE}ℹ${NC} $1"
    log "INFO: $1"
}

# Progress tracking
show_progress() {
    local current=$1
    local total=$2
    local description=$3
    local percent=$((current * 100 / total))
    local completed=$((current * 50 / total))
    local remaining=$((50 - completed))
    
    printf "\r${CYAN}Progress:${NC} ["
    printf "%${completed}s" | tr ' ' '='
    printf "%${remaining}s" | tr ' ' '-'
    printf "] %d%% - %s" "$percent" "$description"
    
    if [ "$current" -eq "$total" ]; then
        echo ""
    fi
}

# Time formatting
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    
    if [ $hours -gt 0 ]; then
        printf "%dh %dm %ds" $hours $minutes $secs
    elif [ $minutes -gt 0 ]; then
        printf "%dm %ds" $minutes $secs
    else
        printf "%ds" $secs
    fi
}

# ===============================================================================
# CONFIGURATION LOADING
# ===============================================================================

get_site_name() {
    print_step "Loading site configuration..."
    
    # Load site name from config_demo.json
    if [ -f "$SCRIPT_DIR/config_demo.json" ]; then
        SITE_NAME=$(python3 -c "import json; print(json.load(open('$SCRIPT_DIR/config_demo.json'))['site'])")
        print_substep "Site name: $SITE_NAME"
    else
        print_error "Configuration file config_demo.json not found"
        exit 1
    fi
}

# ===============================================================================
# DEPENDENCY CHECKING
# ===============================================================================

check_dependencies() {
    print_step "Checking dependencies and environment..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 not found. Please install Python 3.7+"
        exit 1
    fi
    
    local python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_substep "Python version: $python_version"
    
    # Check virtual environment and required packages
    if ! (source "$SCRIPT_DIR/flameICU/bin/activate" && python3 -c "import pandas" 2>/dev/null); then
        print_error "pandas not found in virtual environment. Please activate flameICU and install requirements"
        exit 1
    fi
    
    # Check model-specific packages
    local required_packages=("pandas" "numpy" "sklearn" "torch" "xgboost")
    for package in "${required_packages[@]}"; do
        if (source "$SCRIPT_DIR/flameICU/bin/activate" && python3 -c "import $package" 2>/dev/null); then
            print_substep "$package: ✓"
        else
            print_error "Required Python package '$package' not found in virtual environment"
            exit 1
        fi
    done
    
    # Check that preprocessing data exists
    local data_files=(
        "protected_outputs/preprocessing/by_event_wide_df.parquet"
        "protected_outputs/preprocessing/by_hourly_wide_df.parquet"
        "protected_outputs/intermediate/data/train_df.parquet"
        "protected_outputs/intermediate/data/test_df.parquet"
        "protected_outputs/intermediate/lstm/train_sequences.pkl"
        "protected_outputs/intermediate/lstm/test_sequences.pkl"
    )
    
    print_step "Checking preprocessing outputs..."
    for file in "${data_files[@]}"; do
        if [ -f "$SCRIPT_DIR/$file" ]; then
            print_substep "$file: ✓"
        else
            print_error "Required data file not found: $file"
            print_info "Please run the preprocessing pipeline first: ./run_full_preprocessing.sh"
            exit 1
        fi
    done
    
    print_success "All dependencies checked successfully"
}

# ===============================================================================
# MODEL TRAINING
# ===============================================================================

train_model() {
    local model_name=$1
    local model_dir=$2
    local description=$3
    
    print_step "Training $description..."
    print_substep "Model directory: $model_dir"
    
    local start_time=$(date +%s)
    
    # Change to model directory and run training
    if source "$SCRIPT_DIR/flameICU/bin/activate" && \
       cd "$SCRIPT_DIR/$model_dir" && \
       python3 training.py 2>&1 | tee -a "$LOG_FILE"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "$description completed in $(format_duration $duration)"
        
        # Check if outputs were created
        local site_output_dir="$SCRIPT_DIR/protected_outputs/models_stage_1/$SITE_NAME/$model_name"
        if [ -d "$site_output_dir" ]; then
            print_substep "Model outputs saved to: models_stage_1/$SITE_NAME/$model_name/"
            
            # List key outputs
            if [ -f "$site_output_dir/metrics.json" ]; then
                local roc_auc=$(python3 -c "import json; metrics=json.load(open('$site_output_dir/metrics.json')); print(f\"{metrics.get('test_roc_auc', 'N/A'):.4f}\" if isinstance(metrics.get('test_roc_auc'), (int, float)) else 'N/A')")
                print_substep "Test ROC-AUC: $roc_auc"
            fi
        fi
        
        return 0
    else
        local exit_code=$?
        print_error "$description failed with exit code $exit_code"
        return $exit_code
    fi
}

# ===============================================================================
# DIRECTORY SETUP
# ===============================================================================

create_output_directories() {
    print_step "Setting up output directory structure..."
    
    # Create site-specific model directories
    local model_types=("lstm" "xgboost" "nn")
    
    for model in "${model_types[@]}"; do
        local dir="protected_outputs/models_stage_1/$SITE_NAME/$model"
        if [ ! -d "$SCRIPT_DIR/$dir" ]; then
            mkdir -p "$SCRIPT_DIR/$dir"
            print_substep "Created: $dir"
        else
            print_substep "Exists: $dir"
        fi
        
        # Create plots subdirectory
        if [ ! -d "$SCRIPT_DIR/$dir/plots" ]; then
            mkdir -p "$SCRIPT_DIR/$dir/plots"
        fi
    done
    
    print_success "Output directory structure ready"
}

# ===============================================================================
# SUMMARY REPORTING
# ===============================================================================

generate_summary_report() {
    print_step "Generating comprehensive summary report..."
    
    local summary_file="${LOG_DIR}/training_summary_$(date +%Y%m%d_%H%M%S).txt"
    local current_time=$(date +%s)
    local total_duration=$((current_time - START_TIME))
    
    {
        echo "==============================================================================="
        echo "FLAME-ICU MODEL TRAINING PIPELINE SUMMARY"
        echo "==============================================================================="
        echo "Execution Date: $(date)"
        echo "Total Duration: $(format_duration $total_duration)"
        echo "Site Name: $SITE_NAME"
        echo "Working Directory: $SCRIPT_DIR"
        echo "Log File: $LOG_FILE"
        echo ""
        
        echo "MODELS TRAINED:"
        echo ""
        
        # LSTM Summary
        local lstm_dir="$SCRIPT_DIR/protected_outputs/models_stage_1/$SITE_NAME/lstm"
        if [ -f "$lstm_dir/metrics.json" ]; then
            echo "1. LSTM Model:"
            local metrics=$(python3 -c "
import json
m = json.load(open('$lstm_dir/metrics.json'))
print(f'   - Test ROC-AUC: {m.get(\"test_roc_auc\", \"N/A\"):.4f}' if isinstance(m.get('test_roc_auc'), (int, float)) else '   - Test ROC-AUC: N/A')
print(f'   - Test Accuracy: {m.get(\"test_accuracy\", \"N/A\"):.4f}' if isinstance(m.get('test_accuracy'), (int, float)) else '   - Test Accuracy: N/A')
print(f'   - Test AUPRC: {m.get(\"test_auprc\", \"N/A\"):.4f}' if isinstance(m.get('test_auprc'), (int, float)) else '   - Test AUPRC: N/A')
")
            echo "$metrics"
            echo "   - Model Path: models_stage_1/$SITE_NAME/lstm/lstm_icu_mortality_model.pt"
        else
            echo "1. LSTM Model: Training failed or incomplete"
        fi
        echo ""
        
        # XGBoost Summary
        local xgb_dir="$SCRIPT_DIR/protected_outputs/models_stage_1/$SITE_NAME/xgboost"
        if [ -f "$xgb_dir/metrics.json" ]; then
            echo "2. XGBoost Model:"
            local metrics=$(python3 -c "
import json
m = json.load(open('$xgb_dir/metrics.json'))
print(f'   - Test ROC-AUC: {m.get(\"test_roc_auc\", \"N/A\"):.4f}' if isinstance(m.get('test_roc_auc'), (int, float)) else '   - Test ROC-AUC: N/A')
print(f'   - Test Accuracy: {m.get(\"test_accuracy\", \"N/A\"):.4f}' if isinstance(m.get('test_accuracy'), (int, float)) else '   - Test Accuracy: N/A')
print(f'   - Test AUPRC: {m.get(\"test_auprc\", \"N/A\"):.4f}' if isinstance(m.get('test_auprc'), (int, float)) else '   - Test AUPRC: N/A')
")
            echo "$metrics"
            echo "   - Model Path: models_stage_1/$SITE_NAME/xgboost/xgb_icu_mortality_model.json"
        else
            echo "2. XGBoost Model: Training failed or incomplete"
        fi
        echo ""
        
        # Neural Network Summary
        local nn_dir="$SCRIPT_DIR/protected_outputs/models_stage_1/$SITE_NAME/nn"
        if [ -f "$nn_dir/metrics.json" ]; then
            echo "3. Neural Network Model:"
            local metrics=$(python3 -c "
import json
m = json.load(open('$nn_dir/metrics.json'))
print(f'   - Test ROC-AUC: {m.get(\"test_roc_auc\", \"N/A\"):.4f}' if isinstance(m.get('test_roc_auc'), (int, float)) else '   - Test ROC-AUC: N/A')
print(f'   - Test Accuracy: {m.get(\"test_accuracy\", \"N/A\"):.4f}' if isinstance(m.get('test_accuracy'), (int, float)) else '   - Test Accuracy: N/A')
print(f'   - Test AUPRC: {m.get(\"test_auprc\", \"N/A\"):.4f}' if isinstance(m.get('test_auprc'), (int, float)) else '   - Test AUPRC: N/A')
")
            echo "$metrics"
            echo "   - Model Path: models_stage_1/$SITE_NAME/nn/nn_icu_mortality_model.pt"
        else
            echo "3. Neural Network Model: Training failed or incomplete"
        fi
        echo ""
        
        echo "OUTPUT FILES CREATED:"
        echo "All models saved in: protected_outputs/models_stage_1/$SITE_NAME/"
        echo ""
        
        # List files created
        if [ -d "$SCRIPT_DIR/protected_outputs/models_stage_1/$SITE_NAME" ]; then
            echo "Directory structure:"
            tree -L 3 "$SCRIPT_DIR/protected_outputs/models_stage_1/$SITE_NAME" 2>/dev/null || \
                find "$SCRIPT_DIR/protected_outputs/models_stage_1/$SITE_NAME" -type f -name "*.json" -o -name "*.pt" -o -name "*.pkl" | sort
        fi
        echo ""
        
        echo "NEXT STEPS:"
        echo "1. Review model performance metrics in each model's metrics.json"
        echo "2. Examine plots in each model's plots/ directory"
        echo "3. Use inference scripts for predictions:"
        echo "   - LSTM: cd code/models/lstm && python3 inference.py"
        echo "   - XGBoost: cd code/models/xgboost && python3 inference.py"
        echo "   - Neural Network: cd code/models/nn && python3 inference.py"
        echo ""
        echo "4. For transfer learning at other sites:"
        echo "   - Share models from models_stage_1/$SITE_NAME/"
        echo "   - Use transfer_learning.py scripts in each model directory"
        echo ""
        echo "==============================================================================="
        echo "MODEL TRAINING PIPELINE COMPLETED!"
        echo "==============================================================================="
        
    } | tee "$summary_file"
    
    print_success "Summary report saved to: $summary_file"
    print_info "Total execution time: $(format_duration $total_duration)"
}

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

main() {
    # Print banner
    echo ""
    echo -e "${PURPLE}===============================================================================${NC}"
    echo -e "${PURPLE}              FLAME-ICU COMPLETE MODEL TRAINING PIPELINE                      ${NC}"
    echo -e "${PURPLE}===============================================================================${NC}"
    echo ""
    print_info "Starting comprehensive model training pipeline..."
    print_info "Log file: $LOG_FILE"
    echo ""
    
    # Initialize log
    log "=== FLAME-ICU MODEL TRAINING PIPELINE STARTED ==="
    log "Start time: $(date)"
    log "Working directory: $SCRIPT_DIR"
    
    # Get site name
    get_site_name
    
    # Create output directories
    create_output_directories
    
    # Check dependencies
    check_dependencies
    
    # Training Phase
    print_header "MODEL TRAINING PHASE"
    
    show_progress 0 3 "Starting model training..."
    
    # Train LSTM
    show_progress 0 3 "Training LSTM model..."
    if train_model "lstm" "code/models/lstm" "LSTM Model"; then
        lstm_success=true
    else
        lstm_success=false
        print_warning "LSTM training failed, continuing with other models..."
    fi
    
    # Train XGBoost
    show_progress 1 3 "Training XGBoost model..."
    if train_model "xgboost" "code/models/xgboost" "XGBoost Model"; then
        xgb_success=true
    else
        xgb_success=false
        print_warning "XGBoost training failed, continuing with other models..."
    fi
    
    # Train Neural Network
    show_progress 2 3 "Training Neural Network model..."
    if train_model "nn" "code/models/nn" "Neural Network Model"; then
        nn_success=true
    else
        nn_success=false
        print_warning "Neural Network training failed"
    fi
    
    show_progress 3 3 "Model training complete"
    
    # Generate summary
    generate_summary_report
    
    # Final status
    local end_time=$(date +%s)
    local total_duration=$((end_time - START_TIME))
    
    echo ""
    print_header "PIPELINE COMPLETED!"
    
    # Report successes and failures
    local total_models=3
    local successful_models=0
    
    [ "$lstm_success" = true ] && ((successful_models++))
    [ "$xgb_success" = true ] && ((successful_models++))
    [ "$nn_success" = true ] && ((successful_models++))
    
    if [ $successful_models -eq $total_models ]; then
        print_success "All models trained successfully!"
    elif [ $successful_models -eq 0 ]; then
        print_error "All model training failed!"
        exit 1
    else
        print_warning "$successful_models out of $total_models models trained successfully"
    fi
    
    print_success "Total execution time: $(format_duration $total_duration)"
    print_info "Check the summary report in: ${LOG_DIR}/"
    echo ""
    
    log "=== FLAME-ICU MODEL TRAINING PIPELINE COMPLETED ==="
    log "End time: $(date)"
    log "Total duration: $(format_duration $total_duration)"
    log "Models trained: $successful_models/$total_models"
}

# ===============================================================================
# ERROR HANDLING
# ===============================================================================

# Trap for cleanup on exit
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        print_error "Pipeline failed with exit code $exit_code"
        print_error "Check the log file for details: $LOG_FILE"
        
        local end_time=$(date +%s)
        local duration=$((end_time - START_TIME))
        log "=== PIPELINE FAILED ==="
        log "Exit code: $exit_code"
        log "Duration before failure: $(format_duration $duration)"
    fi
}

trap cleanup EXIT

# ===============================================================================
# SCRIPT ENTRY POINT
# ===============================================================================

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi