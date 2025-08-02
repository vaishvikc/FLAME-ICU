#!/bin/bash

# ===============================================================================
# FLAME-ICU Complete Preprocessing Pipeline
# ===============================================================================
# This script runs the complete preprocessing pipeline:
# 1. Core preprocessing: cohort generation, feature engineering, statistics
# 2. Model-specific data splitting: LSTM, XGBoost, Neural Network
# 3. Validation and summary reporting
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
LOG_FILE="${LOG_DIR}/preprocessing_$(date +%Y%m%d_%H%M%S).log"
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
    
    # Check marimo in virtual environment
    if ! (source "$SCRIPT_DIR/flameICU/bin/activate" && python3 -c "import marimo" 2>/dev/null); then
        print_error "marimo not found in virtual environment. Please activate flameICU and install marimo: pip install marimo"
        exit 1
    fi
    print_substep "marimo found"
    
    # Check required Python packages in virtual environment
    local required_packages=("pandas" "numpy" "pyclif" "sklearn")
    for package in "${required_packages[@]}"; do
        if (source "$SCRIPT_DIR/flameICU/bin/activate" && python3 -c "import $package" 2>/dev/null); then
            print_substep "$package: ✓"
        else
            print_error "Required Python package '$package' not found in virtual environment"
            exit 1
        fi
    done
    
    # Check configuration files
    if [ ! -f "$SCRIPT_DIR/config_demo.json" ]; then
        print_error "Configuration file config_demo.json not found in $SCRIPT_DIR"
        exit 1
    fi
    print_substep "Configuration file: ✓"
    
    # Check model config files exist
    local model_dirs=("code/models/lstm" "code/models/xgboost" "code/models/nn")
    for model_dir in "${model_dirs[@]}"; do
        if [ -f "$SCRIPT_DIR/$model_dir/config.json" ]; then
            print_substep "Model config ($model_dir): ✓"
        else
            print_warning "Model config not found: $model_dir/config.json"
        fi
    done
    
    print_success "All dependencies checked successfully"
}

# ===============================================================================
# SCRIPT EXECUTION
# ===============================================================================

execute_script() {
    local script_name=$1
    local description=$2
    
    print_step "Executing: $description"
    print_substep "Script: $script_name"
    
    local start_time=$(date +%s)
    
    # Activate virtual environment and execute script
    if source "$SCRIPT_DIR/flameICU/bin/activate" && \
       python3 "$script_name" 2>&1 | tee -a "$LOG_FILE"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "$description completed in $(format_duration $duration)"
    else
        local exit_code=$?
        print_error "$description failed with exit code $exit_code"
        print_error "Check the log file for details: $LOG_FILE"
        
        exit 1
    fi
}

# ===============================================================================
# DATA VALIDATION
# ===============================================================================

validate_file() {
    local file_path=$1
    local description=$2
    local min_size_mb=${3:-0}  # Minimum size in MB
    
    if [ -f "$file_path" ]; then
        local file_size=$(stat -f%z "$file_path" 2>/dev/null || stat -c%s "$file_path" 2>/dev/null || echo "0")
        local file_size_mb=$((file_size / 1024 / 1024))
        
        if [ "$file_size_mb" -ge "$min_size_mb" ]; then
            print_substep "$description: ✓ (${file_size_mb}MB)"
            return 0
        else
            print_warning "$description: File too small (${file_size_mb}MB < ${min_size_mb}MB)"
            return 1
        fi
    else
        print_error "$description: File not found - $file_path"
        return 1
    fi
}

validate_outputs() {
    print_step "Validating output files..."
    
    # Use absolute paths based on script directory
    local outputs_dir="$SCRIPT_DIR/protected_outputs/preprocessing"
    local errors=0
    
    # Core preprocessing outputs
    validate_file "$outputs_dir/icu_cohort.parquet" "ICU Cohort" 1 || ((errors++))
    validate_file "$outputs_dir/cohort_metadata.json" "Cohort Metadata" 0 || ((errors++))
    validate_file "$outputs_dir/by_event_wide_df.parquet" "Event-wide Dataset" 10 || ((errors++))
    validate_file "$outputs_dir/by_hourly_wide_df.parquet" "Hourly Dataset" 5 || ((errors++))
    validate_file "$outputs_dir/dataset_statistics.parquet" "Dataset Statistics" 0 || ((errors++))
    
    # Model-specific splits
    local intermediate_data="$SCRIPT_DIR/protected_outputs/intermediate/data"
    validate_file "$intermediate_data/train_df.parquet" "Training Data" 5 || ((errors++))
    validate_file "$intermediate_data/test_df.parquet" "Test Data" 1 || ((errors++))
    validate_file "$intermediate_data/split_metadata.json" "Split Metadata" 0 || ((errors++))
    
    # LSTM sequences
    local lstm_data="$SCRIPT_DIR/protected_outputs/intermediate/lstm"
    validate_file "$lstm_data/train_sequences.pkl" "LSTM Training Sequences" 1 || ((errors++))
    validate_file "$lstm_data/test_sequences.pkl" "LSTM Test Sequences" 0 || ((errors++))
    validate_file "$lstm_data/split_metadata.json" "LSTM Split Metadata" 0 || ((errors++))
    
    if [ $errors -eq 0 ]; then
        print_success "All output files validated successfully"
        return 0
    else
        print_error "$errors validation errors found"
        return 1
    fi
}

# ===============================================================================
# PIPELINE PHASES
# ===============================================================================

run_phase1_core_preprocessing() {
    print_header "PHASE 1: CORE PREPROCESSING"
    
    show_progress 0 3 "Starting core preprocessing..."
    
    # Step 1: Cohort Generation
    show_progress 1 3 "Generating ICU cohort..."
    execute_script "01_cohort.py" "ICU Cohort Generation"
    
    # Step 2: Feature Engineering
    show_progress 2 3 "Engineering features..."
    execute_script "02_feature_engineering.py" "Feature Engineering"
    
    # Step 3: Dataset Statistics
    show_progress 3 3 "Computing dataset statistics..."
    execute_script "03_dataset_statistics.py" "Dataset Statistics"
    
    print_success "Phase 1: Core preprocessing completed successfully"
}

run_phase2_data_splitting() {
    print_header "PHASE 2: MODEL-SPECIFIC DATA SPLITTING"
    
    show_progress 0 3 "Starting data splitting..."
    
    # Step 1: XGBoost Data Splitting
    show_progress 1 3 "Splitting XGBoost data..."
    print_step "Running XGBoost data splitter..."
    if source "$SCRIPT_DIR/flameICU/bin/activate" && \
       cd "$SCRIPT_DIR/code/preprocessing" && \
       python3 xgboost_data_splitter.py 2>&1 | tee -a "$LOG_FILE"; then
        print_success "XGBoost data splitting completed"
    else
        print_error "XGBoost data splitting failed"
        exit 1
    fi
    
    # Step 2: LSTM Data Splitting
    show_progress 2 3 "Splitting LSTM data..."
    print_step "Running LSTM data splitter..."
    if source "$SCRIPT_DIR/flameICU/bin/activate" && \
       cd "$SCRIPT_DIR/code/preprocessing" && \
       python3 lstm_data_split.py 2>&1 | tee -a "$LOG_FILE"; then
        print_success "LSTM data splitting completed"
    else
        print_error "LSTM data splitting failed"
        exit 1
    fi
    
    # Step 3: Neural Network Data Preparation
    show_progress 3 3 "Preparing NN data..."
    print_step "Neural Network data uses the same splits as XGBoost"
    print_substep "NN data preparation handled by nn_data_loader.py during training"
    
    print_success "Phase 2: Data splitting completed successfully"
}

run_phase3_validation_summary() {
    print_header "PHASE 3: VALIDATION & SUMMARY"
    
    show_progress 0 2 "Starting validation..."
    
    # Step 1: Validate Outputs
    show_progress 1 2 "Validating outputs..."
    if ! validate_outputs; then
        print_error "Output validation failed"
        exit 1
    fi
    
    # Step 2: Generate Summary
    show_progress 2 2 "Generating summary..."
    generate_summary_report
    
    print_success "Phase 3: Validation and summary completed successfully"
}

# ===============================================================================
# SUMMARY REPORTING
# ===============================================================================

generate_summary_report() {
    print_step "Generating comprehensive summary report..."
    
    local summary_file="${LOG_DIR}/preprocessing_summary_$(date +%Y%m%d_%H%M%S).txt"
    local current_time=$(date +%s)
    local total_duration=$((current_time - START_TIME))
    
    {
        echo "==============================================================================="
        echo "FLAME-ICU PREPROCESSING PIPELINE SUMMARY"
        echo "==============================================================================="
        echo "Execution Date: $(date)"
        echo "Total Duration: $(format_duration $total_duration)"
        echo "Working Directory: $SCRIPT_DIR"
        echo "Log File: $LOG_FILE"
        echo ""
        
        echo "PHASE 1: CORE PREPROCESSING"
        echo "  ✓ ICU Cohort Generation (01_cohort.py)"
        echo "  ✓ Feature Engineering (02_feature_engineering.py)"
        echo "  ✓ Dataset Statistics (03_dataset_statistics.py)"
        echo ""
        
        echo "PHASE 2: MODEL-SPECIFIC DATA SPLITTING"
        echo "  ✓ XGBoost Data Splitting (xgboost_data_splitter.py)"
        echo "  ✓ LSTM Data Splitting (lstm_data_split.py)"
        echo "  ✓ Neural Network Data Preparation"
        echo ""
        
        echo "PHASE 3: VALIDATION & SUMMARY"
        echo "  ✓ Output File Validation"
        echo "  ✓ Summary Report Generation"
        echo ""
        
        echo "OUTPUT FILES CREATED:"
        echo "Core Preprocessing:"
        
        local outputs_dir="$SCRIPT_DIR/protected_outputs/preprocessing"
        if [ -f "$outputs_dir/icu_cohort.parquet" ]; then
            local size=$(stat -f%z "$outputs_dir/icu_cohort.parquet" 2>/dev/null || stat -c%s "$outputs_dir/icu_cohort.parquet" 2>/dev/null)
            echo "  - ICU Cohort: icu_cohort.parquet ($((size / 1024))KB)"
        fi
        
        if [ -f "$outputs_dir/by_event_wide_df.parquet" ]; then
            local size=$(stat -f%z "$outputs_dir/by_event_wide_df.parquet" 2>/dev/null || stat -c%s "$outputs_dir/by_event_wide_df.parquet" 2>/dev/null)
            echo "  - Event-wide Dataset: by_event_wide_df.parquet ($((size / 1024 / 1024))MB)"
        fi
        
        if [ -f "$outputs_dir/by_hourly_wide_df.parquet" ]; then
            local size=$(stat -f%z "$outputs_dir/by_hourly_wide_df.parquet" 2>/dev/null || stat -c%s "$outputs_dir/by_hourly_wide_df.parquet" 2>/dev/null)
            echo "  - Hourly Dataset: by_hourly_wide_df.parquet ($((size / 1024 / 1024))MB)"
        fi
        
        echo ""
        echo "Model-Specific Splits:"
        
        local intermediate_data="$SCRIPT_DIR/protected_outputs/intermediate/data"
        if [ -f "$intermediate_data/train_df.parquet" ]; then
            local size=$(stat -f%z "$intermediate_data/train_df.parquet" 2>/dev/null || stat -c%s "$intermediate_data/train_df.parquet" 2>/dev/null)
            echo "  - Training Data: train_df.parquet ($((size / 1024 / 1024))MB)"
        fi
        
        if [ -f "$intermediate_data/test_df.parquet" ]; then
            local size=$(stat -f%z "$intermediate_data/test_df.parquet" 2>/dev/null || stat -c%s "$intermediate_data/test_df.parquet" 2>/dev/null)
            echo "  - Test Data: test_df.parquet ($((size / 1024 / 1024))MB)"
        fi
        
        local lstm_data="$SCRIPT_DIR/protected_outputs/intermediate/lstm"
        if [ -f "$lstm_data/train_sequences.pkl" ]; then
            local size=$(stat -f%z "$lstm_data/train_sequences.pkl" 2>/dev/null || stat -c%s "$lstm_data/train_sequences.pkl" 2>/dev/null)
            echo "  - LSTM Training Sequences: train_sequences.pkl ($((size / 1024 / 1024))MB)"
        fi
        
        echo ""
        echo "NEXT STEPS:"
        echo "1. Review the preprocessing outputs in ../protected_outputs/"
        echo "2. Check data quality using the dataset statistics"
        echo "3. Run model training scripts:"
        echo "   - LSTM: cd ../models/lstm && python3 training.py"
        echo "   - XGBoost: cd ../models/xgboost && python3 training.py"
        echo "   - Neural Network: cd ../models/nn && python3 training.py"
        echo ""
        echo "4. Use optimization scripts if needed:"
        echo "   - LSTM: cd ../models/lstm && ./run_optimization.sh"
        echo "   - XGBoost: cd ../models/xgboost && ./train_at_site.sh"
        echo ""
        echo "==============================================================================="
        echo "PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!"
        echo "==============================================================================="
        
    } | tee "$summary_file"
    
    print_success "Summary report saved to: $summary_file"
    print_info "Total execution time: $(format_duration $total_duration)"
}

# ===============================================================================
# DIRECTORY SETUP
# ===============================================================================

create_output_directories() {
    print_step "Setting up output directory structure..."
    
    # First ensure the protected_outputs base directory exists
    if [ ! -d "$SCRIPT_DIR/protected_outputs" ]; then
        mkdir -p "$SCRIPT_DIR/protected_outputs"
        print_substep "Created base directory: protected_outputs"
    fi
    
    # Create all required directories relative to the script location
    local dirs=(
        "protected_outputs/preprocessing"
        "protected_outputs/intermediate"
        "protected_outputs/intermediate/data"
        "protected_outputs/intermediate/lstm"
        "protected_outputs/intermediate/nn"
        "protected_outputs/models"
        "protected_outputs/models/lstm"
        "protected_outputs/models/xgboost"
        "protected_outputs/models/nn"
        "logs"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$SCRIPT_DIR/$dir" ]; then
            mkdir -p "$SCRIPT_DIR/$dir"
            print_substep "Created: $dir"
        else
            print_substep "Exists: $dir"
        fi
    done
    
    print_success "Output directory structure ready"
}

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

main() {
    # Print banner
    echo ""
    echo -e "${PURPLE}===============================================================================${NC}"
    echo -e "${PURPLE}               FLAME-ICU COMPLETE PREPROCESSING PIPELINE                      ${NC}"
    echo -e "${PURPLE}===============================================================================${NC}"
    echo ""
    print_info "Starting comprehensive preprocessing pipeline..."
    print_info "Log file: $LOG_FILE"
    print_info "Working directory: $SCRIPT_DIR"
    echo ""
    
    # Create output directories
    create_output_directories
    
    # Change to preprocessing directory for notebook execution
    cd "$SCRIPT_DIR/code/preprocessing"
    
    # Initialize log
    log "=== FLAME-ICU PREPROCESSING PIPELINE STARTED ==="
    log "Start time: $(date)"
    log "Working directory: $SCRIPT_DIR"
    
    # Check dependencies
    check_dependencies
    
    # Run pipeline phases
    run_phase1_core_preprocessing
    run_phase2_data_splitting
    run_phase3_validation_summary
    
    # Final success message
    local end_time=$(date +%s)
    local total_duration=$((end_time - START_TIME))
    
    echo ""
    print_header "PIPELINE COMPLETED SUCCESSFULLY!"
    print_success "Total execution time: $(format_duration $total_duration)"
    print_success "All preprocessing data is now ready for model training"
    print_info "Check the summary report in: ${LOG_DIR}/"
    echo ""
    
    log "=== FLAME-ICU PREPROCESSING PIPELINE COMPLETED ==="
    log "End time: $(date)"
    log "Total duration: $(format_duration $total_duration)"
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