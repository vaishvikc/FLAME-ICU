#!/bin/bash
# train_at_site.sh - Complete training pipeline for XGBoost ICU mortality model
# This script runs the full pipeline: data splitting, hyperparameter optimization, and final training

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_step() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${GREEN}$1${NC}"
}

print_error() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${RED}ERROR: $1${NC}"
}

# Start pipeline
echo "=============================================="
echo "XGBoost ICU Mortality Model Training Pipeline"
echo "=============================================="
print_step "Starting pipeline at $(date)"

# Step 1: Data Splitting
print_step "Step 1/3: Splitting data and creating aggregated features..."
python ../../preprocessing/xgboost_data_splitter.py
if [ $? -eq 0 ]; then
    print_step "✅ Data splitting completed successfully"
else
    print_error "Data splitting failed"
    exit 1
fi

# Step 2: Hyperparameter Optimization
print_step "Step 2/3: Running hyperparameter optimization (using training data only)..."
python hyperparameter_optimization.py
if [ $? -eq 0 ]; then
    print_step "✅ Hyperparameter optimization completed successfully"
    print_step "Optimal parameters saved to config.json"
else
    print_error "Hyperparameter optimization failed"
    exit 1
fi

# Step 3: Final Training with Optimal Parameters
print_step "Step 3/3: Training final model with optimal parameters..."
python training.py
if [ $? -eq 0 ]; then
    print_step "✅ Final model training completed successfully"
else
    print_error "Final model training failed"
    exit 1
fi

# Summary
echo ""
echo "=============================================="
print_step "Pipeline completed successfully!"
echo "=============================================="
echo ""
echo "Results summary:"
echo "- Train/test splits created in: ../../protected_outputs/intermediate/data/"
echo "- Optimization plots saved in: ../../protected_outputs/models/xgboost/optimization_plots/"
echo "- Final model saved in: ../../protected_outputs/models/xgboost/"
echo "- Evaluation plots saved in: ../../protected_outputs/models/xgboost/plots/"
echo ""
print_step "Total execution time: $SECONDS seconds"