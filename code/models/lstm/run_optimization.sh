#!/bin/bash

# LSTM Model Optimization Pipeline
# This script runs the complete LSTM optimization workflow:
# 1. Data splitting
# 2. Architecture exploration and hyperparameter optimization
# 3. Optional: Final model training

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

# Header
echo "=========================================="
echo "    LSTM Model Optimization Pipeline      "
echo "=========================================="
echo ""

# Check Python environment
print_status "Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found. Please install Python 3.7+"
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python version: $python_version"

# Step 1: Data Splitting
echo ""
echo "=========================================="
echo "    Step 1: Data Splitting                "
echo "=========================================="
print_status "Creating train/test splits for LSTM..."

if python3 data_split.py; then
    print_success "Data splitting completed successfully"
else
    print_error "Data splitting failed"
    exit 1
fi

# Step 2: Architecture & Hyperparameter Optimization
echo ""
echo "=========================================="
echo "    Step 2: Model Optimization            "
echo "=========================================="
print_status "Running architecture exploration and hyperparameter optimization..."
print_warning "This may take 30-60 minutes depending on your system"

if python3 optimize.py; then
    print_success "Optimization completed successfully"
else
    print_error "Optimization failed"
    exit 1
fi

# Step 3: Optional Training
echo ""
echo "=========================================="
echo "    Step 3: Final Model Training          "
echo "=========================================="

read -p "Do you want to train the final model with optimized parameters? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Training final LSTM model..."
    
    if python3 training.py; then
        print_success "Training completed successfully"
    else
        print_error "Training failed"
        exit 1
    fi
else
    print_warning "Skipping final training. You can run it later with: python3 training.py"
fi

# Summary
echo ""
echo "=========================================="
echo "    Pipeline Complete!                    "
echo "=========================================="
print_success "All optimization steps completed successfully"
echo ""
echo "Results saved in:"
echo "  - Data splits: ../../protected_outputs/intermediate/lstm/"
echo "  - Architecture results: ../../protected_outputs/models/lstm/architecture_exploration/"
echo "  - Optimization plots: ../../protected_outputs/models/lstm/optimization_plots/"
echo "  - Updated config: config.json"
echo ""
echo "Next steps:"
echo "  1. Review the optimization results"
echo "  2. Run 'python3 training.py' if you haven't already"
echo "  3. Use 'python3 inference.py' for predictions"
echo ""