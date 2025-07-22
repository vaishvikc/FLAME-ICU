# Hospital Sites in WUPERR Federated Learning

This document provides detailed information about the 8 hospital sites participating in the WUPERR federated learning system for ICU mortality prediction.

## Site Overview

The federated learning system simulates a realistic healthcare network with varying hospital sizes, patient populations, and resource availability. Sites are ordered by size and capability, with Site 1 representing the main teaching hospital (equivalent to RUSH).

## Detailed Site Characteristics

### Site 1: Large Academic Medical Center (Main Site - RUSH Equivalent)
- **Type**: Large Academic Hospital
- **Data Proportion**: 25% of total data
- **Patient Characteristics**: High acuity, complex cases
- **Bias Implementation**: 
  - Increased representation of mortality cases
  - Higher severity scores
  - More complete lab data
- **Role in Federated Learning**: 
  - Initiates model training in each round
  - Sets the baseline model performance
  - Has the most comprehensive data

### Site 2: Large Community Hospital
- **Type**: Large Community Hospital
- **Data Proportion**: 20% of total data
- **Patient Characteristics**: Balanced patient mix
- **Bias Implementation**: 
  - Representative sample of the overall population
  - No specific bias applied
- **Role in Federated Learning**: 
  - Provides stable, generalizable updates
  - Helps maintain model balance

### Site 3: Medium Academic Medical Center
- **Type**: Medium Academic Hospital
- **Data Proportion**: 15% of total data
- **Patient Characteristics**: Research-focused cases
- **Bias Implementation**: 
  - Enhanced data quality
  - Complete feature sets
  - Research protocol patients
- **Role in Federated Learning**: 
  - Contributes high-quality training data
  - Helps improve model precision

### Site 4: Medium Community Hospital
- **Type**: Medium Community Hospital
- **Data Proportion**: 15% of total data
- **Patient Characteristics**: Balanced community patients
- **Bias Implementation**: 
  - Standard patient distribution
  - Typical community hospital cases
- **Role in Federated Learning**: 
  - Provides community hospital perspective
  - Enhances model generalizability

### Site 5: Small Community Hospital
- **Type**: Small Community Hospital
- **Data Proportion**: 10% of total data
- **Patient Characteristics**: Lower acuity patients
- **Bias Implementation**: 
  - Reduced mortality rate (5% vs general ~10%)
  - Less severe cases
  - Shorter ICU stays
- **Role in Federated Learning**: 
  - Helps model learn from less severe cases
  - Prevents overfitting to high-acuity patients

### Site 6: Small Rural Hospital
- **Type**: Small Rural Hospital
- **Data Proportion**: 8% of total data
- **Patient Characteristics**: Limited resources
- **Bias Implementation**: 
  - 30% missing lab values
  - Incomplete feature sets
  - Resource-constrained care patterns
- **Role in Federated Learning**: 
  - Tests model robustness to missing data
  - Represents resource-limited settings

### Site 7: Specialty Hospital
- **Type**: Specialty Hospital
- **Data Proportion**: 5% of total data
- **Patient Characteristics**: Specific condition focus
- **Bias Implementation**: 
  - Boosted cardiac-related features (20% increase)
  - Specialized patient population
  - Focused on specific organ systems
- **Role in Federated Learning**: 
  - Adds specialized knowledge
  - Helps model with specific conditions

### Site 8: Critical Access Hospital
- **Type**: Critical Access Hospital
- **Data Proportion**: 2% of total data
- **Patient Characteristics**: Basic care only
- **Bias Implementation**: 
  - Most basic patient cases
  - Limited diagnostic capabilities
  - Transfers complex cases
- **Role in Federated Learning**: 
  - Represents smallest facilities
  - Tests model on minimal data

## Data Distribution Strategy

The data split follows a realistic healthcare system distribution:
- **Large hospitals (Sites 1-2)**: 45% of data - representing major medical centers
- **Medium hospitals (Sites 3-4)**: 30% of data - typical regional hospitals
- **Small hospitals (Sites 5-6)**: 18% of data - community facilities
- **Specialized/Critical (Sites 7-8)**: 7% of data - niche providers

## Sequential Training Order

The training proceeds sequentially in each round:
```
Round 1: Site 1 → Site 2 → Site 3 → ... → Site 8
Round 2: Site 1 → Site 2 → Site 3 → ... → Site 8
Round 3: Site 1 → Site 2 → Site 3 → ... → Site 8
```

This order ensures:
1. The largest, most resourced site initiates training
2. Model complexity builds progressively
3. Smaller sites benefit from larger sites' knowledge
4. Each round refines the global model

## Site-Specific Biases

### High Acuity Bias (Site 1)
```python
# Increases mortality cases by 10%
# Simulates tertiary care center with sickest patients
```

### Low Acuity Bias (Site 5)
```python
# Reduces mortality rate to 5%
# Simulates community hospital with stable patients
```

### Limited Resources Bias (Site 6)
```python
# Sets 30% of lab values to missing
# Simulates rural hospital constraints
```

### Specific Conditions Bias (Site 7)
```python
# Boosts cardiac features by 20%
# Simulates cardiac specialty hospital
```

## Model Evolution Across Sites

As the model trains sequentially:

1. **Site 1**: Establishes strong baseline with high-quality data
2. **Sites 2-4**: Generalizes model to broader populations
3. **Sites 5-6**: Adapts to resource constraints and missing data
4. **Sites 7-8**: Fine-tunes for edge cases and specialties

## Performance Expectations

Typical performance patterns:
- **Best AUC**: Usually achieved at Sites 1-3 (most data, best quality)
- **Most Robust**: After Site 6 (handles missing data well)
- **Most Specialized**: After Site 7 (cardiac case improvements)
- **Final Model**: Balances all site contributions

## Using Site Data

### Access Site Data
```bash
# After running simulate_multisite.py
ls data/sites/
# site_1_data.parquet
# site_2_data.parquet
# ...
# site_metadata.json
```

### Load Site Metadata
```python
import json
with open('data/sites/site_metadata.json', 'r') as f:
    metadata = json.load(f)
    
# View Site 1 statistics
print(metadata['site_1'])
```

### Train Specific Site
```bash
# Train only Site 3 for Round 2
python sequential_train.py --site_id 3 --round_num 2
```

## Customizing Sites

To modify site characteristics, edit `simulate_multisite.py`:

```python
# In MultisiteSimulator.__init__()
self.site_config = {
    1: {'proportion': 0.25, 'type': 'large_academic', 'bias': 'high_acuity'},
    # Modify proportions, types, or biases here
}
```

To add new bias types, implement in `create_site_specific_bias()` method.