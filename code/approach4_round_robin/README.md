# Approach 4: Round Robin Federated Training

## Overview

This approach implements collaborative sequential training where models are passed between sites, accumulating knowledge from multiple data sources while maintaining data privacy. The model learns from each site's data in sequence, creating a globally trained model.

## Who Runs What

### All Sites in Predetermined Order
- **Script**: `round_robin_train.py`
- **Purpose**: Sequential collaborative training
- **Order**: RUSH → Site A → Site B → Site C → Site D → Site E → Site F
- **Data**: Local 2018-2022 data at each site

## Training Sequence

1. **RUSH** starts training with their data (10-20 epochs)
2. **Site A** continues training the RUSH model with their data
3. **Site B** continues training the Site A model with their data
4. Continue until all sites have contributed
5. Final model tested at each site on 2023-2024 data

## Quick Start

### Check Your Site Position
```bash
# Site order: RUSH, SITE_A, SITE_B, SITE_C, SITE_D, SITE_E, SITE_F
```

### For Each Site (Run When Your Turn)
```bash
cd stage1/approach4_round_robin
python round_robin_train.py --site_name "YOUR_SITE_NAME" --round_number X
# Model automatically passed to next site in sequence
```

## Process Details

### For First Site (RUSH)
1. **Initialize Training**: Start with fresh model
2. **Train**: 10-20 epochs on RUSH data
3. **Pass Model**: Upload to BOX for Site A

### For Middle Sites
1. **Download Model**: Get model from previous site
2. **Continue Training**: 10-20 epochs on local data
3. **Pass Model**: Upload to BOX for next site

### For Last Site
1. **Download Model**: Get model from previous site
2. **Final Training**: 10-20 epochs on local data
3. **Final Model**: Upload completed round robin model

## Expected Outputs

- `round_robin_final_xgboost_model/` - Final collaborative XGBoost
- `round_robin_final_nn_model/` - Final collaborative Neural Network
- Training progress logs from all sites
- Performance tracking across rounds

## Round Robin Protocol

### Training Configuration
- **Epochs per Site**: 10-20 (adjusted by data size)
- **Learning Rate**: Decreases with each round
- **Regularization**: Increased to prevent overfitting to later sites
- **Validation**: Each site validates before passing

### Model Passing Rules
- **Secure Transfer**: Only model parameters (no data)
- **Validation Check**: Performance must improve or maintain
- **Progress Tracking**: Log performance at each round
- **Timeout**: 1 week maximum per site

## Configuration

Key parameters in `config.json`:
```json
{
  "round_robin": {
    "epochs_per_site": 15,
    "learning_rate_decay": 0.9,
    "regularization_increase": 1.1,
    "validation_patience": 5,
    "passing_timeout_days": 7
  }
}
```

## Coordination Requirements

### Site Responsibilities
- **Monitor BOX folder** for incoming models
- **Train within 1 week** of receiving model
- **Validate performance** before passing
- **Upload promptly** to next site
- **Report status** to coordination team

### Communication Protocol
- Email notification when model passed
- Slack updates in #flame-icu-federated
- Weekly progress meetings
- Issue escalation to coordination team

## Data Requirements

- **Training Data**: Local 2018-2022 data (each site)
- **Testing Data**: Local 2023-2024 data (final evaluation)
- **Validation Data**: 20% of training for monitoring
- **Model Compatibility**: Consistent feature sets

## Success Criteria

- All sites successfully participate in sequence
- Model performance improves or maintains across rounds
- Final model uploaded to BOX successfully
- No data privacy violations

## Expected Outcomes

- **Globally Trained Model**: Knowledge from all participating sites
- **Training Dynamics**: Analysis of how performance evolves
- **Site Contribution**: Understanding of each site's impact
- **Collaborative Benefits**: Comparison with independent training

## Troubleshooting

**Common Issues:**
- **Model not received**: Check BOX folder permissions
- **Performance degradation**: Check learning rate and regularization
- **Site delays**: Contact coordination team for timeline adjustment
- **Compatibility errors**: Verify feature alignment

**Emergency Protocols:**
- **Site dropout**: Skip to next site in sequence
- **Performance collapse**: Revert to previous round model
- **Technical issues**: Coordination team intervention

## Performance Monitoring

Each site tracks:
- **Local Validation Loss**: Monitor overfitting
- **Cross-validation AUC**: Performance stability
- **Training Time**: Resource utilization
- **Model Size**: Parameter growth

## Timeline

- **Week 1**: RUSH initialization
- **Week 2**: Sites A-B training
- **Week 3**: Sites C-D training
- **Week 4**: Sites E-F training
- **Week 5**: Final testing and analysis

## Quality Assurance

Before passing model:
- Validate performance hasn't degraded >5%
- Check model serialization integrity
- Verify training logs completeness
- Confirm next site notification

## Support

For issues:
- **Technical problems**: flame-icu-coordination@rush.edu
- **Timeline delays**: Notify coordination team immediately
- **Performance issues**: Check training logs and parameters
- **BOX access**: Verify permissions with IT team

## Advanced Features

- **Adaptive Learning Rate**: Automatically adjusts based on performance
- **Early Stopping**: Prevents overfitting at each site
- **Performance Tracking**: Real-time monitoring across sites
- **Rollback Capability**: Revert to previous round if needed