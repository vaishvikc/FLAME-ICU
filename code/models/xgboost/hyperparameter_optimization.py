import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, 
    brier_score_loss, classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_config():
    """Load configuration from config file"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config):
    """Save configuration back to config file"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def load_preprocessing_config():
    """Load preprocessing configuration to get site name"""
    preprocessing_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'preprocessing', 'config_demo.json'
    )
    try:
        with open(preprocessing_config_path, 'r') as f:
            preprocessing_config = json.load(f)
        return preprocessing_config
    except FileNotFoundError:
        print(f"Warning: Preprocessing config not found at {preprocessing_config_path}")
        return {"site": "unknown"}

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def load_presplit_data():
    """Load pre-split training data for optimization"""
    print("Loading pre-split training data...")
    
    # Load configuration
    config = load_config()
    preprocessing_config = load_preprocessing_config()
    site_name = preprocessing_config.get('site', 'unknown')
    
    # Get train file path from config
    train_file = config['data_split']['train_file']
    
    # Load training data
    print(f"Loading training data from: {train_file}")
    train_df = pd.read_parquet(train_file)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Number of unique hospitalization_ids: {train_df['hospitalization_id'].nunique()}")
    
    # Prepare features and target
    X = train_df.drop(['hospitalization_id', 'disposition'], axis=1)
    y = train_df['disposition']

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Training mortality rate: {y.mean():.3f}")
    
    return X, y, site_name

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    global X_train, y_train, feature_names
    
    # Suggest hyperparameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5),
        'seed': 42,
        'verbosity': 0
    }
    
    # Add scale_pos_weight for imbalanced data
    scale_pos_weight = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))
    params['scale_pos_weight'] = scale_pos_weight
    
    # 5-fold stratified cross-validation
    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Scale features (let XGBoost handle NaN values natively)
        scaler = StandardScaler()
        X_fold_train_scaled = scaler.fit_transform(X_fold_train)
        X_fold_val_scaled = scaler.transform(X_fold_val)
        
        # Create DMatrix
        dtrain_fold = xgb.DMatrix(X_fold_train_scaled, label=y_fold_train)
        dval_fold = xgb.DMatrix(X_fold_val_scaled, label=y_fold_val)
        
        # Train model
        model = xgb.train(
            params, 
            dtrain_fold, 
            num_boost_round=500,
            evals=[(dtrain_fold, 'train'), (dval_fold, 'eval')],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        # Get predictions and calculate ROC-AUC
        y_pred_proba = model.predict(dval_fold)
        roc_auc = roc_auc_score(y_fold_val, y_pred_proba)
        cv_scores.append(roc_auc)
    
    return np.mean(cv_scores)

def hyperparameter_optimization():
    """Main function for hyperparameter optimization"""
    global X_train, y_train, feature_names
    
    print("=== XGBoost Hyperparameter Optimization ===")
    
    # Load pre-split training data
    X_train, y_train, site_name = load_presplit_data()
    
    # Store feature names globally for the objective function
    feature_names = list(X_train.columns)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Starting hyperparameter optimization with {len(feature_names)} features...")
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize hyperparameters
    print("Running Optuna optimization (this may take several minutes)...")
    study.optimize(objective, n_trials=20, timeout=600)  # 10 minutes timeout, 20 trials for demo
    
    print(f"Best ROC-AUC: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best parameters to config
    config = load_config()
    
    # Update model_params with best parameters
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 42
    })
    
    config['model_params'] = best_params
    
    # Add optimization metadata
    config['optimization_results'] = {
        'best_roc_auc': float(study.best_value),
        'n_trials': len(study.trials),
        'optimization_date': pd.Timestamp.now().isoformat()
    }
    
    save_config(config)
    print(f"✅ Best parameters saved to config.json")
    print(f"Best cross-validation ROC-AUC: {study.best_value:.4f}")
    
    # Create output directory for optimization plots  
    plots_dir = "../../protected_outputs/models/xgboost/optimization_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot optimization history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('Optimization History')
    plt.savefig(os.path.join(plots_dir, f'optimization_history_{site_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot parameter importance
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('Parameter Importance')
    plt.savefig(os.path.join(plots_dir, f'param_importance_{site_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Optimization complete!")
    print(f"Plots saved to: {plots_dir}")
    print(f"Updated config saved with best parameters")
    print(f"Run training.py to train the final model with optimized parameters")
    
    return study.best_params

if __name__ == "__main__":
    hyperparameter_optimization()