import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
import xgboost as xgb
# Removed train_test_split import - using pre-split data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)

def load_config():
    """Load configuration from config file"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

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

def train_xgboost_model():
    """Train XGBoost model for ICU mortality prediction using pre-split data"""
    # Load configuration
    config = load_config()
    
    # Load preprocessing config to get site name
    preprocessing_config = load_preprocessing_config()
    site_name = preprocessing_config.get('site', 'unknown')
    print(f"Training model for site: {site_name}")
    
    # Make a copy of output config to modify
    output_config = config['output_config'].copy()
    
    # Update output paths with site name
    for key in output_config:
        if isinstance(output_config[key], str):
            # Replace the model name with site-specific name
            output_config[key] = output_config[key].replace(
                'xgb_icu_mortality_model', f'xgb_{site_name}_icu_mortality_model'
            ).replace(
                'xgb_feature_scaler', f'xgb_{site_name}_feature_scaler'
            ).replace(
                'xgb_feature_columns', f'xgb_{site_name}_feature_columns'
            ).replace(
                'metrics.json', f'{site_name}_metrics.json'
            )
    
    # Create output directories
    output_model_dir = os.path.dirname(output_config['model_path'])
    plots_dir = output_config['plots_dir']
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load pre-split data
    print("Loading pre-split data...")
    train_file = config['data_split']['train_file']
    test_file = config['data_split']['test_file']
    
    print(f"Loading training data from: {train_file}")
    train_df = pd.read_parquet(train_file)
    print(f"Training data shape: {train_df.shape}")
    
    print(f"Loading test data from: {test_file}")
    test_df = pd.read_parquet(test_file)
    print(f"Test data shape: {test_df.shape}")
    
    # Prepare features and targets
    X_train = train_df.drop(['hospitalization_id', 'disposition'], axis=1)
    y_train = train_df['disposition']
    X_test = test_df.drop(['hospitalization_id', 'disposition'], axis=1)
    y_test = test_df['disposition']
    
    print(f"Training set: {X_train.shape} features, {len(y_train)} patients")
    print(f"Test set: {X_test.shape} features, {len(y_test)} patients")
    print(f"Training mortality rate: {y_train.mean():.3f}")
    print(f"Test mortality rate: {y_test.mean():.3f}")

    # Scale features and handle NaN values
    print("Checking for missing values before scaling...")
    print(f"NaN values in training set: {np.isnan(X_train).sum()}")
    print(f"NaN values in test set: {np.isnan(X_test).sum()}")

    # Fill NaN values with mean of column
    X_train_filled = np.nan_to_num(X_train, nan=0.0)
    X_test_filled = np.nan_to_num(X_test, nan=0.0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filled)
    X_test_scaled = scaler.transform(X_test_filled)

    print("Checking for NaN values after scaling...")
    print(f"NaN values in scaled training set: {np.isnan(X_train_scaled).sum()}")
    print(f"NaN values in scaled test set: {np.isnan(X_test_scaled).sum()}")

    # Train XGBoost model
    print("Training XGBoost model...")

    # Calculate class weights for handling imbalanced data
    scale_pos_weight = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))

    # Set up XGBoost parameters from config
    params = config['model_params']
    params['scale_pos_weight'] = scale_pos_weight

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)

    # Set up evaluation list
    eval_list = [(dtrain, 'train'), (dtest, 'eval')]

    # Train model with early stopping
    num_rounds = config['training_config']['num_rounds']
    early_stopping_rounds = config['training_config']['early_stopping_rounds']
    print("Training model with cross-validation...")

    # Create a watchlist for monitoring
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]

    # Train the model
    model = xgb.train(
        params, 
        dtrain, 
        num_rounds,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50
    )

    print(f"Best iteration: {model.best_iteration}")

    # Evaluate model
    print("Evaluating model...")

    # Get predictions on test set
    y_pred_proba = model.predict(dtest)

    # Print raw probabilities to debug
    print("\nPredicted probabilities (first 10):")
    print(y_pred_proba[:10])

    # Try different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print("\nEvaluating different thresholds:")
    for threshold in thresholds:
        y_pred_t = (y_pred_proba > threshold).astype(int)
        acc = accuracy_score(y_test, y_pred_t)
        print(f"Threshold {threshold}: accuracy = {acc:.4f}, predictions sum = {y_pred_t.sum()}")

    # Use the standard threshold for the main evaluation
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    brier_score = brier_score_loss(y_test, y_pred_proba)
    ece = expected_calibration_error(y_test, y_pred_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    print(f"Expected Calibration Error: {ece:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Save comprehensive metrics
    metrics = {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'brier_score': float(brier_score),
        'expected_calibration_error': float(ece),
        'confusion_matrix': cm.tolist(),
        'best_iteration': model.best_iteration
    }
    
    with open(output_config['metrics_path'], 'w') as f:
        json.dump(metrics, f, indent=2)

    # Enhanced feature importance visualization with original column names
    try:
        # Get feature importance with original names
        importance_types = ['weight', 'gain', 'cover']
        
        for importance_type in importance_types:
            importance = model.get_score(importance_type=importance_type)
            
            if importance:
                # Map XGBoost feature names back to original column names
                feature_names = list(X_train.columns)
                mapped_importance = {}
                
                for xgb_feature, score in importance.items():
                    # XGBoost uses f0, f1, f2, ... for feature names
                    if xgb_feature.startswith('f'):
                        try:
                            feature_idx = int(xgb_feature[1:])
                            if feature_idx < len(feature_names):
                                original_name = feature_names[feature_idx]
                                mapped_importance[original_name] = score
                        except (ValueError, IndexError):
                            mapped_importance[xgb_feature] = score
                    else:
                        mapped_importance[xgb_feature] = score
                
                # Sort by importance and get top 20
                sorted_importance = sorted(mapped_importance.items(), key=lambda x: x[1], reverse=True)[:20]
                
                if sorted_importance:
                    # Create horizontal bar plot
                    plt.figure(figsize=(12, 8))
                    features, scores = zip(*sorted_importance)
                    y_pos = np.arange(len(features))
                    
                    plt.barh(y_pos, scores)
                    plt.yticks(y_pos, features)
                    plt.xlabel(f'Feature Importance ({importance_type.title()})')
                    plt.title(f'Top 20 Features - XGBoost Feature Importance ({importance_type.title()})')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    
                    # Save plot
                    plt.savefig(os.path.join(plots_dir, f'xgb_{site_name}_feature_importance_{importance_type}.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Feature importance plot ({importance_type}) saved successfully.")
                    
                    # Print top 10 features
                    print(f"\nTop 10 Features ({importance_type}):")
                    for i, (feature, score) in enumerate(sorted_importance[:10], 1):
                        print(f"{i:2d}. {feature}: {score:.3f}")
            else:
                print(f"Warning: No feature importance ({importance_type}) could be calculated.")
                
    except Exception as e:
        print(f"Warning: Could not create feature importance plots: {e}")
        print("Printing feature names for reference:")
        for i, col in enumerate(X_train.columns[:20]):
            print(f"{i:2d}: {col}")

    # Enhanced calibration analysis
    try:
        print("\n=== Calibration Analysis ===")
        
        # Create reliability diagram
        plt.figure(figsize=(15, 5))
        
        # Subplot 1: Calibration curve (reliability diagram)
        plt.subplot(1, 3, 1)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=10
        )
        
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"XGBoost (ECE={ece:.4f})")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Reliability Diagram")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Prediction histogram
        plt.subplot(1, 3, 2)
        plt.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, label='Survived', density=True)
        plt.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, label='Died', density=True)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Density")
        plt.title("Prediction Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Brier score decomposition
        plt.subplot(1, 3, 3)
        # Calculate Brier score components
        reliability = np.sum((fraction_of_positives - mean_predicted_value) ** 2 * 
                           np.histogram(y_pred_proba, bins=10)[0] / len(y_pred_proba))
        resolution = np.sum((fraction_of_positives - np.mean(y_test)) ** 2 * 
                          np.histogram(y_pred_proba, bins=10)[0] / len(y_pred_proba))
        uncertainty = np.mean(y_test) * (1 - np.mean(y_test))
        
        components = ['Reliability', 'Resolution', 'Uncertainty', 'Brier Score']
        values = [reliability, -resolution, uncertainty, brier_score]  # Resolution is subtracted in Brier
        colors = ['red', 'green', 'blue', 'orange']
        
        bars = plt.bar(components, values, color=colors, alpha=0.7)
        plt.title("Brier Score Decomposition")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'xgb_{site_name}_calibration_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Enhanced calibration analysis saved successfully.")
        
        # Print calibration summary
        print(f"Calibration Summary:")
        print(f"  Expected Calibration Error (ECE): {ece:.4f}")
        print(f"  Brier Score: {brier_score:.4f}")
        print(f"  Reliability: {reliability:.4f}")
        print(f"  Resolution: {resolution:.4f}")
        print(f"  Uncertainty: {uncertainty:.4f}")
        
    except Exception as e:
        print(f"Warning: Could not create calibration analysis: {e}")
        print("Skipping calibration analysis.")

    # Save the trained model
    model_path = output_config['model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Save the scaler for preprocessing new data
    scaler_path = output_config['scaler_path']
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    # Save feature column names for inference
    feature_cols_path = output_config['feature_cols_path']
    os.makedirs(os.path.dirname(feature_cols_path), exist_ok=True)
    with open(feature_cols_path, 'wb') as f:
        pickle.dump(list(X_train.columns), f)
    print(f"Feature columns saved to {feature_cols_path}")

    print(f"Evaluation completed. Results saved in {plots_dir}")
    
    return model, scaler, list(X_train.columns)

def main():
    """Main function to train the XGBoost model"""
    train_xgboost_model()

if __name__ == "__main__":
    main()