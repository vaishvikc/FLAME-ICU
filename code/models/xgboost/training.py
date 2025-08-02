import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
import xgboost as xgb
# Removed train_test_split import - using pre-split data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix, 
    average_precision_score, brier_score_loss, roc_curve, precision_recall_curve
)
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
    # Get the absolute path to the project root (3 levels up from this script)
    # Script is at: code/models/xgboost/training.py
    script_dir = os.path.dirname(os.path.abspath(__file__))  # code/models/xgboost
    models_dir = os.path.dirname(script_dir)  # code/models
    code_dir = os.path.dirname(models_dir)  # code
    project_root = os.path.dirname(code_dir)  # project root
    
    # Try top-level config_demo.json first (new location)
    preprocessing_config_path = os.path.join(project_root, 'config_demo.json')
    
    if not os.path.exists(preprocessing_config_path):
        # Fallback to old location
        preprocessing_config_path = os.path.join(project_root, 'preprocessing', 'config_demo.json')
    
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


def plot_training_history(evals_result, best_iteration, plots_dir, site_name):
    """Plot XGBoost training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Extract training history
    train_loss = evals_result['train']['logloss']
    val_loss = evals_result['eval']['logloss']
    epochs = list(range(len(train_loss)))
    
    # Loss plot
    axes[0].plot(epochs, train_loss, label='Train Loss', alpha=0.8)
    axes[0].plot(epochs, val_loss, label='Val Loss', alpha=0.8)
    axes[0].axvline(x=best_iteration, color='r', linestyle='--', alpha=0.5, label=f'Best Iteration ({best_iteration})')
    axes[0].set_xlabel('Boosting Round')
    axes[0].set_ylabel('Log Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning curve zoom (last 50% of training)
    start_idx = len(epochs) // 2
    axes[1].plot(epochs[start_idx:], train_loss[start_idx:], label='Train Loss', alpha=0.8)
    axes[1].plot(epochs[start_idx:], val_loss[start_idx:], label='Val Loss', alpha=0.8)
    if best_iteration >= start_idx:
        axes[1].axvline(x=best_iteration, color='r', linestyle='--', alpha=0.5, label=f'Best Iteration ({best_iteration})')
    axes[1].set_xlabel('Boosting Round')
    axes[1].set_ylabel('Log Loss')
    axes[1].set_title('Training and Validation Loss (Zoomed)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'xgb_{site_name}_training_history.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_evaluation_metrics(y_true, y_pred_proba, plots_dir, site_name):
    """Create comprehensive evaluation plots similar to NN model"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. ROC Curve
    ax1 = plt.subplot(3, 3, 1)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    ax1.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    ax2 = plt.subplot(3, 3, 2)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    ax2.plot(recall, precision, label=f'PR (AUC = {pr_auc:.3f})', linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Calibration Plot
    ax3 = plt.subplot(3, 3, 3)
    # Determine number of bins based on dataset size
    n_bins = min(10, len(y_true) // 10)
    if n_bins < 3:
        n_bins = 3
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
    )
    ece = expected_calibration_error(y_true, y_pred_proba, n_bins=n_bins)
    ax3.plot(mean_predicted_value, fraction_of_positives, "s-", 
             label=f'XGBoost (ECE={ece:.4f})', linewidth=2, markersize=8)
    ax3.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)
    ax3.set_xlabel('Mean Predicted Probability')
    ax3.set_ylabel('Fraction of Positives')
    ax3.set_title('Calibration Plot')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Confusion Matrix
    ax4 = plt.subplot(3, 3, 4)
    y_pred = (y_pred_proba > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, 
                xticklabels=['Survived', 'Died'],
                yticklabels=['Survived', 'Died'])
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    ax4.set_title('Confusion Matrix')
    
    # 5. Prediction Distribution
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.6, label='Survived', density=True, color='blue')
    ax5.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.6, label='Died', density=True, color='red')
    ax5.axvline(x=0.5, color='k', linestyle='--', alpha=0.5, label='Default threshold')
    ax5.set_xlabel('Predicted Probability')
    ax5.set_ylabel('Density')
    ax5.set_title('Prediction Distribution by Outcome')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Threshold Analysis
    ax6 = plt.subplot(3, 3, 6)
    thresholds = np.linspace(0, 1, 100)
    sensitivities = []
    specificities = []
    accuracies = []
    
    for threshold in thresholds:
        y_pred_t = (y_pred_proba > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        accuracies.append(accuracy)
    
    ax6.plot(thresholds, sensitivities, label='Sensitivity', linewidth=2)
    ax6.plot(thresholds, specificities, label='Specificity', linewidth=2)
    ax6.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
    ax6.axvline(x=0.5, color='k', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Threshold')
    ax6.set_ylabel('Metric Value')
    ax6.set_title('Threshold Analysis')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Score Distribution (Box plot)
    ax7 = plt.subplot(3, 3, 7)
    data_for_box = [y_pred_proba[y_true == 0], y_pred_proba[y_true == 1]]
    box = ax7.boxplot(data_for_box, tick_labels=['Survived', 'Died'], patch_artist=True)
    box['boxes'][0].set_facecolor('blue')
    box['boxes'][1].set_facecolor('red')
    ax7.set_ylabel('Predicted Probability')
    ax7.set_title('Score Distribution by Outcome')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Performance Summary Text
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    brier_score = brier_score_loss(y_true, y_pred_proba)
    
    summary_text = f"""Performance Summary:
    
    ROC AUC: {auc:.4f}
    PR AUC: {pr_auc:.4f}
    Brier Score: {brier_score:.4f}
    ECE: {ece:.4f}
    
    Accuracy: {accuracy_score(y_true, y_pred):.4f}
    Sensitivity: {sensitivities[50]:.4f}
    Specificity: {specificities[50]:.4f}
    
    Class Distribution:
    Survived: {(y_true == 0).sum()} ({(y_true == 0).mean():.1%})
    Died: {(y_true == 1).sum()} ({(y_true == 1).mean():.1%})
    """
    
    ax8.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center', transform=ax8.transAxes)
    
    # 9. Probability Calibration Histogram
    ax9 = plt.subplot(3, 3, 9)
    ax9.hist(y_pred_proba, bins=20, alpha=0.7, edgecolor='black')
    ax9.set_xlabel('Predicted Probability')
    ax9.set_ylabel('Count')
    ax9.set_title('Distribution of Predicted Probabilities')
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'xgb_{site_name}_evaluation_metrics.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

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
    
    # Update output paths to use site-specific directory structure
    for key in output_config:
        if isinstance(output_config[key], str):
            # Replace {SITE_NAME} placeholder with actual site name in paths
            output_config[key] = output_config[key].replace('/{SITE_NAME}/', f'/{site_name}/')
    
    # Convert output paths to absolute paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for key in output_config:
        if isinstance(output_config[key], str) and not os.path.isabs(output_config[key]):
            output_config[key] = os.path.abspath(os.path.join(script_dir, output_config[key]))
    
    # Create output directories
    output_model_dir = os.path.dirname(output_config['model_path'])
    plots_dir = output_config['plots_dir']
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load pre-split data
    print("Loading pre-split data...")
    train_file = config['data_split']['train_file']
    test_file = config['data_split']['test_file']
    
    # Convert relative paths to absolute paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(train_file):
        train_file = os.path.abspath(os.path.join(script_dir, train_file))
    if not os.path.isabs(test_file):
        test_file = os.path.abspath(os.path.join(script_dir, test_file))
    
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

    # Handle scaling based on configuration
    use_scaling = config['training_config'].get('use_scaling', True)
    print(f"Scaling enabled: {use_scaling}")
    
    if use_scaling:
        print("Applying StandardScaler to features...")
        # Check for NaN values
        print(f"NaN values in training set: {X_train.isna().sum().sum()}")
        print(f"NaN values in test set: {X_test.isna().sum().sum()}")
        
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)
        
        print("Scaling completed.")
    else:
        print("Scaling disabled, using raw features...")
        X_train_processed = X_train.values
        X_test_processed = X_test.values
        scaler = None

    # Train XGBoost model
    print("Training XGBoost model...")

    # Calculate class weights for handling imbalanced data
    scale_pos_weight = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))

    # Set up XGBoost parameters from config
    params = config['model_params']
    params['scale_pos_weight'] = scale_pos_weight

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_processed, label=y_train)
    dtest = xgb.DMatrix(X_test_processed, label=y_test)

    # Set up evaluation list
    eval_list = [(dtrain, 'train'), (dtest, 'eval')]

    # Train model with early stopping
    num_rounds = config['training_config']['num_rounds']
    early_stopping_rounds = config['training_config']['early_stopping_rounds']
    print("Training model with cross-validation...")

    # Create a watchlist for monitoring
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]

    # Train the model and capture evaluation results
    evals_result = {}
    model = xgb.train(
        params, 
        dtrain, 
        num_rounds,
        evals=watchlist,
        evals_result=evals_result,
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
    
    # Add feature importance using gain metric (most meaningful for tree models)
    try:
        feature_importance = model.get_score(importance_type='gain')
        feature_importance_dict = {}
        feature_names = list(X_train.columns)
        
        if feature_importance:
            for xgb_feature, score in feature_importance.items():
                # XGBoost uses f0, f1, f2, ... for feature names
                if xgb_feature.startswith('f'):
                    try:
                        feature_idx = int(xgb_feature[1:])
                        if feature_idx < len(feature_names):
                            original_name = feature_names[feature_idx]
                            feature_importance_dict[original_name] = float(score)
                    except (ValueError, IndexError):
                        pass
                else:
                    feature_importance_dict[xgb_feature] = float(score)
            
            metrics['feature_importance'] = feature_importance_dict
            print(f"Added feature importance for {len(feature_importance_dict)} features to metrics")
    except Exception as e:
        print(f"Warning: Could not add feature importance to metrics: {e}")
        metrics['feature_importance'] = {}
    
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

    # Generate comprehensive evaluation plots
    print("\nGenerating evaluation plots...")
    plot_evaluation_metrics(y_test, y_pred_proba, plots_dir, site_name)
    
    # Generate training history plot
    print("Generating training history plot...")
    plot_training_history(evals_result, model.best_iteration, plots_dir, site_name)

    # Save the trained model
    model_path = output_config['model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Save the scaler for preprocessing new data (only if scaling was used)
    if scaler is not None:
        scaler_path = output_config['scaler_path']
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")
    else:
        print("No scaler to save (scaling was disabled)")

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