import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, 
    confusion_matrix, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import os
import json
import time
from datetime import datetime

from model import ICUMortalityNN, EarlyStopping
from ..preprocessing.nn_data_loader import load_data, create_data_loaders, save_preprocessing_artifacts, load_config


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


def train_epoch(model, train_loader, criterion, optimizer, device, gradient_clip=None):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Debug: check for invalid values
        if batch_idx == 0:  # Only check first batch
            print(f"Output range: {outputs.min().item():.6f} to {outputs.max().item():.6f}")
            print(f"Any NaN in outputs: {torch.isnan(outputs).any().item()}")
            print(f"Any inf in outputs: {torch.isinf(outputs).any().item()}")
            print(f"Labels range: {labels.min().item():.6f} to {labels.max().item():.6f}")
        
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        # Update weights
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader)
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    epoch_auc = roc_auc_score(all_labels, all_preds)
    
    return epoch_loss, epoch_auc


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    epoch_loss = running_loss / len(test_loader)
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    return epoch_loss, all_preds, all_labels


def plot_training_history(history, plots_dir, site_name):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # AUC plot
    axes[0, 1].plot(history['train_auc'], label='Train AUC')
    axes[0, 1].plot(history['val_auc'], label='Val AUC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].set_title('Training and Validation AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate plot
    axes[1, 0].plot(history['lr'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    
    # Remove empty subplot
    fig.delaxes(axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'nn_{site_name}_training_history.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_evaluation_metrics(y_true, y_pred_proba, plots_dir, site_name):
    """Create comprehensive evaluation plots"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. ROC Curve
    ax1 = plt.subplot(3, 3, 1)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    ax1.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Precision-Recall Curve
    ax2 = plt.subplot(3, 3, 2)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    ax2.plot(recall, precision, label=f'PR (AUC = {pr_auc:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Calibration Plot
    ax3 = plt.subplot(3, 3, 3)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=10
    )
    ece = expected_calibration_error(y_true, y_pred_proba)
    ax3.plot(mean_predicted_value, fraction_of_positives, "s-", 
             label=f'Neural Network (ECE={ece:.4f})')
    ax3.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax3.set_xlabel('Mean Predicted Probability')
    ax3.set_ylabel('Fraction of Positives')
    ax3.set_title('Calibration Plot')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Confusion Matrix
    ax4 = plt.subplot(3, 3, 4)
    y_pred = (y_pred_proba > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    ax4.set_title('Confusion Matrix')
    
    # 5. Prediction Distribution
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.5, label='Survived', density=True)
    ax5.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.5, label='Died', density=True)
    ax5.set_xlabel('Predicted Probability')
    ax5.set_ylabel('Density')
    ax5.set_title('Prediction Distribution by Outcome')
    ax5.legend()
    ax5.grid(True)
    
    # 6. Threshold Analysis
    ax6 = plt.subplot(3, 3, 6)
    thresholds = np.linspace(0, 1, 100)
    sensitivities = []
    specificities = []
    accuracies = []
    
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba > thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        accuracies.append(accuracy)
    
    ax6.plot(thresholds, sensitivities, label='Sensitivity')
    ax6.plot(thresholds, specificities, label='Specificity')
    ax6.plot(thresholds, accuracies, label='Accuracy')
    ax6.axvline(x=0.5, color='k', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Threshold')
    ax6.set_ylabel('Metric Value')
    ax6.set_title('Performance vs Threshold')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'nn_{site_name}_evaluation_metrics.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def train_neural_network():
    """Main training function"""
    print("=== Neural Network Training for ICU Mortality Prediction ===\n")
    
    # Load configuration
    config = load_config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    X_train, X_test, y_train, y_test, feature_columns, site_name = load_data()
    print(f"Site: {site_name}")
    
    # Create data loaders
    batch_size = config['training_config']['batch_size']
    use_scaling = config['training_config']['use_scaling']
    train_loader, test_loader, scaler = create_data_loaders(
        X_train, X_test, y_train, y_test, 
        batch_size=batch_size, 
        use_scaling=use_scaling
    )
    
    # Initialize model
    input_size = len(feature_columns)
    model_params = config['model_params']
    
    print(f"\nInitializing neural network...")
    print(f"Input size: {input_size}")
    print(f"Hidden layers: {model_params['hidden_sizes']}")
    print(f"Dropout rate: {model_params['dropout_rate']}")
    
    model = ICUMortalityNN(
        input_size=input_size,
        hidden_sizes=model_params['hidden_sizes'],
        dropout_rate=model_params['dropout_rate'],
        activation=model_params.get('activation', 'relu'),
        batch_norm=model_params.get('batch_norm', True)
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training components
    training_config = config['training_config']
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0)
    )
    
    # Learning rate scheduler
    lr_scheduler_config = training_config.get('lr_scheduler', {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize AUC
        factor=lr_scheduler_config.get('factor', 0.5),
        patience=lr_scheduler_config.get('patience', 5),
        min_lr=lr_scheduler_config.get('min_lr', 1e-6)
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=training_config.get('early_stopping_rounds', 15),
        mode='max'
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_auc': [], 'val_auc': [],
        'lr': []
    }
    
    # Create output directories
    output_config = config['output_config']
    output_model_dir = os.path.dirname(output_config['model_path'])
    plots_dir = output_config['plots_dir']
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save preprocessing artifacts
    save_preprocessing_artifacts(scaler, feature_columns, output_config, site_name)
    
    # Training loop
    print(f"\nStarting training...")
    print(f"Epochs: {training_config['num_epochs']}")
    print(f"Early stopping patience: {training_config.get('early_stopping_rounds', 15)}")
    
    best_val_auc = 0
    best_model_state = None
    start_time = time.time()
    
    for epoch in range(training_config['num_epochs']):
        # Train
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            gradient_clip=training_config.get('gradient_clip_value')
        )
        
        # Evaluate
        val_loss, val_preds, val_labels = evaluate(model, test_loader, criterion, device)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        if epoch % 10 == 0 or epoch == training_config['num_epochs'] - 1:
            print(f"Epoch {epoch+1}/{training_config['num_epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
        
        # Early stopping
        if early_stopping(val_auc):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    _, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    # Calculate comprehensive metrics
    test_preds_binary = (test_preds > 0.5).astype(int)
    
    accuracy = accuracy_score(test_labels, test_preds_binary)
    roc_auc = roc_auc_score(test_labels, test_preds)
    pr_auc = average_precision_score(test_labels, test_preds)
    brier_score = brier_score_loss(test_labels, test_preds)
    ece = expected_calibration_error(test_labels, test_preds)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    print(f"Expected Calibration Error: {ece:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds_binary, 
                              target_names=['Survived', 'Died']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds_binary)
    print(cm)
    
    # Save model
    model_path = output_config['model_path'].replace(
        'nn_icu_mortality_model', f'nn_{site_name}_icu_mortality_model'
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_params,
        'input_size': input_size,
        'best_epoch': best_epoch,
        'training_time': training_time,
        'site_name': site_name
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'brier_score': float(brier_score),
        'expected_calibration_error': float(ece),
        'confusion_matrix': cm.tolist(),
        'best_epoch': best_epoch,
        'training_time': training_time,
        'total_parameters': total_params,
        'site_name': site_name,
        'training_date': datetime.now().isoformat()
    }
    
    metrics_path = output_config['metrics_path'].replace(
        'metrics.json', f'{site_name}_metrics.json'
    )
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    plot_training_history(history, plots_dir, site_name)
    plot_evaluation_metrics(test_labels, test_preds, plots_dir, site_name)
    
    # Feature importance
    print("\nCalculating feature importance...")
    try:
        model.eval()
        X_sample = torch.tensor(X_test.values[:1000], dtype=torch.float32).to(device)
        
        if scaler is not None:
            X_sample_scaled = torch.tensor(
                scaler.transform(X_test.values[:1000]), 
                dtype=torch.float32
            ).to(device)
        else:
            X_sample_scaled = X_sample
        
        importance_scores = model.get_feature_importance(X_sample_scaled, method='gradient')
        
        # Create feature importance plot
        feature_importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        plt.xlabel('Feature Importance (Gradient-based)')
        plt.title(f'Top 20 Features - Neural Network Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'nn_{site_name}_feature_importance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nTop 10 Most Important Features:")
        for idx, row in feature_importance_df.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
            
    except Exception as e:
        print(f"Warning: Could not calculate feature importance: {e}")
    
    print("\n✅ Training completed successfully!")
    

if __name__ == "__main__":
    train_neural_network()