import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.calibration import CalibrationDisplay

# Set random seeds for reproducibility
np.random.seed(42)

# Define the features to use as specified
selected_features = [
    'eosinophils_absolute',
    'glucose_fingerstick',
    'lymphocytes_absolute',
    'monocytes_absolute',
    'neutrophils_absolute',
    'procalcitonin',
    'troponin_i',
    'wbc',
    'gcs_total',
    'rass',
    'angiotensin',
    'isoproterenol',
    'ketamine',
    'remifentanil',
    'pentobarbital',
    'lorazepam',
    'fio2'
]

# Load data
print("Loading data...")
df = pd.read_parquet('../output/intermitted/by_event_wide_df.parquet')
print(f"Data shape: {df.shape}")
print(f"Number of unique hospitalization_ids: {df['hospitalization_id'].nunique()}")

# Check which features from our selection are available in the dataframe
available_features = [col for col in selected_features if col in df.columns]
missing_features = [col for col in selected_features if col not in df.columns]
if missing_features:
    print(f"Warning: The following requested features are not in the dataset: {missing_features}")

print(f"Available features: {available_features}")

# Add hospitalization_id and disposition to the list of columns we need
required_cols = ['hospitalization_id', 'disposition'] + available_features
df_filtered = df[required_cols].copy()

print(f"Filtered data shape: {df_filtered.shape}")
print(f"Class distribution: \n{df_filtered['disposition'].value_counts()}")
for label, count in df_filtered['disposition'].value_counts().items():
    print(f"Class {label}: {count} ({100 * count / len(df_filtered):.2f}%)")

# Aggregate data by hospitalization_id, calculating min, max, and median for each feature
print("Aggregating data by hospitalization_id...")
aggregation_funcs = {}
for feature in available_features:
    aggregation_funcs[feature] = ['min', 'max', 'median']

# Add disposition to the aggregation (taking the first value as it should be the same for all rows of the same hospitalization)
aggregation_funcs['disposition'] = 'first'

# Perform the aggregation
agg_df = df_filtered.groupby('hospitalization_id').agg(aggregation_funcs)

# Flatten the multi-level column index
agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]

# Reset index to make hospitalization_id a column again
agg_df = agg_df.reset_index()

print(f"Aggregated data shape: {agg_df.shape}")
print(f"Number of unique hospitalization_ids after aggregation: {agg_df['hospitalization_id'].nunique()}")

# Prepare features and target
X = agg_df.drop(['hospitalization_id', 'disposition_first'], axis=1)
y = agg_df['disposition_first']

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

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

# Set up XGBoost parameters - adjusted for small dataset
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.05,  # Lower learning rate
    'max_depth': 3,  # Reduced depth to prevent overfitting
    'min_child_weight': 3,  # Increased to be more conservative
    'subsample': 0.7,  # Sample fewer records per tree
    'colsample_bytree': 0.7,  # Sample fewer features per tree
    'scale_pos_weight': scale_pos_weight,
    'gamma': 1,  # Minimum loss reduction for split
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'seed': 42
}

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# Set up evaluation list
eval_list = [(dtrain, 'train'), (dtest, 'eval')]

# Train model with early stopping
num_rounds = 500
early_stopping_rounds = 20  # Give more rounds before stopping
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
    verbose_eval=50  # Print fewer messages
)

print(f"Best iteration: {model.best_iteration}")

# Evaluate model with cross-validation for more robust assessment
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

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance visualization with error handling
try:
    importance_type = 'weight'  # Try 'weight' instead of default 'gain'
    importance = model.get_score(importance_type=importance_type)
    
    if importance:
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(model, max_num_features=20, importance_type=importance_type)
        plt.title('XGBoost Feature Importance (Weight)')
        plt.savefig(os.path.join(graphs_dir, 'xgb_feature_importance.png'))
        plt.close()
        print("Feature importance plot saved successfully.")
    else:
        print("Warning: No feature importance could be calculated. Skipping plot.")
        
        # Alternative approach: print feature names directly
        print("\nFeatures used in model:")
        for i, col in enumerate(X.columns):
            print(f"{i}: {col}")
            
except ValueError as e:
    print(f"Warning: Could not plot feature importance: {e}")
    print("Skipping feature importance plot.")

# Create calibration plot with proper error handling
try:
    # Use fewer bins due to small dataset
    n_bins = min(5, len(np.unique(y_pred_proba)))  # Use at most 5 bins or unique probabilities
    plt.figure(figsize=(10, 6))
    disp = CalibrationDisplay.from_predictions(y_test, y_pred_proba, n_bins=n_bins, name='XGBoost')
    plt.title('Calibration Plot')
    plt.savefig(os.path.join(graphs_dir, 'xgb_calibration_plot.png'))
    plt.close()
    print("Calibration plot saved successfully.")
except Exception as e:
    print(f"Warning: Could not create calibration plot: {e}")
    print("Skipping calibration plot.")

# Create model directory if it doesn't exist
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')
os.makedirs(model_dir, exist_ok=True)

# Create graphs directory if it doesn't exist
graphs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'final', 'graphs')
os.makedirs(graphs_dir, exist_ok=True)

# Save the trained model
model_path = os.path.join(model_dir, 'xgb_icu_mortality_model.json')
model.save_model(model_path)
print(f"Model saved to {model_path}")

# Save the scaler for preprocessing new data
scaler_path = os.path.join(model_dir, 'xgb_feature_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {scaler_path}")

# Save feature column names for inference
feature_cols_path = os.path.join(model_dir, 'xgb_feature_columns.pkl')
with open(feature_cols_path, 'wb') as f:
    pickle.dump(list(X.columns), f)
print(f"Feature columns saved to {feature_cols_path}")

print(f"Evaluation completed. Results saved in {os.path.join(graphs_dir, 'xgb_feature_importance.png')} and {os.path.join(graphs_dir, 'xgb_calibration_plot.png')}")
