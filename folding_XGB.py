"""
XGBoost Training and Inference Script with N-Fold Validation.

This script executes a complete training workflow for XGBoost on astronomical FITS data.
It extends the standard training process by implementing deterministic N-Fold validation, 
allowing for rigorous cross-validation across the entire dataset.

Key Features:
- Data Loading: robustly loads training and test data from FITS files.
- Preprocessing: performs outlier clipping, median imputation, and robust scaling.
- N-Fold Deterministic Validation: 
    - Automatically partitions the dataset into N unique validation sets based on `test_size`.
    - Fold 0 corresponds to the standard single-run split.
    - Subsequent folds (1 to N-1) deterministically partition the remaining training data
      to serve as validation sets.
    - Ensures every data point is used for validation exactly once across the full set of folds.
- Model Training: uses class weighting and early stopping to handle imbalance and prevent overfitting.
- Inference: generates predictions, confidence scores, and entropy metrics.
- Caching: skips training if output files exist and hyperparameters match the current configuration.
- Selective Output: saves a lightweight FITS file containing only identifiers, features, and predictions.

Usage:
    python folding_xgb.py <train_file> <test_file> <output_file> <fold_index>
"""
import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from astropy.io import fits
from astropy.table import Table
import time
import gc
import joblib
import subprocess
import warnings
import math
from astropy.io.fits.card import VerifyWarning

# Suppress warnings about FITS header keywords being too long.
# Astropy handles this automatically using HIERARCH cards.
warnings.simplefilter('ignore', category=VerifyWarning)

# Import visualization functions from vis.py
try:
    from vis import (
        plot_xgb_training_loss, plot_bailey_diagram,
        plot_xgb_feature_importance, plot_confidence_distribution,
        plot_confidence_entropy, plot_and_print_auc_ap
    )
except ImportError:
    print("Warning: 'vis' module not found. Visualization will be skipped.")
    # Define placeholder functions to avoid NameError if vis.py is missing
    def plot_xgb_training_loss(*args, **kwargs): pass
    def plot_bailey_diagram(*args, **kwargs): pass
    def plot_xgb_feature_importance(*args, **kwargs): pass
    def plot_confidence_distribution(*args, **kwargs): pass
    def plot_confidence_entropy(*args, **kwargs): pass
    def plot_and_print_auc_ap(*args, **kwargs): pass

#########################################
# SECTION 1: UTILITY FUNCTIONS
#########################################

def load_fits_to_df(path):
    """
    Load data from a FITS file into a pandas DataFrame.

    This function handles potential endianness mismatches between FITS (Big Endian)
    and modern CPU architectures (Little Endian) to ensure compatibility with 
    NumPy and pandas.

    Args:
        path (str): The file path to the FITS file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    print(f"Loading {path}...")
    try:
        with fits.open(path) as hdul:
            # Check if data is in the primary HDU or the first extension
            data = hdul[1].data if len(hdul) > 1 and hdul[1].data is not None else hdul[0].data
            
            if not hasattr(data, 'names') or not data.names:
                df = pd.DataFrame(np.asarray(data))
            else:
                data_dict = {}
                for col in data.names:
                    col_data = np.asarray(data[col])
                    # Fix byte order if necessary for pandas compatibility
                    if col_data.dtype.byteorder not in ('=', '|'):
                        col_data = col_data.astype(col_data.dtype.newbyteorder('='))
                    data_dict[col] = col_data
                df = pd.DataFrame(data_dict)
            print(f"Loaded {len(df)} samples with {len(df.columns)} features")
            return df
    except Exception as e:
        print(f"Error loading FITS file: {e}")
        raise

def propagate_labels(target_df, source_df, label_col):
    """
    Transfer known labels from a source dataframe to a target dataframe based on unique IDs.
    
    This is critical when the test/inference dataset lacks explicit labels but contains 
    objects that are present in the training dataset (e.g., during validation runs).

    Args:
        target_df (pd.DataFrame): The dataframe receiving labels (e.g., test_df).
        source_df (pd.DataFrame): The dataframe providing labels (e.g., train_df).
        label_col (str): The name of the label column.
        
    Returns:
        pd.DataFrame: The target dataframe with the label column filled or updated.
    """
    print(f"Attempting to match labels from source to target for column '{label_col}'...")
    id_col = None
    potential_id_cols = ['uniqueid', 'source_id', 'id']
    
    # Identify the common unique identifier column
    for col in potential_id_cols:
        if col in target_df.columns and col in source_df.columns:
            id_col = col
            break
            
    if id_col and label_col in source_df.columns:
        print(f"  Matching based on unique ID column: {id_col}")
        label_map = dict(zip(source_df[id_col], source_df[label_col]))
        
        # Map source labels to the target dataframe using the ID
        matched_labels = target_df[id_col].map(label_map)
        
        if label_col not in target_df.columns:
             target_df[label_col] = matched_labels
        else:
             # Fill only missing values to preserve any existing labels in target
             target_df[label_col] = target_df[label_col].fillna(matched_labels)
        
        # Mark any remaining missing labels as 'UNKNOWN'
        target_df[label_col] = target_df[label_col].fillna('UNKNOWN')
            
        known_count = (target_df[label_col] != 'UNKNOWN').sum()
        print(f"  Labels propagated. {known_count}/{len(target_df)} samples now have known labels.")
    else:
        print("  Could not propagate labels (missing ID column or source labels).")
        if label_col not in target_df.columns:
            target_df[label_col] = 'UNKNOWN'
            
    return target_df

def get_feature_list(train, test):
    """
    Define the specific features to be used for model training.

    This function specifies a curated list of physical parameters (e.g., Period, Amplitude)
    and statistical metrics, along with embeddings. It then filters this list to include 
    only those features present in both the training and testing datasets.

    Args:
        train (pd.DataFrame): The training dataframe.
        test (pd.DataFrame): The testing dataframe.

    Returns:
        list: A list of feature names (strings) validated for use.
    """
    # Curated feature set for variable star classification.
    features = [
        # --- Basic Parameters & Quality Metrics ---
        "true_period", "true_amplitude",

        # --- Photometry & Color Indices ---
        "z_med_mag-ks_med_mag",
        "y_med_mag-ks_med_mag",
        "j_med_mag-ks_med_mag",

        # --- Light Curve Shape & Variability Analysis ---
        "Cody_M", "stet_k",
        "eta_e", "med_BRP", "range_cum_sum", "max_slope", "MAD", "mean_var",
        "percent_amp", "roms", "p_to_p_var", "lag_auto", "AD", "std_nxs",
        "skew", "kurt"
    ]
    
    # Generate embedding column names (0-127)
    embeddings = [str(i) for i in range(128)]
    full_feature_set = features + embeddings
    
    # Find the intersection of requested features and available columns
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    common_cols = train_cols.intersection(test_cols)
    usable_features = [f for f in full_feature_set if f in common_cols]
    
    print(f"Using {len(usable_features)} of {len(full_feature_set)} predefined features.")
    return usable_features

def get_gpu_count():
    """
    Detect available NVIDIA GPUs using the nvidia-smi command line tool.

    Returns:
        int: The number of GPUs detected. Returns 0 if nvidia-smi fails.
    """
    try:
        result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, check=True)
        return len(result.stdout.decode('utf-8').strip().split('\n'))
    except:
        return 0

def check_hyperparameters(fits_header, current_params):
    """
    Validate if the hyperparameters in an existing FITS file match the current script configuration.

    This ensures that cached results are only used if the model configuration is identical.

    Args:
        fits_header (astropy.io.fits.Header): The header from the existing FITS file.
        current_params (dict): Dictionary of the current script's hyperparameters.

    Returns:
        bool: True if all parameters match, False otherwise.
    """
    print("Checking hyperparameters in existing FITS file...")
    for key, current_val in current_params.items():
        header_val = fits_header.get(key)

        if header_val is None:
            print(f"  Mismatch: Key '{key}' not found in FITS header.")
            return False
        
        # Handle 'None' values explicitly
        if current_val is None:
            if str(header_val) == 'None':
                continue
            else:
                return False
        
        try:
            # Cast header value to the type of the current value for accurate comparison
            header_val_typed = type(current_val)(header_val)
            
            # Use np.isclose for floating point comparisons to avoid precision errors
            if isinstance(current_val, float):
                if not np.isclose(header_val_typed, current_val):
                    print(f"  Mismatch: Key '{key}'. Current: {current_val}, Header: {header_val}")
                    return False
            else:
                if header_val_typed != current_val:
                    print(f"  Mismatch: Key '{key}'. Current: {current_val}, Header: {header_val}")
                    return False
        except:
            return False
            
    return True

#########################################
# SECTION 2: DATA PREPARATION
#########################################

def preprocess_data(train_df, test_df, features, label_col):
    """
    Apply standard preprocessing steps to the training and test data.

    Steps:
    1. Replace infinite values with NaN.
    2. Clip outliers to the 0.1st and 99.9th percentiles to improve scaler stability.
    3. Impute missing values using the median.
    4. Scale features using RobustScaler (insensitive to outliers).
    5. Encode class labels into integers.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        test_df (pd.DataFrame): Test dataframe.
        features (list): List of feature column names.
        label_col (str): Name of the target label column.

    Returns:
        tuple: (X_train_processed, X_test_processed, y_train_encoded, label_encoder)
    """
    print("Preprocessing data...")
    X_train = train_df[features].copy().replace([np.inf, -np.inf], np.nan)
    X_test = test_df[features].copy().replace([np.inf, -np.inf], np.nan)

    # 1. Clip outliers based on training data distribution
    q001, q999 = X_train.quantile(0.001), X_train.quantile(0.999)
    X_train = X_train.clip(lower=q001, upper=q999, axis=1)
    X_test = X_test.clip(lower=q001, upper=q999, axis=1)

    # 2. Impute missing values with the median
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # 3. Scale features using RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # 4. Encode labels
    label_encoder = LabelEncoder()
    train_labels = train_df[label_col].fillna('UNKNOWN')
    y_train_encoded = label_encoder.fit_transform(train_labels)
    
    print(f"Number of classes: {len(label_encoder.classes_)}")
    for i, cls in enumerate(label_encoder.classes_):
        count = (y_train_encoded == i).sum()
        print(f"  {i}: {cls} - {count} samples")

    return (pd.DataFrame(X_train_scaled, columns=features, index=train_df.index), 
            pd.DataFrame(X_test_scaled, columns=features, index=test_df.index), 
            y_train_encoded, label_encoder)

#########################################
# SECTION 3: WORKFLOW FUNCTIONS
#########################################

def generate_visualizations(model, label_encoder, test_df_result, label_col, evals_result=None):
    """
    Generate and save a suite of visualization plots to the 'figures' directory.
    
    Includes:
    - Classification Report
    - Feature Importance Plot
    - Training Loss Curve (if eval history provided)
    - AUC/AP Metrics and ROC Curves
    - Bailey Diagram (Period vs Amplitude)
    - Confidence Distribution Histogram
    - Confidence vs Entropy 2D Histogram
    
    Args:
        model (xgb.Booster): The trained XGBoost model.
        label_encoder (LabelEncoder): The fitted LabelEncoder.
        test_df_result (pd.DataFrame): DataFrame containing test data and predictions.
        label_col (str): Name of the true label column.
        evals_result (dict, optional): Results from training (for loss plot).
    """
    print("\nGenerating visualizations...")
    os.makedirs('figures', exist_ok=True)
    
    # 1. Classification Report
    # Prefer the 'xgb_training_class' which comes from propagation, falling back to original label_col
    true_label_col_to_use = 'xgb_training_class'
    if true_label_col_to_use not in test_df_result.columns:
        true_label_col_to_use = label_col

    if true_label_col_to_use in test_df_result.columns:
        y_true = test_df_result[true_label_col_to_use].fillna('UNKNOWN')
        pred_labels = test_df_result['xgb_predicted_class']
        print("\nTest set evaluation:")
        print(classification_report(y_true, pred_labels, zero_division=0))
    
    # 2. Feature Importance
    importance = model.get_score(importance_type='gain')
    top_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    plot_xgb_feature_importance([x[0] for x in top_feats], [x[1] for x in top_feats], output_dir='figures')

    # 3. Training Loss
    if evals_result:
        plot_xgb_training_loss(evals_result, output_dir='figures')

    # 4. Model performance metrics
    confs = test_df_result['xgb_confidence']
    preds_int = label_encoder.transform(test_df_result['xgb_predicted_class'])
    
    plot_and_print_auc_ap(test_df_result, true_label_col_to_use, label_encoder, output_dir='figures')
    plot_bailey_diagram(test_df_result, "xgb_predicted_class", output_dir='figures')
    plot_confidence_distribution(confs, preds_int, label_encoder.classes_, output_dir="figures")
    plot_confidence_entropy(test_df_result, "xgb_predicted_class", output_dir='figures')

def train_xgb(train_df, test_df, features, label_col, out_file, learning_rate, max_depth,
              subsample, colsample_bytree, reg_alpha, reg_lambda, num_boost_round, early_stopping_rounds, 
              test_size, fold_index=None, num_folds=None, use_adaptive_lr=True):
    """
    Execute the XGBoost training and inference workflow with N-Fold validation support.

    This function handles the deterministic splitting of data for cross-validation:
    - If `fold_index` is 0 or None, it performs a standard random split defined by `test_size`.
    - If `fold_index` > 0, it retrieves the original training set from Fold 0 and 
      partitions it to create a unique validation set for this specific fold. This ensures 
      that across N folds, every data point is used for validation exactly once.

    Args:
        train_df (pd.DataFrame): The training data.
        test_df (pd.DataFrame): The test data.
        features (list): List of feature names.
        label_col (str): The name of the label column.
        out_file (str): Path to save the output FITS file.
        learning_rate (float): Initial step size shrinkage.
        max_depth (int): Maximum depth of a tree.
        subsample (float): Subsample ratio of the training instance (for XGBoost).
        colsample_bytree (float): Subsample ratio of columns.
        reg_alpha, reg_lambda (float): Regularization terms.
        num_boost_round (int): Number of boosting rounds.
        early_stopping_rounds (int): Activates early stopping.
        test_size (float): Proportion of dataset to use for validation in Fold 0.
        fold_index (int, optional): The index of the current fold (0 to N-1).
        num_folds (int, optional): Total number of folds. Calculated from test_size if not provided.
        use_adaptive_lr (bool): If True, applies cosine decay to the learning rate.
    """
    print("\n=== XGBoost Training Workflow ===")
    start_time = time.time()
    
    # --- 0. Label Propagation ---
    # Populate labels in the test set using IDs from the training set where possible
    test_df = propagate_labels(test_df, train_df, label_col)
    
    # --- 1. Data Preparation ---
    X_train, X_test, y, label_encoder = preprocess_data(train_df, test_df, features, label_col)
    
    # --- 2. Split Logic (N-Fold Validation) ---
    # Determine stratification strategy for small classes
    stratify_flag = y if np.min(np.bincount(y)) >= 2 else None
    
    # 2a. Generate the Base Split (Fold 0)
    # This split is deterministic (random_state=42) and serves as the reference for all folds.
    X_train_0, X_val_0, y_train_0, y_val_0 = train_test_split(
        X_train, y, test_size=test_size, random_state=42, stratify=stratify_flag
    )
    
    # Calculate number of folds implied by the test_size (e.g., 0.2 test_size -> 5 folds)
    if num_folds is None:
        num_folds = int(1 / test_size)

    # 2b. Select Data for the Current Fold
    if fold_index is None or fold_index == 0:
        print(f"Fold 0/None: Using standard base split (test_size={test_size}).")
        # Fold 0 uses the standard random split generated above.
        X_train_main, X_val = X_train_0, X_val_0
        y_train_main, y_val = y_train_0, y_val_0
    else:
        # Fold > 0: Construct a Unique Validation Set
        # We must select a validation set from the data that was *training* data in Fold 0.
        # This ensures we don't reuse the Fold 0 validation data.
        print(f"Fold {fold_index}: partitioning Fold 0 training data for unique validation set...")
        
        # We partition the 'Fold 0 Training Data' into (num_folds - 1) stratified chunks.
        # shuffle=False maintains the deterministic order from train_test_split.
        skf = StratifiedKFold(n_splits=num_folds - 1, shuffle=False)
        
        # Determine which chunk corresponds to the current fold_index.
        # Fold 1 takes chunk 0, Fold 2 takes chunk 1, etc.
        target_chunk = fold_index - 1
        
        split_gen = skf.split(X_train_0, y_train_0)
        
        found_split = False
        for i, (train_idx_sub, val_idx_sub) in enumerate(split_gen):
            if i == target_chunk:
                # 1. Define Validation Set for this Fold
                # It is a specific slice of the Fold 0 Training Data.
                X_val = X_train_0.iloc[val_idx_sub]
                y_val = y_train_0[val_idx_sub]
                
                # 2. Define Training Set for this Fold
                # It combines:
                #   a. The remaining parts of Fold 0 Training Data
                #   b. The original Fold 0 Validation Data
                X_train_sub = X_train_0.iloc[train_idx_sub]
                y_train_sub = y_train_0[train_idx_sub]
                
                X_train_main = pd.concat([X_train_sub, X_val_0])
                y_train_main = np.concatenate([y_train_sub, y_val_0])
                
                found_split = True
                break
        
        if not found_split:
            raise ValueError(f"Fold index {fold_index} is out of bounds for the calculated splits.")
            
        print(f"Fold {fold_index} prepared. Train: {len(X_train_main)}, Val: {len(X_val)}")

    # --- 3. Class Weighting ---
    # Apply smoothed weights to handle class imbalance
    class_counts = np.bincount(y_train_main)
    class_weights = {i: np.sqrt(len(y_train_main) / (len(np.unique(y)) * count)) if count > 0 else 1 
                     for i, count in enumerate(class_counts)}
    sample_weights = np.array([class_weights[l] for l in y_train_main])

    # --- 4. Data Setup for XGBoost ---
    dtrain = xgb.DMatrix(X_train_main, label=y_train_main, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    num_gpus = get_gpu_count()
    params = {
        'objective': 'multi:softprob',
        'num_class': len(label_encoder.classes_),
        'eval_metric': 'mlogloss',
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'min_child_weight': 10,
        'subsample': subsample, 
        'colsample_bytree': colsample_bytree,
        'colsample_bylevel': 0.7,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'tree_method': 'hist',
        'grow_policy': 'lossguide',
        'device': 'cuda' if num_gpus > 0 else 'cpu'
    }
    
    # --- 5. Adaptive Learning Rate Scheduler (Cosine Decay) ---
    callbacks = []
    if use_adaptive_lr:
        def cosine_scheduler(br):
            # Decays LR from initial value to a minimum of 1e-3 following a cosine curve
            return 1e-3 + 0.5 * (learning_rate - 1e-3) * (1 + math.cos(math.pi * br / num_boost_round))
        callbacks.append(xgb.callback.LearningRateScheduler(cosine_scheduler))

    # --- 6. Training ---
    evals_result = {}
    print("\nStarting XGBoost training...")
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds,
                      evals=[(dtrain, 'train'), (dval, 'validation')], evals_result=evals_result, 
                      verbose_eval=100, callbacks=callbacks)

    # --- 7. Inference ---
    best_iter = model.best_iteration
    print(f"\nTraining complete. Best iteration: {best_iter}")
    
    probs = model.predict(dtest, iteration_range=(0, best_iter + 1))
    preds = np.argmax(probs, axis=1)
    
    # Calculate Confidence and Entropy based on probability distribution
    top_two = np.partition(probs, -2, axis=1)[:, -2:]
    confs = np.max(top_two, axis=1) - np.min(top_two, axis=1)
    entropy = np.abs(np.sum(probs * np.log(probs + 1e-12), axis=1))
    
    test_df_result = test_df.copy()
    test_df_result['xgb_predicted_class'] = label_encoder.inverse_transform(preds)
    test_df_result['xgb_confidence'] = confs
    test_df_result['xgb_entropy'] = entropy
    for i, cls in enumerate(label_encoder.classes_):
        test_df_result[f'prob_{cls}'] = probs[:, i]

    # --- 8. Selective Output Construction ---
    print("Building selective output FITS file...")
    # Find the first available identifier column
    id_col = next(c for c in ['uniqueid', 'source_id', 'id'] if c in test_df.columns)
    
    output_df = pd.DataFrame(index=test_df.index)
    output_df[id_col] = test_df[id_col]
    # Add input features
    output_df = pd.concat([output_df, test_df[features]], axis=1)
    
    # Add predictions
    output_df['xgb_predicted_class'] = test_df_result['xgb_predicted_class']
    output_df['xgb_confidence'] = confs
    output_df['xgb_entropy'] = entropy
    
    # Add training labels if available
    if label_col in test_df_result.columns:
        test_df_result['xgb_training_class'] = test_df_result[label_col]
        output_df['xgb_training_class'] = test_df_result['xgb_training_class']

    # Add probability vectors
    for i, cls in enumerate(label_encoder.classes_):
        output_df[f'prob_{cls}'] = probs[:, i]

    out_table = Table.from_pandas(output_df)
    
    # Attach Hyperparameter metadata to FITS header for validation/caching
    header_info = {
        **params, 
        'fold_index': fold_index, 
        'num_folds': num_folds, 
        'best_iteration': best_iter, 
        'test_size': test_size
    }
    for k, v in header_info.items():
        out_table.meta[k] = str(v)
    
    out_table.write(out_file, format='fits', overwrite=True)
    # Save model binary
    joblib.dump((model, label_encoder), out_file.replace('.fits', '.joblib'))
    
    elapsed_time = time.time() - start_time
    print(f"Saved selective results to {out_file} in {elapsed_time:.1f}s")
    return test_df_result, evals_result, model, label_encoder

#########################################
# SECTION 4: SCRIPT ENTRY POINT
#########################################

def main():
    """Main execution entry point."""
    # Default Hyperparameters
    set_learning_rate = 1E-3
    set_max_depth = 50
    set_subsample = 0.95
    set_colsample_bytree = 0.1
    set_reg_alpha = 0.01
    set_reg_lambda = 0.1
    set_num_boost_round = 1000000 
    set_early_stopping_rounds = 5000
    set_use_adaptive_lr = True
    set_test_size = 0.05
    
    # CLI Argument Parsing
    # Expected format: script.py <train> <test> <output> <fold_index>
    if len(sys.argv) >= 3 and sys.argv[1] != "" and sys.argv[2] != "":
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        out_file = sys.argv[3] if len(sys.argv) > 3 else "xgb_predictions.fits"
        set_fold_index = int(sys.argv[4]) if len(sys.argv) > 4 else None
    else:
        # Fallback to default paths for local debugging
        train_path = "../.data/PRIMVS_P_training_new.fits"
        test_path = "../.data/PRIMVS_P.fits"
        out_file = "xgb_predictions.fits"
        set_fold_index = None

    model_file_path = out_file.replace('.fits', '.joblib')

    try:
        # Load training data primarily to determine number of classes and fold count
        train_df = load_fits_to_df(train_path)
        # Auto-detect the label column name
        label_col = "Type" if "Type" in train_df.columns else [c for c in train_df.columns if 'class' in c.lower() or 'type' in c.lower()][0]
        
        # Calculate N folds. E.g., if test_size=0.05, num_folds=20.
        num_folds = int(round(1 / set_test_size)) if set_test_size > 0 else 1
        num_gpus = get_gpu_count()

        # Build parameter dictionary to check against existing FITS headers
        current_params = {
            'objective': 'multi:softprob',
            'learning_rate': set_learning_rate,
            'max_depth': set_max_depth,
            'subsample': set_subsample, 
            'colsample_bytree': set_colsample_bytree,
            'reg_alpha': set_reg_alpha,
            'reg_lambda': set_reg_lambda,
            'device': 'cuda' if num_gpus > 0 else 'cpu',
            'fold_index': set_fold_index,
            'num_folds': num_folds,
            'test_size': set_test_size
        }

        # --- Caching Mechanism ---
        # If output exists and hyperparameters match, skip training and just visualize.
        if os.path.exists(out_file) and os.path.exists(model_file_path):
            with fits.open(out_file) as hdul:
                header = hdul[1].header 
                if check_hyperparameters(header, current_params):
                    print("Cache hit! Loading existing results...")
                    test_df_full = load_fits_to_df(test_path)
                    test_df_full = propagate_labels(test_df_full, train_df, label_col)
                    
                    minimal_df = Table(hdul[1].data).to_pandas()
                    # Fix for potential byte-string encoding issues in FITS
                    if not minimal_df.empty and isinstance(minimal_df['xgb_predicted_class'].iloc[0], bytes):
                        minimal_df['xgb_predicted_class'] = minimal_df['xgb_predicted_class'].str.decode('utf-8')
                    
                    # Merge predictions back into the full test dataframe for visualization
                    pred_cols = [c for c in minimal_df.columns if any(x in c for x in ['xgb_', 'prob_'])]
                    test_df_result = test_df_full.join(minimal_df[pred_cols])
                    
                    (model, label_encoder) = joblib.load(model_file_path)
                    generate_visualizations(model, label_encoder, test_df_result, label_col)
                    return

        # --- Standard Training Workflow ---
        # If cache miss, proceed with full training
        test_df = load_fits_to_df(test_path)
        features = get_feature_list(train_df, test_df)

        res_df, evals, model, le = train_xgb(
            train_df, test_df, features, label_col, out_file, 
            set_learning_rate, set_max_depth, set_subsample, set_colsample_bytree,
            set_reg_alpha, set_reg_lambda, set_num_boost_round, set_early_stopping_rounds,
            set_test_size,
            fold_index=set_fold_index, num_folds=num_folds, use_adaptive_lr=set_use_adaptive_lr
        )

        # Final Visualization Call
        generate_visualizations(model, le, res_df, label_col, evals_result=evals)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
