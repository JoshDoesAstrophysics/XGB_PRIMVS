"""
XGBoost Training and Inference Script.

This script provides a complete workflow for training an XGBoost model on astronomical
data from FITS files. It is designed for robust performance and includes several
key features:

The main workflow consists of:
- Loading training and test data from FITS files.
- Preprocessing features, including robust outlier clipping and imputation.
- Training an XGBoost model with class balancing and early stopping to prevent overfitting.
- Performing inference on the test set.
- Saving the final predictions and additional data to a FITS file.
- Saving the trained model object using joblib.
- Generating a visualization dashboard of the results.
- Caching: If the output .fits and .joblib files are detected and the
  hyperparameters in the .fits header match the current script settings,
  training and inference will be skipped, and the script will proceed
  directly to visualization.
- Selective Output: The output FITS file contains only the unique ID,
  the input feature (parameter) columns, and the inference output columns.
- N-Fold Deterministic Validation: Based on the test_size, it partitions the 
  data into N folds. Fold 0 is identical to the standard single-run split.
  Subsequent folds partition the remaining data for validation, ensuring
  every data point is used for validation exactly once across all folds.

To run from the command line:
    python folding_xgb.py <path_to_train.fits> <path_to_test.fits> [output_file.fits] [fold_index]
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
# astropy handles this automatically by using HIERARCH cards.
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
    # Define placeholder functions to avoid NameError
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

    Handles potential endianness issues and gracefully manages FITS file structures.

    Args:
        path (str): The file path to the FITS file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    print(f"Loading {path}...")
    try:
        with fits.open(path) as hdul:
            data = hdul[1].data if len(hdul) > 1 and hdul[1].data is not None else hdul[0].data
            if not hasattr(data, 'names') or not data.names:
                df = pd.DataFrame(np.asarray(data))
            else:
                data_dict = {}
                for col in data.names:
                    col_data = np.asarray(data[col])
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
    Propagate labels from source_df to target_df based on a common unique ID.
    Useful when the test/inference dataframe is missing labels but shares objects
    with the training dataframe.
    
    Args:
        target_df (pd.DataFrame): The dataframe receiving labels (e.g. test_df).
        source_df (pd.DataFrame): The dataframe providing labels (e.g. train_df).
        label_col (str): The name of the label column.
        
    Returns:
        pd.DataFrame: The target dataframe with labels filled/updated.
    """
    print(f"Attempting to match labels from source to target for column '{label_col}'...")
    id_col = None
    potential_id_cols = ['uniqueid', 'source_id', 'id']
    
    # Find common ID column
    for col in potential_id_cols:
        if col in target_df.columns and col in source_df.columns:
            id_col = col
            break
            
    if id_col and label_col in source_df.columns:
        print(f"  Matching based on unique ID column: {id_col}")
        # Create mapping dictionary
        label_map = dict(zip(source_df[id_col], source_df[label_col]))
        
        # Map to a matched series
        matched_labels = target_df[id_col].map(label_map)
        
        # Fill target label column
        if label_col not in target_df.columns:
             target_df[label_col] = matched_labels
        else:
             # Only fill missing values to respect existing data if present
             target_df[label_col] = target_df[label_col].fillna(matched_labels)
        
        # Fill remaining NaNs with 'UNKNOWN' to prevent downstream errors
        target_df[label_col] = target_df[label_col].fillna('UNKNOWN')
            
        known_count = (target_df[label_col] != 'UNKNOWN').sum()
        print(f"  Labels propagated. {known_count}/{len(target_df)} samples now have known labels.")
    else:
        print("  Could not propagate labels (missing ID column or source labels).")
        # Ensure column exists even if matching failed
        if label_col not in target_df.columns:
            target_df[label_col] = 'UNKNOWN'
            
    return target_df

def get_feature_list(train, test):
    """
    Define and filter a list of features to be used for training.

    This function defines a curated list of features and also includes embeddings.
    It then finds the intersection of these features with the columns available
    in both the training and testing dataframes.

    Args:
        train (pd.DataFrame): The training dataframe.
        test (pd.DataFrame): The testing dataframe.

    Returns:
        list: A list of feature names (strings) to be used in the model.
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
    
    # Assumes 128-dimensional embeddings from a contrastive learning model.
    embeddings = [str(i) for i in range(128)]
    full_feature_set = features + embeddings
    
    # Determine which of the predefined features are actually available.
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    common_cols = train_cols.intersection(test_cols)
    usable_features = [f for f in full_feature_set if f in common_cols]
    
    print(f"Using {len(usable_features)} of {len(full_feature_set)} predefined features.")
    return usable_features

def get_gpu_count():
    """
    Detect available NVIDIA GPUs via nvidia-smi.

    Returns:
        int: The number of GPUs detected.
    """
    try:
        result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, check=True)
        return len(result.stdout.decode('utf-8').strip().split('\n'))
    except:
        return 0

def check_hyperparameters(fits_header, current_params):
    """
    Compare current hyperparameters with those in a FITS header for caching.

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
        
        if current_val is None:
            if str(header_val) == 'None':
                continue
            else:
                return False
        
        try:
            # Cast header value to the type of the current value for comparison
            header_val_typed = type(current_val)(header_val)
            
            # Use np.isclose for floating point comparisons
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
    Preprocess training and test data for the XGBoost model.

    This involves:
    1. Handling infinite values.
    2. Clipping outliers to the 0.1st and 99.9th percentiles for robustness.
    3. Filling any remaining NaN values with the column median.
    4. Scaling features using RobustScaler.
    5. Encoding labels into integer format.

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

    # 1. Clipping outliers based on training data distribution.
    q001, q999 = X_train.quantile(0.001), X_train.quantile(0.999)
    X_train = X_train.clip(lower=q001, upper=q999, axis=1)
    X_test = X_test.clip(lower=q001, upper=q999, axis=1)

    # 2. Impute missing values with the median from the training data.
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # 3. Scale features using a scaler robust to outliers.
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # 4. Encode labels.
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
    Generate and save all visualization plots to the 'figures' directory.
    
    Args:
        model (xgb.Booster): The trained XGBoost model.
        label_encoder (LabelEncoder): The fitted LabelEncoder.
        test_df_result (pd.DataFrame): DataFrame with test data and predictions.
        label_col (str): Name of the true label column.
        evals_result (dict, optional): Results from training (for loss plot).
    """
    print("\nGenerating visualizations...")
    os.makedirs('figures', exist_ok=True)
    
    # 1. Classification Report
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
    Execute the XGBoost training and inference workflow with N-Fold support.

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
        fold_index (int, optional): The current partition to validate on.
        num_folds (int, optional): Total number of folds for partitioning.
        use_adaptive_lr (bool): Applies cosine decay if True.
    """
    print("\n=== XGBoost Training Workflow ===")
    start_time = time.time()
    
    # --- 0. Label Propagation ---
    test_df = propagate_labels(test_df, train_df, label_col)
    
    # --- 1. Data Preparation ---
    X_train, X_test, y, label_encoder = preprocess_data(train_df, test_df, features, label_col)
    
    # --- 2. Split Logic (Custom N-Fold based on test_size) ---
    # We enforce that Fold 0 is ALWAYS identical to the standard XGB.py split.
    stratify_flag = y if np.min(np.bincount(y)) >= 2 else None
    
    # Generate the base "Fold 0" split using strict reproducibility
    # idx_train_0 and idx_val_0 represent the split for Fold 0
    X_train_0, X_val_0, y_train_0, y_val_0 = train_test_split(
        X_train, y, test_size=test_size, random_state=42, stratify=stratify_flag
    )
    
    if num_folds is None:
        num_folds = int(1 / test_size)

    if fold_index is None or fold_index == 0:
        print(f"Fold 0/None: Using standard split (test_size={test_size}).")
        X_train_main, X_val = X_train_0, X_val_0
        y_train_main, y_val = y_train_0, y_val_0
    else:
        # For folds > 0, we must NOT use the data in X_val_0 for validation.
        # It must go into the training set.
        # We need to pick a new validation set from X_train_0.
        # To ensure "unique data" for validation across all folds, we divide X_train_0
        # into (num_folds - 1) partitions.
        print(f"Fold {fold_index}: Determining unique validation split...")
        
        # We perform stratified K-Fold on the TRAINING data from Fold 0.
        # shuffle=False ensures we respect the existing order (which is already shuffled deterministically by train_test_split)
        # effectively carving up the remaining data into distinct chunks.
        skf = StratifiedKFold(n_splits=num_folds - 1, shuffle=False)
        
        # We need to iterate to find the correct chunk for this fold_index
        # fold_index 1 corresponds to chunk 0 of the sub-split
        # fold_index 2 corresponds to chunk 1 of the sub-split, etc.
        target_chunk = fold_index - 1
        
        split_gen = skf.split(X_train_0, y_train_0)
        
        found_split = False
        for i, (train_idx_sub, val_idx_sub) in enumerate(split_gen):
            if i == target_chunk:
                # These indices are relative to X_train_0 / y_train_0
                # 1. Create the new validation set for this fold
                X_val = X_train_0.iloc[val_idx_sub]
                y_val = y_train_0[val_idx_sub]
                
                # 2. Create the new training set
                # It consists of the rest of X_train_0 AND the original X_val_0
                X_train_sub = X_train_0.iloc[train_idx_sub]
                y_train_sub = y_train_0[train_idx_sub]
                
                X_train_main = pd.concat([X_train_sub, X_val_0])
                y_train_main = np.concatenate([y_train_sub, y_val_0])
                
                found_split = True
                break
        
        if not found_split:
            raise ValueError(f"Fold index {fold_index} is out of bounds for the calculated splits.")
            
        print(f"Fold {fold_index} created. Train: {len(X_train_main)}, Val: {len(X_val)}")

    # --- 3. Class Weighting ---
    class_counts = np.bincount(y_train_main)
    class_weights = {i: np.sqrt(len(y_train_main) / (len(np.unique(y)) * count)) if count > 0 else 1 
                     for i, count in enumerate(class_counts)}
    sample_weights = np.array([class_weights[l] for l in y_train_main])

    # --- 4. Data Setup ---
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
        'subsample': subsample, # Now correctly using the XGBoost parameter
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
    id_col = next(c for c in ['uniqueid', 'source_id', 'id'] if c in test_df.columns)
    
    output_df = pd.DataFrame(index=test_df.index)
    output_df[id_col] = test_df[id_col]
    output_df = pd.concat([output_df, test_df[features]], axis=1)
    
    output_df['xgb_predicted_class'] = test_df_result['xgb_predicted_class']
    output_df['xgb_confidence'] = confs
    output_df['xgb_entropy'] = entropy
    
    if label_col in test_df_result.columns:
        test_df_result['xgb_training_class'] = test_df_result[label_col]
        output_df['xgb_training_class'] = test_df_result['xgb_training_class']

    for i, cls in enumerate(label_encoder.classes_):
        output_df[f'prob_{cls}'] = probs[:, i]

    out_table = Table.from_pandas(output_df)
    
    # Attach Hyperparameter metadata to FITS header for caching validation
    header_info = {**params, 'fold_index': fold_index, 'num_folds': num_folds, 'best_iteration': best_iter, 'test_size': test_size}
    for k, v in header_info.items():
        out_table.meta[k] = str(v)
    
    out_table.write(out_file, format='fits', overwrite=True)
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
    set_test_size = 0.1
    
    # CLI Argument Parsing
    if len(sys.argv) >= 3 and sys.argv[1] != "" and sys.argv[2] != "":
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        out_file = sys.argv[3] if len(sys.argv) > 3 else "xgb_predictions.fits"
        set_fold_index = int(sys.argv[4]) if len(sys.argv) > 4 else None
    else:
        # Default local paths
        train_path = "../.data/PRIMVS_P_training_new.fits"
        test_path = "../.data/PRIMVS_P.fits"
        out_file = "xgb_predictions.fits"
        set_fold_index = None

    model_file_path = out_file.replace('.fits', '.joblib')

    try:
        # Load training data first to check num_classes for hyperparam dict
        train_df = load_fits_to_df(train_path)
        label_col = "Type" if "Type" in train_df.columns else [c for c in train_df.columns if 'class' in c.lower() or 'type' in c.lower()][0]
        
        # Determine number of folds based on test_size inverse
        num_folds = int(round(1 / set_test_size)) if set_test_size > 0 else 1
        num_gpus = get_gpu_count()

        # Build parameter dictionary for Cache Check
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

        # --- Caching Check ---
        if os.path.exists(out_file) and os.path.exists(model_file_path):
            with fits.open(out_file) as hdul:
                header = hdul[1].header 
                if check_hyperparameters(header, current_params):
                    print("Cache hit! Loading existing results...")
                    test_df_full = load_fits_to_df(test_path)
                    test_df_full = propagate_labels(test_df_full, train_df, label_col)
                    
                    minimal_df = Table(hdul[1].data).to_pandas()
                    # Fix for FITS byte-to-string conversion issue
                    if not minimal_df.empty and isinstance(minimal_df['xgb_predicted_class'].iloc[0], bytes):
                        minimal_df['xgb_predicted_class'] = minimal_df['xgb_predicted_class'].str.decode('utf-8')
                    
                    pred_cols = [c for c in minimal_df.columns if any(x in c for x in ['xgb_', 'prob_'])]
                    test_df_result = test_df_full.join(minimal_df[pred_cols])
                    
                    (model, label_encoder) = joblib.load(model_file_path)
                    generate_visualizations(model, label_encoder, test_df_result, label_col)
                    return

        # --- Standard Training Workflow ---
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
