"""
XGBoost Training and Inference Script for Astronomical Data.

This script performs the complete machine learning workflow for classifying variable stars
(or other astronomical objects) using the XGBoost algorithm. It takes data stored in
FITS files as input and produces classified catalogues and diagnostic plots.

Intended Audience:
    This script is designed for researchers and students. Comments are written to explain
    *why* specific steps are taken (e.g., scaling, outlier clipping, stratified splitting)
    to help you understand the machine learning pipeline.

Workflow Overview:
    1.  **Data Loading**: Reads training and testing data from FITS files.
    2.  **Label Propagation**: If the test data lacks labels but shares object IDs with
        the training data, labels are copied over.
    3.  **Preprocessing**:
        * **Clipping**: Removes extreme outliers (top/bottom 0.1%) to prevent them from
            skewing the model.
        * **Imputation**: Fills missing values (NaNs) with the median of that column.
        * **Scaling**: Standardizes the range of features using RobustScaler (insensitive to outliers).
        * **Encoding**: Converts string class names (e.g., "RR Lyrae") into integers (0, 1, 2...).
    4.  **Training**: Trains an XGBoost classifier.
        * Uses **Class Weighting** to handle imbalanced data (lots of one class, few of another).
        * Uses **Early Stopping** to stop training when the model stops improving (prevents overfitting).
        * Uses **Cosine Decay** for the learning rate to fine-tune the model as it learns.
    5.  **Inference**: Applies the trained model to the test set to predict classes and probabilities.
    6.  **Output**: Saves a FITS file with predictions and a .joblib file with the trained model.
    7.  **Visualization**: Generates performance plots (Confusion matrices, Feature importance, etc.).

Usage:
    Run this script from the command line:
    python XGB.py <path_to_train.fits> <path_to_test.fits> [output_file.fits]

    Example:
    python XGB.py training_data.fits survey_data.fits results.fits
"""

import sys
import os
import time
import gc
import math
import subprocess
import warnings
import joblib

# Data manipulation libraries
import pandas as pd
import numpy as np

# Machine Learning libraries
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Astronomy file handling
from astropy.io import fits
from astropy.table import Table
from astropy.io.fits.card import VerifyWarning

# Suppress specific FITS header warnings. 
# Long keywords are automatically handled by Astropy's HIERARCH convention, so these warnings are safe to ignore.
warnings.simplefilter('ignore', category=VerifyWarning)

# Import visualization functions from a local module named 'vis'.
# We use a try-except block here so the script can still run (just without plots)
# if the 'vis.py' file is missing.
try:
    from vis import (
        plot_xgb_training_loss, plot_bailey_diagram,
        plot_xgb_feature_importance, plot_confidence_distribution,
        plot_confidence_entropy, plot_and_print_auc_ap
    )
except ImportError:
    print("Warning: 'vis' module not found. Visualization will be skipped.")
    # Define placeholder functions that do nothing if 'vis' is missing.
    def plot_xgb_training_loss(*args, **kwargs): pass
    def plot_bailey_diagram(*args, **kwargs): pass
    def plot_xgb_feature_importance(*args, **kwargs): pass
    def plot_confidence_distribution(*args, **kwargs): pass
    def plot_confidence_entropy(*args, **kwargs): pass
    def plot_and_print_auc_ap(*args, **kwargs): return {}

#########################################
# SECTION 1: UTILITY FUNCTIONS
#########################################

def load_fits_to_df(path):
    """
    Loads data from a FITS file into a Pandas DataFrame.
    
    FITS (Flexible Image Transport System) is the standard data format in astronomy.
    Tables in FITS files are usually in extension [1], but sometimes [0].
    
    Args:
        path (str): The file path to the .fits file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    print(f"Loading {path}...")
    try:
        with fits.open(path) as hdul:
            # Check if data is in extension 1 (standard for tables) or extension 0 (primary)
            data = hdul[1].data if len(hdul) > 1 and hdul[1].data is not None else hdul[0].data
            
            # Convert FITS data (which is a structured array) to a Pandas DataFrame
            if not hasattr(data, 'names') or not data.names:
                df = pd.DataFrame(np.asarray(data))
            else:
                data_dict = {}
                for col in data.names:
                    col_data = np.asarray(data[col])
                    # Fix Endianness: FITS is Big-Endian, most modern CPUs are Little-Endian.
                    # This converts the byte order so Pandas/NumPy can read it efficiently.
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
    Fills missing labels in the target dataset using labels from the source dataset.
    
    This is useful if you have a test set that technically contains "known" objects 
    from your training set, but the label column is missing or empty in the test file.
    It matches rows based on unique identifiers (like 'sourceid').

    Args:
        target_df (pd.DataFrame): The dataframe receiving labels (e.g., test_df).
        source_df (pd.DataFrame): The dataframe providing labels (e.g., train_df).
        label_col (str): The name of the column containing the class labels (e.g., "Type").
        
    Returns:
        pd.DataFrame: The target dataframe with the label column filled where possible.
    """
    print(f"Attempting to match labels from source to target for column '{label_col}'...")
    id_col = None
    # We look for common names used for unique identifiers in astronomical catalogues.
    potential_id_cols = ['uniqueid', 'sourceid', 'source_id', 'id']
    
    # Find which ID column exists in both dataframes
    for col in potential_id_cols:
        if col in target_df.columns and col in source_df.columns:
            id_col = col
            break
            
    if id_col and label_col in source_df.columns:
        print(f"  Matching based on unique ID column: {id_col}")
        # Create a dictionary mapping ID -> Label from the source
        label_map = dict(zip(source_df[id_col], source_df[label_col]))
        
        # Apply this map to the target's ID column
        matched_labels = target_df[id_col].map(label_map)
        
        # If the column doesn't exist in target, create it.
        if label_col not in target_df.columns:
             target_df[label_col] = matched_labels
        else:
             # If it exists, only fill the missing (NaN) values to preserve existing data
             target_df[label_col] = target_df[label_col].fillna(matched_labels)
        
        # Fill any remaining NaNs with 'UNKNOWN' so the code doesn't crash later
        na_count = target_df[label_col].isna().sum()
        if na_count > 0:
            target_df[label_col] = target_df[label_col].fillna('UNKNOWN')
            
        known_count = (target_df[label_col] != 'UNKNOWN').sum()
        print(f"  Labels propagated. {known_count}/{len(target_df)} samples now have known labels.")
    else:
        print("  Could not propagate labels (missing ID column or source labels).")
        # Ensure column exists even if matching failed so downstream code works
        if label_col not in target_df.columns:
            target_df[label_col] = 'UNKNOWN'
            
    return target_df

def save_thresholds_to_header(fits_path, thresholds_dict):
    """
    Saves the optimal probability thresholds to the FITS file header.
    
    After training, we calculate the best probability threshold for each class 
    (e.g., "Only classify as Star X if probability > 0.8"). We save this into the 
    data file itself so future scripts know how to interpret the predictions.

    Args:
        fits_path (str): Path to the FITS file to update.
        thresholds_dict (dict): Dictionary mapping class names to threshold floats.
    """
    if not thresholds_dict:
        return

    print(f"Updating FITS header in {fits_path} with optimal thresholds...")
    try:
        # Open in 'update' mode to modify the header without rewriting the whole file
        with fits.open(fits_path, mode='update') as hdul:
            # Target the table extension (usually 1, fallback to 0)
            target_hdu = hdul[1] if len(hdul) > 1 else hdul[0]
            header = target_hdu.header
            
            header['COMMENT'] = '--- OPTIMAL PROBABILITY THRESHOLDS (PR CURVE) ---'
            
            for cls, thresh in thresholds_dict.items():
                # We prefix keys with 'THR_' to keep the header organized.
                # 'HIERARCH' allows keys longer than 8 characters (standard FITS limit).
                key = f"THR_{cls}"
                header[key] = (float(thresh), f"Optimal threshold for {cls}")
                
            hdul.flush()
            print("  Header updated successfully.")
    except Exception as e:
        print(f"  Warning: Failed to update FITS header: {e}")

def get_feature_list(train, test):
    """
    Determines which columns to use as input features for the model.

    This function contains a manually curated list of features relevant to 
    variable star physics (periods, amplitudes, colors). It checks which of these 
    features exist in BOTH the training and testing files.

    Args:
        train (pd.DataFrame): The training dataframe.
        test (pd.DataFrame): The testing dataframe.

    Returns:
        list: A list of column names (strings) to be used as inputs.
    """
    # Curated feature set for variable star classification.
    features = [
        # --- Basic Parameters & Quality Metrics ---
        #"best_fap",
        "true_period", "true_amplitude",
        #"chisq", "uwe",

        # --- Photometry & Color Indices ---
        #"ks_med_mag", "j_med_mag", "h_med_mag",
        "z_med_mag-ks_med_mag",
        "y_med_mag-ks_med_mag",
        "j_med_mag-ks_med_mag",
        #"h_med_mag-ks_med_mag",

        # --- Positional Data ---
        #"l",
        #"b",
        #"parallax",
        #"pmra", "pmdec",

        # --- Periodogram Fitting Results ---
        #"ls_y_y_0", "ls_peak_width_0",
        #"pdm_y_y_0", "pdm_peak_width_0",
        #"ce_y_y_0", "ce_peak_width_0",

        # --- Light Curve Shape & Variability Analysis ---
        "Cody_M", "stet_k",
        "eta_e", "med_BRP", "range_cum_sum", "max_slope", "MAD", "mean_var",
        "percent_amp", "roms", "p_to_p_var", "lag_auto", "AD", "std_nxs",
        "skew", "kurt"
    ]
    
    # 2. Add Embedding features (Machine Learning extracted features)
    # Assumes we have 128 columns named "0", "1", ... "127" from a pre-trained neural net.
    embeddings = [str(i) for i in range(128)]
    full_feature_set = features + embeddings
    
    # 3. Find intersection: Use only features present in both datasets
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    common_cols = train_cols.intersection(test_cols)
    usable_features = [f for f in full_feature_set if f in common_cols]
    
    print(f"Available features in train: {len(train_cols)}")
    print(f"Available features in test: {len(test_cols)}")
    print(f"Common features: {len(common_cols)}")
    print(f"Using {len(usable_features)} of {len(full_feature_set)} predefined features.")
    
    # Fallback: If none of our curated features are found, use ALL common numeric columns.
    if not usable_features:
        print("Warning: No predefined features found. Falling back to all common numeric columns.")
        exclude_cols = {'sourceid', 'source_id', 'id', 'index', 'best_class_name', 'uniqueid'}
        usable_features = [
            col for col in common_cols
            if pd.api.types.is_numeric_dtype(train[col]) and col not in exclude_cols
        ]
        print(f"Using {len(usable_features)} common numeric features as a fallback.")

    return usable_features

def get_gpu_count():
    """
    Checks for available NVIDIA GPUs to accelerate training.

    Returns:
        int: The number of GPUs detected.
    """
    try:
        result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, check=True)
        return len(result.stdout.decode('utf-8').strip().split('\n'))
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0

def check_hyperparameters(fits_header, current_params):
    """
    Checks if the settings in an existing output file match the current script settings.
    
    This is used for caching: if the file already exists and was created with the 
    exact same parameters, we don't need to re-run the expensive training.

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
            if header_val == 'None':
                continue # Match
            else:
                print(f"  Mismatch: Key '{key}'. Current: None, Header: {header_val}")
                return False
        
        try:
            # Convert header string to the correct type (float/int) for comparison
            header_val_typed = type(current_val)(header_val)
            
            # Use specific comparison for floats to handle slight precision differences
            if isinstance(current_val, float):
                if not np.isclose(header_val_typed, current_val):
                    print(f"  Mismatch: Key '{key}'. Current: {current_val}, Header: {header_val}")
                    return False
            else:
                if header_val_typed != current_val:
                    print(f"  Mismatch: Key '{key}'. Current: {current_val}, Header: {header_val}")
                    return False
        except (ValueError, TypeError):
            print(f"  Mismatch: Type conversion failed for key '{key}'. Current: {current_val}, Header: {header_val}")
            return False
            
    return True

#########################################
# SECTION 2: DATA PREPARATION
#########################################

def preprocess_data(train_df, test_df, features, label_col):
    """
    Prepares raw data for the XGBoost model.

    Machine learning models require clean, numerical data. This function performs:
    1. **Clipping**: Replaces extreme outliers (top/bottom 0.1%) with boundary values.
    2. **Imputation**: Fills missing values (NaNs) with the column median.
    3. **Scaling**: Centers and scales data using RobustScaler (better for data with outliers).
    4. **Encoding**: Converts string labels (e.g., "RR Lyrae") to integers (0, 1, ...).

    Args:
        train_df (pd.DataFrame): Training dataframe.
        test_df (pd.DataFrame): Test dataframe.
        features (list): List of feature column names to process.
        label_col (str): Name of the target label column.

    Returns:
        tuple: (X_train_processed, X_test_processed, y_train_encoded, label_encoder)
    """
    print("Preprocessing data...")
    if label_col not in train_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in training data")

    # Replace infinite values with NaN so they can be handled by the imputer
    X_train = train_df[features].copy().replace([np.inf, -np.inf], np.nan)
    X_test = test_df[features].copy().replace([np.inf, -np.inf], np.nan)

    # --- Step 1: Clip Outliers ---
    # We clip data to the 0.1st and 99.9th percentiles. This prevents a single 
    # garbage value (e.g., magnitude = 9999) from ruining the scaling.
    q001 = X_train.quantile(0.001)
    q999 = X_train.quantile(0.999)
    X_train = X_train.clip(lower=q001, upper=q999, axis=1)
    X_test = X_test.clip(lower=q001, upper=q999, axis=1)

    # --- Step 2: Impute Missing Values ---
    # Fills NaNs with the median of that column from the training set.
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # --- Step 3: Scale Features ---
    # RobustScaler uses the median and Interquartile Range (IQR). 
    # Unlike Standard Scaler (mean/std), it doesn't get distorted by outliers.
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Convert back to readable DataFrames
    X_train_processed = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)
    
    # --- Step 4: Encode Labels ---
    label_encoder = LabelEncoder()
    train_labels = train_df[label_col].fillna('UNKNOWN')
    y_train_encoded = label_encoder.fit_transform(train_labels)
    
    # Print class stats for the user
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print("Class distribution:")
    class_names = label_encoder.classes_
    for i, cls in enumerate(class_names):
        count = (y_train_encoded == i).sum().item()
        print(f"  {i}: {cls} - {count} samples ({count/len(y_train_encoded)*100:.1f}%)")

    return X_train_processed, X_test_processed, y_train_encoded, label_encoder


#########################################
# SECTION 3: WORKFLOW FUNCTIONS
#########################################

def train_xgb(train_df, test_df, features, label_col, out_file, learning_rate, max_depth,
              subsample, colsample_bytree, reg_alpha, reg_lambda, num_boost_round, early_stopping_rounds, 
              test_size, use_adaptive_lr=True):
    """
    Executes the full XGBoost training and inference pipeline.

    Flow of Control:
    1. **Label Propagation**: Fills labels in test_df if matching IDs are found in train_df.
    2. **Preprocessing**: Cleans, imputes, and scales the data.
    3. **Splitting**: Separates a validation set from the training data (stratified).
    4. **Weighting**: Calculates class weights to handle dataset imbalance.
    5. **Training**: Runs XGBoost with specified hyperparameters and callbacks.
    6. **Inference**: Predicts classes and probabilities for the test set.
    7. **Formatting**: Selects columns and builds the final output dataframe.
    8. **Deduplication**: Removes duplicate source IDs, keeping the highest confidence result.
    9. **Saving**: Saves the model (.joblib) and data (.fits).

    Args:
        train_df (pd.DataFrame): The training data.
        test_df (pd.DataFrame): The test data.
        features (list): List of feature names to use.
        label_col (str): The name of the label column.
        out_file (str): Path to save the output FITS file.
        learning_rate (float): Initial step size for the optimizer.
        max_depth (int): Max depth of a tree (controls complexity).
        subsample (float): Fraction of samples used per tree (prevents overfitting).
        colsample_bytree (float): Fraction of features used per tree.
        reg_alpha (float): L1 regularization (lasso).
        reg_lambda (float): L2 regularization (ridge).
        num_boost_round (int): Maximum number of training iterations.
        early_stopping_rounds (int): Stop if validation loss doesn't improve for N rounds.
        test_size (float): Fraction of data to use for validation (e.g., 0.05).
        use_adaptive_lr (bool): If True, reduces learning rate over time.
        
    Returns:
        tuple: (output_df, evals_result, model, label_encoder)
        Note: output_df is the final, deduplicated dataframe used for saving.
    """
    print("\n=== XGBoost Training Workflow ===")
    start_time = time.time()
    
    # --- 1. Label Propagation ---
    test_df = propagate_labels(test_df, train_df, label_col)
    
    # --- 2. Data Preparation ---
    X_train, X_test, y, label_encoder = preprocess_data(
        train_df, test_df, features, label_col
    )
    
    # --- 3. Create Train/Validation Split ---
    # Stratify ensures that the validation set has the same proportion of classes 
    # as the training set, which is crucial for imbalanced astronomical data.
    class_counts = np.bincount(y)
    stratify_flag = y if np.min(class_counts) >= 2 else None
    
    print(f"Splitting training data with test_size={test_size}...")
    # -- BASH INJECTION ANCHOR --
    
    # --- SPLIT INJECTION POINT ---
    # The bash folding script uses sed to replace this block. Do not change these marker lines.
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y, test_size=test_size, random_state=37, stratify=stratify_flag
    )
    # --- END SPLIT INJECTION POINT ---

    # --- 4. Class Weighting ---
    # Astronomical datasets are often imbalanced (e.g., many variable stars, few supernovae).
    # We calculate weights so the model pays more attention to rare classes.
    total_samples = len(y_train_main)
    num_classes = len(class_counts)
    class_weights_map = {
        i: np.sqrt(total_samples / (num_classes * count)) if count > 0 else 1
        for i, count in enumerate(np.bincount(y_train_main))
    }
    sample_weights = np.array([class_weights_map[label] for label in y_train_main])
    print("Applied class weights to training data.")

    # --- 5. XGBoost Setup ---
    # DMatrix is an internal data structure that XGBoost uses for speed and memory efficiency.
    dtrain = xgb.DMatrix(X_train_main, label=y_train_main, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    num_gpus = get_gpu_count()
    print(f"Auto-detected {num_gpus} GPUs.")
    
    params = {
        'objective': 'multi:softprob', # Output probability distribution over classes
        'num_class': len(label_encoder.classes_),
        'eval_metric': 'mlogloss',     # Multi-class Log Loss (standard error metric)
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'min_child_weight': 10,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'colsample_bylevel': 0.7,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'tree_method': 'hist',         # Histogram-based method (fast)
        'grow_policy': 'lossguide',
        'device': 'cuda' if num_gpus > 0 else 'cpu',
        'seed': 37                     # Fix XGBoost randomness for reproducibility
    }
    
    # Copy params to a dict for saving later.
    hyperparams_to_save = params.copy()
    hyperparams_to_save['num_boost_round_config'] = num_boost_round
    hyperparams_to_save['early_stopping_rounds_config'] = early_stopping_rounds
    hyperparams_to_save['test_size_config'] = test_size
    hyperparams_to_save['use_adaptive_lr'] = use_adaptive_lr

    # --- Adaptive Learning Rate (Cosine Decay) ---
    callbacks_list = []
    if use_adaptive_lr:
        print(f"Enabling Adaptive Learning Rate (Cosine Decay). Initial: {learning_rate}")
        
        def cosine_decay_scheduler(boosting_round):
            """
            Slowly lowers learning rate following a cosine curve.
            This helps the model 'settle' into a minimum as it gets closer to the solution.
            """
            min_lr = 1e-3 # Minimum learning rate floor
            progress = boosting_round / num_boost_round 
            new_lr = min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * progress))
            return new_lr

        callbacks_list.append(xgb.callback.LearningRateScheduler(cosine_decay_scheduler))

    # --- 6. Model Training ---
    evals_result = {}
    print("\nStarting XGBoost training...")
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        evals_result=evals_result,
        verbose_eval=100,
        callbacks=callbacks_list
    )
    
    # --- 7. Prediction ---
    best_iter = model.best_iteration
    hyperparams_to_save['best_iteration_result'] = best_iter
    print(f"\nTraining complete. Best iteration: {best_iter}")

    # Log best losses for the FITS header
    if 'train' in evals_result and 'mlogloss' in evals_result['train'] and len(evals_result['train']['mlogloss']) > best_iter:
        best_train_loss = evals_result['train']['mlogloss'][best_iter]
        hyperparams_to_save['best_train_loss'] = float(best_train_loss)
        print(f"  Best Train mlogloss: {best_train_loss:.6f}")
    
    if 'validation' in evals_result and 'mlogloss' in evals_result['validation'] and len(evals_result['validation']['mlogloss']) > best_iter:
        best_val_loss = evals_result['validation']['mlogloss'][best_iter]
        hyperparams_to_save['best_val_loss'] = float(best_val_loss)
        print(f"  Best Validation mlogloss: {best_val_loss:.6f}")

    # Generate predictions using the best state of the model
    probs = model.predict(dtest, iteration_range=(0, best_iter + 1))
    preds = np.argmax(probs, axis=1) # The class index with highest probability
    
    # Calculate Confidence: Difference between top-1 and top-2 probability
    top_two = np.partition(probs, -2, axis=1)[:, -2:]
    confs = np.max(top_two, axis=1) - np.min(top_two, axis=1)
    
    # Convert integer predictions back to string labels
    pred_labels = label_encoder.inverse_transform(preds)
    
    # Calculate Entropy (uncertainty measure)
    log_probs = np.log(probs, out=np.zeros_like(probs, dtype=float), where=(probs > 0))
    entropy = np.abs(np.sum(probs * log_probs, axis=1))

    # Create the FULL results dataframe (internal use only, for debugging if needed)
    test_df_result = test_df.copy()

    if label_col in test_df.columns:
        test_df_result['xgb_training_class'] = test_df[label_col]
    else:
        print(f"Warning: Label column '{label_col}' missing in test data even after propagation attempt.")
        test_df_result[label_col] = 'UNKNOWN'
        test_df_result['xgb_training_class'] = 'UNKNOWN'
    
    test_df_result['xgb_predicted_class'] = pred_labels
    test_df_result['xgb_confidence'] = confs
    test_df_result['xgb_entropy'] = entropy
    for i, class_name in enumerate(label_encoder.classes_):
        test_df_result[f'prob_{class_name}'] = probs[:, i]

    # --- 8. Build Selective Output FITS File ---
    print("Building selective output FITS file...")
    
    # We create a cleaner output file that only contains IDs, coordinates, inputs, and results.
    cols_to_export = []
    
    # Find ID columns (Changed to include ALL id columns for deduplication logic)
    # We explicitly look for 'uniqueid', 'sourceid', 'source_id', and 'id' to be robust
    potential_id_cols = ['uniqueid', 'sourceid', 'source_id', 'id']
    for col in potential_id_cols:
        if col in test_df.columns and col not in cols_to_export:
            cols_to_export.append(col)
            print(f"  Adding ID column to output: {col}")

    if not cols_to_export:
        print("  Warning: No 'uniqueid', 'sourceid', 'source_id' or 'id' column found.")

    # Find Coordinate columns
    coord_cols = ['ra', 'ra_error', 'dec', 'dec_error', 'l', 'b']
    print(f"  Adding coordinate columns: {coord_cols}")
    for col in coord_cols:
        if col in test_df.columns:
            if col not in cols_to_export:
                cols_to_export.append(col)

    # Add Feature columns
    print(f"  Adding {len(features)} parameter (feature) columns...")
    for col in features:
        if col not in cols_to_export:
            cols_to_export.append(col)

    output_df = test_df[cols_to_export].copy()
    
    # Add Prediction columns
    print("  Adding inference output columns...")
    output_df['xgb_predicted_class'] = pred_labels
    output_df['xgb_confidence'] = confs
    output_df['xgb_entropy'] = entropy

    if 'xgb_training_class' in test_df_result.columns:
        print("  Adding 'xgb_training_class' (true label) to output.")
        output_df['xgb_training_class'] = test_df_result['xgb_training_class']
    
    for i, class_name in enumerate(label_encoder.classes_):
        output_df[f'prob_{class_name}'] = probs[:, i]
    
    # For each sourceid (or source_id), keep only the uniqueid with the highest confidence.
    # We allow fallback to 'source_id' if 'sourceid' is missing to be robust.
    dedupe_col = None
    if 'sourceid' in output_df.columns:
        dedupe_col = 'sourceid'
    elif 'source_id' in output_df.columns:
        dedupe_col = 'source_id'
        
    if dedupe_col:
        print(f"  Deduplicating based on '{dedupe_col}' (keeping highest confidence, preserving ID in output)...")
        len_before = len(output_df)
        # Sort by confidence descending, so highest confidence comes first
        output_df = output_df.sort_values(by='xgb_confidence', ascending=False)
        # Drop duplicates based on the ID, keeping the first one (which is the highest confidence)
        output_df = output_df.drop_duplicates(subset=[dedupe_col], keep='first')
        # Sort by ID for clean output
        output_df = output_df.sort_values(by=dedupe_col)
        print(f"  Removed {len_before - len(output_df)} duplicate rows. Final count: {len(output_df)}")
    else:
        print("  Skipping deduplication: Neither 'sourceid' nor 'source_id' found in output.")

    # --- 9. Save Model and Data ---
    
    # Save model object (joblib)
    model_file_path = 'xgb_model.joblib'
    joblib.dump((model, label_encoder, evals_result), model_file_path)
    print(f"Saved trained XGBoost model, label encoder, and training history to {model_file_path}")
    
    # Save FITS file
    out_fits_file, _ = os.path.splitext(out_file)
    out_fits_file += ".fits"
    
    out_table = Table.from_pandas(output_df) 

    print("Adding hyperparameters to FITS header...")
    out_table.meta['COMMENT'] = '--- XGBOOST HYPERPARAMETERS ---'
    for key, value in hyperparams_to_save.items():
        if value is None:
            value_to_save = 'None'
        else:
            value_to_save = value
        out_table.meta[key] = value_to_save

    out_table.write(out_fits_file, format='fits', overwrite=True)
    print(f"Saved selective predictions and data to {out_fits_file}")
    
    elapsed_time = time.time() - start_time
    print(f"\nXGBoost workflow completed in {elapsed_time:.1f} seconds.")
    
    # RETURN the deduplicated, final dataframe (output_df) instead of test_df_result
    return output_df, evals_result, model, label_encoder

def generate_visualizations(model, label_encoder, test_df_result, label_col, evals_result=None):
    """
    Generates diagnostic plots to evaluate the model's performance.

    Plots generated (saved in 'figures/' folder):
    1. **Classification Report**: Precision/Recall/F1-score textual report.
    2. **Feature Importance**: Which physics parameters mattered most?
    3. **Training Loss**: Did the model converge? (Log-loss vs Iterations)
    4. **Bailey Diagram**: (Specific to astronomy) Period vs Amplitude plot.
    5. **Confidence Distribution**: How sure is the model about its predictions?
    6. **Confidence Entropy**: Another view of uncertainty.
    
    Args:
        model (xgb.Booster): The trained XGBoost model.
        label_encoder (LabelEncoder): The encoder used to translate class names.
        test_df_result (pd.DataFrame): DataFrame containing predictions and probabilities.
        label_col (str): Name of the true label column.
        evals_result (dict, optional): Training history for loss plotting.
        
    Returns:
        dict: A dictionary of optimal probability thresholds derived from PR curves.
    """
    print("\nGenerating visualizations...")
    os.makedirs('figures', exist_ok=True)
    
    # --- Classification Report ---
    # We prefer the propagated labels ('xgb_training_class') if available.
    true_label_col_to_use = 'xgb_training_class'
    if true_label_col_to_use not in test_df_result.columns:
        print(f"Warning: '{true_label_col_to_use}' not found, falling back to '{label_col}' for classification report.")
        true_label_col_to_use = label_col 

    if true_label_col_to_use in test_df_result.columns:
        y_true = test_df_result[true_label_col_to_use].fillna('UNKNOWN')
        pred_labels = test_df_result['xgb_predicted_class']
        print("\nTest set evaluation (on deduplicated data):")
        print(classification_report(y_true, pred_labels, zero_division=0))
    else:
        print(f"\nSkipping classification report: True label column ('{true_label_col_to_use}') not in test data.")
    
    # --- Feature Importance ---
    importance = model.get_score(importance_type='gain')
    top_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    feat_names = [x[0] for x in top_feats]
    scores = [x[1] for x in top_feats]
    plot_xgb_feature_importance(feat_names, scores, output_dir='figures')

    # --- Training Loss ---
    if evals_result:
        plot_xgb_training_loss(evals_result, output_dir='figures')
    else:
        print("Skipping training loss plot (no evals_result provided).")

    # --- Other Plots ---
    confs = test_df_result['xgb_confidence']
    preds_int = label_encoder.transform(test_df_result['xgb_predicted_class'])
    
    # plot_and_print_auc_ap calculates Precision-Recall curves and determines
    # the optimal probability threshold for each class.
    thresholds_dict = plot_and_print_auc_ap(test_df_result, true_label_col_to_use, label_encoder, output_dir='figures')
    
    plot_bailey_diagram(test_df_result, "xgb_predicted_class", output_dir='figures', thresholds_dict=thresholds_dict)
    plot_confidence_distribution(confs, preds_int, label_encoder.classes_, output_dir="figures")
    plot_confidence_entropy(test_df_result, "xgb_predicted_class", output_dir='figures', use_brg_cmap=True)
    
    print("Visualizations saved to: figures/")
    
    return thresholds_dict


#########################################
# SECTION 4: SCRIPT ENTRY POINT
#########################################

def main():
    """
    Main execution logic.
    
    1. Checks command line arguments for file paths.
    2. Checks if valid output files already exist (Caching).
    3. If cached files match current settings -> Skip training, Load data, Visualize.
    4. If not -> Load data, Train model, Save data, Visualize.
    """
    # --- Hyperparameters ---
    # These control how the XGBoost model learns.
    set_learning_rate = 1E-3         # Initial step size
    set_max_depth = 50               # Max tree depth (higher = more complex model)
    set_subsample = 0.95             # % of data used per tree (prevent overfitting)
    set_colsample_bytree = 0.1       # % of features used per tree
    set_reg_alpha = 0.01             # L1 Regularization
    set_reg_lambda = 0.1             # L2 Regularization
    set_num_boost_round = 1000000    # Max iterations (very high because we use early stopping)
    set_early_stopping_rounds = 5000 # Stop if no improvement for this many rounds
    set_use_adaptive_lr = True       # Use cosine decay scheduler
    set_test_size = 0.05             # % of training data reserved for validation

    # --- File Path Handling ---
    if len(sys.argv) >= 3:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        out_file = sys.argv[3] if len(sys.argv) > 3 else "xgb_predictions.fits"
    else:
        # Default paths for easy testing/debugging
        print("Using default file paths.")
        train_path = ".data/PRIMVS_P_training_new.fits"
        test_path = ".data/PRIMVS_P.fits"
        out_file = "xgb_predictions.fits"
        
    # Define output paths
    model_file_path = 'xgb_model.joblib'
    out_fits_file, _ = os.path.splitext(out_file)
    out_fits_file += ".fits"

    # --- Main Execution Block ---
    try:
        # Load training labels. This file is now assumed to be degenerate 
        # (mostly just IDs and labels, with no features).
        train_labels_df = load_fits_to_df(train_path)
        
        # Identify the column containing the class labels (e.g., "Type")
        label_col = "Type"
        if label_col not in train_labels_df.columns:
            # Fallback: search for columns like 'class' or 'type'
            available_cols = [c for c in train_labels_df.columns if 'class' in c.lower() or 'type' in c.lower()]
            if available_cols:
                label_col = available_cols[0]
                print(f"Warning: Using fallback label column: {label_col}")
            else:
                raise ValueError("Could not find a suitable label column ('Type', 'class', etc.) in the training data.")
        
        # Calculate parameters needed for the check
        le_for_check = LabelEncoder()
        train_labels = train_labels_df[label_col].fillna('UNKNOWN')
        y_train_encoded = le_for_check.fit_transform(train_labels)
        num_classes = len(le_for_check.classes_)
        num_gpus = get_gpu_count()

        # Build dictionary of current parameters to compare against cached file
        current_params = {
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'eval_metric': 'mlogloss',
            'learning_rate': set_learning_rate,
            'max_depth': set_max_depth,
            'min_child_weight': 10,
            'subsample': set_subsample,
            'colsample_bytree': set_colsample_bytree,
            'colsample_bylevel': 0.7,
            'reg_alpha': set_reg_alpha,
            'reg_lambda': set_reg_lambda,
            'tree_method': 'hist',
            'grow_policy': 'lossguide',
            'device': 'cuda' if num_gpus > 0 else 'cpu',
            'seed': 37,
            'num_boost_round_config': set_num_boost_round,
            'early_stopping_rounds_config': set_early_stopping_rounds,
            'test_size_config': set_test_size,
            'use_adaptive_lr': set_use_adaptive_lr
        }
        
        # --- Caching Check ---
        # If output files exist, check if we can reuse them instead of retraining.
        if os.path.exists(out_fits_file) and os.path.exists(model_file_path):
            print(f"Found existing files: {out_fits_file} and {model_file_path}")
            
            with fits.open(out_fits_file) as hdul:
                header = hdul[1].header 
                are_params_identical = check_hyperparameters(header, current_params)
                
                if are_params_identical:
                    print("Hyperparameters match. Skipping training and inference.")
                    
                    # 1. Load cached predictions from FITS
                    # We load directly from the output FITS file, which guarantees we are using
                    # the deduplicated data that was previously saved.
                    print(f"  Loading data from {out_fits_file}...")
                    test_df_result_table = Table(hdul[1].data) 
                    test_df_result_final = test_df_result_table.to_pandas()
                    print("  Data loaded.")

                    # Fix potential byte-string encoding issues in FITS columns
                    if not test_df_result_final.empty and isinstance(test_df_result_final['xgb_predicted_class'].iloc[0], bytes):
                        print("  Converting 'xgb_predicted_class' column from bytes to string for plotting...")
                        test_df_result_final['xgb_predicted_class'] = test_df_result_final['xgb_predicted_class'].str.decode('utf-8')
                    
                    # 2. Load trained model and history
                    print(f"  Loading model from {model_file_path}...")
                    loaded_joblib = joblib.load(model_file_path)
                    
                    # Handle backward compatibility for older joblib files that might not have evals_result
                    if len(loaded_joblib) == 3:
                        model, label_encoder, evals_result = loaded_joblib
                        print("  Model, encoder, and history loaded.")
                    else:
                        model, label_encoder = loaded_joblib
                        evals_result = None
                        print("  Model and encoder loaded (no history found).")

                    # 3. Generate plots using the loaded data
                    thresholds_dict = generate_visualizations(
                        model, label_encoder, test_df_result_final, 
                        label_col, evals_result=evals_result
                    )
                    
                    save_thresholds_to_header(out_fits_file, thresholds_dict)
                    print("\n=== XGBoost Visualization Complete (Skipped Training) ===")
                    return # Exit script successfully
                
                else:
                    print("Hyperparameters mismatch. Re-training model.")
            
        else:
            print("Output files not found. Starting new training session.")

        # --- Full Training Workflow ---
        # If we didn't exit above, we need to train the model.
        
        # 1. Load the full dataset (PRIMVS_P.fits) which acts as our test_df AND provides features for train_df
        print("Loading full dataset to extract features...")
        test_df = load_fits_to_df(test_path)
        
        # 2. Construct train_df by cross-matching IDs and deduplicating based on best_FAP
        print("\nConstructing full training dataset from matched IDs...")
        potential_id_cols = ['sourceid', 'source_id', 'id', 'uniqueid']
        id_col = None
        for col in potential_id_cols:
            if col in train_labels_df.columns and col in test_df.columns:
                id_col = col
                break
                
        if not id_col:
            raise ValueError(f"Could not find a common ID column to match training labels with features. Looked for: {potential_id_cols}")

        print(f"  Matching based on ID column: '{id_col}'")
        
        # Extract rows from the full dataset that belong to our training set
        train_features_df = test_df[test_df[id_col].isin(train_labels_df[id_col])].copy()
        
        # Deduplicate duplicated source IDs using 'best_FAP' (lowest is best)
        fap_col = next((c for c in train_features_df.columns if c.lower() == 'best_fap'), None)
        if fap_col:
            print(f"  Deduplicating training data using '{fap_col}' (keeping minimum value)...")
            train_features_df = train_features_df.sort_values(by=fap_col, ascending=True)
            train_features_df = train_features_df.drop_duplicates(subset=[id_col], keep='first')
        else:
            print("  Warning: 'best_FAP' column not found. Deduplicating by keeping the first occurrence.")
            train_features_df = train_features_df.drop_duplicates(subset=[id_col], keep='first')
            
        # Merge the features with the labels
        # Drop label_col from train_features_df if it happens to exist to prevent '_x'/'_y' suffixing
        if label_col in train_features_df.columns:
            train_features_df = train_features_df.drop(columns=[label_col])
            
        # Clean the original labels just in case it had duplicates itself
        train_labels_clean = train_labels_df.drop_duplicates(subset=[id_col])
        train_df = pd.merge(
            train_features_df, 
            train_labels_clean[[id_col, label_col]], 
            on=id_col, 
            how='inner'
        )
        print(f"  Final training dataset constructed with {len(train_df)} unique sources.\n")

        # Now get our feature list using our constructed train_df and test_df
        features = get_feature_list(train_df, test_df)
        if not features:
            raise ValueError("No common features found between training and test sets.")
            
        # Run training
        # Note: train_xgb returns output_df (the deduplicated dataframe)
        output_df, evals_result, model, label_encoder = train_xgb(
            train_df=train_df, test_df=test_df, features=features, label_col=label_col,
            out_file=out_file, learning_rate=set_learning_rate, max_depth=set_max_depth,
            subsample=set_subsample, colsample_bytree=set_colsample_bytree,
            reg_alpha=set_reg_alpha, reg_lambda=set_reg_lambda,
            num_boost_round=set_num_boost_round, early_stopping_rounds=set_early_stopping_rounds,
            test_size=set_test_size,
            use_adaptive_lr=set_use_adaptive_lr
        )
        
        # Run visualization on the OUTPUT dataframe (which is deduplicated)
        thresholds_dict = generate_visualizations(
            model, label_encoder, output_df, 
            label_col, evals_result=evals_result
        )

        # Update output file
        save_thresholds_to_header(out_fits_file, thresholds_dict)

        print("\n=== XGBoost Training Complete ===")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        gc.collect()
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
