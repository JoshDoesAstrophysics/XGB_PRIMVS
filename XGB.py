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
- Selective Output: The output FITS file now contains only the unique ID,
  the input feature (parameter) columns, and the inference output columns.

To run from the command line:
    python GXGB.py <path_to_train.fits> <path_to_test.fits> [output_file.fits]
"""
import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
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
        na_count = target_df[label_col].isna().sum()
        if na_count > 0:
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
    
    # Assumes 128-dimensional embeddings from a contrastive learning model.
    embeddings = [str(i) for i in range(128)]
    full_feature_set = features + embeddings
    
    # Determine which of the predefined features are actually available.
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    common_cols = train_cols.intersection(test_cols)
    usable_features = [f for f in full_feature_set if f in common_cols]
    
    print(f"Available features in train: {len(train_cols)}")
    print(f"Available features in test: {len(test_cols)}")
    print(f"Common features: {len(common_cols)}")
    print(f"Using {len(usable_features)} of {len(full_feature_set)} predefined features.")
    
    if not usable_features:
        print("Warning: No predefined features found. Falling back to all common numeric columns.")
        exclude_cols = {'source_id', 'id', 'index', 'best_class_name', 'uniqueid'}
        usable_features = [
            col for col in common_cols
            if pd.api.types.is_numeric_dtype(train[col]) and col not in exclude_cols
        ]
        print(f"Using {len(usable_features)} common numeric features as a fallback.")

    return usable_features

def get_gpu_count():
    """
    Detect the number of available NVIDIA GPUs.

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
    Compare current hyperparameters with those in a FITS header.

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
        except (ValueError, TypeError):
            print(f"  Mismatch: Type conversion failed for key '{key}'. Current: {current_val}, Header: {header_val}")
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
        tuple: A tuple containing processed dataframes and the fitted label encoder:
               (X_train_processed, X_test_processed, y_train_encoded, label_encoder).
    """
    print("Preprocessing data...")
    if label_col not in train_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in training data")

    X_train = train_df[features].copy().replace([np.inf, -np.inf], np.nan)
    X_test = test_df[features].copy().replace([np.inf, -np.inf], np.nan)

    # Clip outliers based on training data distribution.
    q001 = X_train.quantile(0.001)
    q999 = X_train.quantile(0.999)
    X_train = X_train.clip(lower=q001, upper=q999, axis=1)
    X_test = X_test.clip(lower=q001, upper=q999, axis=1)

    # Impute missing values with the median from the training data.
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Scale features using a scaler robust to outliers.
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Convert back to DataFrames
    X_train_processed = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)
    
    # Encode labels.
    label_encoder = LabelEncoder()
    train_labels = train_df[label_col].fillna('UNKNOWN')
    y_train_encoded = label_encoder.fit_transform(train_labels)
    
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
    Execute the XGBoost training and inference workflow.

    Args:
        train_df (pd.DataFrame): The training data.
        test_df (pd.DataFrame): The test data.
        features (list): List of feature names.
        label_col (str): The name of the label column.
        out_file (str): Path to save the output FITS file.
        learning_rate (float): Step size shrinkage.
        max_depth (int): Maximum depth of a tree.
        subsample (float): Subsample ratio of the training instance.
        colsample_bytree (float): Subsample ratio of columns when constructing each tree.
        reg_alpha (float): L1 regularization term on weights.
        reg_lambda (float): L2 regularization term on weights.
        num_boost_round (int): Number of boosting rounds.
        early_stopping_rounds (int): Activates early stopping.
        test_size (float): Proportion of the dataset to include in the validation split.
        use_adaptive_lr (bool): If True, applies cosine decay to learning rate.
        
    Returns:
        tuple: (test_df_result, evals_result, model, label_encoder)
    """
    print("\n=== XGBoost Training Workflow ===")
    start_time = time.time()
    
    # --- 0. Label Propagation ---
    # Match labels from train_df to test_df BEFORE preprocessing using unique IDs
    test_df = propagate_labels(test_df, train_df, label_col)
    
    # --- 1. Data Preparation ---
    X_train, X_test, y, label_encoder = preprocess_data(
        train_df, test_df, features, label_col
    )
    
    # --- 2. Create Train/Validation Split ---
    # Stratify to handle imbalanced classes, with a fallback for rare classes.
    class_counts = np.bincount(y)
    stratify_flag = y if np.min(class_counts) >= 2 else None
    
    print(f"Splitting training data with test_size={test_size}...")
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y, test_size=test_size, random_state=42, stratify=stratify_flag
    )

    # --- 3. Class Weighting ---
    # Smoothed class weighting to handle imbalance
    total_samples = len(y_train_main)
    num_classes = len(class_counts)
    class_weights_map = {
        i: np.sqrt(total_samples / (num_classes * count)) if count > 0 else 1
        for i, count in enumerate(np.bincount(y_train_main))
    }
    sample_weights = np.array([class_weights_map[label] for label in y_train_main])
    print("Applied class weights to training data.")

    # --- 4. Model, Parameters, and Data Setup ---
    dtrain = xgb.DMatrix(X_train_main, label=y_train_main, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    num_gpus = get_gpu_count()
    print(f"Auto-detected {num_gpus} GPUs.")
    
    params = {
        'objective': 'multi:softprob',
        'num_class': len(label_encoder.classes_),
        'eval_metric': 'mlogloss',
        'learning_rate': learning_rate, # This is the INITIAL learning rate
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
    
    hyperparams_to_save = params.copy()
    hyperparams_to_save['num_boost_round_config'] = num_boost_round
    hyperparams_to_save['early_stopping_rounds_config'] = early_stopping_rounds
    hyperparams_to_save['test_size_config'] = test_size
    hyperparams_to_save['use_adaptive_lr'] = use_adaptive_lr

    # --- 5. Adaptive Learning Rate Scheduler (Cosine Decay) ---
    callbacks_list = []
    if use_adaptive_lr:
        print(f"Enabling Adaptive Learning Rate (Cosine Decay). Initial: {learning_rate}")
        
        def cosine_decay_scheduler(boosting_round):
            """
            Cosine annealing scheduler.
            Decreases LR from initial 'learning_rate' to 'min_lr' following a cosine curve.
            """
            min_lr = 1e-3 # Floor for the learning rate
            # Total iterations roughly estimated by num_boost_round
            # Note: With early stopping, this might decay slower than expected, but is generally stable.
            progress = boosting_round / num_boost_round 
            
            # Compute new LR
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
        callbacks=callbacks_list # <--- Injected here
    )
    
    # --- 7. Prediction and Evaluation ---
    best_iter = model.best_iteration
    hyperparams_to_save['best_iteration_result'] = best_iter
    print(f"\nTraining complete. Best iteration: {best_iter}")

    # Store best training and validation losses for FITS header
    if 'train' in evals_result and 'mlogloss' in evals_result['train'] and len(evals_result['train']['mlogloss']) > best_iter:
        best_train_loss = evals_result['train']['mlogloss'][best_iter]
        hyperparams_to_save['best_train_loss'] = float(best_train_loss) # Ensure it's a standard float
        print(f"  Best Train mlogloss: {best_train_loss:.6f}")
    
    if 'validation' in evals_result and 'mlogloss' in evals_result['validation'] and len(evals_result['validation']['mlogloss']) > best_iter:
        best_val_loss = evals_result['validation']['mlogloss'][best_iter]
        hyperparams_to_save['best_val_loss'] = float(best_val_loss) # Ensure it's a standard float
        print(f"  Best Validation mlogloss: {best_val_loss:.6f}")

    probs = model.predict(dtest, iteration_range=(0, best_iter + 1))
    preds = np.argmax(probs, axis=1)
    top_two = np.partition(probs, -2, axis=1)[:, -2:]
    confs = np.max(top_two, axis=1) - np.min(top_two, axis=1)
    pred_labels = label_encoder.inverse_transform(preds)
    log_probs = np.log(probs, out=np.zeros_like(probs, dtype=float), where=(probs > 0))
    entropy = np.abs(np.sum(probs * log_probs, axis=1))

    # Create the FULL dataframe for visualization and return
    # This dataframe contains all original test columns + predictions
    test_df_result = test_df.copy()

    # If the original true label column exists in the test data (it should now via propagation),
    # copy it to a new standard column name 'xgb_training_class'.
    if label_col in test_df.columns:
        test_df_result['xgb_training_class'] = test_df[label_col]
    else:
        # Force creation of the column with 'UNKNOWN' values if missing (shouldn't happen with prop)
        print(f"Warning: Label column '{label_col}' missing in test data even after propagation attempt.")
        test_df_result[label_col] = 'UNKNOWN'
        test_df_result['xgb_training_class'] = 'UNKNOWN'
    
    test_df_result['xgb_predicted_class'] = pred_labels
    test_df_result['xgb_confidence'] = confs
    test_df_result['xgb_entropy'] = entropy
    for i, class_name in enumerate(label_encoder.classes_):
        test_df_result[f'prob_{class_name}'] = probs[:, i]


    # --- Create the SELECTIVE output DataFrame for FITS file ---
    print("Building selective output FITS file...")
    
    # Start with a new DataFrame, preserving the index
    output_df = pd.DataFrame(index=test_df.index) 
    
    # 1. Add the uniqueid column (if found)
    id_col = None
    potential_id_cols = ['uniqueid', 'source_id', 'id'] # Prioritize user's request
    for col in potential_id_cols:
        if col in test_df.columns:
            id_col = col
            break
    
    if id_col:
        print(f"  Adding ID column: {id_col}")
        output_df[id_col] = test_df[id_col]
    else:
        print("  Warning: No 'uniqueid', 'source_id', or 'id' column found.")

    # 2. Add the parameter columns (features) from the original test_df
    print(f"  Adding {len(features)} parameter (feature) columns...")
    output_df = pd.concat([output_df, test_df[features]], axis=1)
    
    # 3. Add the output columns from inference
    print("  Adding inference output columns...")
    output_df['xgb_predicted_class'] = pred_labels
    output_df['xgb_confidence'] = confs
    output_df['xgb_entropy'] = entropy

    # Also add the training class, if it was available.
    if 'xgb_training_class' in test_df_result.columns:
        print("  Adding 'xgb_training_class' (true label) to output.")
        output_df['xgb_training_class'] = test_df_result['xgb_training_class']
    
    # Add per-class probabilities to the output
    for i, class_name in enumerate(label_encoder.classes_):
        output_df[f'prob_{class_name}'] = probs[:, i]

    # Save model object and label encoder
    model_file_path = 'xgb_model.joblib'
    joblib.dump((model, label_encoder), model_file_path)
    print(f"Saved trained XGBoost model and label encoder to {model_file_path}")
    
    out_fits_file, _ = os.path.splitext(out_file)
    out_fits_file += ".fits"
    
    # Use the selective output_df to create the Astropy Table
    out_table = Table.from_pandas(output_df) 

    print("Adding hyperparameters to FITS header...")
    out_table.meta['COMMENT'] = '--- XGBOOST HYPERPARAMETERS ---'
    for key, value in hyperparams_to_save.items():
        # FITS headers don't like None, so convert to string.
        if value is None:
            value_to_save = 'None'
        else:
            value_to_save = value
        
        # Add to meta dictionary. astropy will format this into the header.
        out_table.meta[key] = value_to_save

    out_table.write(out_fits_file, format='fits', overwrite=True)
    print(f"Saved selective predictions and data to {out_fits_file}")
    
    elapsed_time = time.time() - start_time
    print(f"\nXGBoost workflow completed in {elapsed_time:.1f} seconds.")
    
    # Return the FULL dataframe for visualizations
    return test_df_result, evals_result, model, label_encoder

def generate_visualizations(model, label_encoder, test_df_result, label_col, evals_result=None):
    """
    Generate and save all visualization plots.
    
    Args:
        model (xgb.Booster): The trained XGBoost model.
        label_encoder (LabelEncoder): The fitted LabelEncoder.
        test_df_result (pd.DataFrame): DataFrame with test data and predictions.
                                     (This is the FULL dataframe, not the selective one)
        label_col (str): Name of the true label column (for evaluation).
        evals_result (dict, optional): Results from training (for loss plot).
    """
    print("\nGenerating visualizations...")
    os.makedirs('figures', exist_ok=True)
    
    # --- Classification Report ---
    # We now check for 'xgb_training_class' first. If it's not there,
    # we fall back to the original 'label_col'. The plotting function
    # 'plot_and_print_auc_ap' has its own internal check, but we do
    # one here for the classification report.
    true_label_col_to_use = 'xgb_training_class'
    if true_label_col_to_use not in test_df_result.columns:
        print(f"Warning: '{true_label_col_to_use}' not found, falling back to '{label_col}' for classification report.")
        true_label_col_to_use = label_col # Fallback

    if true_label_col_to_use in test_df_result.columns:
        y_true = test_df_result[true_label_col_to_use].fillna('UNKNOWN')
        pred_labels = test_df_result['xgb_predicted_class']
        print("\nTest set evaluation:")
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
    # Re-create preds (int) and confs from the dataframe for plotting
    confs = test_df_result['xgb_confidence']
    preds_int = label_encoder.transform(test_df_result['xgb_predicted_class'])
    
    # Pass the correctly determined label column name
    plot_and_print_auc_ap(test_df_result, true_label_col_to_use, label_encoder, output_dir='figures')
    
    plot_bailey_diagram(test_df_result, "xgb_predicted_class", output_dir='figures')
    plot_confidence_distribution(confs, preds_int, label_encoder.classes_, output_dir="figures")
    plot_confidence_entropy(test_df_result, "xgb_predicted_class", output_dir='figures')
    
    print("Visualizations saved to: figures/")


#########################################
# SECTION 4: SCRIPT ENTRY POINT
#########################################

def main():
    """Main execution function of the script."""
    # --- Hyperparameters ---
    set_learning_rate = 1E-3
    set_max_depth = 50
    set_subsample = 0.95
    set_colsample_bytree = 0.1
    set_reg_alpha = 0.01
    set_reg_lambda = 0.1
    set_num_boost_round = 1000000 
    set_early_stopping_rounds = 5000
    set_use_adaptive_lr = True
    set_test_size = 0.2

    # --- File Path Handling ---
    if len(sys.argv) >= 3:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        out_file = sys.argv[3] if len(sys.argv) > 3 else "xgb_predictions.fits"
    else:
        # Provide default paths for easier testing.
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
        # Load training data first to check num_classes for hyperparam dict
        train_df = load_fits_to_df(train_path)
        
        # Auto-detect label column if not 'Type'.
        label_col = "Type"
        if label_col not in train_df.columns:
            available_cols = [c for c in train_df.columns if 'class' in c.lower() or 'type' in c.lower()]
            if available_cols:
                label_col = available_cols[0]
                print(f"Warning: Using fallback label column: {label_col}")
            else:
                raise ValueError("Could not find a suitable label column ('Type', 'class', etc.).")
        
        # Get num_classes for the hyperparameter dictionary
        le_for_check = LabelEncoder()
        train_labels = train_df[label_col].fillna('UNKNOWN')
        y_train_encoded = le_for_check.fit_transform(train_labels)
        num_classes = len(le_for_check.classes_)
        num_gpus = get_gpu_count()

        # Build the *current* hyperparameter dictionary for comparison
        # This MUST match the 'params' dict inside train_xgb
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
            'num_boost_round_config': set_num_boost_round,
            'early_stopping_rounds_config': set_early_stopping_rounds,
            'test_size_config': set_test_size,
            'use_adaptive_lr': set_use_adaptive_lr # Included in check
        }
        
        # --- Caching Check ---
        if os.path.exists(out_fits_file) and os.path.exists(model_file_path):
            print(f"Found existing files: {out_fits_file} and {model_file_path}")
            
            with fits.open(out_fits_file) as hdul:
                # Use header from extension 1 (the table)
                header = hdul[1].header 
            
                are_params_identical = check_hyperparameters(header, current_params)
                
                if are_params_identical:
                    print("Hyperparameters match. Skipping training and inference.")
                    
                    # Load existing data from the (now) open FITS handle
                    print(f"  Loading data from {out_fits_file}...")
                    test_df_result_table = Table(hdul[1].data) 
                    
                    # This 'test_df_result' is the MINIMAL one.
                    # We need to load the FULL test_df for visualizations
                    print(f"  Loading full test data from {test_path} for visualization...")
                    test_df_full_for_vis = load_fits_to_df(test_path)
                    
                    # Propagate labels in cached path to ensure viz works correctly
                    test_df_full_for_vis = propagate_labels(test_df_full_for_vis, train_df, label_col)

                    # Convert the minimal FITS data to pandas
                    test_df_minimal = test_df_result_table.to_pandas()
                    print("  Data loaded.")

                    # Fix for FITS byte-to-string conversion issue
                    if not test_df_minimal.empty and isinstance(test_df_minimal['xgb_predicted_class'].iloc[0], bytes):
                        print("  Converting 'xgb_predicted_class' column from bytes to string for plotting...")
                        test_df_minimal['xgb_predicted_class'] = test_df_minimal['xgb_predicted_class'].str.decode('utf-8')
                    
                    # We must re-create the 'test_df_result' that generate_visualizations expects,
                    # which is the FULL test_df plus the prediction columns.
                    
                    # Get prediction columns from the minimal df
                    pred_cols = [col for col in test_df_minimal.columns if col.startswith('xgb_') or col.startswith('prob_') or col == 'xgb_training_class']
                    
                    # Join them back onto the full test_df
                    # This assumes the index is aligned, which it should be.
                    test_df_result_for_vis = test_df_full_for_vis.join(test_df_minimal[pred_cols])
                    
                    
                    print(f"  Loading model from {model_file_path}...")
                    (model, label_encoder) = joblib.load(model_file_path)
                    print("  Model loaded.")
                    
                    # Generate plots from existing files
                    generate_visualizations(
                        model, label_encoder, test_df_result_for_vis, 
                        label_col, evals_result=None
                    )
                    
                    print("\n=== XGBoost Visualization Complete (Skipped Training) ===")
                    return # Exit script
                
                else:
                    # This 'else' corresponds to 'if are_params_identical'
                    print("Hyperparameters mismatch. Re-training model.")
            
        else:
            print("Output files not found. Starting new training session.")

        # --- Full Training Workflow ---
        # Load test data (train_df is already loaded)
        test_df = load_fits_to_df(test_path)
        
        features = get_feature_list(train_df, test_df)
        if not features:
            raise ValueError("No common features found between training and test sets.")
            
        # Run training and inference
        test_df_result, evals_result, model, label_encoder = train_xgb(
            train_df=train_df, test_df=test_df, features=features, label_col=label_col,
            out_file=out_file, learning_rate=set_learning_rate, max_depth=set_max_depth,
            subsample=set_subsample, colsample_bytree=set_colsample_bytree,
            reg_alpha=set_reg_alpha, reg_lambda=set_reg_lambda,
            num_boost_round=set_num_boost_round, early_stopping_rounds=set_early_stopping_rounds,
            test_size=set_test_size,
            use_adaptive_lr=set_use_adaptive_lr
        )
        
        # Generate plots
        generate_visualizations(
            model, label_encoder, test_df_result, 
            label_col, evals_result=evals_result
        )

        print("\n=== XGBoost Training Complete ===")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        gc.collect()
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
