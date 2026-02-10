import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import sys
import math
import urllib.request
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from matplotlib.colors import ListedColormap, LogNorm, LinearSegmentedColormap, to_rgba
from matplotlib.ticker import FuncFormatter

#########################################
# SECTION 1: STYLE & UTILITY FUNCTIONS
#########################################

def load_fits_to_df(path):
    """
    Loads astronomical data from a FITS file into a Pandas DataFrame.

    Goal:
        To bridge the gap between astronomical data formats (FITS) and data analysis 
        tools (Pandas). This function handles common issues like multiple file extensions 
        and byte-order (endianness) incompatibility between FITS files and modern CPUs.

    Inputs:
        path (str): The relative or absolute file path to the .fits file.

    Outputs:
        pd.DataFrame: A dataframe where columns match the FITS table columns 
                      and rows represent individual stars or observations.
    
    Flow of Control:
        1. Open the FITS file.
        2. Determine which 'extension' holds the data (usually extension 1).
        3. Iterate through columns to fix "Endianness":
           - FITS files are "Big-Endian" (older standard).
           - Most modern processors (Intel/AMD) are "Little-Endian".
           - We swap the bytes so NumPy/Pandas can read the numbers correctly.
        4. Construct and return the DataFrame.
    """
    try:
        # Open the FITS file context manager
        with fits.open(path) as hdul:
            # FITS files are split into "Header Data Units" (HDUs).
            # HDU 0 usually contains metadata or an image. 
            # HDU 1 usually contains the tabular data we want.
            # We check if HDU 1 exists and has data; otherwise, we fall back to HDU 0.
            data = hdul[1].data if len(hdul) > 1 and hdul[1].data is not None else hdul[0].data
            
            # Check if the data has column names (typical for FITS Tables)
            if not hasattr(data, 'names') or not data.names:
                # If no names, treat it as a raw numpy array
                df = pd.DataFrame(np.asarray(data))
            else:
                # Convert the FITS table data into a dictionary of numpy arrays.
                # This step is crucial for fixing the byte order (endianness).
                data_dict = {}
                for col in data.names:
                    col_data = np.asarray(data[col])
                    
                    # Check the byte order of the data.
                    # '>' indicates Big-Endian (FITS standard).
                    # '<' indicates Little-Endian (PC standard).
                    # '=' indicates Native (matches the machine running the code).
                    # If the data is not native ('=' or '|'), we swap bytes.
                    if col_data.dtype.byteorder not in ('=', '|'):
                        col_data = col_data.astype(col_data.dtype.newbyteorder('='))
                    
                    data_dict[col] = col_data
                
                # Create the DataFrame from the cleaned dictionary
                df = pd.DataFrame(data_dict)
                
            return df
            
    except Exception as e:
        print(f"Error loading FITS file: {e}")
        raise

def get_rrlyr_candidates(fits_path, threshold_override=None):
    """
    Filters the dataset to retrieve high-confidence RR Lyrae candidates.

    Goal:
        To isolate stars that the machine learning model has identified as 'RRLyr' 
        with high probability. It also categorizes these stars based on whether 
        they were already known or are new discoveries.

    Inputs:
        fits_path (str): Path to the predictions FITS file.
        threshold_override (float, optional): A manual probability cutoff (0.0 to 1.0). 
                                              If None, the optimal threshold from the file header is used.

    Outputs:
        pd.DataFrame: A filtered dataframe containing only high-confidence RRLyr candidates,
                      with an additional 'Status' column describing their discovery state.
                      Returns None if the file cannot be read or required columns are missing.

    Flow of Control:
        1. Load data from FITS.
        2. Identify the probability threshold (from header or manual override).
        3. Filter rows where:
           - The model's predicted class is 'RRLyr'.
           - The probability of being 'RRLyr' > threshold.
        4. Create a 'Status' column to classify candidates:
           - 'Known RRLyr': The training labels already knew it was RRLyr.
           - 'New Candidate': The training labels marked it as 'UNKNOWN'.
           - 'Reclassified': The training labels thought it was something else (e.g., ECL).
    """
    # --- Step 1: Load the Data ---
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[1].data if len(hdul) > 1 else hdul[0].data
            header = hdul[1].header if len(hdul) > 1 else hdul[0].header
            
            # Convert FITS data to Pandas DataFrame using the same logic as load_fits_to_df
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
    except Exception as e:
        print(f"Error reading {fits_path}: {e}")
        return None

    # --- Step 2: Verification ---
    # Ensure the dataframe has the columns required for filtering and status assignment.
    req_cols = ['xgb_predicted_class', 'xgb_training_class', 'prob_RRLyr']
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        print(f"Missing required columns for RRLyr extraction: {missing}")
        return None

    # --- Step 3: Threshold Determination ---
    # The 'THR_RRLyr' header keyword stores the optimal threshold calculated during training.
    # This is usually the threshold that balances Precision and Recall (F1-score).
    if threshold_override is not None:
        threshold = threshold_override
    elif 'THR_RRLyr' in header:
        threshold = float(header['THR_RRLyr'])
    else:
        # Default to standard 50% probability if no specific threshold exists
        threshold = 0.5

    # --- Step 4: Filtering ---
    # We select only the rows that the model predicts as RRLyr with high confidence.
    subset = df[
        (df['xgb_predicted_class'].str.strip() == 'RRLyr') & 
        (df['prob_RRLyr'] >= threshold)
    ].copy()
    
    if subset.empty:
        return pd.DataFrame() 

    # --- Step 5: Status Categorization ---
    # This step adds scientific context to the results. Are we rediscovering known stars,
    # finding completely new ones, or correcting previous misclassifications?
    subset['xgb_training_class'] = subset['xgb_training_class'].astype(str).str.strip()

    conditions = [
        # Case 1: The ground truth agreed with our prediction.
        (subset['xgb_training_class'] == 'RRLyr'),
        # Case 2: The star had no previous label.
        (subset['xgb_training_class'] == 'UNKNOWN'),
        # Case 3: The star was previously labeled differently (e.g., as an Eclipsing Binary).
        (subset['xgb_training_class'] != 'RRLyr') & (subset['xgb_training_class'] != 'UNKNOWN')
    ]
    choices = ['Known RRLyr', 'New Candidate', 'Reclassified']
    
    # np.select applies the choices based on the conditions list order.
    subset['Status'] = np.select(conditions, choices, default='Unknown')
    
    return subset

def get_all_candidates(fits_path):
    """
    Generalizes the filtering process to retrieve candidates for ALL classes.

    Goal:
        To iterate through every class (e.g., 'ECL', 'CEP', 'RRLyr'), apply class-specific
        probability thresholds, and compile a master list of high-confidence predictions.

    Inputs:
        fits_path (str): Path to the predictions FITS file.

    Outputs:
        pd.DataFrame: A single dataframe containing high-confidence candidates for all classes.
                      Includes 'Status' (specific class context) and 'Global_Status' (general context).

    Flow of Control:
        1. Load data.
        2. Identify all unique predicted classes.
        3. Loop through each class:
           a. Retrieve the optimal threshold for that specific class from the FITS header.
           b. Filter rows where prediction matches the class and probability > threshold.
           c. Assign status labels (Known/New/Reclassified).
        4. Concatenate all class subsets into one final dataframe.
    """
    # --- Step 1: Load the Data ---
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[1].data if len(hdul) > 1 else hdul[0].data
            header = hdul[1].header if len(hdul) > 1 else hdul[0].header
            
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
    except Exception as e:
        print(f"Error reading {fits_path}: {e}")
        return None

    if 'xgb_predicted_class' not in df.columns:
        print("Missing 'xgb_predicted_class' column.")
        return None

    # Clean whitespace from string columns to ensure accurate matching
    df['xgb_predicted_class'] = df['xgb_predicted_class'].astype(str).str.strip()
    if 'xgb_training_class' in df.columns:
        df['xgb_training_class'] = df['xgb_training_class'].astype(str).str.strip()
    else:
        df['xgb_training_class'] = 'UNKNOWN'

    all_candidates = []
    
    # Get the list of all classes the model predicted
    unique_classes = df['xgb_predicted_class'].unique()
    
    for cls in unique_classes:
        # Look for class-specific threshold in header (e.g., 'THR_ECL', 'THR_CEP')
        header_key = f"THR_{cls}"
        if header_key in header:
            threshold = float(header[header_key])
        else:
            threshold = 0.5 # Default fallback
            
        prob_col = f"prob_{cls}"
        if prob_col not in df.columns:
            continue

        # Filter for high-confidence candidates of this specific class
        subset = df[
            (df['xgb_predicted_class'] == cls) & 
            (df[prob_col] >= threshold)
        ].copy()
        
        if subset.empty:
            continue

        # Assign Detailed Status (e.g., "Known ECL", "Known CEP")
        conditions = [
            (subset['xgb_training_class'] == cls), # Match
            (subset['xgb_training_class'] == 'UNKNOWN'), # Discovery
            (subset['xgb_training_class'] != cls) & (subset['xgb_training_class'] != 'UNKNOWN') # Mismatch
        ]
        choices = [f'Known {cls}', 'New Candidate', 'Reclassified']
        subset['Status'] = np.select(conditions, choices, default='Unknown')
        
        # Assign Global Status (Simplified for plotting all classes together)
        global_conditions = [
            (subset['xgb_training_class'] == cls),
            (subset['xgb_training_class'] == 'UNKNOWN'),
            (subset['xgb_training_class'] != cls) & (subset['xgb_training_class'] != 'UNKNOWN')
        ]
        global_choices = ['Known', 'New', 'Reclassified']
        subset['Global_Status'] = np.select(global_conditions, global_choices, default='Unknown')
        
        all_candidates.append(subset)
        
    if not all_candidates:
        return pd.DataFrame()
        
    final_df = pd.concat(all_candidates, ignore_index=True)
    return final_df

def get_consistent_color_map(class_names):
    """
    Generates a consistent dictionary of colors for a list of class names.

    Goal:
        To ensure that a specific class (e.g., "ECL") is always represented by the 
        same color across different plots (Bailey diagram, ROC, etc.). This makes 
        comparative analysis much easier. It uses a colorblind-friendly palette.

    Inputs:
        class_names (list): A list of class name strings.

    Outputs:
        dict: A dictionary mapping class names to hex color codes (e.g., {'ECL': '#E69F00'}).
    """
    # Sort classes alphabetically to ensure the mapping is deterministic.
    # Without sorting, running the code twice could assign different colors if the input order changes.
    sorted_classes = sorted(list(set(class_names)))
    
    # "Okabe-Ito" Palette: Designed to be accessible to people with color blindness.
    palette = [
        '#E69F00', # Orange
        '#56B4E9', # Sky Blue
        '#009E73', # Bluish Green
        '#F0E442', # Yellow
        '#332288', # Indigo 
        '#882255', # Wine/Dark Red 
        '#CC79A7', # Reddish Purple
        '#000000', # Black
        # Extended palette using Tab20 if more colors are needed
        '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#AA4499'
    ]
    
    # If there are more classes than colors, extend with standard matplotlib colors
    if len(sorted_classes) > len(palette):
        tab20 = plt.cm.tab20.colors 
        tab20_hex = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b in tab20]
        palette.extend(tab20_hex)

    # Map each class name to a color
    color_map = {}
    for i, cls in enumerate(sorted_classes):
        color_map[cls] = palette[i % len(palette)]
        
    return color_map

def set_plot_style(large_text=False):
    """
    Configures Matplotlib global settings for publication-quality figures.

    Goal:
        To standardize font sizes, line widths, and resolution across all plots generated
        by this script.

    Inputs:
        large_text (bool): If True, increases font sizes significantly (good for posters/slides).
    """
    plt.rcParams['figure.figsize'] = [8, 6]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    
    base_size = 14
    if large_text:
        base_size = 18
        
    # Set font sizes relative to the base size
    plt.rcParams['font.size'] = base_size
    plt.rcParams['axes.labelsize'] = base_size + 2
    plt.rcParams['axes.titlesize'] = base_size + 4
    plt.rcParams['xtick.labelsize'] = base_size + 2
    plt.rcParams['ytick.labelsize'] = base_size + 2
    plt.rcParams['legend.fontsize'] = base_size - 2
    
    # Visual elements
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8
    
    # Clean background settings
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.color'] = "grey"

#########################################
# SECTION 2: PLOTTING FUNCTIONS
#########################################

def plot_xgb_training_loss(evals_result, output_dir='figures'):
    """
    Plots the training and validation "Log Loss" over iterations (epochs).

    Goal:
        To diagnose if the model is Overfitting or Underfitting.
        - Training Loss: Error on data the model sees. Should always go down.
        - Validation Loss: Error on data the model does NOT see.
        
        If Validation Loss starts going UP while Training Loss goes DOWN, the model 
        is memorizing noise (Overfitting). We want the point where Validation Loss is lowest.

    Inputs:
        evals_result (dict): The dictionary returned by xgb.train() containing loss history.
        output_dir (str): Directory to save the plot.

    Outputs:
        matplotlib.pyplot: The plot object (also saves to file).
    """
    # Extract the loss values from the dictionary
    train_loss = list(evals_result['train'].values())[0]
    val_loss = list(evals_result['validation'].values())[0]
    iterations = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_loss, 'b-', label='Training Loss')
    plt.plot(iterations, val_loss, 'r-', label='Validation Loss')
    
    # Find the iteration where validation loss was minimized (the optimal stopping point)
    best_iter = val_loss.index(min(val_loss)) + 1
    
    # Annotate the best iteration on the plot
    plt.axvline(x=best_iter, color='gray', linestyle='--')
    plt.scatter(best_iter, min(val_loss), color='red', s=100)
    plt.annotate(f'Best: {min(val_loss):.4f} (iter {best_iter})', 
                 xy=(best_iter, min(val_loss)),
                 xytext=(best_iter + 5, min(val_loss)),
                 arrowprops=dict(arrowstyle='->'))
    
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_loss.png")
    return plt

def plot_bailey_diagram(df, class_column, output_dir='class_figures', min_prob=0.7, min_confidence=0.9, max_entropy=0.2, thresholds_dict=None):
    """
    Creates a Bailey Diagram (Period vs. Amplitude) for the predicted candidates.

    Goal:
        The Bailey Diagram is the standard diagnostic tool in variable star astronomy.
        Different types of stars (RRLyr, Cepheids, ECLs) occupy distinct regions in 
        this space. This plot allows us to visually verify if our machine learning 
        predictions align with physical reality.

    Inputs:
        df (pd.DataFrame): Dataframe containing 'true_period', 'true_amplitude', and predictions.
        class_column (str): The column name containing the class labels (e.g., 'xgb_predicted_class').
        output_dir (str): Directory to save the plot.
        thresholds_dict (dict): Map of {class: threshold}. Preferred method for filtering "good" candidates.
        min_prob/conf/entropy (float): Fallback cutoff values if thresholds_dict is not provided.

    Flow of Control:
        1. Clean data: Remove outliers (e.g., Amplitude > 5 mag).
        2. Calculate log(Period) if missing (standard X-axis for Bailey diagrams).
        3. Filter data: Keep only rows that exceed the probability threshold for their specific class.
        4. Sampling: If there are millions of stars, plot a random sample (e.g., 10k per class) to prevent plotting saturation.
        5. Plot: Scatter plot with different colors/markers for each class.
    """
    set_plot_style(large_text=True)
    plt.rcParams['legend.fontsize'] = 15
    
    # Work on a copy to ensure we don't modify the data passed in
    df = df.copy()
    
    # --- Data Cleaning ---
    if 'true_amplitude' in df.columns:
        df = df[df['true_amplitude'] < 5] 
    
    if 'log_true_period' not in df.columns and 'true_period' in df.columns:
        df['log_true_period'] = np.log10(df['true_period'])
    
    # --- Filtering Logic ---
    if thresholds_dict is not None:
        # Method A (Preferred): Use the calculated optimal thresholds per class
        keep_mask = pd.Series(False, index=df.index)
        unique_classes = df[class_column].unique()
        
        for cls in unique_classes:
            prob_col = f'prob_{cls}'
            if prob_col in df.columns:
                threshold = thresholds_dict.get(cls, 0.5) 
                # Keep row IF it is Class X AND Probability(X) > Threshold
                cls_mask = (df[class_column] == cls) & (df[prob_col] >= threshold)
                keep_mask |= cls_mask
        df = df[keep_mask]
    else:
        # Method B (Fallback): Use global hard cutoffs
        prob_col = class_column.replace('predicted_class', 'confidence')
        if prob_col in df.columns:
            df = df[df[prob_col] > min_prob]
        if 'xgb_confidence' in df.columns:
            df = df[df['xgb_confidence'] > min_confidence]
        if 'xgb_entropy' in df.columns:
            df = df[df['xgb_entropy'] < max_entropy]
    
    # --- Sampling ---
    # Take the top 10,000 most confident predictions per class to visualize density cleanly.
    prob_sort_col = class_column.replace('predicted_class', 'confidence')
    if prob_sort_col not in df.columns:
        prob_sort_col = 'xgb_confidence'
        
    if prob_sort_col in df.columns:
        sampled_df = df.sort_values(prob_sort_col, ascending=False).groupby(class_column).head(10000).reset_index(drop=True)
    else:
        sampled_df = df.groupby(class_column).head(10000).reset_index(drop=True)
    
    # Setup plot
    unique_types = sorted(sampled_df[class_column].unique())
    color_map = get_consistent_color_map(unique_types)
    markers = ['o', 's', '^', 'D', 'v', 'X', 'P', '*', 'h']
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    for i, var_type in enumerate(unique_types):
        type_df = sampled_df[sampled_df[class_column] == var_type]
        
        plt.scatter(type_df['log_true_period'], type_df['true_amplitude'], 
                   color=color_map[var_type], marker=markers[i % len(markers)], 
                   label=var_type, s=7, alpha=0.3)
    
    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=min(len(unique_types), 5), markerscale=5, frameon=True)
    for handle in leg.legend_handles: handle.set_alpha(1.0)
    
    ax.set_xlabel(r'log$_{10}$(Period) [days]')
    ax.set_ylabel(r'Amplitude [mag]')
    ax.set_xlim(left=-1) 
    ax.set_ylim(0, 5)    
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/bailey_diagram.jpg', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_bailey_diagram(df, class_column, output_dir='figures'):
    """
    Creates a Bailey Diagram for the TRAINING data (Ground Truth).

    Goal:
        To visualize the input data fed into the model. This serves as a "Truth" 
        reference. Comparing this plot to the prediction Bailey Diagram shows 
        how well the model has learned the physical distribution of stars.

    Inputs:
        df (pd.DataFrame): Training dataframe.
        class_column (str): The column containing true labels.

    Flow of Control:
        1. Clean data: Apply quantile clipping (0.1% to 99.9%) to remove extreme 
           outliers, matching the preprocessing steps used during training.
        2. Sampling: Prioritize plotting stars with the best periodicity statistics 
           ('best_fap') to show the cleanest examples.
        3. Plot: Scatter plot.
    """
    set_plot_style(large_text=True)
    plt.rcParams['legend.fontsize'] = 15
    
    df = df.copy()

    # --- Pre-cleaning ---
    # Clip data to remove extreme statistical outliers that compress the plot
    cols_to_clean = ['true_period', 'true_amplitude']
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            q001 = df[col].quantile(0.001)
            q999 = df[col].quantile(0.999)
            df[col] = df[col].clip(lower=q001, upper=q999)
    
    if 'true_amplitude' in df.columns:
        df = df[df['true_amplitude'] < 5] 
    
    if 'log_true_period' not in df.columns and 'true_period' in df.columns:
        df = df[df['true_period'] > 0]
        df['log_true_period'] = np.log10(df['true_period'])
    
    # --- Sampling ---
    # If 'best_fap' (False Alarm Probability) is available, use it to pick the most periodic stars.
    if 'best_fap' in df.columns:
        sampled_df = df.sort_values('best_fap', ascending=True).groupby(class_column).head(10000).reset_index(drop=True)
    else:
        sampled_df = df.sample(frac=1, random_state=42).groupby(class_column).head(10000).reset_index(drop=True)
    
    unique_types = sorted(sampled_df[class_column].unique())
    color_map = get_consistent_color_map(unique_types)
    markers = ['o', 's', '^', 'D', 'v', 'X', 'P', '*', 'h']
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    for i, var_type in enumerate(unique_types):
        type_df = sampled_df[sampled_df[class_column] == var_type]
        plt.scatter(type_df['log_true_period'], type_df['true_amplitude'], 
                   color=color_map[var_type], marker=markers[i % len(markers)], 
                   label=var_type, s=7, alpha=0.3)
    
    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=min(len(unique_types), 5), markerscale=5, frameon=True)
    for handle in leg.legend_handles: handle.set_alpha(1.0)
    ax.set_xlabel(r'log$_{10}$(Period) [days]')
    ax.set_ylabel(r'Amplitude [mag]')
    ax.set_xlim(left=-1)
    ax.set_ylim(0, 5)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/bailey_diagram_training.jpg', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_entropy(df, class_column, output_dir='class_figures', min_prob=0.0, use_brg_cmap=False):
    """
    Plots Prediction Confidence vs. Prediction Entropy (Uncertainty) using 2D Histograms.

    Goal:
        To assess how "sure" the model is about its predictions.
        - Confidence: The probability of the top class (e.g., 0.95).
        - Entropy: A measure of confusion. 
          * Low Entropy + High Confidence (Bottom Right) = Good.
          * High Entropy + Low Confidence (Top Left) = Confused/Uncertain.

    Inputs:
        df (pd.DataFrame): Dataframe with predictions.
        class_column (str): Column name for labels.

    Flow of Control:
        1. Setup a grid of subplots (one for each class).
        2. Create a custom colormap that goes from Light Tint -> Solid Color -> Black.
           This helps visualize density: where are most of the points concentrated?
        3. Use a 2D Histogram (heatmap) instead of a scatter plot.
           Why? With millions of points, scatter plots become solid blocks of color. 
           Histograms show where the data actually piles up.
    """
    df = df.copy()

    # Determine confidence column and filter
    prob_col = class_column.replace('predicted_class', 'confidence')
    if prob_col in df.columns:
        df = df[df[prob_col] > min_prob]

    # Select axes
    if 'xgb_confidence' not in df.columns or 'xgb_entropy' not in df.columns:
        # Fallback if specific entropy column is missing
        x_col = prob_col
        df['uncertainty'] = 1 - df[prob_col]
        y_col = 'uncertainty'
    else:
        x_col = 'xgb_confidence'
        y_col = 'xgb_entropy'

    unique_types = sorted(df[class_column].unique())
    n_classes = len(unique_types)
    color_map = get_consistent_color_map(unique_types)

    # Calculate grid dimensions
    ncols = 3
    nrows = math.ceil(n_classes / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True)
    if n_classes > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    bins = 50 

    for i, var_type in enumerate(unique_types):
        ax = axes[i]
        type_df = df[df[class_column] == var_type]
        
        if len(type_df) == 0: continue
        
        # Create a custom colormap for this specific class
        if use_brg_cmap:
            cmap = 'brg'
        else:
            base_color = color_map[var_type]
            is_black = (base_color.lower() == '#000000') or (base_color.lower() == 'black')
            
            if is_black:
                colors = ['#D0D0D0', '#606060', '#000000']
            else:
                rgb = to_rgba(base_color)[:3]
                # Mix with white for the light tint
                light_tint = [(0.2 * c + 0.8) for c in rgb] 
                colors = [light_tint, base_color, '#000000']
            cmap = LinearSegmentedColormap.from_list(f"cmap_{var_type}", colors)

        # Plot 2D Histogram with LogNorm to see structure across orders of magnitude
        h = ax.hist2d(
            type_df[x_col], type_df[y_col], bins=bins, 
            range=[[x_min, x_max], [y_min, y_max]],
            cmap=cmap, norm=LogNorm(), cmin=1 
        )
        
        plt.colorbar(h[3], ax=ax)
        ax.set_title(f"{var_type}\n(n={len(type_df)})")
        ax.grid(True, linestyle='--', alpha=0.3)
        
        if i // ncols == nrows - 1: ax.set_xlabel(f'Confidence ({x_col})')
        if i % ncols == 0: ax.set_ylabel(f'Entropy ({y_col})')

    for j in range(i + 1, len(axes)): axes[j].axis('off')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/classification_confidence_entropy.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_distribution(confidences, predictions, class_names, output_dir="figures"):
    """
    Plots a Step Histogram of model confidence scores for each class.

    Goal:
        To visualize the distribution of confidence scores.
        - Are most predictions high confidence (>0.9)? (Good)
        - Is there a long tail of low confidence predictions? (Bad/Needs filtering)

    Inputs:
        confidences (np.array): Array of float confidence scores.
        predictions (np.array): Array of integer class indices.
        class_names (list): List of class names matching the indices.
    """
    color_map = get_consistent_color_map(class_names)
    
    plt.figure(figsize=(12, 6))
    bins = np.linspace(0, 1, 51) # 50 bins from 0.0 to 1.0
    
    for i, class_name in enumerate(class_names):
        class_conf = confidences[predictions == i]
        if len(class_conf) > 0:
            # We use density=True to normalize the histograms.
            # This allows us to compare the shape of the distribution for a rare class
            # (100 stars) against a common class (100,000 stars) on the same plot.
            plt.hist(class_conf, bins=bins, histtype='step', linewidth=2.5, 
                     label=f"{class_name} (n={len(class_conf)})", 
                     color=color_map[class_name], density=True)
    
    plt.xlabel('Model Confidence')
    plt.ylabel('Density (Normalized)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{output_dir}/confidence_distribution.png', dpi=300)
    plt.close()

def plot_xgb_feature_importance(feature_names, importance_values, top_n=20, output_dir='figures'):
    """
    Bar chart of the most important features used by the XGBoost model.

    Goal:
        To understand *what* the model is looking at. 
        High importance means the model splits the data frequently based on this feature.
        If "Period" is top, the model relies heavily on period.

    Inputs:
        feature_names (list): List of feature name strings.
        importance_values (list): Corresponding gain/importance scores.
    """
    indices = np.argsort(importance_values)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importance = [importance_values[i] for i in indices]
    
    plt.figure(figsize=(14, 10))
    plt.barh(range(len(top_features)), top_importance, align='center', color='skyblue', edgecolor='navy', alpha=0.8)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance (Gain)', fontsize=14)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/xgb_feature_importance.png', dpi=300)
    plt.close()

def plot_and_print_auc_ap(df, true_label_col, label_encoder, output_dir='figures'):
    """
    Calculates and plots performance metrics (ROC AUC and Average Precision).

    Goal:
        To evaluate how well the model separates classes.
        - ROC AUC (Area Under Curve): 0.5 is random guessing, 1.0 is perfect.
        - Average Precision (AP): Area under Precision-Recall curve. Good for imbalanced data.
        
        It also calculates the **Optimal Threshold** for each class, which is the 
        probability cutoff that balances Precision and Recall (closest to point 1,1).

    Inputs:
        df (pd.DataFrame): Dataframe with truth and probability columns.
        true_label_col (str): Column name for ground truth.
        label_encoder (LabelEncoder): Encoder to map string classes to integers.

    Outputs:
        dict: A dictionary mapping class names to optimal thresholds (e.g., {'ECL': 0.85}).
    """
    print("\nCalculating AP and ROC AUC metrics...")
    
    optimal_thresholds = {}

    if true_label_col not in df.columns:
        print(f"Warning: True label column '{true_label_col}' not found. Skipping metrics.")
        return optimal_thresholds

    classes = label_encoder.classes_
    prob_cols = [f'prob_{cls}' for cls in classes]
    
    missing_cols = [col for col in prob_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing probability columns, skipping AP/AUC calculation: {missing_cols}")
        return optimal_thresholds
        
    # Binarize labels for One-vs-Rest calculation (e.g., Is 'ECL' vs Is Not 'ECL')
    y_true_raw = df[true_label_col].fillna('UNKNOWN')
    Y_true = label_binarize(y_true_raw, classes=classes)
    Y_proba = df[prob_cols].values
    
    per_class_ap = []
    per_class_roc = []
    class_labels = []
    class_metrics_data = {} 
    
    total_ap = 0
    total_roc = 0
    valid_ap_classes = 0
    valid_roc_classes = 0

    # Calculate metrics for each class individually
    for k, class_name in enumerate(classes):
        y_true_class = Y_true[:, k] # Binary vector: 1 if class k, 0 otherwise
        y_proba_class = Y_proba[:, k] # Probability scores for class k
        
        pos_count = y_true_class.sum()
        neg_count = len(y_true_class) - pos_count
        total_count = len(y_true_class)
        
        class_metrics_data[class_name] = {
            'count': pos_count,
            'ap': np.nan, 'roc': np.nan,
            'pr_curve': None, 'roc_curve': None, 'no_skill': 0
        }
        
        if pos_count > 0:
            try:
                # 1. Average Precision (AP)
                ap = average_precision_score(y_true_class, y_proba_class)
                per_class_ap.append(ap)
                total_ap += ap
                valid_ap_classes += 1
                class_metrics_data[class_name]['ap'] = ap
                
                # 2. Precision-Recall Curve Data
                precision, recall, thresholds = precision_recall_curve(y_true_class, y_proba_class)
                no_skill = pos_count / total_count
                class_metrics_data[class_name]['no_skill'] = no_skill
                class_metrics_data[class_name]['pr_curve'] = (precision, recall, thresholds)
            except ValueError:
                per_class_ap.append(np.nan)

            # 3. ROC AUC
            if neg_count > 0:
                try:
                    roc = roc_auc_score(y_true_class, y_proba_class)
                    per_class_roc.append(roc)
                    total_roc += roc
                    valid_roc_classes += 1
                    class_metrics_data[class_name]['roc'] = roc
                    fpr, tpr, _ = roc_curve(y_true_class, y_proba_class)
                    class_metrics_data[class_name]['roc_curve'] = (fpr, tpr)
                except ValueError:
                    per_class_roc.append(np.nan)
            else:
                per_class_roc.append(np.nan)
                
            class_labels.append(class_name)

    # Print Macro Metrics (Average across classes)
    macro_ap = total_ap / valid_ap_classes if valid_ap_classes > 0 else 0
    macro_roc_auc = total_roc / valid_roc_classes if valid_roc_classes > 0 else 0
    print("--- Macro-Averaged Metrics ---")
    print(f"Macro-Averaged Precision (mAP) (over {valid_ap_classes} classes): {macro_ap:.4f}")
    print(f"Macro-Averaged ROC AUC         (over {valid_roc_classes} classes): {macro_roc_auc:.4f}")

    if not class_labels:
        print("No classes with true samples found.")
        return optimal_thresholds

    # --- Plot 1: Bar Chart of AP and ROC AUC per class ---
    metrics_df = pd.DataFrame({
        'Class': class_labels, 'AP': per_class_ap, 'ROC_AUC': per_class_roc
    })
    metrics_df = metrics_df.sort_values(by='AP', ascending=True).dropna(subset=['AP', 'ROC_AUC'], how='all')

    if not metrics_df.empty:
        try:
            set_plot_style(large_text=False)
        except NameError:
            pass
        
        fig_height = max(5, 0.4 * len(metrics_df))
        fig, ax = plt.subplots(figsize=(12, fig_height))
        y_pos = np.arange(len(metrics_df))
        bar_height = 0.4
        
        ax.barh(y_pos - bar_height / 2, metrics_df['AP'], height=bar_height, label='Avg. Precision (AP)', color='C0')
        ax.barh(y_pos + bar_height / 2, metrics_df['ROC_AUC'], height=bar_height, label='ROC AUC', color='C2')
        
        for i, (ap, roc) in enumerate(zip(metrics_df['AP'], metrics_df['ROC_AUC'])):
            if pd.notna(ap): ax.text(ap + 0.01, y_pos[i] - bar_height / 2, f'{ap:.2f}', va='center', fontsize=9)
            if pd.notna(roc): ax.text(roc + 0.01, y_pos[i] + bar_height / 2, f'{roc:.2f}', va='center', fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics_df['Class'])
        ax.set_xlabel('Score')
        ax.set_xlim(0, 1.15)
        ax.legend(loc='lower right')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/per_class_metrics.jpg', dpi=300, bbox_inches='tight')
        plt.close()

    # --- Plot 2: Combined ROC and PR Curves ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    valid_class_names = [cls for cls, data in class_metrics_data.items() if data['count'] > 0]
    color_map = get_consistent_color_map(valid_class_names)
    
    # Subplot A: ROC Curves
    ax_roc = axes[0]
    has_roc_data = False
    for class_name in valid_class_names:
        data = class_metrics_data[class_name]
        if data['roc_curve'] is not None:
            fpr, tpr = data['roc_curve']
            color = color_map[class_name]
            auc_val = data['roc']
            ap_val = data['ap']
            count = data['count']
            label_str = f"{class_name} (n={int(count)}) AUC={auc_val:.2f} AP={ap_val:.2f}"
            ax_roc.plot(fpr, tpr, lw=2, color=color, label=label_str)
            has_roc_data = True

    if has_roc_data:
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(loc="lower right", fontsize=12, ncol=1)
        ax_roc.grid(alpha=0.3)
    else:
        ax_roc.text(0.5, 0.5, "No valid ROC data available", ha='center', va='center')

    # Subplot B: PR Curves and Threshold Selection
    ax_pr = axes[1]
    has_pr_data = False
    for class_name in valid_class_names:
        data = class_metrics_data[class_name]
        if data['pr_curve'] is not None:
            precision, recall, thresholds = data['pr_curve']
            color = color_map[class_name]
            
            ax_pr.plot(recall, precision, lw=2, color=color)

            # --- Find Optimal Threshold ---
            # We define "optimal" as the point on the curve closest to the top-right (1, 1).
            # This represents the best tradeoff between Precision and Recall.
            # distance^2 = (1-recall)^2 + (1-precision)^2
            distances = (1 - recall)**2 + (1 - precision)**2
            best_idx = np.argmin(distances)
            
            best_r = recall[best_idx]
            best_p = precision[best_idx]
            
            if best_idx < len(thresholds):
                best_thresh = thresholds[best_idx]
            else:
                best_thresh = 1.0
            
            # Save this threshold for return
            optimal_thresholds[class_name] = best_thresh
            
            ax_pr.scatter(best_r, best_p, color=color, s=50, edgecolor='white', zorder=5)
            has_pr_data = True

    if has_pr_data:
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.grid(alpha=0.3)
    else:
        ax_pr.text(0.5, 0.5, "No valid PR data available", ha='center', va='center')

    plt.tight_layout()
    pdf_path = f'{output_dir}/metrics_summary.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return optimal_thresholds


#########################################
# SECTION 3: PCA VISUALIZATION FUNCTIONS
#########################################

def visualize_pca(pca_df, original_df, pca_model, output_dir='./figures'):
    """
    Generates a suite of plots to analyze Principal Component Analysis (PCA) results.

    Goal:
        PCA compresses many features into a few "Principal Components" (PC1, PC2).
        These plots help us understand:
        1. How much data variance is explained? (Scree Plot)
        2. What features make up PC1/PC2? (Loadings Heatmap)
        3. Is there structure/clustering in PC space? (Scatter/Density)

    Inputs:
        pca_df: Dataframe with PC1, PC2 columns.
        original_df: Dataframe with original features (for overlaying color).
        pca_model: The trained scikit-learn PCA model.
    """
    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()

    # 1. Scatter Plot
    plt.figure()
    plt.scatter(pca_df['PC1'], pca_df['PC2'], s=5, alpha=0.5)
    plt.xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.2%})")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_scatter.png")
    plt.close()

    # 2. Density Plot (KDE)
    plt.figure()
    sns.kdeplot(x=pca_df['PC1'], y=pca_df['PC2'], cmap="Blues", fill=True)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_density.png")
    plt.close()

    # 3. 2D Histogram (for large datasets)
    plt.figure()
    h = plt.hist2d(pca_df['PC1'], pca_df['PC2'], bins=100, cmap='viridis', norm=LogNorm())
    plt.colorbar(h[3])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_histogram.png")
    plt.close()

    # 4. Scree Plot (Elbow Method)
    plt.figure()
    plt.bar(range(1, len(pca_model.explained_variance_ratio_)+1), pca_model.explained_variance_ratio_)
    plt.step(range(1, len(pca_model.explained_variance_ratio_)+1), np.cumsum(pca_model.explained_variance_ratio_), label='Cumulative')
    plt.xlabel('PC')
    plt.ylabel('Explained Variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_scree.png")
    plt.close()

    # 5. Loadings Heatmap (Correlation between Features and PCs)
    if hasattr(pca_model, 'feature_names_in_'):
        feature_names = pca_model.feature_names_in_
    else:
        feature_names = [f"Feature {i}" for i in range(pca_model.components_.shape[1])]

    load = pd.DataFrame(pca_model.components_.T, 
                        index=feature_names, 
                        columns=[f'PC{i+1}' for i in range(pca_model.n_components_)])
    
    # Sort to find the most influential features
    top = pd.concat([
        load['PC1'].abs().sort_values(ascending=False), 
        load['PC2'].abs().sort_values(ascending=False)
    ]).index.unique()[:50] 
    
    plt.figure(figsize=(10, max(6, len(top) * 0.4)))
    sns.heatmap(load.loc[top, ['PC1', 'PC2']], annot=True, cmap='coolwarm', center=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_loadings.png")
    plt.close()

    # 6. Coloured Scatter Plots (Overlaying Physical Properties)
    if 'best_fap' in original_df.columns:
        plt.figure()
        plt.scatter(pca_df['PC1'], pca_df['PC2'], c=original_df['best_fap'], cmap='viridis_r', s=10, alpha=0.7)
        plt.colorbar(label='FAP')
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_fap.png")
        plt.close()

    if 'true_period' in original_df.columns:
        plt.figure()
        plt.scatter(pca_df['PC1'], pca_df['PC2'], 
                   c=np.log10(np.clip(original_df['true_period'].values, 0.01, None)), 
                   cmap='plasma', s=10, alpha=0.7)
        plt.colorbar(label='log10(Period)')
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_period.png")
        plt.close()

    if 'true_amplitude' in original_df.columns:
        plt.figure()
        plt.scatter(pca_df['PC1'], pca_df['PC2'], 
                   c=np.log10(np.clip(original_df['true_amplitude'].values, 0.001, None)), 
                   cmap='inferno', s=10, alpha=0.7)
        plt.colorbar(label='log10(Amplitude)')
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_amplitude.png")
        plt.close()

def plot_pca_degeneracy_analysis(pca_df, df, pca_model, features, output_dir='./figures', n_pcs_limit=5, threshold=0.4):
    """
    Analyzes the structural relationship (Degeneracy) between original features and PCs.

    Goal:
        To see which original features are redundant. If multiple features (e.g., 'Period', 
        'Frequency', 'Harmonic 1') all correlate strongly with PC1, they are "degenerate" 
        or measuring the same underlying physical property.

    Outputs:
        A heatmap showing the correlation matrix between Features (rows) and PCs (columns).
    """
    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()
    
    valid_features = [f for f in features if f in df.columns]
    
    if not valid_features:
        print("Warning: No valid features found for degeneracy analysis.")
        return

    n_pcs_full = min(20, pca_df.shape[1]) 
    pc_cols_full = [f'PC{i+1}' for i in range(n_pcs_full)]
    
    if hasattr(pca_model, 'explained_variance_ratio_'):
        explained_variance = pca_model.explained_variance_ratio_[:n_pcs_full]
        pc_labels = [f"PC{i+1} ({var:.1%})" for i, var in enumerate(explained_variance)]
    else:
        pc_labels = pc_cols_full

    analysis_df = pd.concat([pca_df[pc_cols_full].reset_index(drop=True), 
                            df[valid_features].reset_index(drop=True)], axis=1)
    
    corr_matrix_full = analysis_df.corr()
    
    # Extract sub-matrix: Features (rows) vs PCs (cols)
    feature_pc_corr = corr_matrix_full.loc[valid_features, pc_cols_full]
    
    # Create Mask for Annotations to keep the plot clean
    # Only show numbers if correlation is high (> threshold)
    show_text_mask = feature_pc_corr.abs() >= threshold
    row_max_indices = feature_pc_corr.abs().idxmax(axis=1)
    for row_idx, col_name in row_max_indices.items():
        if pd.notna(col_name):
            show_text_mask.loc[row_idx, col_name] = True

    annot_labels = feature_pc_corr.applymap(lambda x: f"{x:.2f}")
    annot_labels = annot_labels.where(show_text_mask, "")
    
    fig_height = max(10, len(valid_features) * 0.6)
    plt.figure(figsize=(20, fig_height))
    
    sns.heatmap(feature_pc_corr, 
                annot=annot_labels, 
                cmap='RdBu', 
                center=0, 
                fmt='', 
                vmin=-1, vmax=1, 
                xticklabels=pc_labels, yticklabels=True,
                cbar_kws={'label': 'Correlation'})
                
    plt.xlabel("Principal Components (Explained Variance)")
    plt.ylabel("Original Features")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filename = f"{output_dir}/pca_degeneracy_matrix.png"
    plt.savefig(filename)
    plt.close()
    
    print(f"Saved PCA degeneracy analysis to {filename}")

#########################################
# SECTION 4: DATA COMPARISON PLOTS
#########################################

def plot_period_comparison(periods_path, output_dir='figures'):
    """
    Comparies recovered periods against true periods to identify Aliases.

    Goal:
        In astronomy, sampling patterns (e.g., observing only at night) create "Aliases".
        If a star pulses every 0.9 days, but we observe it every 1.0 days, we might measure 
        the wrong period.
        
        This function plots True Period vs Measured Period and overlays mathematical 
        alias lines (1-day, 0.5-day) to identify where the algorithm got confused.

    Inputs:
        periods_path (str): FITS file with 'true_period' and 'P_1' (measured period).
    """
    try:
        df = load_fits_to_df(periods_path)
        
        x_col = 'true_period'
        y_col = 'P_1'
        
        if x_col not in df.columns or y_col not in df.columns:
            print(f"Missing period columns. Expected '{x_col}' and '{y_col}' in {periods_path}")
            return
            
        set_plot_style()
        plt.figure(figsize=(8, 8))
        
        valid_data = df[(df[x_col] > 0) & (df[y_col] > 0)]
        x_log = np.log10(valid_data[x_col])
        y_log = np.log10(valid_data[y_col])
        
        plt.scatter(x_log, y_log, s=10, alpha=0.1, c='black', edgecolors='none')
        
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        min_line = min(xlim[0], ylim[0])
        max_line = max(xlim[1], ylim[1])
        
        # Color definitions for alias lines
        c_identity = '#000000' # Black
        c_harmonic = '#009E73' # Bluish Green
        c_alias_1d = '#332288' # Indigo
        c_alias_1d_dbl = '#E69F00' # Orange
        c_alias_05d = '#56B4E9' # Sky Blue
        c_alias_2d = '#882255' # Wine
        c_alias_2d_half = '#CC79A7' # Reddish Purple

        # --- Reference Lines ---
        # 1:1 Identity line (Perfect recovery)
        plt.plot([min_line, max_line], [min_line, max_line], color=c_identity, linestyle='-', alpha=0.6, label='1:1', linewidth=1.5)
        
        # Harmonics (2:1 and 1:2)
        log2 = np.log10(2)
        plt.plot([min_line, max_line], [min_line + log2, max_line + log2], color=c_harmonic, linestyle='--', alpha=0.8, label='2:1 / 1:2', linewidth=1.5)
        log05 = np.log10(0.5)
        plt.plot([min_line, max_line], [min_line + log05, max_line + log05], color=c_harmonic, linestyle='--', alpha=0.8, linewidth=1.5)
        
        # --- Alias Logic ---
        # Formula: f_obs = f_true +/- k * f_sampling
        x_grid_log = np.linspace(min_line, max_line, 2000)
        x_grid_linear = 10**x_grid_log
        
        # 1. Sidereal Day Aliases (Standard 1-day cycle)
        y_alias_plus = x_grid_linear / (1 + x_grid_linear)
        plt.plot(x_grid_log, np.log10(y_alias_plus), color=c_alias_1d, linestyle='-.', alpha=0.8, label='1-day Aliases', linewidth=1.5)
        
        y_alias_minus = x_grid_linear / np.abs(1 - x_grid_linear)
        mask_lower = x_grid_linear < 0.99
        mask_upper = x_grid_linear > 1.01
        plt.plot(x_grid_log[mask_lower], np.log10(y_alias_minus[mask_lower]), color=c_alias_1d, linestyle='-.', alpha=0.8, linewidth=1.5)
        plt.plot(x_grid_log[mask_upper], np.log10(y_alias_minus[mask_upper]), color=c_alias_1d, linestyle='-.', alpha=0.8, linewidth=1.5)

        # 2. Period Doubled 1-day Aliases
        y_dbl_alias_plus = 2 * x_grid_linear / (1 + x_grid_linear)
        plt.plot(x_grid_log, np.log10(y_dbl_alias_plus), color=c_alias_1d_dbl, linestyle=':', alpha=0.8, label='2:1 1-day Aliases', linewidth=1.5)
        
        y_dbl_alias_minus = 2 * x_grid_linear / np.abs(1 - x_grid_linear)
        plt.plot(x_grid_log[mask_lower], np.log10(y_dbl_alias_minus[mask_lower]), color=c_alias_1d_dbl, linestyle=':', alpha=0.8, linewidth=1.5)
        plt.plot(x_grid_log[mask_upper], np.log10(y_dbl_alias_minus[mask_upper]), color=c_alias_1d_dbl, linestyle=':', alpha=0.8, linewidth=1.5)

        # 3. 0.5-day Aliases
        y_alias_2_plus = x_grid_linear / (1 + 2 * x_grid_linear)
        plt.plot(x_grid_log, np.log10(y_alias_2_plus), color=c_alias_05d, linestyle='--', alpha=0.8, label='0.5-day Aliases', linewidth=1.5)
        
        y_alias_2_minus = x_grid_linear / np.abs(1 - 2 * x_grid_linear)
        mask_lower_2 = x_grid_linear < 0.495
        mask_upper_2 = x_grid_linear > 0.505
        plt.plot(x_grid_log[mask_lower_2], np.log10(y_alias_2_minus[mask_lower_2]), color=c_alias_05d, linestyle='--', alpha=0.8, linewidth=1.5)
        plt.plot(x_grid_log[mask_upper_2], np.log10(y_alias_2_minus[mask_upper_2]), color=c_alias_05d, linestyle='--', alpha=0.8, linewidth=1.5)

        # 4. 2-day Aliases
        y_alias_05_plus = x_grid_linear / (1 + 0.5 * x_grid_linear)
        plt.plot(x_grid_log, np.log10(y_alias_05_plus), color=c_alias_2d, linestyle='-.', alpha=0.8, label='2-day Aliases', linewidth=1.5)
        
        y_alias_05_minus = x_grid_linear / np.abs(1 - 0.5 * x_grid_linear)
        mask_lower_05 = x_grid_linear < 1.98
        mask_upper_05 = x_grid_linear > 2.02
        plt.plot(x_grid_log[mask_lower_05], np.log10(y_alias_05_minus[mask_lower_05]), color=c_alias_2d, linestyle='-.', alpha=0.8, linewidth=1.5)
        plt.plot(x_grid_log[mask_upper_05], np.log10(y_alias_05_minus[mask_upper_05]), color=c_alias_2d, linestyle='-.', alpha=0.8, linewidth=1.5)

        # 5. 1:2 2-day Aliases
        y_half_alias_05_plus = 0.5 * x_grid_linear / (1 + 0.5 * x_grid_linear)
        plt.plot(x_grid_log, np.log10(y_half_alias_05_plus), color=c_alias_2d_half, linestyle=':', alpha=0.8, label='1:2 2-day Aliases', linewidth=1.5)
        
        y_half_alias_05_minus = 0.5 * x_grid_linear / np.abs(1 - 0.5 * x_grid_linear)
        plt.plot(x_grid_log[mask_lower_05], np.log10(y_half_alias_05_minus[mask_lower_05]), color=c_alias_2d_half, linestyle=':', alpha=0.8, linewidth=1.5)
        plt.plot(x_grid_log[mask_upper_05], np.log10(y_half_alias_05_minus[mask_upper_05]), color=c_alias_2d_half, linestyle=':', alpha=0.8, linewidth=1.5)

        plt.xlim(xlim)
        plt.ylim(ylim)
        
        plt.xlabel(r'log$_{10}$(PRIMVS Period) [days]')
        plt.ylabel(r'log$_{10}$(OGLE-IV Period) [days]')
        plt.legend(loc='upper left')
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/period_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {output_dir}/period_comparison.png")
        
    except Exception as e:
        print(f"Error creating period plot: {e}")
        import traceback
        traceback.print_exc()

def plot_rrlyr_galactic_distribution(fits_path, output_dir='figures'):
    """
    Plots the position of RRLyr stars in Galactic Coordinates (l, b).

    Goal:
        To see where the stars are located in the Milky Way.
        We transform the Longitude ($l$) from the standard 0-360 range to -180 to 180,
        placing the Galactic Center (l=0) in the middle of the plot.

    Inputs:
        fits_path: File containing 'l' (Longitude) and 'b' (Latitude) columns.
    """
    subset = get_rrlyr_candidates(fits_path)
    if subset is None or subset.empty:
        return

    # Center l coordinates (0..360 -> -180..180) for better visualization of Galactic Center
    subset['l_centered'] = subset['l'].apply(lambda x: x - 360 if x > 180 else x)
    
    # Sort by status to control z-order (Known bottom, New/Reclassified top)
    status_rank = {'Known RRLyr': 0, 'New Candidate': 1, 'Reclassified': 2, 'Unknown': -1}
    subset['rank'] = subset['Status'].map(status_rank)
    subset = subset.sort_values('rank')

    set_plot_style()
    plt.figure(figsize=(12, 7))
    palette = {'Known RRLyr': '#009E73', 'New Candidate': '#56B4E9', 'Reclassified': '#D55E00'}
    markers = {'Known RRLyr': '*', 'New Candidate': 'o', 'Reclassified': 'X'}
    
    for status in ['Known RRLyr', 'New Candidate', 'Reclassified']:
        layer = subset[subset['Status'] == status]
        if layer.empty: continue
        
        color = palette[status]
        marker = markers[status]
        
        if status == 'New Candidate':
            # Render "New Candidate" as HOLLOW circles to distinguish them
            plt.scatter(
                layer['l_centered'], layer['b'],
                c='none', 
                edgecolors=color,
                marker=marker,
                s=8, alpha=0.6, linewidths=0.5,
                label=status
            )
        else:
            plt.scatter(
                layer['l_centered'], layer['b'],
                c=color,
                edgecolors='none',
                marker=marker,
                s=8, alpha=0.6,
                label=status
            )

    plt.xlabel('Galactic Longitude ($l$) [deg]')
    plt.ylabel('Galactic Latitude ($b$) [deg]')
    
    # Set dynamic limits
    l_min = subset['l_centered'].min()
    l_max = subset['l_centered'].max()
    l_pad = max(1, (l_max - l_min) * 0.05)
    plt.xlim(l_max + l_pad, l_min - l_pad) # Inverted x-axis (Astro convention: East Left)

    b_min = subset['b'].min()
    b_max = subset['b'].max()
    b_pad = max(1, (b_max - b_min) * 0.05)
    plt.ylim(b_min - b_pad, b_max + b_pad)
    
    # Formatter to show longitude as 0-360 even though plot uses centered values
    def format_l(x, pos):
        val = x + 360 if x < 0 else x
        return f"{val:.0f}"
    
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_l))
    
    leg = plt.legend(title='Classification Status', loc='lower right', markerscale=3)
    for handle in leg.legend_handles: handle.set_alpha(1.0)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = f"{output_dir}/rrlyr_galactic_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_rrlyr_color_color(fits_path, output_dir='figures'):
    """
    Plots a Color-Color Diagram (Z-J vs J-Ks) for RRLyr stars.

    Goal:
        To compare the spectral properties (temperatures/reddening) of the stars.
        - X-axis ($J - K_s$): Measures redness.
        - Y-axis ($Z - J$): Measures blueness.
        We calculate $Z - J$ from existing columns: $(Z-Ks) - (J-Ks) = Z - J$.

    Inputs:
        fits_path: File containing infrared magnitudes (J, Ks, Z).
    """
    subset = get_rrlyr_candidates(fits_path)
    if subset is None or subset.empty:
        return

    # Define source columns for color calculation
    col_jk = 'j_med_mag-ks_med_mag' # The "Redder" color (J - Ks)
    col_zk = 'z_med_mag-ks_med_mag' # The wide baseline (Z - Ks)

    missing_cols = [c for c in [col_jk, col_zk] if c not in subset.columns]
    if missing_cols:
        print(f"Missing color columns {missing_cols}. Skipping Color-Color plot.")
        return
    
    # Derive the Z-J color index
    subset['Z_J'] = subset[col_zk] - subset[col_jk]
    
    x_axis = col_jk
    y_axis = 'Z_J'

    # Filter extreme outliers
    mask_colors = (
        (subset[x_axis] > -0.5) & (subset[x_axis] < 2.5) &
        (subset[y_axis] > -0.5) & (subset[y_axis] < 2.5)
    )
    subset = subset[mask_colors]

    # Sort for z-order
    status_rank = {'Known RRLyr': 0, 'New Candidate': 1, 'Reclassified': 2, 'Unknown': -1}
    subset['rank'] = subset['Status'].map(status_rank)
    subset = subset.sort_values('rank')

    set_plot_style()
    plt.figure(figsize=(9, 8))
    palette = {'Known RRLyr': '#009E73', 'New Candidate': '#56B4E9', 'Reclassified': '#D55E00'}
    markers = {'Known RRLyr': '*', 'New Candidate': 'o', 'Reclassified': 'X'}
    
    for status in ['Known RRLyr', 'New Candidate', 'Reclassified']:
        layer = subset[subset['Status'] == status]
        if layer.empty: continue
        
        color = palette[status]
        marker = markers[status]
        
        if status == 'New Candidate':
            plt.scatter(
                layer[x_axis], layer[y_axis],
                c='none',
                edgecolors=color,
                marker=marker,
                s=15, alpha=0.6, linewidths=0.5,
                label=status
            )
        else:
            plt.scatter(
                layer[x_axis], layer[y_axis],
                c=color,
                edgecolors='none',
                marker=marker,
                s=15, alpha=0.6,
                label=status
            )
    
    plt.xlabel(r'$J - K_s$ [mag]') # Redder color on X
    plt.ylabel(r'$Z - J$ [mag]')   # Bluer color on Y
    
    leg = plt.legend(title='Classification Status', loc='upper left', markerscale=2)
    for handle in leg.legend_handles: handle.set_alpha(1.0)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = f"{output_dir}/rrlyr_color_color.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def download_surot_map(map_path):
    """
    Downloads the Surot et al. (2020) extinction map if it doesn't exist locally.
    """
    url = "http://cdsarc.u-strasbg.fr/viz-bin/nph-Cat/fits?J/A+A/644/A140/ejkmap.dat.gz"
    
    if not os.path.exists(map_path):
        try:
            os.makedirs(os.path.dirname(map_path), exist_ok=True)
            urllib.request.urlretrieve(url, map_path)
        except Exception as e:
            print(f"Failed to download map: {e}")
            raise

def analyze_rrlyr_reddening(fits_path, output_dir='figures'):
    """
    Compares the Reddening (Extinction) of Known vs New RRLyr candidates.

    Goal:
        Stars behind dust clouds look redder and fainter ("Extinction").
        We cross-match our candidates with an external extinction map (Surot et al. 2020)
        to see if our "New" candidates are in highly obscured regions (which would explain
        why they were missed by previous surveys).

    Inputs:
        fits_path: File containing candidate coordinates.
    """
    subset = get_rrlyr_candidates(fits_path)
    if subset is None or subset.empty:
        return

    if 'l' not in subset.columns or 'b' not in subset.columns:
        print("Error: Missing 'l' or 'b' columns required for reddening lookup.")
        return

    map_path = ".data/surot_ejkmap.fits.gz"
    
    try:
        download_surot_map(map_path)
        
        with fits.open(map_path) as hdul:
            if len(hdul) > 1 and hasattr(hdul[1], 'data') and hdul[1].data is not None:
                data = hdul[1].data
                cols = data.columns.names
            else:
                data = hdul[0].data
                cols = getattr(hdul[0].columns, 'names', [])

            # Dynamic column search (case-insensitive) for map coordinates
            glon_col = next((c for c in cols if 'GLON' in c.upper() or 'L' == c.upper()), None)
            glat_col = next((c for c in cols if 'GLAT' in c.upper() or 'B' == c.upper()), None)
            ejk_col = next((c for c in cols if 'E' in c.upper() and ('J' in c.upper() or 'K' in c.upper()) and 'ERR' not in c.upper()), None)
            
            if not (glon_col and glat_col and ejk_col):
                print(f"Error: Could not identify GLON, GLAT, or Extinction columns in map. Found: {cols}")
                return

            # Create Catalog SkyCoord (Map)
            map_coords = SkyCoord(l=data[glon_col]*u.deg, b=data[glat_col]*u.deg, frame='galactic')
            
            # Create Candidate SkyCoord (Targets)
            cand_coords = SkyCoord(l=subset['l'].values*u.deg, b=subset['b'].values*u.deg, frame='galactic')
            
            # Perform nearest neighbor match using Astropy
            idx, d2d, _ = cand_coords.match_to_catalog_sky(map_coords)
            
            # Filter sources outside the map footprint (distance > 2 arcmin)
            match_limit = 2.0 * u.arcmin
            is_valid_match = d2d < match_limit
            n_dropped = (~is_valid_match).sum()
            
            matched_vals = data[ejk_col][idx]
            
            if hasattr(matched_vals, 'copy'):
                matched_vals = matched_vals.copy()
            else:
                matched_vals = np.array(matched_vals)
                
            matched_vals[~is_valid_match] = np.nan
            
            subset['E(J-Ks)'] = matched_vals
            
        # Calculate summary statistics per status group
        stats = subset.groupby('Status')['E(J-Ks)'].describe()[['count', 'mean', '50%', 'std']]
        stats.rename(columns={'50%': 'median'}, inplace=True)
        
        # --- Histogram Plot ---
        hist_subset = subset[subset['Status'].isin(['Known RRLyr', 'New Candidate'])].dropna(subset=['E(J-Ks)'])
        
        if not hist_subset.empty:
            set_plot_style()
            plt.figure(figsize=(10, 6))
            
            c_known = '#009E73'
            c_new = '#56B4E9'
            
            data_known = hist_subset[hist_subset['Status'] == 'Known RRLyr']['E(J-Ks)']
            data_new = hist_subset[hist_subset['Status'] == 'New Candidate']['E(J-Ks)']
            
            combined_data = pd.concat([data_known, data_new])
            if not combined_data.empty:
                bins = np.linspace(combined_data.min(), combined_data.max(), 50)
                
                # Plot 'New' (Discovery) FIRST (Bottom Layer)
                if not data_new.empty:
                    plt.hist(data_new, bins=bins, color=c_new, alpha=0.6, 
                             label=f'New Candidate (n={len(data_new)})', 
                             log=True, histtype='stepfilled')

                # Plot 'Known' (Reference) SECOND (Top Layer)
                if not data_known.empty:
                    plt.hist(data_known, bins=bins, color=c_known, alpha=0.6, 
                             label=f'Known RRLyr (n={len(data_known)})', 
                             log=True, histtype='stepfilled')
                
                plt.xlabel(r'Extinction $E(J-K_s)$ [mag]')
                plt.ylabel('Number (log scale)')
                plt.legend(loc='upper right')
                plt.xlim(0, max(3.0, combined_data.max()))
                plt.grid(True, which='major', linestyle='--', alpha=0.3)
                
                os.makedirs(output_dir, exist_ok=True)
                out_path = f"{output_dir}/rrlyr_reddening_hist.png"
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                print("No valid reddening data to plot.")
        else:
            print("No Known or New candidates found with valid extinction data.")
        
    except Exception as e:
        print(f"Error processing Surot map: {e}")
        import traceback
        traceback.print_exc()

def analyze_global_reddening(fits_path, output_dir='figures'):
    """
    Analyzes reddening for ALL high-confidence candidates (not just RRLyr).
    Identical logic to analyze_rrlyr_reddening, but uses the 'Global_Status'
    column to aggregate across all classes.
    """
    # 1. Get filtered candidates for ALL classes
    subset = get_all_candidates(fits_path)
    if subset is None or subset.empty:
        return

    if 'l' not in subset.columns or 'b' not in subset.columns:
        print("Error: Missing 'l' or 'b' columns required for reddening lookup.")
        return

    map_path = ".data/surot_ejkmap.fits.gz"
    
    try:
        download_surot_map(map_path)
        
        with fits.open(map_path) as hdul:
            if len(hdul) > 1 and hasattr(hdul[1], 'data') and hdul[1].data is not None:
                data = hdul[1].data
                cols = data.columns.names
            else:
                data = hdul[0].data
                cols = getattr(hdul[0].columns, 'names', [])

            glon_col = next((c for c in cols if 'GLON' in c.upper() or 'L' == c.upper()), None)
            glat_col = next((c for c in cols if 'GLAT' in c.upper() or 'B' == c.upper()), None)
            ejk_col = next((c for c in cols if 'E' in c.upper() and ('J' in c.upper() or 'K' in c.upper()) and 'ERR' not in c.upper()), None)
            
            if not (glon_col and glat_col and ejk_col):
                print(f"Error: Could not identify GLON, GLAT, or Extinction columns in map. Found: {cols}")
                return

            map_coords = SkyCoord(l=data[glon_col]*u.deg, b=data[glat_col]*u.deg, frame='galactic')
            cand_coords = SkyCoord(l=subset['l'].values*u.deg, b=subset['b'].values*u.deg, frame='galactic')
            
            idx, d2d, _ = cand_coords.match_to_catalog_sky(map_coords)
            
            match_limit = 2.0 * u.arcmin
            is_valid_match = d2d < match_limit
            n_dropped = (~is_valid_match).sum()
            
            matched_vals = data[ejk_col][idx]
            if hasattr(matched_vals, 'copy'): matched_vals = matched_vals.copy()
            else: matched_vals = np.array(matched_vals)
            matched_vals[~is_valid_match] = np.nan
            
            subset['E(J-Ks)'] = matched_vals
            
        # 3. Plotting Logic (Aggregated)
        hist_subset = subset[subset['Global_Status'].isin(['Known', 'New'])].dropna(subset=['E(J-Ks)'])
        
        if not hist_subset.empty:
            set_plot_style()
            plt.figure(figsize=(10, 6))
            
            c_known = '#009E73'
            c_new = '#56B4E9'
            
            data_known = hist_subset[hist_subset['Global_Status'] == 'Known']['E(J-Ks)']
            data_new = hist_subset[hist_subset['Global_Status'] == 'New']['E(J-Ks)']
            
            combined_data = pd.concat([data_known, data_new])
            if not combined_data.empty:
                bins = np.linspace(combined_data.min(), combined_data.max(), 50)
                
                # Plot 'New' Candidates FIRST (Bottom Layer)
                if not data_new.empty:
                    plt.hist(data_new, bins=bins, color=c_new, alpha=0.6, 
                             label=f'New Candidates (All Classes) (n={len(data_new)})', 
                             log=True, histtype='stepfilled')

                # Plot 'Known' SECOND (Top Layer)
                if not data_known.empty:
                    plt.hist(data_known, bins=bins, color=c_known, alpha=0.6, 
                             label=f'Known (All Classes) (n={len(data_known)})', 
                             log=True, histtype='stepfilled')
                
                plt.xlabel(r'Extinction $E(J-K_s)$ [mag]')
                plt.ylabel('Number (log scale)')
                plt.legend(loc='upper right')
                plt.xlim(0, max(3.0, combined_data.max()))
                plt.grid(True, which='major', linestyle='--', alpha=0.3)
                
                os.makedirs(output_dir, exist_ok=True)
                out_path = f"{output_dir}/global_reddening_hist.png"
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                print("No valid reddening data to plot.")
        else:
            print("No Known or New candidates found with valid extinction data.")

    except Exception as e:
        print(f"Error processing Surot map: {e}")
        import traceback
        traceback.print_exc()

def print_class_statistics(fits_path):
    """
    Prints a summary table of statistics for each class found in the predictions.

    Goal:
        To provide a quick textual overview of how many stars were predicted for each class,
        and how many of those predictions passed the high-confidence threshold.

    Columns Explained:
        - Predicted: Total sources where this class was the top prediction.
        - Candidates: Predicted sources that also passed the probability threshold.
        - In Training: Sources in this file labeled as this class in ground truth.
        - Training & Cut: Ground truth sources that passed the probability threshold (Recovery).
    """
    print(f"\n--- Class Statistics Summary ---")
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[1].data if len(hdul) > 1 else hdul[0].data
            header = hdul[1].header if len(hdul) > 1 else hdul[0].header
            
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
    except Exception as e:
        print(f"Error reading {fits_path}: {e}")
        return

    req_cols = ['xgb_predicted_class', 'xgb_training_class']
    if any(col not in df.columns for col in req_cols):
        print(f"Missing required columns {req_cols} for statistics.")
        return

    # Clean strings
    df['xgb_predicted_class'] = df['xgb_predicted_class'].astype(str).str.strip()
    df['xgb_training_class'] = df['xgb_training_class'].astype(str).str.strip()

    classes = sorted(df['xgb_predicted_class'].unique())
    
    print(f"{'Class':<15} | {'Pred':>8} | {'Cand(Cut)':>10} | {'True(File)':>12} | {'True w/ Cut':>12} | {'Threshold':>9}")
    print("-" * 80)

    total_pred = 0
    total_cand = 0
    total_true = 0
    total_true_cut = 0

    for cls in classes:
        if cls == 'UNKNOWN': continue 

        header_key = f"THR_{cls}"
        if header_key in header:
            threshold = float(header[header_key])
        else:
            threshold = 0.5
        
        prob_col = f"prob_{cls}"
        if prob_col not in df.columns:
            continue

        n_pred = len(df[df['xgb_predicted_class'] == cls])
        n_cand = len(df[(df['xgb_predicted_class'] == cls) & (df[prob_col] >= threshold)])
        n_true = len(df[df['xgb_training_class'] == cls])
        n_true_cut = len(df[(df['xgb_training_class'] == cls) & (df[prob_col] >= threshold)])

        print(f"{cls:<15} | {n_pred:8d} | {n_cand:10d} | {n_true:12d} | {n_true_cut:12d} | {threshold:9.3f}")

        total_pred += n_pred
        total_cand += n_cand
        total_true += n_true
        total_true_cut += n_true_cut

    print("-" * 80)
    print(f"{'TOTAL':<15} | {total_pred:8d} | {total_cand:10d} | {total_true:12d} | {total_true_cut:12d} | {'-':>9}")
    print("\n")

if __name__ == "__main__":
    # --- Main Execution Block ---
    # This allows the script to be run as a standalone program to generate plots.
    # It checks for data files in the default locations and generates whatever figures possible.
    
    # Default file paths
    periods_file = ".data/periods.fits" 
    training_file = ".data/PRIMVS_P_training_new.fits"
    predictions_file = "./xgb_predictions.fits"

    # Allow command line arguments to override defaults
    # Usage: python vis.py [periods_file] [training_file]
    if len(sys.argv) >= 2:
        periods_file = sys.argv[1]
    if len(sys.argv) >= 3:
        training_file = sys.argv[2]
        
    # 1. Period Comparison Plot
    if os.path.exists(periods_file):
        plot_period_comparison(periods_file)
    else:
        print(f"File not found: {periods_file}. Skipping period comparison.")

    # 2. Training Data Bailey Diagram
    if os.path.exists(training_file):
        try:
            df_train = load_fits_to_df(training_file)
            
            # Auto-detect label column if not standard 'Type'
            class_col = "Type"
            if class_col not in df_train.columns:
                 candidates = [c for c in df_train.columns if 'type' in c.lower() or 'class' in c.lower()]
                 if candidates: class_col = candidates[0]
            
            if class_col in df_train.columns:
                plot_training_bailey_diagram(df_train, class_col)
        except Exception as e:
            print(f"Error plotting training diagram: {e}")

    # 3. RRLyr Analysis & Global Reddening
    if os.path.exists(predictions_file):
        print_class_statistics(predictions_file)
        
        plot_rrlyr_galactic_distribution(predictions_file)
        plot_rrlyr_color_color(predictions_file)
        analyze_rrlyr_reddening(predictions_file, output_dir='figures')
        
        analyze_global_reddening(predictions_file, output_dir='figures')
    else:
        print(f"File not found: {predictions_file}. Skipping RRLyr plots.")
