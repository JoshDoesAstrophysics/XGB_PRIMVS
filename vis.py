import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import sys
import math
from astropy.io import fits
from astropy.table import Table
from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from matplotlib.colors import ListedColormap, LogNorm, LinearSegmentedColormap, to_rgba

#########################################
# SECTION 1: STYLE & UTILITY FUNCTIONS
#########################################

def load_fits_to_df(path):
    """
    Load data from a FITS file into a pandas DataFrame.

    Handles potential endianness issues and gracefully manages FITS file structures.
    Identical implementation to XGB.py for consistency.

    Args:
        path (str): The file path to the FITS file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    print(f"Loading {path}...")
    try:
        with fits.open(path) as hdul:
            # Check if extension 1 exists and has data, otherwise use extension 0
            data = hdul[1].data if len(hdul) > 1 and hdul[1].data is not None else hdul[0].data
            
            if not hasattr(data, 'names') or not data.names:
                df = pd.DataFrame(np.asarray(data))
            else:
                data_dict = {}
                for col in data.names:
                    col_data = np.asarray(data[col])
                    # Fix endianness if necessary (pandas requires native byte order)
                    if col_data.dtype.byteorder not in ('=', '|'):
                        col_data = col_data.astype(col_data.dtype.newbyteorder('='))
                    data_dict[col] = col_data
                df = pd.DataFrame(data_dict)
            print(f"Loaded {len(df)} samples with {len(df.columns)} features")
            return df
    except Exception as e:
        print(f"Error loading FITS file: {e}")
        raise

def get_consistent_color_map(class_names):
    """
    Generates a consistent color mapping for class names.
    
    This ensures that a specific class (e.g., "ECL") is always represented by 
    the same color across all plots (Bailey diagram, ROC, etc.), making it 
    easier to track performance visually.

    Args:
        class_names (list): A list of class name strings.

    Returns:
        dict: A dictionary mapping class names to hex color codes.
    """
    # Sort classes alphabetically to ensure the mapping is deterministic regardless of input order.
    # This prevents colors from swapping if the order of classes in the data changes.
    sorted_classes = sorted(list(set(class_names)))
    
    # Primary Palette: High-Contrast Accessible Colors
    # This palette is designed to be accessible to people with color blindness.
    # It modifies the standard Okabe-Ito palette to increase contrast between 
    # specific hues that can appear similar in small markers or thin lines.
    palette = [
        '#E69F00', # Orange
        '#56B4E9', # Sky Blue
        '#009E73', # Bluish Green
        '#F0E442', # Yellow
        '#332288', # Indigo (Contrasts with Sky Blue)
        '#882255', # Wine/Dark Red (Contrasts with Orange)
        '#CC79A7', # Reddish Purple
        '#000000', # Black
        # Extended high-contrast colors to avoid limiting to 8
        '#88CCEE', # Cyan
        '#44AA99', # Teal
        '#117733', # Green
        '#999933', # Olive
        '#DDCC77', # Sand
        '#CC6677', # Rose
        '#AA4499', # Purple
    ]
    
    # Secondary Palette: Tab20
    # If there are more classes than the primary palette can handle, we extend
    # the palette using Matplotlib's Tab20.
    if len(sorted_classes) > len(palette):
        tab20 = plt.cm.tab20.colors # Returns RGB tuples (0-1)
        # Convert RGB tuples to Hex strings
        tab20_hex = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b in tab20]
        palette.extend(tab20_hex)

    # Assign colors to classes cyclically
    color_map = {}
    for i, cls in enumerate(sorted_classes):
        color_map[cls] = palette[i % len(palette)]
        
    return color_map


def set_plot_style(large_text=False):
    """
    Configures Matplotlib's rcParams to produce publication-quality figures.
    
    This centralizes style settings (fonts, line widths, dpi) so all plots 
    share a professional and consistent look.

    Args:
        large_text (bool): If True, significantly increases font sizes. 
                           Useful for posters or slides where standard text is too small.
    """
    plt.rcParams['figure.figsize'] = [8, 6]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    
    # Define base font sizes
    base_size = 14
    if large_text:
        base_size = 18
        
    # Apply font sizes to various plot elements
    plt.rcParams['font.size'] = base_size
    plt.rcParams['axes.labelsize'] = base_size + 2
    plt.rcParams['axes.titlesize'] = base_size + 4
    plt.rcParams['xtick.labelsize'] = base_size + 2
    plt.rcParams['ytick.labelsize'] = base_size + 2
    plt.rcParams['legend.fontsize'] = base_size - 2
    plt.rcParams['axes.linewidth'] = 1.5
    
    # Line and marker styles
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8
    
    # Background and grid settings
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
    Plots the training and validation log-loss over iterations.
    
    This plot helps diagnose model fitting:
    - Diverging lines (Validation rising, Training falling) indicate Overfitting.
    - Both lines high indicate Underfitting.
    - Ideally, Validation loss decreases and then flattens out.

    Args:
        evals_result (dict): The dictionary returned by xgb.train containing evaluation metrics.
        output_dir (str): Directory to save the plot.
    """
    # XGBoost returns a nested dictionary structure; we extract the metric lists here.
    train_loss = list(evals_result['train'].values())[0]
    val_loss = list(evals_result['validation'].values())[0]
    iterations = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_loss, 'b-', label='Training Loss')
    plt.plot(iterations, val_loss, 'r-', label='Validation Loss')
    
    # Identify and highlight the iteration with the minimum validation loss (best model)
    best_iter = val_loss.index(min(val_loss)) + 1
    plt.axvline(x=best_iter, color='gray', linestyle='--')
    plt.scatter(best_iter, min(val_loss), color='red', s=100)
    plt.annotate(f'Best: {min(val_loss):.4f} (iter {best_iter})', 
                 xy=(best_iter, min(val_loss)),
                 xytext=(best_iter + 5, min(val_loss)),
                 arrowprops=dict(arrowstyle='->'))
    
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.title('XGBoost Training Progress')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_loss.png")
    print(f"Saved loss plot to {output_dir}/")
    
    return plt


def plot_bailey_diagram(df, class_column, output_dir='class_figures', min_prob=0.7, min_confidence=0.9, max_entropy=0.2):
    """
    Creates a Bailey diagram (Period vs. Amplitude), a standard diagnostic tool in
    variable star astronomy. 
    
    Points are colored by their predicted class. We filter for high-confidence predictions
    to see the "core" distribution of each class, rather than noisy outliers.

    Args:
        df (pd.DataFrame): The dataframe containing 'true_period', 'true_amplitude', and predictions.
        class_column (str): The column name containing the class labels.
        output_dir (str): Directory to save the plot.
        min_prob (float): Minimum probability threshold for including a point.
        min_confidence (float): Minimum confidence margin threshold.
        max_entropy (float): Maximum entropy threshold (filters out uncertain predictions).
    """
    # Set plot style with larger text for readability
    set_plot_style(large_text=True)
    plt.rcParams['legend.fontsize'] = 15
    
    # Work on a copy to prevent side effects on the main dataframe
    df = df.copy()

    # --- Pre-cleaning (Consistent with Training Diagram & XGB) ---
    # Apply robust quantile clipping to Period and Amplitude to clean up extreme outliers.
    cols_to_clean = ['true_period', 'true_amplitude']
    for col in cols_to_clean:
        if col in df.columns:
            # Handle Infinite values
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Calculate 0.1% and 99.9% quantiles based on the CURRENT dataframe
            # This ensures we clean outliers specific to the dataset being plotted
            q001 = df[col].quantile(0.001)
            q999 = df[col].quantile(0.999)
            
            # Clip values
            df[col] = df[col].clip(lower=q001, upper=q999)
    
    # Filter out unphysical or extreme outliers for a cleaner plot
    if 'true_amplitude' in df.columns:
        df = df[df['true_amplitude'] < 2]
    
    # Ensure log period exists for the x-axis
    if 'log_true_period' not in df.columns and 'true_period' in df.columns:
        # Ensure positive periods before log
        df = df[df['true_period'] > 0]
        df['log_true_period'] = np.log10(df['true_period'])
    
    if 'log_true_period' in df.columns:
        df = df[df['log_true_period'] < 2.7]
    
    # Filter data based on prediction confidence to show only high-quality classifications
    prob_col = class_column.replace('predicted_class', 'confidence')
    if prob_col in df.columns:
        df = df[df[prob_col] > min_prob]
    
    if 'xgb_confidence' in df.columns:
        df = df[df['xgb_confidence'] > min_confidence]
    if 'xgb_entropy' in df.columns:
        df = df[df['xgb_entropy'] < max_entropy]
    
    # Limit the number of points per class to prevent the plot from becoming a solid block of color.
    # We take the top N most confident predictions per class.
    sampled_df = df.groupby(class_column).apply(
        lambda x: x.nlargest(n=min(len(x), 10000), columns=prob_col)
    ).reset_index(drop=True)
    
    # Get consistent colors
    unique_types = sorted(sampled_df[class_column].unique())
    color_map = get_consistent_color_map(unique_types)
    markers = ['o', 's', '^', 'D', 'v', 'X', 'P', '*', 'h']
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    for i, var_type in enumerate(unique_types):
        type_df = sampled_df[sampled_df[class_column] == var_type]
        
        color = color_map[var_type]
        marker = markers[i % len(markers)]
            
        plt.scatter(type_df['log_true_period'], type_df['true_amplitude'], 
                   color=color, 
                   marker=marker, 
                   label=var_type, s=7, alpha=0.3)
    
    leg = plt.legend(bbox_to_anchor=(0.1, 0.8), ncol=2, markerscale=5)
    for handle in leg.legend_handles: handle.set_alpha(1.0)
    ax.set_xlabel(r'log$_{10}$(Period) [days]')
    ax.set_ylabel(r'Amplitude [mag]')
    
    ax.set_xlim(-1, 2.7)
    ax.set_ylim(0, 2)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/bailey_diagram.jpg', dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_bailey_diagram(df, class_column, output_dir='figures'):
    """
    Creates a Bailey diagram for the training data (Ground Truth).
    
    Unlike the prediction plot, this DOES NOT apply any confidence, probability, 
    or entropy filters, ensuring we visualize the raw distribution of the training set.
    
    However, it applies quantile clipping (0.1% - 99.9%) to remove extreme outliers,
    mimicking the XGBoost preprocessing step for a cleaner view.

    Args:
        df (pd.DataFrame): The training dataframe.
        class_column (str): The column name containing the true class labels.
        output_dir (str): Directory to save the plot.
    """
    # Set plot style with larger text for readability
    set_plot_style(large_text=True)
    plt.rcParams['legend.fontsize'] = 15
    
    # Work on a copy to prevent side effects on the main dataframe
    df = df.copy()

    # --- Pre-cleaning (borrowed from XGB.py logic) ---
    # Apply robust quantile clipping to Period and Amplitude to clean up extreme outliers.
    # This aligns the visualization with the data distribution seen by the model.
    cols_to_clean = ['true_period', 'true_amplitude']
    for col in cols_to_clean:
        if col in df.columns:
            # Handle Infinite values
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Calculate 0.1% and 99.9% quantiles based on the CURRENT dataframe
            # This ensures we clean outliers specific to the dataset being plotted
            q001 = df[col].quantile(0.001)
            q999 = df[col].quantile(0.999)
            
            # Clip values
            df[col] = df[col].clip(lower=q001, upper=q999)
    
    # Filter out unphysical or extreme outliers for a cleaner plot (optional but recommended)
    # The quantile clipping above handles the majority, but we keep the < 2 limit 
    # for the y-axis standard of Bailey diagrams unless the data genuinely goes higher.
    if 'true_amplitude' in df.columns:
        df = df[df['true_amplitude'] < 2]
    
    # Ensure log period exists for the x-axis
    if 'log_true_period' not in df.columns and 'true_period' in df.columns:
        # Ensure positive periods before log
        df = df[df['true_period'] > 0]
        df['log_true_period'] = np.log10(df['true_period'])
    
    if 'log_true_period' in df.columns:
        df = df[df['log_true_period'] < 2.7]
    
    # Sample to prevent overplotting. 
    # To mimic the "high confidence" filtering of the prediction plot, we 
    # prioritize samples with the lowest False Alarm Probability (best_fap) if available.
    # This ensures we see the highest quality training data, rather than random noise.
    if 'best_fap' in df.columns:
        print("Using 'best_fap' to prioritize high-quality samples for training diagram.")
        sampled_df = df.groupby(class_column).apply(
            lambda x: x.nsmallest(n=min(len(x), 10000), columns='best_fap')
        ).reset_index(drop=True)
    else:
        sampled_df = df.groupby(class_column).apply(
            lambda x: x.sample(n=min(len(x), 10000), random_state=42)
        ).reset_index(drop=True)
    
    # Get consistent colors
    unique_types = sorted(sampled_df[class_column].unique())
    color_map = get_consistent_color_map(unique_types)
    markers = ['o', 's', '^', 'D', 'v', 'X', 'P', '*', 'h']
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    for i, var_type in enumerate(unique_types):
        type_df = sampled_df[sampled_df[class_column] == var_type]
        
        color = color_map[var_type]
        marker = markers[i % len(markers)]
            
        plt.scatter(type_df['log_true_period'], type_df['true_amplitude'], 
                   color=color, 
                   marker=marker, 
                   label=var_type, s=7, alpha=0.3)
    
    leg = plt.legend(bbox_to_anchor=(0.1, 0.8), ncol=2, markerscale=5)
    for handle in leg.legend_handles: handle.set_alpha(1.0)
    ax.set_xlabel(r'log$_{10}$(Period) [days]')
    ax.set_ylabel(r'Amplitude [mag]')
    ax.set_title('Training Data Bailey Diagram (Clipped & Quality Filtered)')
    
    ax.set_xlim(-1, 2.7)
    ax.set_ylim(0, 2)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/bailey_diagram_training.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training Bailey diagram to {output_dir}/bailey_diagram_training.jpg")


def plot_confidence_entropy(df, class_column, output_dir='class_figures', min_prob=0.0, use_brg_cmap=False):
    """
    Plots Prediction Confidence vs. Prediction Entropy using Faceted 2D Histograms.
    
    This visualization helps assess model certainty:
    - High Confidence + Low Entropy (Bottom Right): The model is certain and precise.
    - Low Confidence + High Entropy (Top Left): The model is confused/uncertain.
    
    Visualization Strategy:
    We use Faceted 2D Histograms (Heatmaps) to handle the density of points.
    To ensure visibility and high dynamic range, we construct a custom colormap 
    for each class.
    
    The Colormap transitions from:
    1. Light Tint (Low Density): A faint version of the class color, ensuring visibility 
       against the white background.
    2. Class Color (Medium Density): The primary visual identifier.
    3. Black (Peak Density): Highlights the core of the distribution.
    
    Args:
        df (pd.DataFrame): Dataframe containing predictions.
        class_column (str): The column containing class labels.
        output_dir (str): Directory to save output.
        min_prob (float): Filter points below this probability (optional).
        use_brg_cmap (bool): If True, overrides custom colormaps with the 'brg' colormap.
    """
    df = df.copy()

    # Determine confidence column and filter
    prob_col = class_column.replace('predicted_class', 'confidence')
    if prob_col in df.columns:
        df = df[df[prob_col] > min_prob]

    # Select X and Y axes. If 'entropy' isn't pre-calculated, we estimate uncertainty.
    if 'xgb_confidence' not in df.columns or 'xgb_entropy' not in df.columns:
        x_col = prob_col
        y_col = None
        for col in df.columns:
            if col != prob_col and pd.api.types.is_numeric_dtype(df[col]) and col != class_column:
                y_col = col
                break
        if y_col is None:
            df['uncertainty'] = 1 - df[prob_col]
            y_col = 'uncertainty'
    else:
        x_col = 'xgb_confidence'
        y_col = 'xgb_entropy'

    unique_types = sorted(df[class_column].unique())
    n_classes = len(unique_types)
    
    # Get consistent colors
    color_map = get_consistent_color_map(unique_types)

    # Calculate grid dimensions (e.g., 3 columns wide)
    ncols = 3
    nrows = math.ceil(n_classes / ncols)
    
    # Create subplots. Share x and y axes to make direct visual comparison between classes easy.
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True)
    
    # Flatten axes array for easy iteration
    if n_classes > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Calculate global bounds so all histograms share the same scale
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    
    bins = 50 # Resolution of the histogram grid

    for i, var_type in enumerate(unique_types):
        ax = axes[i]
        type_df = df[df[class_column] == var_type]
        
        if len(type_df) == 0:
            continue
        
        # Determine which colormap to use
        if use_brg_cmap:
            cmap = 'brg'
        else:
            # --- Create Custom "Density" Colormap for this Class ---
            base_color = color_map[var_type]
            
            # Detect if the base color is Black (or very dark)
            # If so, we need a grayscale gradient to preserve dynamic range.
            is_black = (base_color.lower() == '#000000') or (base_color.lower() == 'black')
            
            if is_black:
                # Special Grayscale Gradient for Black class
                colors = [
                    '#D0D0D0', # Light Gray (Low Density) - Visible on white background
                    '#606060', # Dark Gray (Medium Density)
                    '#000000'  # Black (Peak Density)
                ]
            else:
                # Standard Gradient for Colored classes
                # 1. Low Density: Light tint (mix with white) so it pops against white background
                # 2. Medium Density: The class color itself
                # 3. Peak Density: Black (to show high concentration core)
                
                rgb = to_rgba(base_color)[:3]
                # Mix with white (80% white, 20% color) for the light tint
                light_tint = [(0.2 * c + 0.8) for c in rgb] 
                
                colors = [
                    light_tint, # Very light version of class color
                    base_color, # The class color
                    '#000000'   # Black for the core
                ]
            
            # Create the colormap
            cmap = LinearSegmentedColormap.from_list(f"cmap_{var_type}", colors)

        # Plot 2D Histogram (Heatmap)
        # norm=LogNorm() scales colors logarithmically, ensuring both sparse outliers 
        # and dense cores are visible. cmin=1 ensures empty bins are transparent.
        h = ax.hist2d(
            type_df[x_col], 
            type_df[y_col], 
            bins=bins, 
            range=[[x_min, x_max], [y_min, y_max]],
            cmap=cmap, 
            norm=LogNorm(),
            cmin=1 
        )
        
        # Add a small colorbar to each subplot
        plt.colorbar(h[3], ax=ax)
        
        ax.set_title(f"{var_type}\n(n={len(type_df)})")
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Only label outer axes to reduce clutter
        if i // ncols == nrows - 1:
            ax.set_xlabel(f'Confidence ({x_col})')
        if i % ncols == 0:
            ax.set_ylabel(f'Entropy ({y_col})')

    # Turn off any unused subplots in the grid
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Confidence vs. Entropy Density by Class', fontsize=16, y=1.02)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/classification_confidence_entropy.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_distribution(confidences, predictions, class_names, output_dir="figures"):
    """
    Plots a Step Histogram of the model's confidence scores for each class.
    
    This visualizes how "sure" the model usually is for each class.
    - We use 'step' histograms (unfilled outlines) to allow overlapping distributions 
      to be seen clearly without occluding each other.
    - We use density normalization (density=True) so that classes with different 
      population sizes (e.g., 100 vs 100,000 samples) can be compared on the same scale.
    
    Args:
        confidences (np.array): Array of confidence scores.
        predictions (np.array): Array of integer class predictions.
        class_names (list): List of class names corresponding to the integer predictions.
        output_dir (str): Directory to save the plot.
    """
    color_map = get_consistent_color_map(class_names)
    
    plt.figure(figsize=(12, 6))
    
    # Use fixed bins from 0 to 1 for consistent comparison
    bins = np.linspace(0, 1, 51) # 50 bins
    
    for i, class_name in enumerate(class_names):
        class_conf = confidences[predictions == i]
        if len(class_conf) > 0:
            # Step histogram with clear colored outlines
            plt.hist(class_conf, 
                     bins=bins, 
                     histtype='step', 
                     linewidth=2.5, 
                     label=f"{class_name} (n={len(class_conf)})", 
                     color=color_map[class_name],
                     density=True) # Normalize area to 1 to compare shapes despite imbalance
    
    plt.xlabel('Model Confidence')
    plt.ylabel('Density (Normalized)')
    plt.title('Distribution of Model Confidence by Predicted Class')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{output_dir}/confidence_distribution.png', dpi=300)
    plt.close()


def plot_xgb_feature_importance(feature_names, importance_values, top_n=20, output_dir='figures'):
    """
    Bar chart visualization of the most important features used by the XGBoost model.
    Importance is typically measured by 'Gain' (improvement in accuracy brought by a feature).
    """
    # Sort to find the top N features
    indices = np.argsort(importance_values)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importance = [importance_values[i] for i in indices]
    
    plt.figure(figsize=(14, 10))
    plt.barh(range(len(top_features)), top_importance, align='center', color='skyblue', edgecolor='navy', alpha=0.8)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance (Gain)', fontsize=14)
    plt.title(f'Top {top_n} Feature Importance', fontsize=16)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/xgb_feature_importance.png', dpi=300)
    plt.close()


def plot_and_print_auc_ap(df, true_label_col, label_encoder, output_dir='figures'):
    """
    Comprehensive evaluation of model performance.
    
    Calculates:
    1. Macro-Averaged Precision (mAP).
    2. ROC AUC scores.
    
    Generates:
    1. A Bar Chart comparing AP and ROC AUC per class.
    2. A Combined Plot containing both ROC Curves and Precision-Recall Curves side-by-side.

    Args:
        df (pd.DataFrame): Dataframe containing truth and probability columns.
        true_label_col (str): Column name for the ground truth labels.
        label_encoder (LabelEncoder): Encoder object to map class names to integers.
        output_dir (str): Directory to save plots.
    """
    print("\nCalculating AP and ROC AUC metrics...")
    
    if true_label_col not in df.columns:
        print(f"Warning: True label column '{true_label_col}' not found. Skipping metrics.")
        return

    classes = label_encoder.classes_
    prob_cols = [f'prob_{cls}' for cls in classes]
    
    # Validation: Ensure all necessary probability columns exist in the dataframe
    missing_cols = [col for col in prob_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing probability columns, skipping AP/AUC calculation: {missing_cols}")
        return
        
    # Binarize labels for One-vs-Rest calculation (e.g., "Is Class A" vs "Is NOT Class A")
    y_true_raw = df[true_label_col].fillna('UNKNOWN')
    Y_true = label_binarize(y_true_raw, classes=classes)
    Y_proba = df[prob_cols].values
    
    per_class_ap = []
    per_class_roc = []
    class_labels = []
    
    # Storage for curve plotting
    class_metrics_data = {} 
    
    total_ap = 0
    total_roc = 0
    valid_ap_classes = 0
    valid_roc_classes = 0

    # Iterate through each class to calculate metrics
    for k, class_name in enumerate(classes):
        y_true_class = Y_true[:, k]
        y_proba_class = Y_proba[:, k]
        
        pos_count = y_true_class.sum()
        neg_count = len(y_true_class) - pos_count
        total_count = len(y_true_class)
        
        # Initialize dictionary for this class
        class_metrics_data[class_name] = {
            'count': pos_count,
            'ap': np.nan,
            'roc': np.nan,
            'pr_curve': None,
            'roc_curve': None,
            'no_skill': 0
        }
        
        # We can only calculate metrics if there is at least one positive example
        if pos_count > 0:
            try:
                # 1. Average Precision (AP)
                ap = average_precision_score(y_true_class, y_proba_class)
                per_class_ap.append(ap)
                total_ap += ap
                valid_ap_classes += 1
                class_metrics_data[class_name]['ap'] = ap
                
                # 2. Precision-Recall Curve Data
                # Note: thresholds has length n, precision/recall have length n+1
                precision, recall, thresholds = precision_recall_curve(y_true_class, y_proba_class)
                
                # "No Skill" baseline is the prevalence (ratio of positive cases)
                no_skill = pos_count / total_count
                class_metrics_data[class_name]['no_skill'] = no_skill
                class_metrics_data[class_name]['pr_curve'] = (precision, recall, thresholds)
                
            except ValueError:
                per_class_ap.append(np.nan)

            # 3. ROC AUC
            # Requires both positive and negative examples
            if neg_count > 0:
                try:
                    roc = roc_auc_score(y_true_class, y_proba_class)
                    per_class_roc.append(roc)
                    total_roc += roc
                    valid_roc_classes += 1
                    class_metrics_data[class_name]['roc'] = roc
                    
                    # ROC Curve Data
                    fpr, tpr, _ = roc_curve(y_true_class, y_proba_class)
                    class_metrics_data[class_name]['roc_curve'] = (fpr, tpr)
                    
                except ValueError:
                    per_class_roc.append(np.nan)
            else:
                per_class_roc.append(np.nan)
                
            class_labels.append(class_name)
        else:
            pass

    # --- Print Macro-Averaged Summary ---
    macro_ap = total_ap / valid_ap_classes if valid_ap_classes > 0 else 0
    macro_roc_auc = total_roc / valid_roc_classes if valid_roc_classes > 0 else 0

    print("--- Macro-Averaged Metrics ---")
    print(f"Macro-Averaged Precision (mAP) (over {valid_ap_classes} classes): {macro_ap:.4f}")
    print(f"Macro-Averaged ROC AUC         (over {valid_roc_classes} classes): {macro_roc_auc:.4f}")

    # --- Plot 1: Per-Class Bar Chart ---
    if not class_labels:
        print("No classes with true samples found.")
        return

    metrics_df = pd.DataFrame({
        'Class': class_labels,
        'AP': per_class_ap,
        'ROC_AUC': per_class_roc
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
        
        # Horizontal bars for AP and ROC AUC
        ax.barh(y_pos - bar_height / 2, metrics_df['AP'], height=bar_height, label='Avg. Precision (AP)', color='C0')
        ax.barh(y_pos + bar_height / 2, metrics_df['ROC_AUC'], height=bar_height, label='ROC AUC', color='C2')
        
        # Annotate bars with numeric values
        for i, (ap, roc) in enumerate(zip(metrics_df['AP'], metrics_df['ROC_AUC'])):
            if pd.notna(ap): ax.text(ap + 0.01, y_pos[i] - bar_height / 2, f'{ap:.2f}', va='center', fontsize=9)
            if pd.notna(roc): ax.text(roc + 0.01, y_pos[i] + bar_height / 2, f'{roc:.2f}', va='center', fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics_df['Class'])
        ax.set_xlabel('Score')
        ax.set_title('Per-Class Average Precision (AP) and ROC AUC Scores')
        ax.set_xlim(0, 1.15)
        ax.legend(loc='lower right')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/per_class_metrics.jpg', dpi=300, bbox_inches='tight')
        plt.close()

    # --- Plot 2: Combined ROC and PR Curves ---
    # We use a single figure with two subplots for side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Get consistent colors
    valid_class_names = [cls for cls, data in class_metrics_data.items() if data['count'] > 0]
    color_map = get_consistent_color_map(valid_class_names)
    
    # --- Subplot 1: ROC Curves ---
    ax_roc = axes[0]
    has_roc_data = False
    
    for class_name in valid_class_names:
        data = class_metrics_data[class_name]
        if data['roc_curve'] is not None:
            fpr, tpr = data['roc_curve']
            color = color_map[class_name]
            
            # Label with Class, Population (n), AUC, and AP
            auc_val = data['roc']
            ap_val = data['ap']
            count = data['count']
            
            # Format label string
            label_str = f"{class_name} (n={int(count)}) AUC={auc_val:.2f} AP={ap_val:.2f}"
            
            ax_roc.plot(fpr, tpr, lw=2, color=color, label=label_str)
            has_roc_data = True

    if has_roc_data:
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic (ROC)')
        
        # Legend configuration using consistent font sizing.
        # We enforce a minimum size of 12 for readability.
        ax_roc.legend(loc="lower right", fontsize=12, ncol=1)
        ax_roc.grid(alpha=0.3)
    else:
        ax_roc.text(0.5, 0.5, "No valid ROC data available", ha='center', va='center')

    # --- Subplot 2: Precision-Recall Curves ---
    ax_pr = axes[1]
    has_pr_data = False
    
    for class_name in valid_class_names:
        data = class_metrics_data[class_name]
        if data['pr_curve'] is not None:
            precision, recall, thresholds = data['pr_curve']
            color = color_map[class_name]
            no_skill = data['no_skill']
            
            # Plot without label (legend is already in ROC plot)
            ax_pr.plot(recall, precision, lw=2, color=color)
            
            # Add no skill baseline for this class
            ax_pr.plot([0, 1], [no_skill, no_skill], linestyle=':', lw=1.5, color=color, alpha=0.7)

            # --- Find and Annotate Best Threshold (Closest to 1,1) ---
            # Calculate Euclidean distance from each point on the curve to (1, 1)
            # Distance^2 = (1 - recall)^2 + (1 - precision)^2
            distances = (1 - recall)**2 + (1 - precision)**2
            best_idx = np.argmin(distances)
            
            # Coordinates of the best point
            best_r = recall[best_idx]
            best_p = precision[best_idx]
            
            # Retrieve the corresponding threshold
            # Note: precision/recall arrays are length n_thresholds + 1
            # If best_idx is the last element, it corresponds to threshold=1.0 (implicitly)
            if best_idx < len(thresholds):
                best_thresh = thresholds[best_idx]
            else:
                best_thresh = 1.0
            
            # Plot the specific point
            ax_pr.scatter(best_r, best_p, color=color, s=50, edgecolor='white', zorder=5)
            
            # Annotate with the threshold value
            # We offset the text slightly to avoid overlapping the curve too much
            ax_pr.annotate(
                f"{best_thresh:.2f}",
                xy=(best_r, best_p),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                color=color,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, ec="none")
            )
            
            has_pr_data = True

    if has_pr_data:
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curves\n(Annotated with optimal probability thresholds)')
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        
        # No legend needed here as it relies on the ROC legend
        ax_pr.grid(alpha=0.3)
    else:
        ax_pr.text(0.5, 0.5, "No valid PR data available", ha='center', va='center')

    # Save as PDF for high quality tiling
    plt.tight_layout()
    pdf_path = f'{output_dir}/metrics_summary.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined metrics plot to {pdf_path}")


#########################################
# SECTION 3: PCA VISUALIZATION FUNCTIONS
#########################################

def visualize_pca(pca_df, original_df, pca_model, output_dir='./figures'):
    """
    Generates a suite of visualizations to analyze Principal Component Analysis (PCA) results.

    Plots generated:
    1. PC1 vs PC2 Scatter Plot: Basic projection of data points.
    2. Density Plot: KDE plot to show concentrations in PCA space.
    3. 2D Histogram: Binned heatmap of PC density.
    4. Scree Plot: Explained variance per component to judge dimensionality.
    5. Loadings Heatmap: Shows which original features contribute most to PC1/PC2.
    6. Feature Overlays: Scatter plots colored by 'best_fap', 'Period', and 'Amplitude'.

    Args:
        pca_df (pd.DataFrame): DataFrame containing the PCA components (PC1, PC2, etc.).
        original_df (pd.DataFrame): The original dataframe with feature columns (for coloring points).
        pca_model (sklearn.decomposition.PCA): The trained PCA model object.
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()

    # 1. Scatter Plot
    plt.figure()
    plt.scatter(pca_df['PC1'], pca_df['PC2'], s=5, alpha=0.5)
    plt.xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.2%})")
    plt.title("PCA: PC1 vs PC2")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_scatter.png")
    plt.close()

    # 2. Density Plot
    plt.figure()
    sns.kdeplot(x=pca_df['PC1'], y=pca_df['PC2'], cmap="Blues", fill=True)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Density")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_density.png")
    plt.close()

    # 3. 2D Histogram
    plt.figure()
    h = plt.hist2d(pca_df['PC1'], pca_df['PC2'], bins=100, cmap='viridis', norm=LogNorm())
    plt.colorbar(h[3])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Histogram")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_histogram.png")
    plt.close()

    # 4. Scree Plot
    plt.figure()
    plt.bar(range(1, len(pca_model.explained_variance_ratio_)+1), pca_model.explained_variance_ratio_)
    plt.step(range(1, len(pca_model.explained_variance_ratio_)+1), np.cumsum(pca_model.explained_variance_ratio_), label='Cumulative')
    plt.xlabel('PC')
    plt.ylabel('Explained Variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_scree.png")
    plt.close()

    # 5. Loadings Heatmap
    if hasattr(pca_model, 'feature_names_in_'):
        feature_names = pca_model.feature_names_in_
    else:
        # Fallback if feature names weren't stored in the model
        feature_names = [f"Feature {i}" for i in range(pca_model.components_.shape[1])]

    load = pd.DataFrame(pca_model.components_.T, 
                        index=feature_names, 
                        columns=[f'PC{i+1}' for i in range(pca_model.n_components_)])
    
    # Select features. We want to show all significant features.
    # We increase the limit to ensure we don't accidentally cut off features if the user has ~15.
    top = pd.concat([
        load['PC1'].abs().sort_values(ascending=False), 
        load['PC2'].abs().sort_values(ascending=False)
    ]).index.unique()[:50] # Increased limit to 50 to ensure all 15+ features are shown
    
    plt.figure(figsize=(10, max(6, len(top) * 0.4)))
    sns.heatmap(load.loc[top, ['PC1', 'PC2']], annot=True, cmap='coolwarm', center=0)
    plt.title("Feature Loadings (PC1 & PC2)")
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
        # Log scale for Period to handle large dynamic range
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
        # Log scale for Amplitude
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
    Analyzes the structural relationship between original features and Principal Components.
    
    Generates a Structure Matrix Heatmap:
    - Shows correlations between Features (rows) and PCs (columns).
    - Features are ordered by the input list (grouping physical variables together).
    - Numbers are hidden if they are below the threshold, unless they are the 
      max correlation for that feature.
    - X-axis labels include the explained variance percentage.

    Args:
        pca_df (pd.DataFrame): DataFrame containing the PCA components.
        df (pd.DataFrame): Original dataframe containing the raw feature values.
        pca_model (sklearn.decomposition.PCA): The trained PCA model.
        features (list): List of feature names used for PCA.
        output_dir (str): Directory to save plots.
        n_pcs_limit (int): Unused (kept for compatibility). 
        threshold (float): Correlation threshold for masking text.
    """
    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()
    
    # Ensure we only use features that exist in the dataframe
    valid_features = [f for f in features if f in df.columns]
    
    if not valid_features:
        print("Warning: No valid features found for degeneracy analysis.")
        return

    # --- FULL PLOT WITH CUSTOM ANNOTATIONS ---
    
    # Use top 20 PCs or whatever is available
    n_pcs_full = min(20, pca_df.shape[1]) 
    pc_cols_full = [f'PC{i+1}' for i in range(n_pcs_full)]
    
    # Construct labels with Explained Variance Percentage
    if hasattr(pca_model, 'explained_variance_ratio_'):
        explained_variance = pca_model.explained_variance_ratio_[:n_pcs_full]
        pc_labels = [f"PC{i+1} ({var:.1%})" for i, var in enumerate(explained_variance)]
    else:
        pc_labels = pc_cols_full

    # Combine PCs and Features into one temp dataframe for correlation calculation
    analysis_df = pd.concat([pca_df[pc_cols_full].reset_index(drop=True), 
                            df[valid_features].reset_index(drop=True)], axis=1)
    
    # Calculate correlation matrix
    corr_matrix_full = analysis_df.corr()
    
    # Extract only the sub-matrix: Features (rows) vs PCs (cols)
    # CRITICAL: We use 'valid_features' to enforce the user's input order (Physical, then others...)
    # We do NOT sort by magnitude.
    feature_pc_corr = corr_matrix_full.loc[valid_features, pc_cols_full]
    
    # --- Create Mask for Annotations (Clean Look) ---
    # Rule 1: Show text if abs(value) >= threshold
    show_text_mask = feature_pc_corr.abs() >= threshold
    
    # Rule 2: ALWAYS show the max association per feature (row)
    # This ensures every feature has at least one number explaining where it went.
    row_max_indices = feature_pc_corr.abs().idxmax(axis=1)
    for row_idx, col_name in row_max_indices.items():
        if pd.notna(col_name):
            show_text_mask.loc[row_idx, col_name] = True

    # Generate custom annotation strings
    annot_labels = feature_pc_corr.applymap(lambda x: f"{x:.2f}")
    annot_labels = annot_labels.where(show_text_mask, "")
    
    # Plotting
    # Make width significantly larger to accommodate "PC1 (25.0%)" labels
    fig_height = max(10, len(valid_features) * 0.6)
    plt.figure(figsize=(20, fig_height))
    
    sns.heatmap(feature_pc_corr, 
                annot=annot_labels, 
                cmap='RdBu', 
                center=0, 
                fmt='', # fmt='' is required because annot contains strings
                vmin=-1, 
                vmax=1, 
                xticklabels=pc_labels,
                yticklabels=True,
                cbar_kws={'label': 'Correlation'})
                
    plt.title(f"Feature-Component Structure Matrix")
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
    Plots a comparison of periods from a single FITS file containing both data sources.
    
    Loads data from a single FITS file and plots 'true_period' vs 'P_1' on a log-log scale.

    Args:
        periods_path (str): Path to the periods FITS file.
        output_dir (str): Directory to save the output plot.
    """
    print(f"Generating Period-Period comparison plot using {periods_path}...")
    
    # Load data
    try:
        # Load Periods using robust loader
        df = load_fits_to_df(periods_path)
        
        # Check period columns
        x_col = 'true_period'
        y_col = 'P_1'
        
        if x_col not in df.columns or y_col not in df.columns:
            print(f"Missing period columns. Expected '{x_col}' and '{y_col}' in {periods_path}")
            print(f"Columns found: {df.columns.tolist()}")
            return
            
        # Plot
        set_plot_style()
        plt.figure(figsize=(8, 8))
        
        # Filter for positive periods for log scale
        valid_data = df[(df[x_col] > 0) & (df[y_col] > 0)]
        x = valid_data[x_col]
        y = valid_data[y_col]
        
        # Log scale
        plt.xscale('log')
        plt.yscale('log')
        
        plt.scatter(x, y, s=10, alpha=0.1, c='black', edgecolors='none')
        
        # Capture the auto-scaled limits from the scatter plot BEFORE adding lines.
        # This is critical for avoiding excessive whitespace.
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        
        # Determine plot bounds for the reference lines.
        # We calculate the union of x and y ranges to ensure the lines span the 
        # entire relevant area (e.g., if x goes to 10 but y goes to 100, the 1:1 line needs to go to 100).
        min_line = min(xlim[0], ylim[0])
        max_line = max(xlim[1], ylim[1])
        
        # Identity line (1:1)
        plt.plot([min_line, max_line], [min_line, max_line], 'r--', alpha=0.5, label='1:1')
        
        # 2:1 Harmonic (y = 2x) - The line above the diagonal
        plt.plot([min_line, max_line], [min_line * 2, max_line * 2], 'g-.', alpha=0.5, label='2:1')
        
        # Restore the original limits to ensure the lines don't force the plot to expand.
        plt.xlim(xlim)
        plt.ylim(ylim)
        
        plt.xlabel("PRIMVS Periods")
        plt.ylabel("OGLE-IV periods")
        plt.title("Period Comparison")
        plt.legend(loc='upper left')
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/period_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {output_dir}/period_comparison.png")
        
    except Exception as e:
        print(f"Error creating period plot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # --- Main execution block for running specific plots directly ---
    
    # Default file paths
    periods_file = ".data/periods.fits" 
    training_file = ".data/PRIMVS_P_training_new.fits"

    # Allow command line arguments to override defaults
    if len(sys.argv) >= 2:
        periods_file = sys.argv[1]
    if len(sys.argv) >= 3:
        training_file = sys.argv[2]
        
    # 1. Run Period Comparison Plot (if file exists)
    if os.path.exists(periods_file):
        plot_period_comparison(periods_file)
    else:
        print(f"File not found: {periods_file}. Skipping period comparison.")

    # 2. Run Training Bailey Diagram Plot (if file exists)
    if os.path.exists(training_file):
        print(f"Generating Training Bailey Diagram using {training_file}...")
        try:
            df_train = load_fits_to_df(training_file)
            
            # Auto-detect label column if not standard 'Type'
            class_col = "Type"
            if class_col not in df_train.columns:
                 candidates = [c for c in df_train.columns if 'type' in c.lower() or 'class' in c.lower()]
                 if candidates:
                     class_col = candidates[0]
                     print(f"Using detected class column: {class_col}")
            
            if class_col in df_train.columns:
                plot_training_bailey_diagram(df_train, class_col)
            else:
                print("Could not find class column (e.g., 'Type') in training file.")
        except Exception as e:
            print(f"Error plotting training diagram: {e}")
    else:
        print(f"File not found: {training_file}. Skipping training Bailey diagram.")
