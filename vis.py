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
from matplotlib.ticker import FuncFormatter

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


def plot_bailey_diagram(df, class_column, output_dir='class_figures', min_prob=0.7, min_confidence=0.9, max_entropy=0.2, thresholds_dict=None):
    """
    Creates a Bailey diagram (Period vs. Amplitude), a standard diagnostic tool in
    variable star astronomy. 
    
    Points are colored by their predicted class. We filter for high-confidence predictions
    to see the "core" distribution of each class, rather than noisy outliers.

    Args:
        df (pd.DataFrame): The dataframe containing 'true_period', 'true_amplitude', and predictions.
        class_column (str): The column name containing the class labels.
        output_dir (str): Directory to save the plot.
        min_prob (float): Minimum probability threshold (used only if thresholds_dict is None).
        min_confidence (float): Minimum confidence margin threshold (used only if thresholds_dict is None).
        max_entropy (float): Maximum entropy threshold (used only if thresholds_dict is None).
        thresholds_dict (dict): Dictionary mapping class names to optimal probability thresholds.
                                If provided, this overrides other filter arguments.
    """
    # Set plot style with larger text for readability
    set_plot_style(large_text=True)
    plt.rcParams['legend.fontsize'] = 15
    
    # Work on a copy to prevent side effects on the main dataframe
    df = df.copy()
    
    # Filter out unphysical or extreme outliers for a cleaner plot (Keep limits broad for now)
    if 'true_amplitude' in df.columns:
        df = df[df['true_amplitude'] < 5] # Increased to 5 per request
    
    # Ensure log period exists for the x-axis
    if 'log_true_period' not in df.columns and 'true_period' in df.columns:
        df['log_true_period'] = np.log10(df['true_period'])
    
    # Dynamic Filtering Logic
    if thresholds_dict is not None:
        print("Using optimal PR curve thresholds for Bailey Diagram filtering.")
        # Filter based on per-class thresholds
        # Create a boolean mask initialized to False
        keep_mask = pd.Series(False, index=df.index)
        
        unique_classes = df[class_column].unique()
        for cls in unique_classes:
            prob_col = f'prob_{cls}'
            if prob_col in df.columns:
                threshold = thresholds_dict.get(cls, 0.5) # Default to 0.5 if not found
                # Select rows where (Pred Class is cls) AND (Prob > Threshold)
                cls_mask = (df[class_column] == cls) & (df[prob_col] >= threshold)
                keep_mask |= cls_mask
        
        df = df[keep_mask]
    else:
        # Fallback to old logic if no dictionary provided
        prob_col = class_column.replace('predicted_class', 'confidence')
        if prob_col in df.columns:
            df = df[df[prob_col] > min_prob]
        
        if 'xgb_confidence' in df.columns:
            df = df[df['xgb_confidence'] > min_confidence]
        if 'xgb_entropy' in df.columns:
            df = df[df['xgb_entropy'] < max_entropy]
    
    # Limit the number of points per class to prevent the plot from becoming a solid block of color.
    # We take the top N most confident predictions per class using sort_values + groupby + head.
    prob_sort_col = class_column.replace('predicted_class', 'confidence')
    if prob_sort_col not in df.columns:
        prob_sort_col = 'xgb_confidence' # Fallback
        
    if prob_sort_col in df.columns:
        sampled_df = df.sort_values(prob_sort_col, ascending=False).groupby(class_column).head(10000).reset_index(drop=True)
    else:
        sampled_df = df.groupby(class_column).head(10000).reset_index(drop=True)
    
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
    
    # Legend at Top Center
    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=min(len(unique_types), 5), markerscale=5, frameon=True)
    for handle in leg.legend_handles: handle.set_alpha(1.0)
    
    ax.set_xlabel(r'log$_{10}$(Period) [days]')
    ax.set_ylabel(r'Amplitude [mag]')
    
    # Updated Limits
    ax.set_xlim(left=-1) # Remove upper limit
    ax.set_ylim(0, 5)    # Increase y limit to 5
    
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
        df = df[df['true_amplitude'] < 5] # Increased to 5 per request
    
    # Ensure log period exists for the x-axis
    if 'log_true_period' not in df.columns and 'true_period' in df.columns:
        # Ensure positive periods before log
        df = df[df['true_period'] > 0]
        df['log_true_period'] = np.log10(df['true_period'])
    
    if 'log_true_period' in df.columns:
        # df = df[df['log_true_period'] < 2.7] # Removed upper limit
        pass
    
    # Sample to prevent overplotting. 
    # To mimic the "high confidence" filtering of the prediction plot, we 
    # prioritize samples with the lowest False Alarm Probability (best_fap) if available.
    # This ensures we see the highest quality training data, rather than random noise.
    # We use sort_values + groupby + head instead of apply to avoid DeprecationWarnings.
    if 'best_fap' in df.columns:
        print("Using 'best_fap' to prioritize high-quality samples for training diagram.")
        sampled_df = df.sort_values('best_fap', ascending=True).groupby(class_column).head(10000).reset_index(drop=True)
    else:
        # For random sampling, we shuffle the whole DF first, then take the head of each group.
        sampled_df = df.sample(frac=1, random_state=42).groupby(class_column).head(10000).reset_index(drop=True)
    
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
    
    # Legend at Top Center
    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=min(len(unique_types), 5), markerscale=5, frameon=True)
    for handle in leg.legend_handles: handle.set_alpha(1.0)
    ax.set_xlabel(r'log$_{10}$(Period) [days]')
    ax.set_ylabel(r'Amplitude [mag]')
    ax.set_title('Training Data Bailey Diagram (Clipped & Quality Filtered)')
    
    # Updated Limits
    ax.set_xlim(left=-1) # Remove upper limit
    ax.set_ylim(0, 5)    # Increase y limit to 5
    
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
    
    Returns:
        dict: A dictionary mapping class names to their optimal probability threshold found on the PR curve.
    """
    print("\nCalculating AP and ROC AUC metrics...")
    
    optimal_thresholds = {}

    if true_label_col not in df.columns:
        print(f"Warning: True label column '{true_label_col}' not found. Skipping metrics.")
        return optimal_thresholds

    classes = label_encoder.classes_
    prob_cols = [f'prob_{cls}' for cls in classes]
    
    # Validation: Ensure all necessary probability columns exist in the dataframe
    missing_cols = [col for col in prob_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing probability columns, skipping AP/AUC calculation: {missing_cols}")
        return optimal_thresholds
        
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
        return optimal_thresholds

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
            
            # Store this best threshold for return
            optimal_thresholds[class_name] = best_thresh
            
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
        ax_pr.set_title('Precision-Recall Curves')
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
    
    return optimal_thresholds


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
    
    Loads data from a single FITS file and plots 'log10(true_period)' vs 'log10(P_1)'.

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
        
        # Filter for positive periods for log scale calculation
        valid_data = df[(df[x_col] > 0) & (df[y_col] > 0)]
        
        # Transform to log10 space
        x_log = np.log10(valid_data[x_col])
        y_log = np.log10(valid_data[y_col])
        
        # Plot scatter in log-log space (linear axes of log values)
        plt.scatter(x_log, y_log, s=10, alpha=0.1, c='black', edgecolors='none')
        
        # Capture the auto-scaled limits from the scatter plot BEFORE adding lines.
        # This is critical for avoiding excessive whitespace.
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        
        # Determine plot bounds for the reference lines.
        # We calculate the union of x and y ranges to ensure the lines span the 
        # entire relevant area.
        min_line = min(xlim[0], ylim[0])
        max_line = max(xlim[1], ylim[1])
        
        # Use colorblind-friendly palette from get_consistent_color_map (hardcoded for stability)
        # 1:1 Line: Reddish Purple (#CC79A7)
        # Harmonics (2:1, 1:2): Bluish Green (#009E73)
        # 1-day Alias: Sky Blue (#56B4E9)
        # 1/2-day Alias: Cyan (#88CCEE)
        # 2-day Alias: Purple (#AA4499)

        # Identity line (1:1) -> log(y) = log(x)
        plt.plot([min_line, max_line], [min_line, max_line], color='#CC79A7', linestyle='--', alpha=0.8, label='1:1', linewidth=2)
        
        # 2:1 Harmonic (y = 2x) -> log(y) = log(x) + log10(2)
        log2 = np.log10(2)
        plt.plot([min_line, max_line], [min_line + log2, max_line + log2], color='#009E73', linestyle='-.', alpha=0.8, label='2:1 / 1:2', linewidth=1.5)

        # 1:2 Harmonic (y = 0.5x) -> log(y) = log(x) + log10(0.5)
        log05 = np.log10(0.5)
        plt.plot([min_line, max_line], [min_line + log05, max_line + log05], color='#009E73', linestyle='-.', alpha=0.8, linewidth=1.5)
        
        # --- Sidereal Day Aliases (k=1, 1 cycle/day) ---
        # Frequency domain: f_obs = |f_true +/- k * f_sampling|
        # Period domain: P_obs = 1 / |1/P_true +/- k| = P_true / |1 +/- k*P_true|
        
        # Create a grid in log space, convert to linear for formula, then back to log for plotting
        x_grid_log = np.linspace(min_line, max_line, 2000)
        x_grid_linear = 10**x_grid_log
        
        # k=1 (f_obs = f_true + 1)
        y_alias_plus = x_grid_linear / (1 + x_grid_linear)
        plt.plot(x_grid_log, np.log10(y_alias_plus), color='#56B4E9', linestyle='--', alpha=0.8, label='1-day Aliases', linewidth=1.5)
        
        # k=-1 (f_obs = |f_true - 1|)
        # Note: Pole at P=1 (log P=0)
        y_alias_minus = x_grid_linear / np.abs(1 - x_grid_linear)
        
        mask_lower = x_grid_linear < 0.99
        mask_upper = x_grid_linear > 1.01
        plt.plot(x_grid_log[mask_lower], np.log10(y_alias_minus[mask_lower]), color='#56B4E9', linestyle='--', alpha=0.8, linewidth=1.5)
        plt.plot(x_grid_log[mask_upper], np.log10(y_alias_minus[mask_upper]), color='#56B4E9', linestyle='--', alpha=0.8, linewidth=1.5)

        # --- Period Doubled 1-day Aliases (P_obs = 2 * P_alias_1d) ---
        # This represents an alias (f +/- 1) that is period doubled (2:1)
        # Formula: y = 2 * [ P / (1 +/- P) ]
        
        # k=1
        y_dbl_alias_plus = 2 * x_grid_linear / (1 + x_grid_linear)
        plt.plot(x_grid_log, np.log10(y_dbl_alias_plus), color='#E69F00', linestyle=':', alpha=0.8, label='2:1 1-day Aliases', linewidth=1.5)
        
        # k=-1 (Pole at P=1, same as standard 1-day alias)
        y_dbl_alias_minus = 2 * x_grid_linear / np.abs(1 - x_grid_linear)
        plt.plot(x_grid_log[mask_lower], np.log10(y_dbl_alias_minus[mask_lower]), color='#E69F00', linestyle=':', alpha=0.8, linewidth=1.5)
        plt.plot(x_grid_log[mask_upper], np.log10(y_dbl_alias_minus[mask_upper]), color='#E69F00', linestyle=':', alpha=0.8, linewidth=1.5)

        # --- Sidereal Day Aliases (k=2, 2 cycles/day -> 1/2 day period) ---
        # k=2 (f_obs = f_true + 2)
        y_alias_2_plus = x_grid_linear / (1 + 2 * x_grid_linear)
        plt.plot(x_grid_log, np.log10(y_alias_2_plus), color='#88CCEE', linestyle='--', alpha=0.6, label='0.5-day Aliases', linewidth=1.2)
        
        # k=-2 (f_obs = |f_true - 2|)
        # Note: Pole at P=0.5 (log P=-0.301)
        y_alias_2_minus = x_grid_linear / np.abs(1 - 2 * x_grid_linear)

        # Handle singularity at P=0.5
        mask_lower_2 = x_grid_linear < 0.495
        mask_upper_2 = x_grid_linear > 0.505
        plt.plot(x_grid_log[mask_lower_2], np.log10(y_alias_2_minus[mask_lower_2]), color='#88CCEE', linestyle='--', alpha=0.6, linewidth=1.2)
        plt.plot(x_grid_log[mask_upper_2], np.log10(y_alias_2_minus[mask_upper_2]), color='#88CCEE', linestyle='--', alpha=0.6, linewidth=1.2)

        # --- Sidereal Day Aliases (k=0.5, 0.5 cycles/day -> 2 day period) ---
        # k=0.5 (f_obs = f_true + 0.5)
        y_alias_05_plus = x_grid_linear / (1 + 0.5 * x_grid_linear)
        plt.plot(x_grid_log, np.log10(y_alias_05_plus), color='#AA4499', linestyle='--', alpha=0.6, label='2-day Aliases', linewidth=1.2)
        
        # k=-0.5 (f_obs = |f_true - 0.5|)
        # Note: Pole at P=2.0 (log P=0.301)
        y_alias_05_minus = x_grid_linear / np.abs(1 - 0.5 * x_grid_linear)

        # Handle singularity at P=2.0
        mask_lower_05 = x_grid_linear < 1.98
        mask_upper_05 = x_grid_linear > 2.02
        plt.plot(x_grid_log[mask_lower_05], np.log10(y_alias_05_minus[mask_lower_05]), color='#AA4499', linestyle='--', alpha=0.6, linewidth=1.2)
        plt.plot(x_grid_log[mask_upper_05], np.log10(y_alias_05_minus[mask_upper_05]), color='#AA4499', linestyle='--', alpha=0.6, linewidth=1.2)

        # --- Period Halved (1:2) 2-day Aliases (P_obs = 0.5 * P_alias_2d) ---
        # This represents an alias (f +/- 0.5) that is period halved (1:2)
        # Formula: y = 0.5 * [ P / (1 +/- 0.5*P) ]
        
        # k=0.5
        y_half_alias_05_plus = 0.5 * x_grid_linear / (1 + 0.5 * x_grid_linear)
        plt.plot(x_grid_log, np.log10(y_half_alias_05_plus), color='#AA4499', linestyle=':', alpha=0.8, label='1:2 2-day Aliases', linewidth=1.5)
        
        # k=-0.5 (Pole at P=2.0, same as standard 2-day alias)
        y_half_alias_05_minus = 0.5 * x_grid_linear / np.abs(1 - 0.5 * x_grid_linear)
        plt.plot(x_grid_log[mask_lower_05], np.log10(y_half_alias_05_minus[mask_lower_05]), color='#AA4499', linestyle=':', alpha=0.8, linewidth=1.5)
        plt.plot(x_grid_log[mask_upper_05], np.log10(y_half_alias_05_minus[mask_upper_05]), color='#AA4499', linestyle=':', alpha=0.8, linewidth=1.5)

        # Restore the original limits to ensure the lines don't force the plot to expand.
        plt.xlim(xlim)
        plt.ylim(ylim)
        
        # Update axis labels to be scientific and consistent with Bailey diagram style
        plt.xlabel(r'log$_{10}$(PRIMVS Period) [days]')
        plt.ylabel(r'log$_{10}$(OGLE-IV Period) [days]')
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

def plot_rrlyr_galactic_distribution(fits_path, output_dir='figures'):
    """
    Plots the Galactic distribution (l vs b) of predicted RRLyr stars.
    
    This function:
    1. Loads predictions from the FITS file.
    2. Extracts the RRLyr confidence threshold from the header (THR_RRLyr).
    3. Filters for sources predicted as RRLyr with probability > threshold.
    4. Categorizes sources based on their training status:
       - Known RRLyr (Training Class == RRLyr)
       - New Candidate (Training Class == UNKNOWN)
       - Reclassified (Training Class != RRLyr & != UNKNOWN)
    
    Args:
        fits_path (str): Path to the XGBoost predictions FITS file.
        output_dir (str): Directory to save the plot.
    """
    print(f"Generating RRLyr Galactic Distribution plot using {fits_path}...")
    
    try:
        # Load data AND header
        with fits.open(fits_path) as hdul:
            data = hdul[1].data if len(hdul) > 1 else hdul[0].data
            header = hdul[1].header if len(hdul) > 1 else hdul[0].header
            
            # Convert to Pandas
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
        
        # Verify required columns
        req_cols = ['l', 'b', 'xgb_predicted_class', 'xgb_training_class', 'prob_RRLyr']
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            print(f"Missing required columns for RRLyr plot: {missing}")
            return

        # 1. Get Threshold from Header
        # Look for THR_RRLyr (generated by XGB.py). Default to 0.5 if not found.
        threshold = 0.5
        if 'THR_RRLyr' in header:
            threshold = float(header['THR_RRLyr'])
            print(f"Using RRLyr threshold from header: {threshold:.3f}")
        else:
            print("Warning: THR_RRLyr not found in header. Defaulting to 0.5")

        # 2. Filter Data
        # Predicted as RRLyr AND Probability > Threshold
        subset = df[
            (df['xgb_predicted_class'].str.strip() == 'RRLyr') & 
            (df['prob_RRLyr'] >= threshold)
        ].copy()
        
        if subset.empty:
            print("No RRLyr candidates found above threshold.")
            return

        print(f"Found {len(subset)} RRLyr candidates above threshold.")

        # 3. Categorize (Hue Logic)
        # Clean training class strings just in case
        subset['xgb_training_class'] = subset['xgb_training_class'].astype(str).str.strip()

        conditions = [
            (subset['xgb_training_class'] == 'RRLyr'),
            (subset['xgb_training_class'] == 'UNKNOWN'),
            (subset['xgb_training_class'] != 'RRLyr') & (subset['xgb_training_class'] != 'UNKNOWN')
        ]
        choices = ['Known RRLyr', 'New Candidate', 'Reclassified']
        
        subset['Status'] = np.select(conditions, choices, default='Unknown')

        # --- MODIFICATION START ---
        # Center coordinates on l=0 (Galactic Center)
        # Standard range becomes [-180, 180] roughly, where values > 180 are wrapped to negative.
        # This maps 350 -> -10, 10 -> 10, putting 0 in the middle.
        subset['l_centered'] = subset['l'].apply(lambda x: x - 360 if x > 180 else x)
        # --- MODIFICATION END ---
        
        # Sort by status to control z-order (Known -> New -> Reclassified)
        # We want Reclassified on top (last), Known at bottom (first)
        status_rank = {'Known RRLyr': 0, 'New Candidate': 1, 'Reclassified': 2, 'Unknown': -1}
        subset['rank'] = subset['Status'].map(status_rank)
        subset = subset.sort_values('rank')

        # 4. Plot
        set_plot_style()
        plt.figure(figsize=(12, 7))
        
        # Define colors mapping to logic
        # Known = Green (Confirmation)
        # New = Blue (Discovery)
        # Reclassified = Red/Orange (Correction)
        palette = {
            'Known RRLyr': '#009E73',    # Bluish Green
            'New Candidate': '#56B4E9',  # Sky Blue
            'Reclassified': '#D55E00'    # Vermilion
        }
        
        # Plot using Seaborn for easy categorical handling
        sns.scatterplot(
            data=subset, 
            x='l_centered',  # Changed from 'l'
            y='b', 
            hue='Status', 
            palette=palette,
            style='Status',
            markers={'Known RRLyr': 'o', 'New Candidate': '*', 'Reclassified': 'X'},
            s=7, 
            alpha=0.8,
            edgecolor='none'
        )

        plt.xlabel('Galactic Longitude ($l$) [deg]')
        plt.ylabel('Galactic Latitude ($b$) [deg]')
        plt.title(f'Galactic Distribution of Predicted RRLyr (p > {threshold:.2f})')
        
        # --- MODIFICATION START ---
        # Set dynamic limits based on the actual centered data range plus padding
        l_min = subset['l_centered'].min()
        l_max = subset['l_centered'].max()
        l_pad = max(1, (l_max - l_min) * 0.05) # Min padding of 5 degrees
        
        # Invert x-axis: Start at positive (left), end at negative (right)
        # This matches standard astronomical convention (East Left, West Right)
        plt.xlim(l_max + l_pad, l_min - l_pad)

        # Dynamic Y limits based on data range
        b_min = subset['b'].min()
        b_max = subset['b'].max()
        b_pad = max(1, (b_max - b_min) * 0.05)
        plt.ylim(b_min - b_pad, b_max + b_pad)
        
        # Format ticks to show positive values (0-360)
        def format_l(x, pos):
            val = x + 360 if x < 0 else x
            return f"{val:.0f}"
        
        plt.gca().xaxis.set_major_formatter(FuncFormatter(format_l))
        # --- MODIFICATION END ---
        
        # Improve Legend
        # Increase markerscale to make legend symbols larger, similar to Bailey diagram
        leg = plt.legend(title='Classification Status', bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3)
        for handle in leg.legend_handles: 
            handle.set_alpha(1.0)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_path = f"{output_dir}/rrlyr_galactic_distribution.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved RRLyr galactic plot to {out_path}")

    except Exception as e:
        print(f"Error creating RRLyr galactic plot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # --- Main execution block for running specific plots directly ---
    
    # Default file paths
    periods_file = ".data/periods.fits" 
    training_file = ".data/PRIMVS_P_training_new.fits"
    predictions_file = "./xgb_predictions.fits"

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

    # 3. Run RRLyr Galactic Distribution Plot (if file exists)
    if os.path.exists(predictions_file):
        plot_rrlyr_galactic_distribution(predictions_file)
    else:
        print(f"File not found: {predictions_file}. Skipping RRLyr galactic plot.")
