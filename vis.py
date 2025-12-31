import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import math
from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from matplotlib.colors import ListedColormap, LogNorm, LinearSegmentedColormap, to_rgba

#########################################
# SECTION 1: STYLE & UTILITY FUNCTIONS
#########################################

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
        '#000000'  # Black
    ]
    
    # Secondary Palette: Tab20
    # If there are more classes than the primary palette can handle (8), we extend
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
    
    # Filter out unphysical or extreme outliers for a cleaner plot
    if 'true_amplitude' in df.columns:
        df = df[df['true_amplitude'] < 2]
    
    # Ensure log period exists for the x-axis
    if 'log_true_period' not in df.columns and 'true_period' in df.columns:
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


def plot_confidence_entropy(df, class_column, output_dir='class_figures', min_prob=0.0):
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
                precision, recall, _ = precision_recall_curve(y_true_class, y_proba_class)
                
                # "No Skill" baseline is the prevalence (ratio of positive cases)
                no_skill = pos_count / total_count
                class_metrics_data[class_name]['no_skill'] = no_skill
                class_metrics_data[class_name]['pr_curve'] = (precision, recall)
                
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
            precision, recall = data['pr_curve']
            color = color_map[class_name]
            
            # Plot without label (legend is already in ROC plot)
            ax_pr.plot(recall, precision, lw=2, color=color)
            
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
