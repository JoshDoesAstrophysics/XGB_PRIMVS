import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from matplotlib.colors import ListedColormap

# --- Unified Color Strategy ---
def get_consistent_color_map(class_names):
    """
    Generate a consistent color mapping for a list of class names.

    Ensures that the same class always gets the same color, utilizing a 
    colorblind-friendly palette extended with Tab20 for larger numbers of classes.

    Args:
        class_names (list): A list of class names to generate colors for.

    Returns:
        dict: A dictionary mapping class names (keys) to hex color codes (values).
    """
    # Sort classes to ensure the mapping is deterministic regardless of data order
    sorted_classes = sorted(list(set(class_names)))
    
    # 1. Okabe-Ito Palette (Colorblind Friendly)
    # 2. Tab20 (for larger number of classes)
    palette = [
        '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000'
    ]
    
    # If we have more classes than the base palette, extend with Tab20
    if len(sorted_classes) > len(palette):
        tab20 = plt.cm.tab20.colors # Returns RGB tuples (0-1)
        # Convert RGB to Hex
        tab20_hex = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b in tab20]
        palette.extend(tab20_hex)

    color_map = {}
    for i, cls in enumerate(sorted_classes):
        # Cycle through palette if we somehow still run out
        color_map[cls] = palette[i % len(palette)]
        
    return color_map


def set_plot_style(large_text=False):
    """
    Configure global matplotlib settings for publication-quality figures.

    Sets figure size, DPI, font sizes, line widths, and grid styles.

    Args:
        large_text (bool): If True, increases font sizes for better visibility 
                           in presentations or posters. Defaults to False.
    """
    plt.rcParams['figure.figsize'] = [8, 6]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    
    # Text size settings
    base_size = 14
    if large_text:
        base_size = 18
        
    plt.rcParams['font.size'] = base_size
    plt.rcParams['axes.labelsize'] = base_size + 2
    plt.rcParams['axes.titlesize'] = base_size + 4
    plt.rcParams['xtick.labelsize'] = base_size + 2
    plt.rcParams['ytick.labelsize'] = base_size + 2
    plt.rcParams['legend.fontsize'] = base_size - 2
    plt.rcParams['axes.linewidth'] = 1.5
    
    # Line widths
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8
    
    # Colors and grid
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.color'] = "grey"


def plot_xgb_training_loss(evals_result, output_dir='figures'):
    """
    Plot the training and validation loss curves from XGBoost evaluation results.

    Args:
        evals_result (dict): The evaluation result dictionary returned by 
                             xgb.train(). Expected structure: {'train': {...}, 'validation': {...}}.
        output_dir (str): Directory where the plot will be saved. Defaults to 'figures'.
    
    Returns:
        matplotlib.pyplot: The plt object containing the plot.
    """
    # Extract loss values - handles XGBoost's nested dictionary structure
    train_loss = list(evals_result['train'].values())[0]
    val_loss = list(evals_result['validation'].values())[0]
    iterations = range(1, len(train_loss) + 1)
    
    # Create simple plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_loss, 'b-', label='Training Loss')
    plt.plot(iterations, val_loss, 'r-', label='Validation Loss')
    
    # Mark best model
    best_iter = val_loss.index(min(val_loss)) + 1
    plt.axvline(x=best_iter, color='gray', linestyle='--')
    plt.scatter(best_iter, min(val_loss), color='red', s=100)
    plt.annotate(f'Best: {min(val_loss):.4f} (iter {best_iter})', 
                 xy=(best_iter, min(val_loss)),
                 xytext=(best_iter + 5, min(val_loss)),
                 arrowprops=dict(arrowstyle='->'))
    
    # Basic formatting
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
    Create a Bailey diagram (Period vs. Amplitude) for variable stars.

    Points are colored by class, with filters applied for probability, confidence, 
    and entropy to show high-quality classifications.

    Args:
        df (pd.DataFrame): The dataframe containing star data (periods, amplitudes, predictions).
        class_column (str): The column name containing the predicted class labels.
        output_dir (str): Directory to save the figure. Defaults to 'class_figures'.
        min_prob (float): Minimum probability threshold for including a point. Defaults to 0.7.
        min_confidence (float): Minimum confidence score threshold. Defaults to 0.9.
        max_entropy (float): Maximum entropy threshold. Defaults to 0.2.
    """
    # Set up figure parameters
    set_plot_style(large_text=True)
    plt.rcParams['legend.fontsize'] = 15
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Apply filters
    if 'true_amplitude' in df.columns:
        df = df[df['true_amplitude'] < 2]
    
    # Add log period if not present
    if 'log_true_period' not in df.columns and 'true_period' in df.columns:
        df['log_true_period'] = np.log10(df['true_period'])
    
    if 'log_true_period' in df.columns:
        df = df[df['log_true_period'] < 2.7]
    
    # Apply confidence thresholds
    prob_col = class_column.replace('predicted_class', 'confidence')
    if prob_col in df.columns:
        df = df[df[prob_col] > min_prob]
    
    # Additional confidence columns if they exist
    if 'xgb_confidence' in df.columns:
        df = df[df['xgb_confidence'] > min_confidence]
    if 'xgb_entropy' in df.columns:
        df = df[df['xgb_entropy'] < max_entropy]
    
    # Sample to avoid overcrowding (using largest confidence to see clear trends)
    sampled_df = df.groupby(class_column).apply(
        lambda x: x.nlargest(n=min(len(x), 10000), columns=prob_col)
    ).reset_index(drop=True)
    
    # Use consistent colors
    unique_types = sorted(sampled_df[class_column].unique())
    color_map = get_consistent_color_map(unique_types)
    
    markers = ['o', 's', '^', 'D', 'v', 'X', 'P', '*', 'h']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    for i, var_type in enumerate(unique_types):
        type_df = sampled_df[sampled_df[class_column] == var_type]
        
        color = color_map[var_type]
        marker = markers[i % len(markers)]
            
        # Plot the data
        plt.scatter(type_df['log_true_period'], type_df['true_amplitude'], 
                   color=color, 
                   marker=marker, 
                   label=var_type, s=7, alpha=0.3)
    
    # Add labels and legend
    leg = plt.legend(bbox_to_anchor=(0.1, 0.8), ncol=2, markerscale=5)
    for handle in leg.legend_handles: handle.set_alpha(1.0)
    ax.set_xlabel(r'log$_{10}$(Period) [days]')
    ax.set_ylabel(r'Amplitude [mag]')
    
    # Set axis limits
    ax.set_xlim(-1, 2.7)
    ax.set_ylim(0, 2)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/bailey_diagram.jpg', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_entropy(df, class_column, output_dir='class_figures', min_prob=0.0, max_points_per_class=2000):
    """
    Generate a scatter plot of Model Confidence vs. Entropy.

    Visualizes the relationship between the model's confidence in its prediction
    and the entropy (uncertainty) of the prediction distribution.

    Args:
        df (pd.DataFrame): Dataframe containing prediction results (confidence, entropy columns).
        class_column (str): The column name for predicted class labels.
        output_dir (str): Directory to save the plot. Defaults to 'class_figures'.
        min_prob (float): Minimum probability threshold for data inclusion. Defaults to 0.0.
        max_points_per_class (int): Maximum number of points to plot per class 
                                    (via random sampling) to prevent overplotting. Defaults to 2000.
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()

    # Determine the confidence column and apply filter based on probability
    prob_col = class_column.replace('predicted_class', 'confidence')
    if prob_col in df.columns:
        df = df[df[prob_col] > min_prob]

    # --- Determine which columns to use for X and Y axes ---
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

    # --- Plotting Setup ---
    # Increased width for the external legend
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use consistent colors
    unique_types = sorted(df[class_column].unique())
    color_map = get_consistent_color_map(unique_types)
    
    markers = ['o', 's', '^', 'D', 'v', 'X', 'P', '*', 'h']

    # --- Create Scatter Plot ---
    for i, var_type in enumerate(unique_types):
        type_df = df[df[class_column] == var_type]
        
        # --- Per-Class Subsampling ---
        # Randomly sample to preserve distribution shape while reducing density
        if len(type_df) > max_points_per_class:
            type_df = type_df.sample(n=max_points_per_class, random_state=42)
        
        color = color_map[var_type]
        marker = markers[i % len(markers)]
        
        ax.scatter(
            type_df[x_col],
            type_df[y_col],
            color=color,
            marker=marker,
            label=f"{var_type} (n={len(type_df)})", # Include sample count in legend
            s=10,       # Increased size for visibility after subsampling
            alpha=0.4   # Increased opacity for visibility after subsampling
        )

    # --- Final Touches ---
    # Legend Outside
    leg = ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0, markerscale=2)
    for handle in leg.legend_handles: handle.set_alpha(1.0)
    
    ax.set_xlabel(f'Confidence ({x_col})')
    ax.set_ylabel(f'Entropy ({y_col})')
    ax.set_title('Confidence vs. Entropy')

    ax.grid(True, linestyle='--', alpha=0.6)

    # Use bbox_inches='tight' to include the external legend
    plt.savefig(f"{output_dir}/classification_confidence_entropy.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_distribution(confidences, predictions, class_names, output_dir="figures"):
    """
    Plot the Kernel Density Estimate (KDE) of model confidence for each class.

    Args:
        confidences (np.array): Array of confidence scores.
        predictions (np.array): Array of integer class predictions.
        class_names (list): List of class names corresponding to the integer predictions.
        output_dir (str): Directory to save the plot. Defaults to "figures".
    """
    # Get consistent colors
    color_map = get_consistent_color_map(class_names)
    
    plt.figure(figsize=(12, 6))
    for i, class_name in enumerate(class_names):
        class_conf = confidences[predictions == i]
        if len(class_conf) > 0:
            sns.kdeplot(class_conf, label=f"{class_name} (n={len(class_conf)})", 
                        clip=(0.0, 1.0), color=color_map[class_name])
    
    plt.xlabel('Model Confidence')
    plt.ylabel('Density')
    plt.title('Distribution of Model Confidence by Predicted Class')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{output_dir}/confidence_distribution.png', dpi=300)
    plt.close()


def plot_xgb_feature_importance(feature_names, importance_values, top_n=20, output_dir='figures'):
    """
    Plot the top N most important features from the XGBoost model.

    Args:
        feature_names (list): List of feature names.
        importance_values (list or np.array): List of importance scores corresponding to feature names.
        top_n (int): Number of top features to display. Defaults to 20.
        output_dir (str): Directory to save the plot. Defaults to 'figures'.
    """
    # Sort features by importance
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
    Calculate metrics and generate a combined ROC/PR visualization.

    Calculates macro-averaged Average Precision (AP) and ROC AUC scores.
    Generates two files:
    1. A bar chart comparing per-class AP and ROC AUC scores.
    2. A combined PDF plot containing ROC curves and Precision-Recall curves.

    Args:
        df (pd.DataFrame): Dataframe containing true labels and predicted probabilities.
        true_label_col (str): Name of the column containing the ground truth labels.
        label_encoder (LabelEncoder): The encoder used to transform class labels.
        output_dir (str): Directory to save the figures. Defaults to 'figures'.
    """
    print("\nCalculating AP and ROC AUC metrics...")
    
    if true_label_col not in df.columns:
        print(f"Warning: True label column '{true_label_col}' not found. Skipping metrics.")
        return

    classes = label_encoder.classes_
    prob_cols = [f'prob_{cls}' for cls in classes]
    
    # Ensure all probability columns exist
    missing_cols = [col for col in prob_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing probability columns, skipping AP/AUC calculation: {missing_cols}")
        return
        
    y_true_raw = df[true_label_col].fillna('UNKNOWN')
    Y_true = label_binarize(y_true_raw, classes=classes)
    Y_proba = df[prob_cols].values
    
    per_class_ap = []
    per_class_roc = []
    class_labels = []
    
    # Storage for curve plotting
    # We use a dict to associate metrics with class names for the legend
    class_metrics_data = {} 
    
    total_ap = 0
    total_roc = 0
    valid_ap_classes = 0
    valid_roc_classes = 0

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
        
        # Check if class has any true samples
        if pos_count > 0:
            try:
                # 1. Calculate Average Precision Score (Scalar)
                ap = average_precision_score(y_true_class, y_proba_class)
                per_class_ap.append(ap)
                total_ap += ap
                valid_ap_classes += 1
                class_metrics_data[class_name]['ap'] = ap
                
                # 2. Calculate Precision-Recall Curve (Vector)
                precision, recall, _ = precision_recall_curve(y_true_class, y_proba_class)
                
                # Calculate No Skill (Prevalence)
                no_skill = pos_count / total_count
                class_metrics_data[class_name]['no_skill'] = no_skill
                class_metrics_data[class_name]['pr_curve'] = (precision, recall)
                
            except ValueError:
                per_class_ap.append(np.nan)

            # 3. Calculate ROC AUC
            if neg_count > 0:
                try:
                    roc = roc_auc_score(y_true_class, y_proba_class)
                    per_class_roc.append(roc)
                    total_roc += roc
                    valid_roc_classes += 1
                    class_metrics_data[class_name]['roc'] = roc
                    
                    # Calculate ROC curve points
                    fpr, tpr, _ = roc_curve(y_true_class, y_proba_class)
                    class_metrics_data[class_name]['roc_curve'] = (fpr, tpr)
                    
                except ValueError:
                    per_class_roc.append(np.nan)
            else:
                per_class_roc.append(np.nan)
                
            class_labels.append(class_name)
        else:
            pass

    # --- Print Macro-Averaged Metrics ---
    macro_ap = total_ap / valid_ap_classes if valid_ap_classes > 0 else 0
    macro_roc_auc = total_roc / valid_roc_classes if valid_roc_classes > 0 else 0

    print("--- Macro-Averaged Metrics ---")
    print(f"Macro-Averaged Precision (mAP) (over {valid_ap_classes} classes): {macro_ap:.4f}")
    print(f"Macro-Averaged ROC AUC         (over {valid_roc_classes} classes): {macro_roc_auc:.4f}")

    # --- Plot Per-Class Bar Chart ---
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
        
        # Use simple colors for bar chart
        ax.barh(y_pos - bar_height / 2, metrics_df['AP'], height=bar_height, label='Avg. Precision (AP)', color='C0')
        ax.barh(y_pos + bar_height / 2, metrics_df['ROC_AUC'], height=bar_height, label='ROC AUC', color='C2')
        
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

    # --- Combined ROC and PR Metric Plot ---
    # Prepare unified figure with two subplots
    # Reduced figsize to make the plot less huge and text relatively larger
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Get consistent colors
    valid_class_names = [cls for cls, data in class_metrics_data.items() if data['count'] > 0]
    color_map = get_consistent_color_map(valid_class_names)
    
    # Subplot 1: ROC Curves
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
            
            # Format the label string
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
        
        # Legend on bottom right for ROC
        # Increased fontsize (removed 'small', set to 10)
        ax_roc.legend(loc="lower right", fontsize=10, ncol=1)
        ax_roc.grid(alpha=0.3)
    else:
        ax_roc.text(0.5, 0.5, "No valid ROC data available", ha='center', va='center')

    # Subplot 2: Precision-Recall Curves
    ax_pr = axes[1]
    has_pr_data = False
    
    for class_name in valid_class_names:
        data = class_metrics_data[class_name]
        if data['pr_curve'] is not None:
            precision, recall = data['pr_curve']
            color = color_map[class_name]
            
            # Plot without label (legend is in ROC plot)
            ax_pr.plot(recall, precision, lw=2, color=color)
            
            has_pr_data = True

    if has_pr_data:
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curves')
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        
        # No legend here (relies on ROC legend)
        ax_pr.grid(alpha=0.3)
    else:
        ax_pr.text(0.5, 0.5, "No valid PR data available", ha='center', va='center')

    # Save as PDF for high quality tiling
    plt.tight_layout()
    pdf_path = f'{output_dir}/metrics_summary.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined metrics plot to {pdf_path}")
