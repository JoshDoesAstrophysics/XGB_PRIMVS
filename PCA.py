#!/usr/bin/env python3
"""
Principal Component Analysis (PCA) Script for Astronomical Data.

This script performs dimensionality reduction on astronomical feature data using PCA.
It is designed to help explore the structure of the data, identify clusters, and 
understand which physical parameters drive the variability in the dataset.

Intended Audience:
    This script is designed for researchers and students. Comments explain *why* data is transformed (e.g., log-scaling periods, handling skewness) and how 
    PCA works in this context.

Workflow Overview:
    1.  **Data Loading**: Reads data from FITS or CSV files.
    2.  **Preprocessing**: Crucial for PCA.
        * **Clipping**: Removes extreme outliers that would dominate the variance.
        * **Transformation**: Applies log or root transforms to make feature distributions 
            more Gaussian (bell-curve shaped), which PCA prefers.
        * **De-correlation**: Removes highly correlated (redundant) features.
        * **Scaling**: Standardizes features so they have the same scale (PCA is very 
            sensitive to magnitudes; without this, a feature ranging 0-10000 would 
            overpower a feature ranging 0-1).
    3.  **PCA Application**: Computes the Principal Components.
    4.  **Visualization**: Generates plots to visualize the data in the new PC space.
    5.  **Output**: Saves the transformed data and the "loadings" (which explain 
        what each PC actually represents physically).

Usage:
    Run this script from the command line:
    python PCA.py <n_components>

    Example:
    python PCA.py 12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA as PCA
import seaborn as sns
from astropy.table import Table
from astropy.io import fits
import os
from matplotlib.colors import LogNorm

# Import visualization functions from local module 'vis'
# These generate the diagnostic plots (scree plots, scatter plots, etc.)
from vis import visualize_pca, plot_pca_degeneracy_analysis

def load_data(filepath):
    """
    Loads data from a FITS table or CSV file into a Pandas DataFrame.

    Args:
        filepath (str): Path to the input file (.fits or .csv).

    Returns:
        pd.DataFrame: A DataFrame containing the raw data.
    """
    print(f"Loading data from {filepath}...")
    if filepath.endswith('.fits'):
        with fits.open(filepath) as hdul:
            # FITS data usually resides in extension 1
            df = pd.DataFrame(hdul[1].data)
    else:
        df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records with {len(df.columns)} features")
    return df

def prepare_data(df, features, scale_method='robust'):
    """
    Cleans and transforms data specifically for PCA.

    PCA is a linear algorithm that assumes data is roughly normally distributed (Gaussian)
    and sensitive to outliers. This function forces the data to adhere to these assumptions
    as closely as possible.

    Steps taken:
    1.  **Selection**: Keeps only the requested feature columns.
    2.  **Clipping**: Limits values to the 1st and 99th percentiles to suppress outliers.
    3.  **Transformation**: Applies math functions (log, sqrt) to fix skewed distributions.
    4.  **Redundancy Removal**: Drops features that are >85% correlated with others.
    5.  **Scaling**: Centers and scales the data (RobustScaler recommended).

    Args:
        df (pd.DataFrame): The raw input dataframe.
        features (list): List of column names to use.
        scale_method (str): 'robust' (uses median/IQR) or 'standard' (uses mean/std).

    Returns:
        pd.DataFrame: The processed, scaled dataframe ready for PCA.
    """
    # Select only desired columns
    features = [f for f in features if f in df.columns]
    print(f"Using {len(features)} features: {features}")
    df = df[features].copy()

    # --- Step 1: Outlier Clipping ---
    # PCA tries to maximize variance. Extreme outliers have huge variance, so PCA will 
    # obsess over them if we don't clip them. We cap values at the 1st and 99th percentiles.
    print("\nApplying percentile clipping...")
    for f in df.columns:
        # Skip embedding columns (usually named '0', '1', etc.) if present
        if f.isdigit(): continue
        
        p01, p99 = np.nanpercentile(df[f], 1), np.nanpercentile(df[f], 99)
        df[f] = df[f].clip(p01, p99)

    # --- Step 2: Distribution Transformation ---
    # Many astronomical variables (Period, Amplitude) span orders of magnitude or are 
    # heavily skewed. We apply transformations to make them more symmetric.
    print("\nTransforming features...")
    for f in df.columns:
        x = df[f].values
        if f.isdigit(): continue
        
        # Calculate skewness to decide on transformation
        sk = pd.Series(x).skew()
        rng = np.nanmax(x) / (np.nanmin(x) + 1e-10)

        # Domain-specific transformations
        if f == 'true_period' or f == 'true_amplitude':
            # Periods and Amplitudes follow a power law; log-scale is standard in astronomy.
            # We handle <= 0 values by clamping them to a small positive number.
            x[x <= 0] = max(np.nanmin(x[x > 0]), 1e-10) / 10
            x = np.log10(x)
            print(f"  log10({f})")
            
        elif f == 'best_fap':
            # False Alarm Probability is usually very small (e.g., 1e-5). 
            # -log10 makes this a manageable positive number (e.g., 5).
            x[x <= 0] = max(np.nanmin(x[x > 0]), 1e-10) / 10
            x = -np.log10(x)
            print(f"  -log10({f})")
            
        # Statistical transformations based on skewness
        elif abs(sk) > 2:
            # Highly skewed: Log1p (log(1+x)) suppresses large tails effectively.
            offset = abs(np.nanmin(x)) + 1e-10 if np.nanmin(x) <= 0 else 0
            x = np.log1p(x + offset)
            print(f"  log1p({f})")
            
        elif abs(sk) > 1:
            # Moderately skewed: Cube root or Square root handles negative values while compacting the range.
            x = np.cbrt(x) if np.nanmin(x) < 0 else np.sqrt(np.abs(x)) * np.sign(x)
            print(f"  root transform ({f})")
            
        elif rng > 1000:
            # Huge dynamic range: Log transform compresses the range.
            offset = abs(np.nanmin(x)) + 1e-10 if np.nanmin(x) <= 0 else 0
            x = np.log1p(x + offset)
            print(f"  log1p(range fix: {f})")
            
        df[f] = x

    # --- Step 3: Redundancy Removal (Correlation Check) ---
    # If two features are 99% correlated, PCA will just split the weight between them, 
    # which adds noise and makes interpretation harder. We drop one of them.
    print("\nChecking correlations...")
    keep = df.columns[~df.columns.str.isdigit()]
    corr = df[keep].corr().abs()
    
    # Create upper triangle matrix of correlation to avoid duplicates
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = []
    
    for col in upper.columns:
        # Find features correlated > 0.85 with this column
        hits = upper[col][upper[col] > 0.85].index.tolist()
        if hits:
            # If correlated, keep the one with higher variance (more signal)
            all_feats = [col] + hits
            var = {k: df[k].var() for k in all_feats}
            top = max(var, key=var.get)
            to_drop += [k for k in all_feats if k != top and k not in to_drop]
            
    df.drop(columns=to_drop, inplace=True)
    if to_drop:
        print(f"  Dropped due to high correlation: {to_drop}")

    # --- Step 4: Final Scaling ---
    # RobustScaler is preferred here because despite our clipping, some outliers might persist.
    # It scales using the Median and Interquartile Range rather than Mean and Variance.
    scaler = RobustScaler(quantile_range=(5, 95)) if scale_method == 'robust' else StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    # Fill any remaining NaNs (created by transforms) with 0 (the new median/mean)
    df.fillna(0, inplace=True)
    print(f"\nFinal feature count: {len(df.columns)}")
    return df

def apply_pca(data, n_components=2):
    """
    Fits the PCA model to the data.

    Args:
        data (pd.DataFrame): The preprocessed, scaled data.
        n_components (int): The number of Principal Components to calculate.

    Returns:
        tuple: (pca_df, pca)
            - pca_df: DataFrame containing the data projected onto the new PC axes.
            - pca: The fitted sklearn PCA object (contains loadings, explained variance, etc.).
    """
    # random_state ensures the results are reproducible every time we run the script
    pca = PCA(n_components=n_components, random_state=37)
    comps = pca.fit_transform(data)
    
    # Create a DataFrame for the results
    df = pd.DataFrame(comps, columns=[f'PC{i+1}' for i in range(n_components)])
    
    print(f"Explained variance ratio (first 2 components): {pca.explained_variance_ratio_[:2]}")
    return df, pca

def save_pca_data(pca_df, original_df, pca_model, output_file='pca_results.fits'):
    """
    Saves the PCA results to files.

    1. **FITS File**: Contains the original data PLUS the new PC columns.
       Useful for plotting color-magnitude diagrams colored by PC value.
    2. **CSV File**: Contains the "Loadings".
       This tells you which physical features (Period, Amplitude, etc.) make up
       each Principal Component. Essential for interpreting what "PC1" actually means.

    Args:
        pca_df (pd.DataFrame): The PCA coordinates.
        original_df (pd.DataFrame): The original input data.
        pca_model (PCA): The trained PCA model object.
        output_file (str): Path for the output FITS file.
    """
    # Merge original data with PCA coordinates
    df_out = original_df.copy()
    for col in pca_df.columns:
        df_out[col] = pca_df[col]
        
    # Save as FITS
    astro_table = Table.from_pandas(df_out)
    astro_table.write(output_file, format='fits', overwrite=True)

    # Save Loadings (Eigenvectors) to CSV
    # Transpose so rows are features, columns are PCs
    load = pd.DataFrame(
        pca_model.components_.T, 
        columns=[f'PC{i+1}' for i in range(pca_model.n_components_)], 
        index=pca_model.feature_names_in_
    )
    load.to_csv(output_file.replace('.fits', '_loadings.csv'))

def main(n_components=12):
    """
    Main execution workflow.
    
    1. Defines the list of features to use.
    2. Loads the dataset.
    3. Runs the complex preprocessing pipeline.
    4. Computes PCA.
    5. Generates visualizations.
    6. Saves results.
    """
    sample_size = None # Set to an integer (e.g., 10000) to test on a small subset quickly

    input_file = "./.data/PRIMVS_P.fits"
    output_dir = f"./pca/"
    output_file = f"{output_dir}pca_results.fits"

    # Curated feature set for variable star classification.
    # We comment out features that might not be useful or present, but keep them listed
    # to show the full potential set.
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

    # --- 1. Load Data ---
    df = load_data(input_file)
    
    # Optional sampling for faster debugging
    if sample_size and sample_size < len(df):
        print(f"Subsampling to {sample_size} random records...")
        df = df.sample(sample_size, random_state=37)

    # --- 2. Preprocess Data ---
    # This cleans, transforms, de-correlates, and scales the features.
    df_scaled = prepare_data(df, features)
    
    # --- 3. Apply PCA ---
    pca_df, pca_model = apply_pca(df_scaled, n_components=n_components)
    
    # --- 4. Visualization ---
    # These functions (from vis.py) generate plots to help interpret the PCs.
    visualize_pca(pca_df, df, pca_model, output_dir=output_dir)
    plot_pca_degeneracy_analysis(pca_df, df, pca_model, features, output_dir=output_dir)
    
    # --- 5. Save Results ---
    import joblib
    # Save the full state for later use in other scripts
    joblib.dump(
        {"pca_df": pca_df, "df": df, "pca_model": pca_model, "features": features}, 
        f"{output_dir}pca_dump.joblib"
    )
    
    save_pca_data(pca_df, df, pca_model, output_file)
    print("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run PCA on astronomical data.")
    parser.add_argument("n_components", type=int, help="number of PCA components to fit")
    args = parser.parse_args()
    
    main(args.n_components)
