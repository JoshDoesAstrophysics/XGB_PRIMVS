import os
import sys
import argparse
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.timeseries import LombScargle
from astrobase import periodbase
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize_scalar
import traceback

# ==========================================
# Configuration
# ==========================================
# Paths relative to the project directory
INPUT_FITS = "./xgb_predictions.fits"
LIGHT_CURVE_DIR = "./.data/light_curves/"
PLOT_DIR = "./figures/light_curves/"

# Filtering Configuration
TARGET_CLASS = "RRLyr"
CLASS_COL = "xgb_predicted_class"

# Search Range
START_P = 0.1
END_P = 5.0
NHARMS_AOV = 2 # Default harmonics for AOV

# ==========================================
# Functions
# ==========================================

def resolve_path(source_id, base_dir):
    """Resolves file path based on source_id hierarchy."""
    source_str = str(int(source_id))
    if len(source_str) < 6:
        source_str = source_str.zfill(6)
    subdir1 = source_str[:3]
    subdir2 = source_str[3:6]
    return os.path.join(base_dir, subdir1, subdir2, f"{source_str}.csv")

def remove_outliers(times, mags, errs):
    """
    Removes outliers using a percentile cut (1st to 99th percentile).
    """
    if len(mags) < 5: 
        return times, mags, errs
    
    try:
        # IQR-style cut using 0.01 and 0.99 quantiles
        q_low = np.quantile(mags, 0.01)
        q_high = np.quantile(mags, 0.99)
        
        # Keep data within the range [q_low, q_high]
        mask = (mags >= q_low) & (mags <= q_high)
        
        if np.sum(mask) > 5:
            return times[mask], mags[mask], errs[mask]
        return times, mags, errs
    except Exception as e:
        print(f"Warning: Outlier removal failed: {e}")
        return times, mags, errs

def get_amplitude(times, mags, period, nbins=20):
    if period is None or np.isnan(period) or period == 0:
        return 0.0
    phased_times = (times % period) / period
    bins = np.linspace(0, 1, nbins + 1)
    bin_means = []
    for i in range(nbins):
        mask = (phased_times >= bins[i]) & (phased_times < bins[i+1])
        if np.any(mask):
            bin_means.append(np.median(mags[mask]))
    if not bin_means:
        return 0.0
    return (np.max(bin_means) - np.min(bin_means)) / 2.0

def run_methods(times, mags, errs):
    results = {}
    pgram_data = {} # To store periodogram arrays for plotting
    
    # Ensure standard types
    times = np.array(times, dtype=np.float64)
    mags = np.array(mags, dtype=np.float64)
    errs = np.array(errs, dtype=np.float64)

    # --- OPTIMIZATION: Dynamic Step Size ---
    baseline = times.max() - times.min()
    if baseline <= 0:
        print("Error: Time baseline is zero or negative.")
        return {}, {}

    # 1. Frequency Grid for Astropy (GLS)
    f_min = 1.0 / END_P
    f_max = 1.0 / START_P
    oversample = 5 
    df = 1.0 / (oversample * baseline)
    
    freqs = np.arange(f_min, f_max, df)
    if len(freqs) == 0:
        print("Error: Generated frequency grid is empty.")
        return {}, {}
    periods = 1.0 / freqs # Used for GLS peak checking

    # 2. Step Size for Astrobase (PDM, AOV)
    # Using autofreq=True is efficient and safe for 0.1-5.0 range
    
    # Common kwargs
    kwargs = {'verbose': False, 'nworkers': 1, 'nbestpeaks': 3}

    # --- 1. GLS (Astropy) ---
    try:
        ls = LombScargle(times, mags, errs)
        power = ls.power(freqs) # Default normalization
        
        # Store for plotting (Period, Power)
        pgram_data['GLS'] = (periods, power)

        idx_sorted = np.argsort(power)[::-1]
        top_periods = []
        
        count = 0
        for idx in idx_sorted:
            p_cand = periods[idx]
            is_distinct = True
            for existing_p in top_periods:
                if abs(p_cand - existing_p) / existing_p < 0.05:
                    is_distinct = False
                    break
            if is_distinct:
                top_periods.append(p_cand)
                count += 1
            if count >= 3:
                break
        
        while len(top_periods) < 3:
            top_periods.append(np.nan)
            
        results['GLS'] = top_periods
    except Exception:
        print("GLS Error:")
        traceback.print_exc()
        results['GLS'] = [np.nan] * 3
        pgram_data['GLS'] = ([], [])

    # --- 2. PDM (Astrobase) ---
    try:
        # Fixed: Use phasebinsize instead of bins. 
        # 10 bins ~= 0.1 phasebinsize
        pdm = periodbase.spdm.stellingwerf_pdm(
            times, mags, errs, 
            startp=START_P, endp=END_P, 
            autofreq=True,
            phasebinsize=0.1,
            **kwargs
        )

        # Extract Data for Plotting
        pdm_periods = None
        pdm_vals = None
        
        # 1. Get Periods/Frequency
        if 'periods' in pdm:
            pdm_periods = pdm['periods']
        elif 'frequency' in pdm:
            pdm_periods = 1.0 / pdm['frequency']

        # 2. Get Values (Theta)
        if 'lspvals' in pdm:
            pdm_vals = pdm['lspvals']
        elif 'statistic' in pdm:
            pdm_vals = pdm['statistic']
            
        if pdm_periods is not None and pdm_vals is not None:
             # Force numpy array conversion immediately
             pgram_data['PDM'] = (np.array(pdm_periods), np.array(pdm_vals))
        else:
             pgram_data['PDM'] = ([], [])

        # Extract Peaks
        if 'nbestperiods' in pdm:
             results['PDM'] = pdm['nbestperiods'][:3]
        elif 'nbestpeaks' in pdm:
             results['PDM'] = pdm['nbestpeaks'][:3]
        else:
            print(f"PDM Warning: Peak Keys missing.")
            results['PDM'] = [np.nan] * 3
            
        # Ensure length 3
        while len(results['PDM']) < 3:
            results['PDM'].append(np.nan)

    except Exception:
        print("PDM Error:")
        traceback.print_exc()
        results['PDM'] = [np.nan] * 3
        pgram_data['PDM'] = ([], [])

    # --- 3. AOV (Astrobase Multi-Harmonic) ---
    try:
        # SMAV uses harmonic fitting (nharms) instead of bins
        # nharms argument corrected to nharmonics based on user feedback.
        aov = periodbase.smav.aovhm_periodfind(
            times, mags, errs, 
            startp=START_P, endp=END_P, 
            autofreq=True,
            nharmonics=NHARMS_AOV, 
            **kwargs
        )
        
        # Extract Data for Plotting
        aov_periods = None
        aov_vals = None
        
        # 1. Get Periods/Frequency
        if 'periods' in aov:
            aov_periods = aov['periods']
        elif 'frequency' in aov:
            aov_periods = 1.0 / aov['frequency']

        # 2. Get Values
        if 'lspvals' in aov:
            aov_vals = aov['lspvals']
        elif 'statistic' in aov:
            aov_vals = aov['statistic']

        if aov_periods is not None and aov_vals is not None:
             # Force numpy array conversion immediately
             # FIX: Cast to real to avoid ComplexWarning in plots
             pgram_data['AOVMH'] = (np.array(aov_periods), np.real(np.array(aov_vals)))
        else:
             pgram_data['AOVMH'] = ([], [])

        # Extract Peaks
        if 'nbestperiods' in aov:
            results['AOVMH'] = aov['nbestperiods'][:3]
        else:
            print(f"AOV Warning: Peak Keys missing.")
            results['AOVMH'] = [np.nan] * 3

        # Ensure length 3
        while len(results['AOVMH']) < 3:
            results['AOVMH'].append(np.nan)
            
    except Exception:
        print("AOV Error:")
        traceback.print_exc()
        results['AOVMH'] = [np.nan] * 3
        pgram_data['AOVMH'] = ([], [])

    return results, pgram_data

def determine_true_period(results):
    top_periods = []
    # Include all methods in voting if they have results
    for m in ['GLS', 'PDM', 'AOVMH']:
        if m in results and len(results[m]) > 0:
            p = results[m][0]
            if not np.isnan(p) and p > 0: 
                top_periods.append({'period': p, 'method': m})
    
    if not top_periods: return np.nan

    clusters = []
    tolerance = 0.01 
    for item in top_periods:
        p = item['period']; added = False
        for cluster in clusters:
            cluster_vals = [x['period'] for x in cluster]
            if len(cluster_vals) > 0:
                mean_p = np.mean(cluster_vals)
                if abs(p - mean_p) / mean_p < tolerance:
                    cluster.append(item); added = True; break
        if not added: clusters.append([item])
    
    if not clusters: return np.nan
    
    clusters.sort(key=len, reverse=True)
    best_vals = [x['period'] for x in clusters[0]]
    return np.mean(best_vals)

def plot_phased_lc(source_id, times, mags, errs, true_period, true_amp, results, pgram_data):
    # Ensure plot directory exists
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    filename = os.path.join(PLOT_DIR, f"{source_id}_phased.png")
    
    # Updated Grid: 2 Rows. Top for LC+Table, Bottom for Periodograms
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, height_ratios=[1, 1], hspace=0.3)
    
    # --- Row 1: Phased LC and Stats Table ---
    ax_lc = fig.add_subplot(gs[0, :2]) # LC takes 2/3 of top row
    ax_table = fig.add_subplot(gs[0, 2]) # Table takes 1/3 of top row
    
    # 1. Phased Light Curve
    if np.isnan(true_period) or true_period == 0:
        ax_lc.text(0.5, 0.5, "No Valid Period Found", ha='center', va='center')
    else:
        t0 = times[np.argmax(mags)]
        phase = ((times - t0) / true_period) % 1.0
        phase_concat = np.concatenate([phase, phase + 1.0])
        mags_concat = np.concatenate([mags, mags])
        errs_concat = np.concatenate([errs, errs])
        
        ax_lc.errorbar(phase_concat, mags_concat, yerr=errs_concat, fmt='o', color='black', ecolor='gray', markersize=3, alpha=0.6, capsize=0)
        ax_lc.invert_yaxis()
        ax_lc.set_xlabel("Phase")
        ax_lc.set_ylabel("Magnitude (Ks)")
        ax_lc.set_title(f"Source {source_id} | P = {true_period:.6f} d")
        ax_lc.grid(True, linestyle='--', alpha=0.5)

    # 2. Stats Table
    ax_table.axis('off')
    columns = ["Method", "Period (d)", "Amp"]
    table_data = [["TRUE", f"{true_period:.6f}", f"{true_amp:.4f}"]]
    
    method_order = ['PDM', 'GLS', 'AOVMH']
    for m in method_order:
        res_list = results.get(m, [np.nan]*3)
        p = res_list[0]
        p_str = f"{p:.6f}" if not np.isnan(p) else "-"
        
        table_data.append([m, p_str, f"{get_amplitude(times, mags, p):.4f}"])
        
    table = ax_table.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.8)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0: cell.set_text_props(weight='bold'); cell.set_facecolor('#f0f0f0')

    # Highlight winner
    if not np.isnan(true_period) and true_period > 0:
        for i, m in enumerate(method_order):
            res_list = results.get(m, [np.nan]*3)
            p = res_list[0]
            if not np.isnan(p) and p > 0 and abs(p - true_period) / true_period < 0.01:
                 row_idx = i + 2
                 for c in range(len(columns)):
                     if (row_idx, c) in table.get_celld(): table[row_idx, c].set_facecolor('#d9f2d9')

    # --- Row 2: Periodograms ---
    
    # Helper to plot pgram
    def plot_single_pgram(ax, name, color, xlabel=False):
        if name in pgram_data:
            x, y = pgram_data[name]
            
            # Ensure proper array types and shapes
            x = np.array(x).flatten()
            y = np.array(y).flatten()
            
            # Filter NaNs/Infs
            mask = np.isfinite(x) & np.isfinite(y)
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) > 0:
                ax.plot(x_clean, y_clean, color=color, linewidth=1)
                
                # Mark best peak
                res_list = results.get(name, [np.nan])
                best_p = res_list[0]
                if not np.isnan(best_p):
                    ax.axvline(best_p, color='red', linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, "No Valid Data", ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', transform=ax.transAxes)
            
        ax.set_title(name)
        if xlabel: ax.set_xlabel("Period (d)")
        
    # PDM (Lower is better)
    ax_pdm = fig.add_subplot(gs[1, 0])
    plot_single_pgram(ax_pdm, 'PDM', 'blue', xlabel=True)
    if 'PDM' in pgram_data and len(pgram_data['PDM'][1]) > 0:
        ax_pdm.invert_yaxis() # Invert PDM so dip is peaky visually
        ax_pdm.set_ylabel("Theta")

    # GLS (Higher is better)
    ax_gls = fig.add_subplot(gs[1, 1])
    plot_single_pgram(ax_gls, 'GLS', 'green', xlabel=True)
    ax_gls.set_ylabel("Power")

    # AOV (Higher is better)
    ax_aov = fig.add_subplot(gs[1, 2])
    plot_single_pgram(ax_aov, 'AOVMH', 'purple', xlabel=True)
    ax_aov.set_ylabel("Statistic")

    # Replaced plt.tight_layout() to avoid table conflict
    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.95, hspace=0.3, wspace=0.3)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def load_data(source_id):
    csv_path = resolve_path(source_id, LIGHT_CURVE_DIR)
    if not os.path.exists(csv_path): return None, None, None
    try:
        df = pd.read_csv(csv_path)
        required = ['mjd', 'ks_mag', 'ks_err']
        if not all(col in df.columns for col in required): return None, None, None
        
        # --- NEW: Filter based on ast_res_chisq ---
        if 'ast_res_chisq' in df.columns:
            df = df[df['ast_res_chisq'] <= 10]
            
        df = df[required].dropna()
        if len(df) < 10: return None, None, None
        
        times, mags, errs = df['mjd'].values, df['ks_mag'].values, df['ks_err'].values
        return remove_outliers(times, mags, errs)
    except Exception as e: 
        print(f"Error loading {source_id}: {e}")
        return None, None, None

def list_ids():
    if not os.path.exists(INPUT_FITS):
        print(f"Error: {INPUT_FITS} not found", file=sys.stderr)
        sys.exit(1)
    try:
        t = Table.read(INPUT_FITS)
        
        # Filter by class if column exists
        if CLASS_COL in t.colnames:
            t = t[t[CLASS_COL] == TARGET_CLASS]

        col = 'source_id' if 'source_id' in t.colnames else 'sourceid'
        for sid in t[col]:
            print(sid)
    except Exception as e:
        print(f"Error reading fits: {e}", file=sys.stderr)
        sys.exit(1)

def process_single(source_id, output_dir=None, make_plot=False):
    times, mags, errs = load_data(source_id)
    if times is None: 
        print(f"Data load failed for {source_id}")
        return

    results, pgram_data = run_methods(times, mags, errs)
    
    # 1. Determine base period from methods
    base_period = determine_true_period(results)
    
    # 2. Use base period directly (String Length/Alias check removed)
    true_period = base_period
    
    true_amp = get_amplitude(times, mags, true_period)

    row = {
        'source_id': source_id, 
        'calc_period': true_period, 
        'calc_amplitude': true_amp
    }
    
    methods = ['PDM', 'GLS', 'AOVMH']
    for m in methods:
        res_list = results.get(m, [np.nan]*3)
        while len(res_list) < 3: res_list.append(np.nan)
        for i in range(3):
            p = res_list[i]
            row[f'{m}_period_{i+1}'] = p
            row[f'{m}_amp_{i+1}'] = get_amplitude(times, mags, p)

    if output_dir:
        out_file = os.path.join(output_dir, f"{source_id}.csv")
        pd.DataFrame([row]).to_csv(out_file, index=False)
    else:
        print(f"True Period: {true_period}")
            
        for m in methods:
            if m in results and len(results[m]) > 0:
                p = results[m][0]
                print(f"  {m}: {p:.6f} d")

    if make_plot:
        plot_phased_lc(source_id, times, mags, errs, true_period, true_amp, results, pgram_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, help="Source ID to process")
    parser.add_argument("--list-ids", action="store_true", help="Print all source IDs from FITS and exit")
    parser.add_argument("--output-dir", type=str, help="Directory to save CSV result", default=None)
    parser.add_argument("--plot", action="store_true", help="Generate phased plot")
    args = parser.parse_args()

    if args.list_ids:
        list_ids()
    elif args.id:
        process_single(args.id, args.output_dir, args.plot)
