import sys
import os
import re
import argparse
from datetime import datetime

# Try to import tqdm for progress bars, handle it if missing
try:
    from tqdm import tqdm
except ImportError:
    # Fallback dummy iterator if tqdm is missing
    def tqdm(iterable, **kwargs):
        return iterable

# -----------------------------------------------------------------------------
# GPU Configuration & TFLOPS Estimates (FP32)
# -----------------------------------------------------------------------------
GPU_SPECS = {
    "H100": 67.0,   # FP32 TFLOPS (approx, SXM5)
    "P100": 9.3,    # FP32 TFLOPS (PCIe)
    "A30": 10.3,    # FP32 TFLOPS
    "L40S": 91.6,   # FP32 TFLOPS
    "A6000": 38.7,  # FP32 TFLOPS
}

def get_gpu_model_from_host(hostname):
    """
    Identifies GPU model based on hostname patterns.
    Returns the key in GPU_SPECS or None if unknown.
    """
    h = hostname.lower().strip()
    
    # H100: mbh100...
    if h.startswith("mbh100"):
        return "H100"
    
    # L40S: mbl40s...
    if h.startswith("mbl40s"):
        return "L40S"
    
    # A30: mba30... or b525-b531
    if h.startswith("mba30"):
        return "A30"
    if h.startswith("b5"):
        # Check range b525 to b531
        match = re.match(r"^b5(\d{2})", h)
        if match:
            num = int(match.group(1))
            if 25 <= num <= 31:
                return "A30"
                
    # P100: t527... or t528...
    if h.startswith("t527") or h.startswith("t528"):
        return "P100"
        
    # A6000: contains 'a6000'
    if "a6000" in h:
        return "A6000"
        
    return None

def get_file_gpu_tflops(eff_filepath, debug=False):
    """
    Finds ANY .out file in the same directory as the .eff file
    and scans it for the GPU model.
    """
    eff_dir = os.path.dirname(eff_filepath)
    if not eff_dir:
        eff_dir = "."
        
    out_filepath = None
    
    # Search for the first .out file in the directory
    try:
        if os.path.exists(eff_dir):
            for fname in os.listdir(eff_dir):
                if fname.endswith(".out"):
                    out_filepath = os.path.join(eff_dir, fname)
                    break 
    except Exception as e:
        if debug:
            print(f"[DEBUG] Error listing dir {eff_dir}: {e}")
        return 0.0, None

    if not out_filepath:
        if debug:
            print(f"[DEBUG] No .out file found in directory: {eff_dir}")
        return 0.0, None

    try:
        with open(out_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            # Scan first 2000 lines
            for i, line in enumerate(f):
                # Case insensitive check for "Running on:"
                if "running on:" in line.lower():
                    # Split by the colon, take the second part
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        hostname = parts[1].strip()
                        model = get_gpu_model_from_host(hostname)
                        
                        if model:
                            if debug:
                                print(f"[DEBUG] Found host '{hostname}' in {os.path.basename(out_filepath)} -> Detected {model}")
                            return GPU_SPECS[model], model
                        else:
                            if debug:
                                print(f"[DEBUG] Found host '{hostname}' in {os.path.basename(out_filepath)} but NO GPU MATCH found.")
                    return 0.0, None
                
                if i > 2000: 
                    break
                    
        if debug:
            print(f"[DEBUG] Scanned {os.path.basename(out_filepath)} but did not find 'Running on:' line.")
            
    except Exception as e:
        if debug:
            print(f"[DEBUG] Error reading {out_filepath}: {e}")
        pass
        
    return 0.0, None

# -----------------------------------------------------------------------------
# Parsing Logic
# -----------------------------------------------------------------------------

def parse_timestamp_fast(ts_str):
    """
    Manually parses 'YYYY/MM/DD HH:MM:SS.fff' to avoid strptime overhead.
    """
    try:
        # Indices: 01234567890123456789012
        # Value:   2025/12/18 12:01:03.944
        return datetime(
            int(ts_str[0:4]),   # Year
            int(ts_str[5:7]),   # Month
            int(ts_str[8:10]),  # Day
            int(ts_str[11:13]), # Hour
            int(ts_str[14:16]), # Minute
            int(ts_str[17:19]), # Second
            int(ts_str[20:]) * 1000 # Milliseconds to Microseconds
        )
    except (ValueError, IndexError):
        try:
             return datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S.%f")
        except ValueError:
             return datetime.now() # Fallback safe

def calculate_file_metrics(filepath):
    """
    Parses a single .eff file to calculate total GPU hours.
    Returns tuple: (wall_seconds, weighted_gpu_seconds, row_count)
    """
    total_weighted_seconds = 0.0
    total_wall_seconds = 0.0
    row_count = 0
    
    last_time = None
    last_util = 0.0
    
    try:
        with open(filepath, 'r') as f:
            header_line = f.readline()
            if not header_line:
                return 0.0, 0.0, 0
            
            headers = [h.strip().lower() for h in header_line.split(',')]
            
            gpu_idx = -1
            timestamp_idx = -1
            
            for i, col in enumerate(headers):
                if 'utilization.gpu' in col:
                    gpu_idx = i
                if 'timestamp' in col:
                    timestamp_idx = i
            
            if gpu_idx == -1 or timestamp_idx == -1:
                return 0.0, 0.0, 0

            for line in f:
                parts = line.split(',')
                if len(parts) <= gpu_idx: 
                    continue
                
                try:
                    ts_str = parts[timestamp_idx].strip()
                    curr_time = parse_timestamp_fast(ts_str)
                    
                    util_part = parts[gpu_idx]
                    if util_part.strip().endswith('%'):
                        util_str = util_part.replace('%', '')
                    else:
                        util_str = util_part
                    
                    curr_util = float(util_str)
                    
                    if last_time is not None:
                        dt = (curr_time - last_time).total_seconds()
                        if dt > 0:
                            avg_util_percent = (last_util + curr_util) * 0.5
                            weighted_seconds = (avg_util_percent * 0.01) * dt
                            total_weighted_seconds += weighted_seconds
                            total_wall_seconds += dt
                    
                    last_time = curr_time
                    last_util = curr_util
                    row_count += 1
                    
                except (ValueError, IndexError):
                    continue

    except Exception as e:
        print(f"[Error] Failed to read {filepath}: {e}")
        return 0.0, 0.0, 0

    return total_wall_seconds, total_weighted_seconds, row_count

def print_summary(name, wall_sec, gpu_sec, rows, estimated_ops=None, is_total=False):
    """Prints a formatted summary block."""
    gpu_hours = gpu_sec / 3600.0
    wall_hours = wall_sec / 3600.0
    avg_eff = (gpu_hours / wall_hours * 100) if wall_hours > 0 else 0.0
    
    if not is_total:
        print("-" * 40)
        print(f"File: {name}")

    print(f"Rows processed:       {rows}")
    print(f"Total Wall Time:      {wall_hours:.4f} hours")
    print(f"Weighted GPU Time:    {gpu_hours:.4f} GPU-hours")
    print(f"Average Efficiency:   {avg_eff:.2f}%")
    
    if estimated_ops is not None:
        print(f"Est. Operations:      {estimated_ops:,.2f} Trillion (FP32)")

    if not is_total:
        print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Calculate GPU hours and estimated operations from .eff files.")
    parser.add_argument("path", help="File or directory path to scan for .eff files")
    parser.add_argument("--debug", action="store_true", help="Print debug info for file matching and GPU detection")

    args = parser.parse_args()
    target_path = args.path
    
    grand_total_wall = 0.0
    grand_total_gpu = 0.0
    grand_total_rows = 0
    grand_total_ops = 0.0 
    files_processed = 0

    # Collect files
    eff_files = []
    if os.path.isfile(target_path):
        eff_files.append(target_path)
    elif os.path.isdir(target_path):
        print(f"Scanning directory: {target_path} ...")
        for root, dirs, files in os.walk(target_path):
            for filename in files:
                if filename.endswith(".eff"):
                    eff_files.append(os.path.join(root, filename))
    else:
        print(f"Error: '{target_path}' is not a valid file or directory.")
        sys.exit(1)

    if not eff_files:
        print("No .eff files found.")
        sys.exit(0)

    print(f"Found {len(eff_files)} files. Processing...")

    # Process
    if args.debug:
        print(f"Debug Mode: ON")
        iterator = eff_files # Don't use tqdm in debug mode so prints are clean
    else:
        iterator = tqdm(eff_files, unit="file", ncols=80)
    
    for full_path in iterator:
        w, g, r = calculate_file_metrics(full_path)
        
        # Calculate Operations (Default behavior now)
        file_ops = None
        if r > 0:
            tflops, model = get_file_gpu_tflops(full_path, debug=args.debug)
            if tflops > 0:
                # Weighted GPU Seconds * TFLOPS = Trillions of Operations
                file_ops = g * tflops
                grand_total_ops += file_ops
            
        if r > 0:
            grand_total_wall += w
            grand_total_gpu += g
            grand_total_rows += r
            files_processed += 1
            
            # For single file, print immediately
            if len(eff_files) == 1:
                print_summary(full_path, w, g, r, estimated_ops=file_ops)

    if files_processed > 0:
        print_summary(f"{files_processed} files", grand_total_wall, grand_total_gpu, grand_total_rows, estimated_ops=grand_total_ops, is_total=True)

if __name__ == "__main__":
    main()
