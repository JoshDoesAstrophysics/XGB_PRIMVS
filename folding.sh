#!/bin/bash

#=======================================================================
# SLURM N-FOLD PARALLEL SUBMISSION SCRIPT
#
# This script launches N parallel jobs for a single set of 
# hyperparameters, partitioned by fold.
#
# It automatically reads the 'set_test_size' value from 
# your folding_xgb.py to calculate folds.
#=======================================================================

# --- Configuration ---

PYTHON_SCRIPT_NAME="./folding_XGB.py"
VIS_SCRIPT_NAME="./vis.py"

# --- Manual Data Path Declaration ---
# Update these relative paths as needed
TRAIN_DATA_PATH=".data/PRIMVS_P_training_new.fits"
TEST_DATA_PATH=".data/PRIMVS_P.fits"

# Use relative path for the output directory
GRID_RUNS_DIR="./n_folds"

# SLURM Parameters
SBATCH_PARTITION="mb-h100"
SBATCH_ACCOUNT="gr-vvv"
SBATCH_GRES="gpu:1"
SBATCH_MEM="96G"
SBATCH_TIME="1-00:00:00"
MODULES_TO_LOAD="gcc/14.2.0 python/3.12.0"

#=======================================================================
# LOGIC
#=======================================================================

# --- Automatically detect TEST_SIZE value from the python script ---
if [ -f "$PYTHON_SCRIPT_NAME" ]; then
    # Extract set_test_size
    TEST_SIZE=$(grep "set_test_size =" "$PYTHON_SCRIPT_NAME" | head -n 1 | sed -E 's/.*= *([0-9.]+).*/\1/')
    echo "Detected TEST_SIZE=$TEST_SIZE from $PYTHON_SCRIPT_NAME"
else
    echo "Error: $PYTHON_SCRIPT_NAME not found."
    exit 1
fi

if [[ -z "$TEST_SIZE" ]]; then
    echo "Error: Could not parse TEST_SIZE from $PYTHON_SCRIPT_NAME"
    exit 1
fi

mkdir -p "$GRID_RUNS_DIR"

# Resolve absolute paths for the scripts and data
# This ensures that when the job runs in a subdirectory, it still knows 
# where the source files and datasets are located.
ABS_PYTHON_SCRIPT_PATH=$(realpath "$PYTHON_SCRIPT_NAME")
ABS_VIS_SCRIPT_PATH=$(realpath "$VIS_SCRIPT_NAME")
ABS_TRAIN_PATH=$(realpath "$TRAIN_DATA_PATH")
ABS_TEST_PATH=$(realpath "$TEST_DATA_PATH")

# Calculate N folds using the detected TEST_SIZE
N_FOLDS=$(python3 -c "print(int(round(1/$TEST_SIZE)))")

echo "Preparing to launch $N_FOLDS parallel folds..."

for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    
    # Create a unique directory for this fold
    RUN_DIR="${GRID_RUNS_DIR}/fold_${FOLD}"
    mkdir -p "$RUN_DIR"
    
    # Local copies of scripts to ensure job isolation
    RUN_SCRIPT_PATH="${RUN_DIR}/folding_xgb_run.py"
    RUN_OUTPUT_FITS="xgb_preds_fold_${FOLD}.fits"
    
    cp "$ABS_PYTHON_SCRIPT_PATH" "$RUN_SCRIPT_PATH"
    cp "$ABS_VIS_SCRIPT_PATH" "${RUN_DIR}/vis.py"
    
    # Generate the SLURM submission script for this specific fold
    ABS_RUN_DIR=$(realpath "$RUN_DIR")

    cat << EOF > "${RUN_DIR}/submit_fold.sh"
#!/bin/bash
#SBATCH --job-name=XGB_fold${FOLD}
#SBATCH --output=fold_${FOLD}.out
#SBATCH --error=fold_${FOLD}.err
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --gres=${SBATCH_GRES}
#SBATCH --mem=${SBATCH_MEM}
#SBATCH --time=${SBATCH_TIME}
#SBATCH --chdir=${ABS_RUN_DIR}

# Print info for debugging
echo "Job ID: \$SLURM_JOB_ID"
echo "Run ID: ${RUN_ID}"
echo "Running on: \$(hostname)"
echo "Starting at: \$(date)"
echo "Working Directory: \$(pwd)"

# Load environment
module load ${MODULES_TO_LOAD}

echo "Starting Fold ${FOLD} at \$(date)"

# Start nvidia-smi in the background to log GPU usage
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1 > XGB_gpu_\${SLURM_JOB_ID}.eff &
NVSMI_PID=\$!

# Run the script using the absolute paths resolved from the manual config
python -u folding_xgb_run.py "${ABS_TRAIN_PATH}" "${ABS_TEST_PATH}" "${RUN_OUTPUT_FITS}" ${FOLD}
PY_EXIT_CODE=\$?

# Stop the nvidia-smi logging
kill \$NVSMI_PID

echo "Fold ${FOLD} finished at \$(date)"
exit \$PY_EXIT_CODE
EOF

    # Submit the job
    sbatch "${RUN_DIR}/submit_fold.sh"
    
done

echo "---"
echo "Successfully submitted $N_FOLDS jobs."
echo "Check progress with 'squeue -u $USER'"
echo "Outputs will be found in: ${GRID_RUNS_DIR}/"
