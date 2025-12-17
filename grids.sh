#!/bin/bash

#=======================================================================
# SLURM HYPERPARAMETER GRID SEARCH SCRIPT
#
# This script launches a separate SLURM job for each combination
# of hyperparameters defined in the "Grid Search Configuration" section.
#
# It creates a unique directory for each run, copies the base Python
# script, uses 'sed' to modify the hyperparameters, and then
# submits that specific job.
#=======================================================================

# --- Grid Search Configuration ---

# Define the hyperparameter arrays to test
# Add or remove values from these lists as needed
LEARNING_RATES=(1E-3)
MAX_DEPTHS=(20 50 80)
SUBSAMPLES=(0.8 0.9 0.95)
COLSAMPLES_BT=(1E-1 1E0 1E1)
REG_ALPHAS=(1E-1 1E0 1E1)
REG_LAMBDAS=(1E-9 1E-7 1E-5)
NUM_BOOST_ROUNDS=(1000000)
EARLY_STOPPING_ROUNDS=(5000)

# --- Path Configuration ---

# Set the path to your base Python script
PYTHON_SCRIPT_NAME="XGB.py"
VIS_SCRIPT_NAME="vis.py"

# Set the paths to your training and test data
# These paths must be accessible from the compute nodes
TRAIN_DATA_PATH="../.data/PRIMVS_P_training_new.fits"
TEST_DATA_PATH="../.data/PRIMVS_P.fits"

# Base directory for all project code
# Ensure this ends with a slash / if you want to simply concatenate variables
PROJECT_DIR="/project/gr-vvv/jwanning/UWyo_PRIMVS/PRIMVS_ORDO/"

# Directory to store all run outputs
# This will be created inside your PROJECT_DIR
GRID_RUNS_DIR="grid_search_runs"

# --- SLURM Configuration ---

# These will be used as a template for each submitted job
# You can override these in the job-specific script if needed
SBATCH_PARTITION="mb-h100,mb-l40s,mb-a6000,mb-a30,teton-gpu,non-investor,beartooth-gpu"
SBATCH_ACCOUNT="gr-vvv"
SBATCH_GRES="gpu:1"
SBATCH_MEM="96G"
SBATCH_TIME="1-00:00:00"
SBATCH_CPUS="1"

# --- Modules ---
MODULES_TO_LOAD="gcc/14.2.0 python/3.12.0"

#=======================================================================
# SCRIPT LOGIC - Do not edit below this line unless you are sure
#=======================================================================

echo "Starting Grid Search..."
echo "Project Directory: $PROJECT_DIR"
echo "Base Python Script: $PYTHON_SCRIPT_NAME"

# Ensure we are in the project directory
cd "$PROJECT_DIR" || { echo "Failed to cd into $PROJECT_DIR. Exiting."; exit 1; }

# Create the main directory for all grid search runs
mkdir -p "$GRID_RUNS_DIR"

# Get the absolute path for the data files
ABS_TRAIN_PATH=$(realpath "$TRAIN_DATA_PATH")
ABS_TEST_PATH=$(realpath "$TEST_DATA_PATH")
ABS_PYTHON_SCRIPT_PATH=$(realpath "$PYTHON_SCRIPT_NAME")
ABS_VIS_SCRIPT_PATH=$(realpath "$VIS_SCRIPT_NAME")

if [ ! -f "$ABS_PYTHON_SCRIPT_PATH" ]; then
    echo "Error: Base Python script not found at $ABS_PYTHON_SCRIPT_PATH"
    exit 1
fi
if [ ! -f "$ABS_VIS_SCRIPT_PATH" ]; then
    echo "Error: Visualization script not found at $ABS_VIS_SCRIPT_PATH"
    exit 1
fi
if [ ! -f "$ABS_TRAIN_PATH" ]; then
    echo "Error: Training data not found at $ABS_TRAIN_PATH"
    exit 1
fi
if [ ! -f "$ABS_TEST_PATH" ]; then
    echo "Error: Test data not found at $ABS_TEST_PATH"
    exit 1
fi

TOTAL_JOBS=0

# --- Nested loops for grid search ---
for LR in "${LEARNING_RATES[@]}"; do
for MD in "${MAX_DEPTHS[@]}"; do
for SS in "${SUBSAMPLES[@]}"; do
for CS in "${COLSAMPLES_BT[@]}"; do
for RA in "${REG_ALPHAS[@]}"; do
for RL in "${REG_LAMBDAS[@]}"; do
for NBR in "${NUM_BOOST_ROUNDS[@]}"; do
for ESR in "${EARLY_STOPPING_ROUNDS[@]}"; do

    # Create a unique identifier for this run
    RUN_ID="lr_${LR}_md_${MD}_ss_${SS}_cs_${CS}_ra_${RA}_rl_${RL}_nbr_${NBR}_esr_${ESR}"
    
    # *** FIX: Use Absolute Path for RUN_DIR ***
    # We combine PROJECT_DIR and GRID_RUNS_DIR to ensure the path is absolute.
    # This prevents SLURM from nesting the output directories.
    RUN_DIR="${PROJECT_DIR}/${GRID_RUNS_DIR}/${RUN_ID}"
    
    # Remove double slashes if PROJECT_DIR ended with one
    RUN_DIR=$(echo "$RUN_DIR" | sed 's|//|/|g')
    
    echo "---"
    echo "Preparing Job: $RUN_ID"
    
    # Create the run's specific directory
    mkdir -p "$RUN_DIR"
    
    # Define file paths for this specific run
    RUN_SCRIPT_PATH="${RUN_DIR}/XGB_run.py"
    RUN_SLURM_PATH="${RUN_DIR}/submit_job.sh"
    RUN_OUTPUT_FITS="${RUN_DIR}/xgb_preds_${RUN_ID}.fits"
    
    # 1. Create the temporary, modified Python script for this run
    cp "$ABS_PYTHON_SCRIPT_PATH" "$RUN_SCRIPT_PATH"
    cp "$ABS_VIS_SCRIPT_PATH" "${RUN_DIR}/vis.py"
    
    # Use robust sed commands to replace default values
    sed -i -E "s/^(\s*set_learning_rate\s*=\s*).*/\1${LR}/" "$RUN_SCRIPT_PATH"
    sed -i -E "s/^(\s*set_max_depth\s*=\s*).*/\1${MD}/" "$RUN_SCRIPT_PATH"
    sed -i -E "s/^(\s*set_subsample\s*=\s*).*/\1${SS}/" "$RUN_SCRIPT_PATH"
    sed -i -E "s/^(\s*set_colsample_bytree\s*=\s*).*/\1${CS}/" "$RUN_SCRIPT_PATH"
    sed -i -E "s/^(\s*set_reg_alpha\s*=\s*).*/\1${RA}/" "$RUN_SCRIPT_PATH"
    sed -i -E "s/^(\s*set_reg_lambda\s*=\s*).*/\1${RL}/" "$RUN_SCRIPT_PATH"
    sed -i -E "s/^(\s*set_num_boost_round\s*=\s*).*/\1${NBR}/" "$RUN_SCRIPT_PATH"
    sed -i -E "s/^(\s*set_early_stopping_rounds\s*=\s*).*/\1${ESR}/" "$RUN_SCRIPT_PATH"

    # 2. Create the SLURM submission script for this run
    cat << EOF > "$RUN_SLURM_PATH"
#!/bin/bash
#SBATCH --job-name=XGB_${RUN_ID}
#SBATCH --output=${RUN_DIR}/XGB_%j.out
#SBATCH --error=${RUN_DIR}/XGB_%j.err
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --gres=${SBATCH_GRES}
#SBATCH --mem=${SBATCH_MEM}
#SBATCH --time=${SBATCH_TIME}
#SBATCH --cpus-per-task=${SBATCH_CPUS}
#SBATCH --chdir=${RUN_DIR}

# Print info for debugging
echo "Job ID: \$SLURM_JOB_ID"
echo "Run ID: ${RUN_ID}"
echo "Running on: \$(hostname)"
echo "Starting at: \$(date)"
echo "Working Directory: \$(pwd)"

# Start nvidia-smi in the background
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1 > XGB_gpu_\${SLURM_JOB_ID}.eff &
NVSMI_PID=\$!

# Activate virtual environment
module load ${MODULES_TO_LOAD}

echo "Running Python script..."
# Run the modified Python script
python -u XGB_run.py "${ABS_TRAIN_PATH}" "${ABS_TEST_PATH}" "${RUN_OUTPUT_FITS}"

PY_EXIT_CODE=\$?
echo "Python script finished with exit code: \$PY_EXIT_CODE"

# Stop the nvidia-smi logging
kill \$NVSMI_PID

echo "Finished at: \$(date)"
exit \$PY_EXIT_CODE
EOF

    # 3. Submit the job
    sbatch "$RUN_SLURM_PATH"
    
    echo "Submitted job for ${RUN_ID}"
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
    
    # Optional: Add a small sleep to avoid overwhelming the scheduler
    # sleep 0.1

done
done
done
done
done
done
done
done

echo "---"
echo "Grid search submission complete. Submitted $TOTAL_JOBS jobs."
echo "Monitor jobs with: squeue -u $USER"
echo "Outputs will be in: ${PROJECT_DIR}/${GRID_RUNS_DIR}/"
