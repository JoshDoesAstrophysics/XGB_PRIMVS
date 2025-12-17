#!/bin/bash

# Directory to search
SEARCH_DIR="grid_search_runs"

# Check if directory exists
if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Directory '$SEARCH_DIR' not found."
    exit 1
fi

# --- ARGUMENT PARSING ---
# Default to "files" if no argument is provided
MODE="${1:-files}"

failed_files=""

# Loop through all .err files to identify failures
# We use a while loop with process substitution to handle variable scope correctly
while read -r err_file; do
    
    # 1. Check if .err file is empty
    # -s checks if file exists and has size > 0
    # If it's empty, we assume no issues (per your instructions)
    if [ ! -s "$err_file" ]; then
        continue
    fi

    # 2. Check corresponding .out file for Success Message
    # This acts as an "OR gate": If .out has success, we ignore the content in .err (likely just warnings)
    
    # Construct .out filename: replace .err with .out
    out_file="${err_file%.err}.out"
    
    if [ -f "$out_file" ]; then
        if grep -q "=== XGBoost Training Complete ===" "$out_file"; then
            continue # Job succeeded, ignore the stderr content
        fi
    fi

    # If we are here: .err is NOT empty AND .out does NOT show success.
    # Add to list of failed files.
    failed_files+="$err_file"$'\n'

done < <(find "$SEARCH_DIR" -name "*.err")

# Check if any failed files were found
if [ -z "$failed_files" ]; then
    echo "Great! All runs in '$SEARCH_DIR' completed successfully (or have empty .err files)."
    exit 0
fi

# --- MODES ---

if [ "$MODE" == "files" ]; then
    echo "The following .err files indicate failed runs:"
    # echo -e interprets the newline characters in the string
    echo -e "$failed_files"

elif [ "$MODE" == "totals" ]; then
    echo "Error Summary (Count | Error Message):"
    echo "--------------------------------------"

    # Iterate through each failed .err file to determine the cause
    echo -e "$failed_files" | while read -r file; do
        if [ -z "$file" ]; then continue; fi
        
        # 1. Check for Slurm Time Limit
        if grep -q "DUE TO TIME LIMIT" "$file"; then
            echo "Slurm: Time Limit Exceeded"
            continue
        fi

        # 2. Check for Slurm/System OOM
        if grep -iqE "oom-kill|out of memory|killed" "$file"; then
            echo "Slurm/System: Out of Memory / Killed"
            continue
        fi

        # 3. Check for Explicit Python Exceptions (Best Match)
        py_error=$(grep -E "^[[:space:]]*[a-zA-Z._]*(Error|Exception):" "$file" | tail -n 1)
        if [ -n "$py_error" ]; then
            echo "$py_error" | sed 's/^[ \t]*//'
            continue
        fi

        # 4. Check for ANY "exit code" pattern
        exit_code_str=$(grep -oi "exit code[: ]*[0-9]\+" "$file" | tail -n 1)
        if [ -n "$exit_code_str" ]; then
            code_num=$(echo "$exit_code_str" | grep -o "[0-9]\+")
            echo "Exit Code: $code_num"
            continue
        fi

        # 5. Check for "Error" keyword generally (Fallback)
        generic_error=$(grep -i "Error" "$file" | grep -v "srun: error" | tail -n 1)
        if [ -n "$generic_error" ]; then
             echo "$generic_error" | sed 's/^[ \t]*//'
             continue
        fi

        # 6. Fallback: Print the very last non-empty line
        last_line=$(tail -n 5 "$file" | sed '/^$/d' | tail -n 1)
        
        if [ -z "$last_line" ]; then
             echo "Unknown Error (File not empty but no obvious error text)"
        else
             echo "$last_line"
        fi

    done | sort | uniq -c | sort -nr

else
    echo "Error: Invalid argument '$1'. Usage: ./parse_errors.sh [files|totals]"
    exit 1
fi
