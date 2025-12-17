#!/bin/bash

# Directory to search
SEARCH_DIR="grid_search_runs"

# Check if directory exists
if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Directory '$SEARCH_DIR' not found."
    exit 1
fi

# --- ARGUMENT PARSING ---
# 1. Use the first argument, or default to "AUC"
# 2. Convert to uppercase
METRIC_ARG="${1:-AUC}"
METRIC_UPPER=$(echo "$METRIC_ARG" | tr '[:lower:]' '[:upper:]')

all_results=""

if [ "$METRIC_UPPER" == "BOTH" ]; then
    echo "Mode: Searching for highest Combined Score (AUC + mAP)"
    
    # 1. Find files
    # 2. Grep for EITHER string (using -E for extended regex with pipe |)
    # 3. Sed cleans it to: "Filename Value"
    # 4. Awk aggregates: It sums the values per filename. 
    #    It only prints the result if it found exactly 2 metrics for that file.
    all_results=$(find "$SEARCH_DIR" -name "XGB*.out" -exec grep -E -H "Macro-Averaged ROC AUC|Macro-Averaged Precision \(mAP\)" {} + | \
    sed -E 's/(.*):.*: ([0-9]+\.[0-9]+)$/\1 \2/' | \
    awk '{sum[$1] += $2; count[$1]++} END {for (f in sum) {if (count[f] == 2) print f, sum[f]}}')

elif [ "$METRIC_UPPER" == "AUC" ] || [ "$METRIC_UPPER" == "MAP" ]; then
    
    # Set search string based on mode
    if [ "$METRIC_UPPER" == "AUC" ]; then
        SEARCH_STR="Macro-Averaged ROC AUC"
        echo "Mode: Searching for ROC AUC"
    else
        SEARCH_STR="Macro-Averaged Precision (mAP)"
        echo "Mode: Searching for mAP"
    fi

    # Standard single-metric extraction
    all_results=$(find "$SEARCH_DIR" -name "XGB*.out" -exec grep -H "$SEARCH_STR" {} + | \
    sed -E 's/(.*):.*: ([0-9]+\.[0-9]+)$/\1 \2/')

else
    echo "Error: Invalid argument '$1'. Please use 'AUC', 'mAP', or 'BOTH'."
    exit 1
fi

# --- OUTPUT RESULT ---

if [ -z "$all_results" ]; then
    echo "No matching scores found for mode: $METRIC_UPPER."
else
    # Sort numerically descending by score (column 2) and take top one to find max value
    highest_score=$(echo "$all_results" | sort -k2 -rn | head -n 1 | awk '{print $2}')

    echo "Found highest score: $highest_score"
    echo "Files with this score:"

    # Filter all results that match the highest score and print just the filename
    echo "$all_results" | awk -v max="$highest_score" '$2 == max {print $1}'
fi
