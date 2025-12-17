#!/bin/bash
# Batch process multiple counter data folders
# Usage: ./process_all_counter_data.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/plot_counter_generic.py"
BASE_DIR="/Users/ya/Documents/Projects/SNSPD/SNSPD_data/SMSPD_3"

echo "Processing all counter measurement folders..."
echo "=============================================="

# Process test data
echo -e "\n>>> Processing test data..."
/usr/bin/python3 "$PYTHON_SCRIPT" "$BASE_DIR/test/2-7/6K" --bias 66,68,70,72,74

# Process 1MHz data  
echo -e "\n>>> Processing 1MHz data..."
/usr/bin/python3 "$PYTHON_SCRIPT" "$BASE_DIR/1MHz/2-7/6K" --bias 66,68,70,72,74

# Process Counter_sweep_power_3 data with 3 lowest points removed
echo -e "\n>>> Processing Counter_sweep_power_3 data..."
/usr/bin/python3 "$PYTHON_SCRIPT" "$BASE_DIR/Counter_sweep_power_3/2-7/6K" --bias 68,70,72,74 --remove-lowest 3

echo -e "\n=============================================="
echo "All processing complete! Check output/ folder for plots."
