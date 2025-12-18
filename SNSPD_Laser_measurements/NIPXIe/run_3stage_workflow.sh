#!/bin/bash
#
# 3-Stage SNSPD Analysis Workflow
#
# This script automates the complete 3-stage analysis workflow:
#   Stage 1: SelfTrigger.py     (TDMS → event JSON)
#   Stage 2: analyze_events.py  (event JSON → statistics JSON)
#   Stage 3: plot_all.py        (statistics JSON → comparison plots)
#
# Directory structure created next to SNSPD_rawdata:
#   SNSPD_rawdata/path/to/data/*.tdms
#   SNSPD_analysis/path/to/data/stage1_events/*.json
#   SNSPD_analysis/path/to/data/stage2_statistics/*.json
#   SNSPD_analysis/path/to/data/stage3_plots/*.png
#
# Usage:
#   ./run_3stage_workflow.sh /path/to/rawdata_directory [--subset N]
#
# Example:
#   ./run_3stage_workflow.sh /Users/ya/SNSPD_rawdata/SMSPD_3/Laser/2-7/20251210/6K/Pulse/515/10000kHz/207nW --subset 1000
#

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <rawdata_directory> [--subset N] [--no-plots]"
    echo ""
    echo "Examples:"
    echo "  $0 /Users/ya/SNSPD_rawdata/SMSPD_3/.../207nW"
    echo "  $0 /Users/ya/SNSPD_rawdata/SMSPD_3/.../207nW --subset 1000"
    echo "  $0 /Users/ya/SNSPD_rawdata/SMSPD_3/.../207nW --no-plots"
    exit 1
fi

RAW_DIR="$1"
shift
EXTRA_ARGS="$@"

# Check if directory exists
if [ ! -d "$RAW_DIR" ]; then
    echo "Error: Directory not found: $RAW_DIR"
    exit 1
fi

# Derive analysis directory
# Convert SNSPD_rawdata to SNSPD_analysis in the path
ANALYSIS_BASE=$(echo "$RAW_DIR" | sed 's|SNSPD_rawdata|SNSPD_analysis|' | sed 's|SNSPD_data|SNSPD_analysis|')

STAGE1_DIR="${ANALYSIS_BASE}/stage1_events"
STAGE2_DIR="${ANALYSIS_BASE}/stage2_statistics"
STAGE3_DIR="${ANALYSIS_BASE}/stage3_plots"

echo "=========================================================================="
echo "  3-STAGE SNSPD ANALYSIS WORKFLOW"
echo "=========================================================================="
echo ""
echo "Raw data:     $RAW_DIR"
echo "Stage 1:      $STAGE1_DIR"
echo "Stage 2:      $STAGE2_DIR"  
echo "Stage 3:      $STAGE3_DIR"
echo ""
echo "=========================================================================="

# ============================================================================
# STAGE 1: Event Extraction
# ============================================================================
echo ""
echo "=========================================================================="
echo "STAGE 1: Event Extraction (TDMS → Event JSON)"
echo "=========================================================================="

# Extract only --subset argument for Stage 1
SUBSET_ARG=""
if [[ "$EXTRA_ARGS" == *"--subset"* ]]; then
    SUBSET_ARG=$(echo "$EXTRA_ARGS" | grep -oE '\--subset [0-9]+')
fi

python3 "${SCRIPT_DIR}/SelfTrigger.py" "$RAW_DIR" $SUBSET_ARG

# Check if any files were created
if [ ! -d "$STAGE1_DIR" ] || [ -z "$(ls -A $STAGE1_DIR 2>/dev/null)" ]; then
    echo "Error: No output files created in Stage 1"
    exit 1
fi

NUM_STAGE1=$(find "$STAGE1_DIR" -name "*_analysis.json" | wc -l | tr -d ' ')
echo "✓ Stage 1 complete: $NUM_STAGE1 event JSON files created"

# ============================================================================
# STAGE 2: Statistical Analysis
# ============================================================================
echo ""
echo "=========================================================================="
echo "STAGE 2: Statistical Analysis (Event JSON → Statistics JSON)"
echo "=========================================================================="

# Create stage2 directory
mkdir -p "$STAGE2_DIR"

# Check if --no-plots was passed
NO_PLOTS_FLAG=""
if [[ "$EXTRA_ARGS" == *"--no-plots"* ]]; then
    NO_PLOTS_FLAG="--no-plots"
fi

python3 "${SCRIPT_DIR}/analyze_events.py" "${STAGE1_DIR}"/*_analysis.json \
    -d "$STAGE2_DIR" -b 100 $NO_PLOTS_FLAG

NUM_STAGE2=$(find "$STAGE2_DIR" -name "statistics_*.json" | wc -l | tr -d ' ')
echo "✓ Stage 2 complete: $NUM_STAGE2 statistics JSON files created"

# ============================================================================
# STAGE 3: Comparison Plots
# ============================================================================
echo ""
echo "=========================================================================="
echo "STAGE 3: Comparison Plotting (Statistics JSON → Plots)"
echo "=========================================================================="

# Create stage3 directory
mkdir -p "$STAGE3_DIR"

python3 "${SCRIPT_DIR}/plot_all.py" \
    -i "$STAGE2_DIR" \
    -p 'statistics_*.json' \
    -d "$STAGE3_DIR" \
    --mode all

echo "✓ Stage 3 complete: Plots saved to $STAGE3_DIR"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================================================="
echo "  WORKFLOW COMPLETE!"
echo "=========================================================================="
echo ""
echo "Results:"
echo "  Stage 1: $NUM_STAGE1 event JSON files"
echo "  Stage 2: $NUM_STAGE2 statistics JSON files"
echo "  Stage 3: Comparison plots in $STAGE3_DIR"
echo ""
echo "To view plots:"
echo "  open $STAGE3_DIR/output/*/*.png"
echo ""
echo "=========================================================================="
