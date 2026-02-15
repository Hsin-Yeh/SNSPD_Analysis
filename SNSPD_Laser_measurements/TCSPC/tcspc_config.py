#!/usr/bin/env python3
"""
Shared configuration for TCSPC analysis.
All settings used by both read_phu.py and create_combined_plot.py are defined here.
"""

from pathlib import Path

# ============ TIME WINDOW SETTINGS ============
# Signal detection window (in nanoseconds)
T_MIN_NS = 75.0
T_MAX_NS = 79.0
SIGNAL_WIDTH_NS = T_MAX_NS - T_MIN_NS

# Fit range settings
FIT_MAX_UW = 2e-1  # Maximum power for low-power fit region (µW)

# ============ DATA FILE PATHS ============
PROJECT_ROOT = Path(__file__).parent.parent

# Power data file path - using 1-degree interpolated data (0-355° full range)
POWER_DATA_FILE = PROJECT_ROOT / "Attenuation" / "Rotation_10MHz_1degree_data_20260205.txt"

# Output directories
OUTPUT_DIR_BASE = Path.home() / 'SNSPD_analyzed_output' / 'TCSPC' / 'SMSPD_3'

# Power sweep outputs
OUTPUT_DIR_POWER_SWEEP = OUTPUT_DIR_BASE / 'power_sweep'
OUTPUT_DIR_COMBINED = OUTPUT_DIR_POWER_SWEEP / 'combined'
OUTPUT_DIR_INDIVIDUAL = OUTPUT_DIR_POWER_SWEEP  # Base for individual bias folders

# Bias sweep outputs
OUTPUT_DIR_BIAS_SWEEP = OUTPUT_DIR_BASE / 'bias_sweep'
OUTPUT_DIR_BIAS_COMBINED = OUTPUT_DIR_BIAS_SWEEP / 'combined'

# ============ PLOT SETTINGS ============
# Auto-generated color and marker palette for plotting
AUTO_COLORS = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 
               'brown', 'pink', 'olive', 'navy', 'teal', 'maroon', 'lime']
AUTO_MARKERS = ['o', '^', 's', 'D', 'v', 'p', '*', 'h', 'X', 'd', '<', '>', 'P', 'H']
