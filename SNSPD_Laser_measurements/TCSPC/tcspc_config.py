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
FIT_MAX_UW = 3e-1  # Maximum power for low-power fit region (µW)

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
# Marker and color settings for different bias voltages
BIAS_SETTINGS = {
    '66mV': {'color': 'purple', 'marker': 'D'},
    '70mV': {'color': 'blue', 'marker': 'o'},
    '73mV': {'color': 'cyan', 'marker': 'v'},
    '74mV': {'color': 'green', 'marker': 's'},
    '78mV': {'color': 'red', 'marker': '^'},
}

# ============ DATA FILES FOR COMBINED PLOT ============
# Bias voltage files for combined analysis
BIAS_FILES = {
    '66mV': '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_66mV_20260205_0754.phu',
    '70mV': '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_70mV_20260205_0122.phu',
    '73mV': '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_73mV_20260205_1213.phu',
    '74mV': '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_74mV_20260205_0102.phu',
    '78mV': '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_78mV_20260205_0230.phu',
}
