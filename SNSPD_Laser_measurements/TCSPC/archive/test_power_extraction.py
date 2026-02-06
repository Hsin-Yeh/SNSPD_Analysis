#!/usr/bin/env python3
"""Test power sweep data extraction."""

from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from extract_and_compare_power_sweeps import extract_power_sweep_data

# Test with one file
test_file = "/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_70mV_20260205_0122.phu"

if Path(test_file).exists():
    print(f"Testing extraction from: {test_file}")
    data = extract_power_sweep_data(test_file)
    print(f"\nExtracted {len(data['power_uW'])} data points")
    print(f"Power range: {min(data['power_uW']):.4f} - {max(data['power_uW']):.4f} µW")
    print(f"Rate range: {min(data['net_rate_cts_s']):.2f} - {max(data['net_rate_cts_s']):.2f} cts/s")
    print(f"\nFirst 5 data points:")
    for i in range(min(5, len(data['power_uW']))):
        print(f"  {data['power_uW'][i]:.4f} µW -> {data['net_rate_cts_s'][i]:.2f} cts/s")
else:
    print(f"ERROR: File not found: {test_file}")
