#!/usr/bin/env python3
"""Debug extraction to compare with read_phu.py output."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from extract_and_compare_power_sweeps import extract_power_sweep_data

# Test with 70mV
test_file = "/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_70mV_20260205_0122.phu"

print("Extracting from 70mV file...")
data = extract_power_sweep_data(test_file)

print(f"\n=== EXTRACTED DATA ===")
print(f"Total points: {len(data['power_uW'])}")
print(f"\nFirst 10 points:")
for i in range(min(10, len(data['power_uW']))):
    print(f"  {data['power_uW'][i]:.6f} µW -> {data['net_rate_cts_s'][i]:.2f} cts/s")

print(f"\nLast 5 points:")
for i in range(max(0, len(data['power_uW'])-5), len(data['power_uW'])):
    print(f"  {data['power_uW'][i]:.6f} µW -> {data['net_rate_cts_s'][i]:.2f} cts/s")

print(f"\nPower range: {min(data['power_uW']):.6f} - {max(data['power_uW']):.6f} µW")
print(f"Rate range: {min(data['net_rate_cts_s']):.2f} - {max(data['net_rate_cts_s']):.2f} cts/s")
