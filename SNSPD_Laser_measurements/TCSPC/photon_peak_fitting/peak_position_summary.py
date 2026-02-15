#!/usr/bin/env python3
"""
Summary comparison of peak positions and widths across power levels.
"""

import pandas as pd
import numpy as np

# Data from multi-Gaussian fits with loose mean bounds
data = {
    'Block': [10, 120, 150],
    'Power (µW)': [2.6230, 6.8600, 2.1130],
    
    # n=4
    'n4_mu (ns)': [75.2356, 75.2235, 75.2365],
    'n4_sigma (ps)': [24.20, 28.57, 23.19],
    'n4_shift (ps)': [0, -12.13, +0.86],
    
    # n=3
    'n3_mu (ns)': [75.3386, 75.3086, 75.3451],
    'n3_sigma (ps)': [40.18, 35.21, 39.18],
    'n3_shift (ps)': [0, -30.00, +6.52],
    
    # n=2
    'n2_mu (ns)': [75.4688, 75.4388, 75.4701],
    'n2_sigma (ps)': [42.78, 47.85, 36.99],
    'n2_shift (ps)': [0, -30.00, +1.34],
    
    # n=1
    'n1_mu (ns)': [75.5880, 75.5742, 75.6135],
    'n1_sigma (ps)': [144.16, 45.29, 143.48],
    'n1_shift (ps)': [0, -13.81, +25.53],
}

df = pd.DataFrame(data)

print("\n" + "="*100)
print("COMPREHENSIVE PEAK POSITION AND WIDTH SUMMARY")
print("="*100)

print("\n1. MEAN POSITIONS (TOA in ns):")
print("-" * 100)
print("Block | Power (µW) |   n=4    |   n=3    |   n=2    |   n=1   ")
print("------|------------|----------|----------|----------|----------")
for idx, row in df.iterrows():
    print(f"{int(row['Block']):5d} | {row['Power (µW)']:9.4f} | {row['n4_mu (ns)']:.4f}  | {row['n3_mu (ns)']:.4f}  | {row['n2_mu (ns)']:.4f}  | {row['n1_mu (ns)']:.4f}")

print("\n2. PEAK WIDTHS (σ in ps):")
print("-" * 100)
print("Block | Power (µW) |  n=4   |  n=3   |  n=2   |  n=1   ")
print("------|------------|--------|--------|--------|--------")
for idx, row in df.iterrows():
    print(f"{int(row['Block']):5d} | {row['Power (µW)']:9.4f} | {row['n4_sigma (ps)']:6.2f} | {row['n3_sigma (ps)']:6.2f} | {row['n2_sigma (ps)']:6.2f} | {row['n1_sigma (ps)']:6.2f}")

print("\n3. MEAN SHIFTS FROM BLOCK 10 (ps):")
print("-" * 100)
print("Block | Power (µW) |  n=4    |  n=3    |  n=2    |  n=1    ")
print("------|------------|---------|---------|---------|----------")
for idx, row in df.iterrows():
    print(f"{int(row['Block']):5d} | {row['Power (µW)']:9.4f} | {row['n4_shift (ps)']:+7.2f} | {row['n3_shift (ps)']:+7.2f} | {row['n2_shift (ps)']:+7.2f} | {row['n1_shift (ps)']:+7.2f}")

print("\n4. WIDTH RATIO (relative to Block 10):")
print("-" * 100)
print("Block | Power (µW) |  n=4   |  n=3   |  n=2   |  n=1   ")
print("------|------------|--------|--------|--------|--------")
for idx, row in df.iterrows():
    if idx == 0:  # Block 10 reference
        print(f"{int(row['Block']):5d} | {row['Power (µW)']:9.4f} | {1.00:6.2f} | {1.00:6.2f} | {1.00:6.2f} | {1.00:6.2f}")
    else:
        r_n4 = row['n4_sigma (ps)'] / df.loc[0, 'n4_sigma (ps)']
        r_n3 = row['n3_sigma (ps)'] / df.loc[0, 'n3_sigma (ps)']
        r_n2 = row['n2_sigma (ps)'] / df.loc[0, 'n2_sigma (ps)']
        r_n1 = row['n1_sigma (ps)'] / df.loc[0, 'n1_sigma (ps)']
        print(f"{int(row['Block']):5d} | {row['Power (µW)']:9.4f} | {r_n4:6.2f} | {r_n3:6.2f} | {r_n2:6.2f} | {r_n1:6.2f}")

print("\n" + "="*100)
print("KEY FINDINGS:")
print("="*100)

print("\n• Peak positions ARE power-dependent:")
print("  - Block 120 (high power, 6.86 µW): All peaks shifted EARLIER (-12 to -30 ps)")
print("  - Block 150 (low power, 2.11 µW): Peaks shifted LATER (+0.86 to +25.53 ps)")

print("\n• n=1 peak width shows dramatic power dependence:")
print("  - Block 10 (2.62 µW): 144.16 ps")
print("  - Block 120 (6.86 µW): 45.29 ps (0.31x = 69% narrower)")
print("  - Block 150 (2.11 µW): 143.48 ps (essentially same as block 10)")

print("\n• n=1 mean shift is also power-dependent:")
print("  - Block 120: -13.81 ps (earlier)")
print("  - Block 150: +25.53 ps (later)")
print("  - Pattern: Peak moves AND width changes")

print("\n• Other peaks show smaller but consistent width variations:")
print("  - n=4: relatively stable (24-29 ps)")
print("  - n=3, n=2: moderate variation (35-48 ps)")

print("\n• Interpretation:")
print("  1. There's a power-dependent time-shift mechanism (early at high power)")
print("  2. n=1 peak broadening at low power is likely detector timing jitter or")
print("     correlation effects that reduce at high power")
print("  3. Reference photon positions need power-dependent calibration")

print("\n" + "="*100)
