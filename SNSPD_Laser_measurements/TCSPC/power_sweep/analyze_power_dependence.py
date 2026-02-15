#!/usr/bin/env python3
"""
Analyze power-dependent peak width trends from all 49 fitted blocks.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load summary data
summary_file = '/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/74mV/peak_fits_75p7/fit_results_summary.json'

with open(summary_file, 'r') as f:
    summary_dict = json.load(f)

# Extract data
blocks = []
powers = []
sigmas_n4 = []
sigmas_n3 = []
sigmas_n2 = []
sigmas_n1 = []
r_squared = []

for block_id in sorted([int(k) for k in summary_dict.keys()]):
    block_str = str(block_id)
    if block_str in summary_dict:
        data = summary_dict[block_str]
        if data['r_squared'] > 0.5:  # Only include good fits
            blocks.append(block_id)
            powers.append(data['power_uw'])
            sigmas = data['sigmas_ps']
            sigmas_n4.append(sigmas[0])
            sigmas_n3.append(sigmas[1])
            sigmas_n2.append(sigmas[2])
            sigmas_n1.append(sigmas[3])
            r_squared.append(data['r_squared'])

# Convert to arrays
powers = np.array(powers)
sigmas_n4 = np.array(sigmas_n4)
sigmas_n3 = np.array(sigmas_n3)
sigmas_n2 = np.array(sigmas_n2)
sigmas_n1 = np.array(sigmas_n1)
r_squared = np.array(r_squared)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: n=4 width vs power
ax = axes[0, 0]
ax.plot(powers, sigmas_n4, 'o-', markersize=8, linewidth=2, color='steelblue', label='n=4')
ax.set_xlabel('Power (µW)', fontsize=12, weight='bold')
ax.set_ylabel('Peak Width σ (ps)', fontsize=12, weight='bold')
ax.set_title('n=4 Peak Width vs Power', fontsize=13, weight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.legend(fontsize=11)

# Plot 2: n=3 width vs power
ax = axes[0, 1]
ax.plot(powers, sigmas_n3, 'o-', markersize=8, linewidth=2, color='darkgreen', label='n=3')
ax.set_xlabel('Power (µW)', fontsize=12, weight='bold')
ax.set_ylabel('Peak Width σ (ps)', fontsize=12, weight='bold')
ax.set_title('n=3 Peak Width vs Power', fontsize=13, weight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.legend(fontsize=11)

# Plot 3: n=2 width vs power
ax = axes[1, 0]
ax.plot(powers, sigmas_n2, 'o-', markersize=8, linewidth=2, color='darkred', label='n=2')
ax.set_xlabel('Power (µW)', fontsize=12, weight='bold')
ax.set_ylabel('Peak Width σ (ps)', fontsize=12, weight='bold')
ax.set_title('n=2 Peak Width vs Power', fontsize=13, weight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.legend(fontsize=11)

# Plot 4: n=1 width vs power (most interesting)
ax = axes[1, 1]
ax.plot(powers, sigmas_n1, 'o-', markersize=8, linewidth=2, color='purple', label='n=1', alpha=0.8)
ax.set_xlabel('Power (µW)', fontsize=12, weight='bold')
ax.set_ylabel('Peak Width σ (ps)', fontsize=12, weight='bold')
ax.set_title('n=1 Peak Width vs Power (Most Power-Dependent)', fontsize=13, weight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.legend(fontsize=11)

plt.tight_layout()
output_file = '/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/74mV/peak_fits_75p7/power_dependence_analysis.png'
fig.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close(fig)

# Create combined plot
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(powers, sigmas_n4, 'o-', markersize=8, linewidth=2.5, label='n=4 (highest photon)', color='steelblue', alpha=0.8)
ax.plot(powers, sigmas_n3, 's-', markersize=8, linewidth=2.5, label='n=3', color='darkgreen', alpha=0.8)
ax.plot(powers, sigmas_n2, '^-', markersize=8, linewidth=2.5, label='n=2', color='darkred', alpha=0.8)
ax.plot(powers, sigmas_n1, 'v-', markersize=9, linewidth=2.5, label='n=1 (lowest photon)', color='purple', alpha=0.8)

ax.set_xlabel('Power (µW)', fontsize=13, weight='bold')
ax.set_ylabel('Peak Width σ (ps)', fontsize=13, weight='bold')
ax.set_title('Photon Peak Width vs Power (75.7 ns tail cut)', fontsize=14, weight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xscale('log')
ax.legend(fontsize=12, loc='upper left', framealpha=0.9)

plt.tight_layout()
output_file = '/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/74mV/peak_fits_75p7/combined_power_dependence.png'
fig.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close(fig)

# Print statistics
print("\n" + "="*100)
print("PEAK WIDTH POWER-DEPENDENCE ANALYSIS (75.7 ns tail cut)")
print("="*100)

print("\nn=4 Peak (highest photon number):")
print(f"  Power range: {powers.min():.4f} - {powers.max():.4f} µW")
print(f"  Width range: {sigmas_n4.min():.2f} - {sigmas_n4.max():.2f} ps")
print(f"  Ratio (max/min): {sigmas_n4.max()/sigmas_n4.min():.2f}x")

print("\nn=3 Peak:")
print(f"  Power range: {powers.min():.4f} - {powers.max():.4f} µW")
print(f"  Width range: {sigmas_n3.min():.2f} - {sigmas_n3.max():.2f} ps")
print(f"  Ratio (max/min): {sigmas_n3.max()/sigmas_n3.min():.2f}x")

print("\nn=2 Peak:")
print(f"  Power range: {powers.min():.4f} - {powers.max():.4f} µW")
print(f"  Width range: {sigmas_n2.min():.2f} - {sigmas_n2.max():.2f} ps")
print(f"  Ratio (max/min): {sigmas_n2.max()/sigmas_n2.min():.2f}x")

print("\nn=1 Peak (lowest photon number) - MOST POWER-DEPENDENT:")
print(f"  Power range: {powers.min():.4f} - {powers.max():.4f} µW")
print(f"  Width range: {sigmas_n1[sigmas_n1 < 200].min():.2f} - {sigmas_n1[sigmas_n1 < 200].max():.2f} ps (excluding very low power)")
print(f"  Ratio (max/min): {sigmas_n1[sigmas_n1 < 200].max()/sigmas_n1[sigmas_n1 < 200].min():.2f}x")

# Find best high-power and best low-power blocks
high_power_idx = np.argmax(powers)
low_power_idx_good = np.argmin(powers[sigmas_n1 < 200])

print(f"\nHigh power example (block {blocks[high_power_idx]}, {powers[high_power_idx]:.4f} µW):")
print(f"  n=4: {sigmas_n4[high_power_idx]:.2f} ps")
print(f"  n=3: {sigmas_n3[high_power_idx]:.2f} ps")
print(f"  n=2: {sigmas_n2[high_power_idx]:.2f} ps")
print(f"  n=1: {sigmas_n1[high_power_idx]:.2f} ps")

low_power_indices = np.where(sigmas_n1 < 200)[0]
low_power_idx = low_power_indices[np.argmin(powers[low_power_indices])]

print(f"\nLow power example (block {blocks[low_power_idx]}, {powers[low_power_idx]:.4f} µW):")
print(f"  n=4: {sigmas_n4[low_power_idx]:.2f} ps")
print(f"  n=3: {sigmas_n3[low_power_idx]:.2f} ps")
print(f"  n=2: {sigmas_n2[low_power_idx]:.2f} ps")
print(f"  n=1: {sigmas_n1[low_power_idx]:.2f} ps")

print(f"\nRatio (high power / low power) for n=1:")
print(f"  {sigmas_n1[high_power_idx]:.2f} / {sigmas_n1[low_power_idx]:.2f} = {sigmas_n1[high_power_idx]/sigmas_n1[low_power_idx]:.2f}x")

print("\n" + "="*100)
print("✓ Analysis complete!")
print("="*100 + "\n")
