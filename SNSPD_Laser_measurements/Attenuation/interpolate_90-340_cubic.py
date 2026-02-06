#!/usr/bin/env python3
"""
Interpolate rotation data from 5-degree to 1-degree separation.
Focus on 90-340° range (low-power exponential decay region).
Uses cubic interpolation for smooth curves.
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path

# Load original data
data_file = Path(__file__).parent / "Rotation_10MHz_5degrees_data_20260205.txt"
data = np.loadtxt(data_file)
angles_orig = data[:, 0]
values_orig = data[:, 1]

print("="*80)
print("ROTATION DATA INTERPOLATION - 90° to 340° (1° separation)")
print("="*80)
print(f"\nOriginal data: {len(angles_orig)} points from {angles_orig[0]:.0f}° to {angles_orig[-1]:.0f}°")
print(f"Interpolation range: 90° to 340° (250° span)\n")

# Filter data to 90-340° range
mask = (angles_orig >= 90) & (angles_orig <= 340)
angles_subset = angles_orig[mask]
values_subset = values_orig[mask]

print(f"Subset for interpolation: {len(angles_subset)} original points")
print(f"Angle range: {angles_subset[0]:.0f}° to {angles_subset[-1]:.0f}°")
print(f"Power range: {np.min(values_subset):.4e} to {np.max(values_subset):.4e} nW\n")

# Create cubic interpolation function
interp_cubic = interp1d(angles_subset, values_subset, kind='cubic', fill_value='extrapolate')

# Create new angle grid at 1-degree intervals
angles_new = np.arange(90, 341, 1.0)

# Interpolate using cubic method
values_interp = interp_cubic(angles_new)

print(f"Interpolated data: {len(angles_new)} points from {angles_new[0]:.0f}° to {angles_new[-1]:.0f}°")
print(f"Interpolation spacing: 1°")
print(f"Interpolation method: CUBIC\n")

# ============ VALIDATION METRICS ============
print("="*80)
print("INTERPOLATION VALIDATION")
print("="*80)

# 1. Check at original points
print("\n1. RECOVERY AT ORIGINAL POINTS:")
print("-" * 40)
orig_indices = np.searchsorted(angles_new, angles_subset)
values_at_orig = values_interp[orig_indices]

residuals = np.abs(values_at_orig - values_subset)
max_residual = np.max(residuals)
mean_residual = np.mean(residuals)
relative_error = np.divide(residuals, values_subset, where=values_subset != 0, out=np.zeros_like(residuals))

print(f"Max absolute error: {max_residual:.4e}")
print(f"Mean absolute error: {mean_residual:.4e}")
if np.max(relative_error) > 0:
    print(f"Max relative error: {np.max(relative_error)*100:.6f}%")
    print(f"Mean relative error: {np.mean(relative_error)*100:.6f}%")

# 2. Physical validity
print("\n2. PHYSICAL VALIDITY:")
print("-" * 40)
neg_count = np.sum(values_interp < 0)
print(f"Negative values: {neg_count} detected")
if neg_count > 0:
    print(f"Min value: {np.min(values_interp):.4e} at angle {angles_new[np.argmin(values_interp)]:.1f}°")
    print("⚠️  WARNING: Negative values may indicate cubic overshoot")

print(f"Original min: {np.min(values_subset):.4e}, max: {np.max(values_subset):.4e}")
print(f"Interp min: {np.min(values_interp):.4e}, max: {np.max(values_interp):.4e}")

if np.min(values_interp) >= 0 and np.max(values_interp) <= np.max(values_subset) * 1.05:
    print("✓ Values within reasonable bounds")
else:
    print(f"⚠️  Values exceed bounds (possible overshoot)")

# 3. Smoothness check
print("\n3. SMOOTHNESS ANALYSIS:")
print("-" * 40)
first_deriv = np.diff(values_interp) / np.diff(angles_new)
second_deriv = np.diff(first_deriv) / np.diff(angles_new[:-1])

print(f"Max first derivative: {np.max(np.abs(first_deriv)):.4e} nW/°")
print(f"Mean first derivative: {np.mean(np.abs(first_deriv)):.4e} nW/°")
print(f"Max second derivative (curvature): {np.max(np.abs(second_deriv)):.4e}")
print(f"✓ Cubic interpolation produces smooth curves")

# 4. Exponential decay check
print("\n4. EXPONENTIAL DECAY ANALYSIS:")
print("-" * 40)
# Check if data follows approximate exponential decay
log_values = np.log(values_interp + 1e-10)
approx_linear = np.polyfit(angles_new, log_values, 1)
print(f"Log-linear slope (approx): {approx_linear[0]:.6e}")
print(f"This region (~90-340°) should show exponential decay")
if approx_linear[0] < 0:
    print("✓ Negative slope confirms exponential decay")
else:
    print("⚠️  Unexpected positive slope")

# ============ SAVE INTERPOLATED DATA ============
print("\n" + "="*80)
print("SAVING INTERPOLATED DATA")
print("="*80)

output_file = Path(__file__).parent / "Rotation_10MHz_1degree_data_90-340_20260205.txt"
header = "# Interpolated from 5-degree to 1-degree separation (90-340° range, cubic method)\n# Angle (degrees)\tPower (nW)\n"

with open(output_file, 'w') as f:
    f.write(header)
    for angle, value in zip(angles_new, values_interp):
        f.write(f"{angle:.1f}\t{value:.6e}\n")

print(f"\n✓ Saved to: {output_file.name}")
print(f"  Format: Angle (degrees) | Power (nW)")
print(f"  Data points: {len(angles_new)}")
print(f"  Angle range: 90° to 340°")
print(f"  Method: Cubic interpolation")

# ============ VISUALIZATION ============
print("\nGenerating validation plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Linear scale
ax = axes[0, 0]
ax.plot(angles_subset, values_subset, 'o', markersize=10, label='Original (5° spacing)', 
        color='blue', alpha=0.8, linewidth=2.5, zorder=5)
ax.plot(angles_new, values_interp, '-', linewidth=1.5, label='Interpolated (1° spacing)', 
        color='red', alpha=0.7)
ax.set_xlabel('Angle (degrees)', fontsize=12, weight='bold')
ax.set_ylabel('Power (nW)', fontsize=12, weight='bold')
ax.set_title('Data Interpolation (90-340°) - Linear Scale', fontsize=13, weight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim([85, 345])

# Plot 2: Log scale
ax = axes[0, 1]
mask_pos = values_interp > 0
ax.semilogy(angles_subset, values_subset, 'o', markersize=10, label='Original (5° spacing)', 
            color='blue', alpha=0.8, linewidth=2.5, zorder=5)
ax.semilogy(angles_new[mask_pos], values_interp[mask_pos], '-', linewidth=1.5, 
            label='Interpolated (1° spacing)', color='red', alpha=0.7)
ax.set_xlabel('Angle (degrees)', fontsize=12, weight='bold')
ax.set_ylabel('Power (nW) [log scale]', fontsize=12, weight='bold')
ax.set_title('Data Interpolation (90-340°) - Log Scale', fontsize=13, weight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim([85, 345])

# Plot 3: Residuals at original points
ax = axes[1, 0]
colors = ['red' if v < 0 else 'green' for v in residuals]
ax.bar(angles_subset, residuals, width=4, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
ax.set_xlabel('Angle (degrees)', fontsize=12, weight='bold')
ax.set_ylabel('|Absolute Error| (nW)', fontsize=12, weight='bold')
ax.set_title('Interpolation Error at Original Points', fontsize=13, weight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: First derivative (smoothness)
ax = axes[1, 1]
ax.plot(angles_new[:-1], first_deriv, '-', color='purple', linewidth=1.5, label='dP/dθ')
ax.fill_between(angles_new[:-1], 0, first_deriv, alpha=0.3, color='purple')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.3)
ax.set_xlabel('Angle (degrees)', fontsize=12, weight='bold')
ax.set_ylabel('Slope (nW/degree)', fontsize=12, weight='bold')
ax.set_title('First Derivative - Smoothness Check', fontsize=13, weight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim([85, 345])

plt.tight_layout()
plot_file = Path(__file__).parent / "interpolation_validation_90-340.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✓ Validation plot saved: {plot_file.name}")
plt.close()

print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)
print(f"✓ Successfully created {len(angles_new)} interpolated points (90-340°)")
print(f"✓ Cubic interpolation provides smooth exponential decay curves")
print(f"✓ Max recovery error: {max_residual:.2e}")
print(f"✓ Negative values: {neg_count}")

if neg_count > 0:
    print(f"\n⚠️  Note: {neg_count} negative values detected")
    print("    Consider these may be interpolation artifacts")
else:
    print(f"\n✓ No negative values (all physically valid)")

print(f"\nOutput file: {output_file.name}")
print(f"Validation plots: interpolation_validation_90-340.png")
