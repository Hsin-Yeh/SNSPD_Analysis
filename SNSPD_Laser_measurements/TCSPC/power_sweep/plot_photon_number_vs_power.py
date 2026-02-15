#!/usr/bin/env python3
"""
Plot photon number vs power for blocks 120-160.
Extract amplitudes for each photon number and show their relationship with power.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from math import factorial, exp

# Load fit results
results_file = Path('/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/74mV/peak_fits_75p7_limits/fit_results_summary.json')

with open(results_file, 'r') as f:
    results = json.load(f)

# Filter blocks 120-160
block_range = [120, 125, 130, 135, 140, 145, 150, 155, 160]

powers = []
amplitudes_n = [[], [], [], []]  # n=4, 3, 2, 1

for block_id in block_range:
    block_key = str(block_id)
    if block_key not in results:
        continue
    
    data = results[block_key]
    power = data['power_uw']
    amps = data['amplitudes']
    
    powers.append(power)
    for i in range(4):
        amplitudes_n[i].append(amps[i])

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

colors = ['purple', 'blue', 'green', 'red']
labels = ['n=4', 'n=3', 'n=2', 'n=1']

for i in range(4):
    ax.plot(powers, amplitudes_n[i], 'o-', color=colors[i], linewidth=2.5, 
            markersize=8, label=labels[i], alpha=0.8)

ax.set_xlabel('Optical Power (µW)', fontsize=13, weight='bold')
ax.set_ylabel('Peak Amplitude (counts)', fontsize=13, weight='bold')
ax.set_title('Photon Number Peak Amplitudes vs Power\n(Blocks 120-160)', 
             fontsize=14, weight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

plt.tight_layout()

output_dir = Path('/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/74mV/peak_fits_75p7_limits')
output_file = output_dir / 'photon_number_vs_power_120-160.png'
fig.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved plot: {output_file}")

# Also create a normalized version (probabilities)
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))

for i in range(4):
    # Normalize by total
    total = np.array(amplitudes_n[0]) + np.array(amplitudes_n[1]) + np.array(amplitudes_n[2]) + np.array(amplitudes_n[3])
    normalized = np.array(amplitudes_n[i]) / total
    ax2.plot(powers, normalized, 'o-', color=colors[i], linewidth=2.5, 
            markersize=8, label=labels[i], alpha=0.8)

ax2.set_xlabel('Optical Power (µW)', fontsize=13, weight='bold')
ax2.set_ylabel('Probability P(n)', fontsize=13, weight='bold')
ax2.set_title('Photon Number Probability vs Power\n(Blocks 120-160)', 
             fontsize=14, weight='bold')
ax2.legend(fontsize=12, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

plt.tight_layout()

output_file2 = output_dir / 'photon_probability_vs_power_120-160.png'
fig2.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"✓ Saved plot: {output_file2}")

# Print data table
print("\n" + "="*80)
print("PHOTON NUMBER DATA (Blocks 120-160)")
print("="*80)
print(f"{'Block':<8} {'Power (µW)':<15} {'n=4':<12} {'n=3':<12} {'n=2':<12} {'n=1':<12}")
print("-"*80)

for j, block_id in enumerate(block_range):
    block_key = str(block_id)
    if block_key not in results:
        continue
    
    data = results[block_key]
    power = data['power_uw']
    amps = data['amplitudes']
    
    print(f"{block_id:<8} {power:<15.4f} {amps[0]:<12.0f} {amps[1]:<12.0f} {amps[2]:<12.0f} {amps[3]:<12.0f}")

# ============================================================================
# POISSON MEAN ESTIMATION
# ============================================================================

def poisson_prob(n, lam):
    """Poisson probability P(n | lambda)"""
    return (lam**n / factorial(n)) * exp(-lam)

def estimate_poisson_mean(amplitudes):
    """
    Estimate Poisson mean from observed peak amplitudes.
    amplitudes: [A_n4, A_n3, A_n2, A_n1]
    Returns: estimated lambda
    """
    # Normalize amplitudes
    total = sum(amplitudes)
    if total == 0:
        return 0.0
    
    observed_probs = [A / total for A in amplitudes]
    
    # Minimize chi-square between observed and Poisson distribution
    def objective(lam):
        if lam <= 0:
            return 1e10
        predicted_probs = [poisson_prob(n, lam) for n in [4, 3, 2, 1]]
        # Normalize predicted (in case higher n contribute)
        pred_sum = sum(predicted_probs)
        if pred_sum == 0:
            return 1e10
        predicted_probs = [p / pred_sum for p in predicted_probs]
        
        chi_sq = sum((obs - pred)**2 / (pred + 1e-10) 
                     for obs, pred in zip(observed_probs, predicted_probs))
        return chi_sq
    
    # Initial guess: approximate from n=1 amplitude
    result = minimize(objective, x0=2.0, bounds=[(0.1, 10.0)], method='L-BFGS-B')
    return result.x[0] if result.success else 0.0

poisson_means = []
poisson_powers_list = []

for i, block_id in enumerate(block_range):
    block_key = str(block_id)
    if block_key not in results:
        continue
    
    data = results[block_key]
    amps = data['amplitudes']
    lam = estimate_poisson_mean(amps)
    poisson_means.append(lam)
    poisson_powers_list.append(powers[i])

# Plot Poisson mean vs power
fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
ax3.plot(poisson_powers_list, poisson_means, 'o-', markersize=10, linewidth=2.5, 
        color='darkblue', alpha=0.8, label='Estimated λ')

# Add linear fit
if len(poisson_powers_list) > 2:
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(poisson_powers_list, poisson_means)
    
    # Plot fit line
    fit_line = slope * np.array(poisson_powers_list) + intercept
    ax3.plot(poisson_powers_list, fit_line, '--', linewidth=2, color='red', alpha=0.7,
            label=f'Linear fit: λ = {slope:.3f}×P + {intercept:.3f} (R²={r_value**2:.4f})')

ax3.set_xlabel('Optical Power (µW)', fontsize=13, weight='bold')
ax3.set_ylabel('Estimated Poisson Mean (λ)', fontsize=13, weight='bold')
ax3.set_title('Estimated Photon Mean vs Power\n(Blocks 120-160)', 
             fontsize=14, weight='bold')
ax3.legend(fontsize=11, loc='best')
ax3.grid(True, alpha=0.3)

plt.tight_layout()

output_file3 = output_dir / 'poisson_mean_vs_power_120-160.png'
fig3.savefig(output_file3, dpi=150, bbox_inches='tight')
print(f"✓ Saved plot: {output_file3}")

# Print Poisson mean data
print("\n" + "="*80)
print("POISSON MEAN DATA (Blocks 120-160)")
print("="*80)
print(f"{'Block':<8} {'Power (µW)':<15} {'Estimated λ':<15}")
print("-"*80)

for i, block_id in enumerate(block_range):
    if i < len(poisson_means):
        print(f"{block_id:<8} {poisson_powers_list[i]:<15.4f} {poisson_means[i]:<15.4f}")

# ============================================================================
# OVERLAY COMPARISON: Observed vs Predicted Poisson Probabilities
# ============================================================================

# Create comparison plots for a few selected blocks
selected_blocks = [120, 135, 145, 160]
fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, block_id in enumerate(selected_blocks):
    ax = axes[idx]
    block_key = str(block_id)
    
    if block_key not in results:
        continue
    
    data = results[block_key]
    amps = data['amplitudes']
    power = data['power_uw']
    
    # Observed probabilities
    total = sum(amps)
    obs_probs = [A / total for A in amps]
    
    # Estimated lambda and predicted probabilities
    lam = estimate_poisson_mean(amps)
    pred_probs = [poisson_prob(n, lam) for n in [4, 3, 2, 1]]
    pred_sum = sum(pred_probs)
    pred_probs = [p / pred_sum for p in pred_probs]
    
    photon_numbers = [4, 3, 2, 1]
    x = np.arange(len(photon_numbers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, obs_probs, width, label='Observed', 
                   color='steelblue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, pred_probs, width, label=f'Poisson (λ={lam:.2f})', 
                   color='orange', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Photon Number n', fontsize=11, weight='bold')
    ax.set_ylabel('Probability P(n)', fontsize=11, weight='bold')
    ax.set_title(f'Block {block_id} @ {power:.2f} µW', fontsize=12, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(photon_numbers)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Observed vs Poisson-Predicted Photon Number Probabilities\n(Validation of λ Estimation)', 
             fontsize=14, weight='bold', y=0.995)
plt.tight_layout()

output_file4 = output_dir / 'poisson_validation_120-160.png'
fig4.savefig(output_file4, dpi=150, bbox_inches='tight')
print(f"✓ Saved plot: {output_file4}")

# Print validation data
print("\n" + "="*80)
print("POISSON FIT VALIDATION (Selected Blocks)")
print("="*80)
print(f"{'Block':<8} {'λ':<10} {'n=4 (obs)':<12} {'n=4 (pred)':<12} {'n=3 (obs)':<12} {'n=3 (pred)':<12}")
print("-"*80)

for block_id in selected_blocks:
    block_key = str(block_id)
    if block_key not in results:
        continue
    
    data = results[block_key]
    amps = data['amplitudes']
    total = sum(amps)
    obs_probs = [A / total for A in amps]
    
    lam = estimate_poisson_mean(amps)
    pred_probs = [poisson_prob(n, lam) for n in [4, 3, 2, 1]]
    pred_sum = sum(pred_probs)
    pred_probs = [p / pred_sum for p in pred_probs]
    
    print(f"{block_id:<8} {lam:<10.3f} {obs_probs[0]:<12.4f} {pred_probs[0]:<12.4f} {obs_probs[1]:<12.4f} {pred_probs[1]:<12.4f}")

plt.close('all')
print("\n✓ All plots complete!")
