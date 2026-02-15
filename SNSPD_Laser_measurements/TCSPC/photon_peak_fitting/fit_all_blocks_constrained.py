#!/usr/bin/env python3
"""
Refit all blocks with tighter constraints based on block 145 reference parameters.
- Display data up to 76.0 ns
- Fit until 75.7 ns  
- Use block 145 fitted parameters as reference with tolerance bands
"""

import subprocess
import pickle
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Extract histograms
code = """
import sys
sys.path.insert(0, '/Users/ya/Documents/Projects/SNSPD/SNSPD_Analysis/SNSPD_Laser_measurements/TCSPC')
import pickle
from read_phu import read_phu_file

phu_file = '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/sweep_power/SMSPD_3_2-7_500kHz_74mV_20260205_0102.phu'
header, histograms = read_phu_file(phu_file, verbose=False)

with open('/tmp/snspd_histograms.pkl', 'wb') as f:
    pickle.dump((header, histograms), f)

print("OK", flush=True)
"""

print("Loading histograms...")
try:
    result = subprocess.run(['/usr/bin/python3', '-c', code], 
                          capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)
except subprocess.TimeoutExpired:
    print("ERROR: Subprocess timed out")
    sys.exit(1)

with open('/tmp/snspd_histograms.pkl', 'rb') as f:
    header, histograms = pickle.load(f)

print(f"✓ Loaded {len(histograms)} histograms\n")

# Multi-Gaussian function
def multi_gaussian(x, *params):
    result = np.zeros_like(x, dtype=float)
    n_peaks = len(params) // 3
    for i in range(n_peaks):
        mu = params[3*i]
        sigma = params[3*i + 1]
        A = params[3*i + 2]
        result += A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return result

def single_gaussian(x, mu, sigma, A):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Block 145 reference parameters (from previous fit with 75.7 ns cut)
# Block 145 @ 2.6200 µW: σ: 25.10, 41.06, 50.20, 80.86 ps | μ: 75.2356, 75.3386, 75.4688, 75.5880 ns
BLOCK_145_SIGMAS = np.array([25.10, 41.06, 50.20, 80.86])  # ps
BLOCK_145_MUS = np.array([75.2356, 75.3386, 75.4688, 75.5880])  # ns

# Tolerance bands for constraints
MU_TOLERANCE_NS = 0.050  # ±50 ps tolerance on means (allows for power-dependent shifts)
SIGMA_TOLERANCE_FACTOR = 1.0  # Allow ±100% variation from block 145 widths (very loose)

print("="*80)
print("REFITTING ALL BLOCKS with Block 145 as Reference Constraints")
print("="*80)
print(f"\nBlock 145 Reference Parameters (2.62 µW):")
print(f"  n=4: μ=75.2356 ns, σ=25.10 ps")
print(f"  n=3: μ=75.3386 ns, σ=41.06 ps")
print(f"  n=2: μ=75.4688 ns, σ=50.20 ps")
print(f"  n=1: μ=75.5880 ns, σ=80.86 ps")
print(f"\nConstraint Bands:")
print(f"  Mean: ±{MU_TOLERANCE_NS*1000:.0f} ps from reference")
print(f"  Width: ±{SIGMA_TOLERANCE_FACTOR*100:.0f}% from block 145 values")
print("\n" + "="*80 + "\n")

# Load powers
with open('/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/74mV/analysis_summary.json', 'r') as f:
    analysis_summary = json.load(f)
powers_uw = analysis_summary.get('plot1_data', {}).get('powers_uw', [])

resolution_s = header.get('MeasDesc_Resolution', 4e-12)
curve_indices = header.get('HistResDscr_CurveIndex', {})

# Parameters
T_MIN_NS = 75.0
T_MAX_NS = 79.0
cut_time_ns = 76.2
tail_cut_ns = 75.7
plot_max_ns = 76.0

results_all = {}
block_ids = []

# Fit all blocks
for block_idx in range(len(histograms)):
    hist = histograms[block_idx]
    block_id = curve_indices.get(block_idx, block_idx)
    
    if block_id == 0:
        continue
    
    block_ids.append(block_id)
    
    power_idx = sum(1 for i in range(block_idx) if curve_indices.get(i, i) != 0)
    power_uw = powers_uw[power_idx] if power_idx < len(powers_uw) else 0.0
    
    # Extract ROI
    bin_min = int(T_MIN_NS * 1e-9 / resolution_s)
    bin_max = int(T_MAX_NS * 1e-9 / resolution_s)
    
    if bin_max > len(hist):
        bin_max = len(hist)
    
    hist_roi = hist[bin_min:bin_max]
    cut_bin = int(cut_time_ns * 1e-9 / resolution_s)
    tail_cut_bin = int(tail_cut_ns * 1e-9 / resolution_s)
    cut_bin_roi = cut_bin - bin_min
    tail_cut_bin_roi = tail_cut_bin - bin_min
    
    # Extract main peak
    hist_before = hist_roi[:cut_bin_roi]
    time_bins_full = np.arange(len(hist_roi)) * resolution_s * 1e9 + (bin_min * resolution_s * 1e9)
    time_before = time_bins_full[:cut_bin_roi]
    
    # Trim at tail_cut
    hist_trimmed = hist_before[:tail_cut_bin_roi]
    time_trimmed = time_before[:tail_cut_bin_roi]
    
    # Initial parameters using block 145 reference
    initial_params = []
    bounds_lower = []
    bounds_upper = []
    
    A_guesses = []
    for mu_ref in BLOCK_145_MUS:
        idx_closest = np.argmin(np.abs(time_trimmed - mu_ref))
        A_guesses.append(max(hist_trimmed[idx_closest], 100))
    
    for i, (mu_ref, sigma_ref) in enumerate(zip(BLOCK_145_MUS, BLOCK_145_SIGMAS)):
        sigma_init = sigma_ref * 1e-12  # Convert ps to seconds, keep in ns for fitting
        A_init = A_guesses[i]
        
        initial_params.extend([mu_ref, sigma_init, A_init])
        
        # Tighter bounds based on block 145 (in nanoseconds)
        sigma_min = sigma_ref * (1 - SIGMA_TOLERANCE_FACTOR) * 1e-12
        sigma_max = sigma_ref * (1 + SIGMA_TOLERANCE_FACTOR) * 1e-12
        
        bounds_lower.extend([mu_ref - MU_TOLERANCE_NS, sigma_min, A_init * 0.2])
        bounds_upper.extend([mu_ref + MU_TOLERANCE_NS, sigma_max, A_init * 3.0])
    
    # Fit
    try:
        popt, pcov = curve_fit(multi_gaussian, time_trimmed, hist_trimmed,
                             p0=initial_params, bounds=(bounds_lower, bounds_upper),
                             maxfev=50000)
        
        # Extract results
        fit_results = {}
        for i in range(4):
            mu = popt[3*i]
            sigma = popt[3*i + 1]
            A = popt[3*i + 2]
            fit_results[i] = {
                'mu': mu,
                'sigma': sigma,
                'A': A
            }
        
        y_pred = multi_gaussian(time_trimmed, *popt)
        ss_res = np.sum((hist_trimmed - y_pred)**2)
        ss_tot = np.sum((hist_trimmed - np.mean(hist_trimmed))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        results_all[block_id] = {
            'power': power_uw,
            'r_squared': r_squared,
            'params': fit_results,
            'popt': popt,
            'time_trimmed': time_trimmed,
            'hist_trimmed': hist_trimmed
        }
        
        # Print progress
        sigmas = [fit_results[i]['sigma']*1e12 for i in range(4)]  # Convert from ns to ps
        print(f"Block {block_id:3d} @ {power_uw:7.4f} µW | σ: {sigmas[0]:6.2f}, {sigmas[1]:6.2f}, {sigmas[2]:6.2f}, {sigmas[3]:6.2f} ps | R²: {r_squared:.4f}")
        
    except Exception as e:
        print(f"Block {block_id:3d} @ {power_uw:7.4f} µW | FIT FAILED: {str(e)[:40]}")
        continue

print("\n" + "="*80)
print(f"Successfully fitted {len(results_all)} blocks with tighter constraints")
print("="*80)

# Generate plots for each block
output_dir = Path('/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/74mV/peak_fits_constrained')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nGenerating plots in: {output_dir}")

for block_id in sorted(results_all.keys()):
    data = results_all[block_id]
    popt = data['popt']
    time_trimmed = data['time_trimmed']
    hist_trimmed = data['hist_trimmed']
    power_uw = data['power']
    r_squared = data['r_squared']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Get plot range up to 76.0 ns
    plot_end_bin = np.searchsorted(time_trimmed, plot_max_ns)
    if plot_end_bin == 0:
        plot_end_bin = len(time_trimmed)
    
    time_plot = time_trimmed[:plot_end_bin]
    hist_plot = hist_trimmed[:plot_end_bin]
    y_pred_plot = multi_gaussian(time_plot, *popt)
    
    # Plot 1: Data with fit
    ax1.plot(time_plot, hist_plot, 'o-', markersize=3, linewidth=1.2, 
             label='Data', color='steelblue', alpha=0.8)
    ax1.plot(time_plot, y_pred_plot, '-', linewidth=2.5, label='Multi-Gaussian Fit', 
             color='red', alpha=0.9)
    
    # Individual components
    colors_comp = plt.cm.Set2(np.linspace(0, 1, 4))
    for i in range(4):
        mu = popt[3*i]
        sigma = popt[3*i + 1]
        A = popt[3*i + 2]
        component = single_gaussian(time_plot, mu, sigma, A)
        n_label = 4 - i
        ax1.plot(time_plot, component, '--', linewidth=1.8, alpha=0.5, 
                 color=colors_comp[i], label=f'n={n_label}')
    
    # Mark tail cut
    ax1.axvline(75.7, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Tail cut (75.7 ns)')
    
    ax1.set_xlabel('TOA (ns)', fontsize=12, weight='bold')
    ax1.set_ylabel('Counts', fontsize=12, weight='bold')
    ax1.set_title(f'Block {block_id} @ {power_uw:.4f} µW | R² = {r_squared:.6f} | Constrained Fit', 
                  fontsize=13, weight='bold')
    ax1.legend(fontsize=9, loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([time_plot[0], plot_max_ns])
    
    # Plot 2: Residuals
    residuals = hist_plot - y_pred_plot
    ax2.plot(time_plot, residuals, 'o-', markersize=3, linewidth=1.2, 
             color='darkgreen', alpha=0.7)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.axvline(75.7, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax2.set_xlabel('TOA (ns)', fontsize=12, weight='bold')
    ax2.set_ylabel('Residuals', fontsize=12, weight='bold')
    ax2.set_title('Fit Residuals', fontsize=11, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([time_plot[0], plot_max_ns])
    
    plt.tight_layout()
    output_file = output_dir / f'block_{block_id:03d}_constrained.png'
    fig.savefig(output_file, dpi=120, bbox_inches='tight')
    plt.close(fig)

print(f"✓ Generated {len(results_all)} plots")

# Save summary data
summary_dict = {}
for block_id in sorted(results_all.keys()):
    data = results_all[block_id]
    summary_dict[str(block_id)] = {
        'power_uw': data['power'],
        'r_squared': data['r_squared'],
        'sigmas_ps': [data['params'][i]['sigma']*1e12 for i in range(4)],
        'mus_ns': [data['params'][i]['mu'] for i in range(4)],
        'amplitudes': [data['params'][i]['A'] for i in range(4)]
    }

summary_file = output_dir / 'fit_results_constrained.json'
with open(summary_file, 'w') as f:
    json.dump(summary_dict, f, indent=2)

print(f"✓ Saved summary to: {summary_file}")

# Create summary table
print("\n" + "="*100)
print("PEAK WIDTH SUMMARY - CONSTRAINED FITTING (Block 145 Reference)")
print("="*100)
print("Block | Power (µW) |    n=4    |    n=3    |    n=2    |    n=1    |   R²")
print("------|------------|-----------|-----------|-----------|-----------|----------")

for block_id in sorted(results_all.keys()):
    data = results_all[block_id]
    sigmas = [data['params'][i]['sigma']*1e12 for i in range(4)]
    print(f"{block_id:5d} | {data['power']:9.4f} | {sigmas[0]:9.2f} | {sigmas[1]:9.2f} | {sigmas[2]:9.2f} | {sigmas[3]:9.2f} | {data['r_squared']:.4f}")

print("="*100)
print("\n✓ All fits complete with Block 145 constraints!")
