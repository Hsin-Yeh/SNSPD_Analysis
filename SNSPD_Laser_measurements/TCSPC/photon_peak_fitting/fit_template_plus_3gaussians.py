#!/usr/bin/env python3
"""
Hybrid fitting: n=1 template + 3 Gaussians (n=2,3,4)
Uses the n=1 template created from low power data (blocks > 190)
Combined with 3 Gaussians for n=2, n=3, n=4 photon states
"""

import subprocess
import pickle
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Extract histograms in subprocess with timeout
code = """
import sys
sys.path.insert(0, '/Users/ya/Documents/Projects/SNSPD/SNSPD_Analysis/SNSPD_Laser_measurements/TCSPC')
import pickle
from read_phu import read_phu_file

phu_file = '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/sweep_power/SMSPD_3_2-7_500kHz_73mV_20260205_1213.phu'
header, histograms = read_phu_file(phu_file, verbose=False)

with open('/tmp/snspd_histograms.pkl', 'wb') as f:
    pickle.dump((header, histograms), f)

print('OK', flush=True)
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

# Load n=1 template
template_file = Path('/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/73mV/n1_spectrum_analysis/n1_template.npz')
template_data = np.load(template_file)
template_time = template_data['time']
template_normalized = template_data['template']
print(f"✓ Loaded n=1 template from {template_file}")
print(f"  Template time range: {template_time[0]:.2f}-{template_time[-1]:.2f} ns")
print(f"  Template resolution: {(template_time[1]-template_time[0])*1000:.2f} ps\n")

# Load powers
with open('/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/73mV/analysis_summary.json', 'r') as f:
    analysis_summary = json.load(f)
powers_uw = analysis_summary.get('plot1_data', {}).get('powers_uw', [])

resolution_s = header.get('MeasDesc_Resolution', 4e-12)
curve_indices = header.get('HistResDscr_CurveIndex', {})

# Reference parameters from block 160 (for n=2,3,4 Gaussians)
# n=1 is now template, so we use means 2,3,4
REF_MEANS_NS = np.array([75.35237035114503, 75.48166022478044, 75.64799999999998])  # n=2,3,4
REF_SIGMAS_NS = np.array([37.24041993238192, 49.4937827839825, 70.58657951667115]) / 1000.0  # ps to ns

# Tolerance settings
MEAN_TOL_FRACTION = 0.30
SIGMA_TOL_FRACTION = 0.30

# Parameters
T_MIN_NS = 75.0
T_MAX_NS = 79.0
TAIL_CUT_NS = 75.7
PLOT_MAX_NS = 76.0

# Hybrid model: template + 3 Gaussians
def hybrid_model(x, amp1, mu2, sigma2, A2, mu3, sigma3, A3, mu4, sigma4, A4):
    """
    amp1: amplitude of n=1 template
    mu2, sigma2, A2: n=2 Gaussian
    mu3, sigma3, A3: n=3 Gaussian
    mu4, sigma4, A4: n=4 Gaussian
    """
    # n=1: interpolate template
    template_interp = np.interp(x, template_time, template_normalized)
    result = amp1 * template_interp
    
    # n=2, n=3, n=4: Gaussians
    result += A2 * np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2))
    result += A3 * np.exp(-(x - mu3) ** 2 / (2 * sigma3 ** 2))
    result += A4 * np.exp(-(x - mu4) ** 2 / (2 * sigma4 ** 2))
    
    return result

results_all = {}
block_ids = []

print("=" * 80)
print("FITTING ALL BLOCKS: Template (n=1) + 3 Gaussians (n=2,3,4)")
print(f"Mean bounds: ±{MEAN_TOL_FRACTION*100:.0f}% of reference means")
print(f"Sigma bounds: ±{SIGMA_TOL_FRACTION*100:.0f}% of reference sigmas")
print("=" * 80)

for hist_idx, hist in enumerate(histograms):
    block_id = curve_indices.get(hist_idx, hist_idx)
    block_ids.append(block_id)
    
    # Get power
    if block_id < len(powers_uw):
        power_uw = powers_uw[block_id]
    else:
        power_uw = np.nan
    
    # Create time axis
    time_ns = np.arange(len(hist)) * resolution_s * 1e9
    
    # Use full spectrum - no tail cut
    mask = (time_ns >= T_MIN_NS) & (time_ns <= T_MAX_NS)
    x_data = time_ns[mask]
    y_data = hist[mask].astype(float)
    
    if len(x_data) == 0 or np.sum(y_data) == 0:
        results_all[block_id] = {
            'power_uw': power_uw,
            'fit_success': False,
            'error_message': 'No data in range'
        }
        continue
    
    # Initial guesses
    # n=1 template amplitude: estimate from peak in early region
    early_mask = (x_data >= 75.2) & (x_data <= 75.4)
    if np.any(early_mask):
        amp1_guess = np.max(y_data[early_mask])
    else:
        amp1_guess = np.max(y_data) * 0.3
    
    # n=2,3,4 Gaussians: use reference means and estimate amplitudes
    p0 = [
        amp1_guess,  # amp1
        REF_MEANS_NS[0], REF_SIGMAS_NS[0], np.max(y_data) * 0.3,  # n=2
        REF_MEANS_NS[1], REF_SIGMAS_NS[1], np.max(y_data) * 0.2,  # n=3
        REF_MEANS_NS[2], REF_SIGMAS_NS[2], np.max(y_data) * 0.1   # n=4
    ]
    
    # Bounds
    bounds_lower = [
        0,  # amp1 >= 0
        REF_MEANS_NS[0] * (1 - MEAN_TOL_FRACTION), REF_SIGMAS_NS[0] * (1 - SIGMA_TOL_FRACTION), 0,  # n=2
        REF_MEANS_NS[1] * (1 - MEAN_TOL_FRACTION), REF_SIGMAS_NS[1] * (1 - SIGMA_TOL_FRACTION), 0,  # n=3
        REF_MEANS_NS[2] * (1 - MEAN_TOL_FRACTION), REF_SIGMAS_NS[2] * (1 - SIGMA_TOL_FRACTION), 0   # n=4
    ]
    
    bounds_upper = [
        np.inf,  # amp1 no upper limit
        REF_MEANS_NS[0] * (1 + MEAN_TOL_FRACTION), REF_SIGMAS_NS[0] * (1 + SIGMA_TOL_FRACTION), np.inf,  # n=2
        REF_MEANS_NS[1] * (1 + MEAN_TOL_FRACTION), REF_SIGMAS_NS[1] * (1 + SIGMA_TOL_FRACTION), np.inf,  # n=3
        REF_MEANS_NS[2] * (1 + MEAN_TOL_FRACTION), REF_SIGMAS_NS[2] * (1 + SIGMA_TOL_FRACTION), np.inf   # n=4
    ]
    
    # Fit
    try:
        popt, pcov = curve_fit(
            hybrid_model, x_data, y_data,
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=10000
        )
        
        perr = np.sqrt(np.diag(pcov))
        
        # Extract parameters
        amp1 = popt[0]
        mu2, sigma2, A2 = popt[1:4]
        mu3, sigma3, A3 = popt[4:7]
        mu4, sigma4, A4 = popt[7:10]
        
        # Calculate R²
        y_fit = hybrid_model(x_data, *popt)
        ss_res = np.sum((y_data - y_fit) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        results_all[block_id] = {
            'power_uw': power_uw,
            'fit_success': True,
            'x_data': x_data,
            'y_data': y_data,
            'y_fit': y_fit,
            'popt': popt,
            'perr': perr,
            'amp1': amp1,
            'means': [mu2, mu3, mu4],
            'sigmas': [sigma2, sigma3, sigma4],
            'amplitudes': [A2, A3, A4],
            'r_squared': r_squared
        }
        
        print(f"Block {block_id:3d} @ {power_uw:7.4f} µW | "
              f"R²: {r_squared:.4f} | "
              f"A1: {amp1:6.0f} | "
              f"A2: {A2:6.0f} | "
              f"A3: {A3:6.0f} | "
              f"A4: {A4:6.0f}")
        
    except Exception as e:
        results_all[block_id] = {
            'power_uw': power_uw,
            'fit_success': False,
            'error_message': str(e)
        }
        print(f"Block {block_id:3d} @ {power_uw:7.4f} µW | FIT FAILED: {str(e)[:40]}")

# Count successful fits
successful_fits = sum(1 for r in results_all.values() if r.get('fit_success', False))
print(f"\n✓ Fitted {successful_fits}/{len(results_all)} blocks successfully")

# Save results
output_dir = Path('/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/73mV/template_plus_3gaussians')
output_dir.mkdir(parents=True, exist_ok=True)

# Plot individual fits for blocks < 200
print("\nPlotting individual fits for blocks < 200...")
for block_id, result in results_all.items():
    if not result.get('fit_success', False):
        continue
    
    if block_id >= 200:
        continue
    
    power_uw = result['power_uw']
    x_data = result['x_data']
    y_data = result['y_data']
    y_fit = result['y_fit']
    popt = result['popt']
    r_squared = result['r_squared']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data and fit
    ax.plot(x_data, y_data, 'o', markersize=3, alpha=0.5, label='Data')
    ax.plot(x_data, y_fit, 'r-', linewidth=2, label='Hybrid Fit')
    
    # Plot components
    # n=1 template
    template_interp = np.interp(x_data, template_time, template_normalized)
    ax.plot(x_data, popt[0] * template_interp, '--', alpha=0.7, label='n=1 (template)')
    
    # n=2,3,4 Gaussians
    mu2, sigma2, A2 = popt[1:4]
    mu3, sigma3, A3 = popt[4:7]
    mu4, sigma4, A4 = popt[7:10]
    
    ax.plot(x_data, A2 * np.exp(-(x_data - mu2)**2 / (2*sigma2**2)), '--', alpha=0.7, label='n=2')
    ax.plot(x_data, A3 * np.exp(-(x_data - mu3)**2 / (2*sigma3**2)), '--', alpha=0.7, label='n=3')
    ax.plot(x_data, A4 * np.exp(-(x_data - mu4)**2 / (2*sigma4**2)), '--', alpha=0.7, label='n=4')
    
    ax.set_xlim(T_MIN_NS, T_MAX_NS)
    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Counts', fontsize=12)
    ax.set_title(f'Block {block_id} | Power: {power_uw:.4f} µW | R²: {r_squared:.4f}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save
    plot_file = output_dir / f'fit_block_{block_id:03d}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

print(f"✓ Saved individual plots to {output_dir}")

# Create combined plot for blocks < 200
print("\nCreating combined plot for blocks < 200...")
successful_blocks = [bid for bid, r in results_all.items() if r.get('fit_success', False) and bid < 200]
successful_blocks.sort()

if len(successful_blocks) > 0:
    n_plots = len(successful_blocks)
    n_cols = 6
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, block_id in enumerate(successful_blocks):
        ax = axes[idx]
        result = results_all[block_id]
        
        x_data = result['x_data']
        y_data = result['y_data']
        y_fit = result['y_fit']
        power_uw = result['power_uw']
        r_squared = result['r_squared']
        
        ax.plot(x_data, y_data, 'o', markersize=2, alpha=0.5)
        ax.plot(x_data, y_fit, 'r-', linewidth=1.5)
        ax.set_xlim(T_MIN_NS, T_MAX_NS)
        ax.set_title(f'B{block_id} | {power_uw:.3f}µW | R²:{r_squared:.3f}', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(len(successful_blocks), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    combined_file = output_dir / 'combined_fits_blocks_lt200.png'
    plt.savefig(combined_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved combined plot: {combined_file}")

print(f"\n✓ Hybrid fitting complete!")
print(f"  Output directory: {output_dir}")
print(f"  Successful fits: {successful_fits}/{len(results_all)}")
