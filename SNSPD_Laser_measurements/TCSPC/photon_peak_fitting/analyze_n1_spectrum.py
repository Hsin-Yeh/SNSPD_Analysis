#!/usr/bin/env python3
"""
Analyze n=1 photon spectrum for low power blocks (> 190)
1. Estimate dark count distribution from OOT region (80-100ns)
2. Subtract dark counts to get n=1 spectrum
3. Normalize and compare across blocks
"""

import subprocess
import pickle
import json
import numpy as np
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

# Load powers
with open('/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/73mV/analysis_summary.json', 'r') as f:
    analysis_summary = json.load(f)
powers_uw = analysis_summary.get('plot1_data', {}).get('powers_uw', [])

resolution_s = header.get('MeasDesc_Resolution', 4e-12)
curve_indices = header.get('HistResDscr_CurveIndex', {})
measurement_time_s = 30.0  # 30 seconds per measurement

# Time regions
SIGNAL_START_NS = 75.5
SIGNAL_END_NS = 78.0
OOT_START_NS = 80.0
OOT_END_NS = 100.0

# Output directory
output_dir = Path('/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/73mV/n1_spectrum_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ANALYZING N=1 SPECTRUM FOR BLOCKS > 190")
print(f"Signal region: {SIGNAL_START_NS}-{SIGNAL_END_NS} ns")
print(f"OOT region: {OOT_START_NS}-{OOT_END_NS} ns")
print("=" * 80)

# Store results
results = {}

for block_idx in range(len(histograms)):
    hist = histograms[block_idx]
    block_id = curve_indices.get(block_idx, block_idx)

    if block_id == 0 or block_id <= 190:
        continue

    power_idx = sum(1 for i in range(block_idx) if curve_indices.get(i, i) != 0)
    power_uw = powers_uw[power_idx] if power_idx < len(powers_uw) else 0.0

    # Convert time to bins
    signal_bin_start = int(SIGNAL_START_NS * 1e-9 / resolution_s)
    signal_bin_end = int(SIGNAL_END_NS * 1e-9 / resolution_s)
    oot_bin_start = int(OOT_START_NS * 1e-9 / resolution_s)
    oot_bin_end = int(OOT_END_NS * 1e-9 / resolution_s)

    # Extract signal region
    signal_hist = hist[signal_bin_start:signal_bin_end]
    signal_time = np.arange(len(signal_hist)) * resolution_s * 1e9 + SIGNAL_START_NS

    # Extract OOT region (for dark count estimation)
    oot_hist = hist[oot_bin_start:oot_bin_end]
    oot_time = np.arange(len(oot_hist)) * resolution_s * 1e9 + OOT_START_NS

    # Calculate dark count rate from OOT region
    signal_width_ns = SIGNAL_END_NS - SIGNAL_START_NS
    oot_width_ns = OOT_END_NS - OOT_START_NS
    
    # Total counts in each region
    total_signal_counts = np.sum(signal_hist)
    total_oot_counts = np.sum(oot_hist)
    
    # Scale OOT to signal window size
    dark_counts_scaled = total_oot_counts * (signal_width_ns / oot_width_ns)
    
    # Average dark count per bin
    avg_dark_per_bin = np.mean(oot_hist)
    
    # NO DARK COUNT SUBTRACTION - use raw signal
    signal_corrected = signal_hist.astype(float)  # Keep as-is, just convert to float for consistency
    
    # Normalize to peak = 1
    if np.max(signal_corrected) > 0:
        signal_normalized = signal_corrected / np.max(signal_corrected)
    else:
        signal_normalized = signal_corrected
    
    # Store results
    results[block_id] = {
        'power_uw': power_uw,
        'signal_time': signal_time,
        'signal_hist': signal_hist,
        'signal_corrected': signal_corrected,
        'signal_normalized': signal_normalized,
        'oot_hist': oot_hist,
        'oot_time': oot_time,
        'total_signal_counts': total_signal_counts,
        'total_oot_counts': total_oot_counts,
        'dark_counts_scaled': dark_counts_scaled,
        'avg_dark_per_bin': avg_dark_per_bin
    }
    
    print(f"Block {block_id:3d} @ {power_uw:7.4f} µW | "
          f"Signal: {total_signal_counts:6.0f} counts | "
          f"OOT: {total_oot_counts:5.0f} counts | "
          f"Dark (scaled): {dark_counts_scaled:5.1f}")

print(f"\n✓ Analyzed {len(results)} blocks")

# ============================================================================
# PLOT 1: Individual plots for each block
# ============================================================================

print("\nGenerating individual plots...")

for block_id in sorted(results.keys()):
    data = results[block_id]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Raw signal histogram
    ax = axes[0, 0]
    ax.plot(data['signal_time'], data['signal_hist'], 'o-', markersize=2, 
            linewidth=1, color='steelblue', alpha=0.8)
    ax.set_xlabel('Time (ns)', fontsize=11, weight='bold')
    ax.set_ylabel('Counts', fontsize=11, weight='bold')
    ax.set_title(f'Raw Signal - Block {block_id} ({data["power_uw"]:.4f} µW)', 
                fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: OOT region (dark counts)
    ax = axes[0, 1]
    ax.plot(data['oot_time'], data['oot_hist'], 'o-', markersize=2, 
            linewidth=1, color='darkred', alpha=0.8)
    ax.axhline(data['avg_dark_per_bin'], color='red', linestyle='--', 
              linewidth=2, label=f'Avg: {data["avg_dark_per_bin"]:.2f}')
    ax.set_xlabel('Time (ns)', fontsize=11, weight='bold')
    ax.set_ylabel('Counts', fontsize=11, weight='bold')
    ax.set_title(f'OOT Region (Dark Counts)', fontsize=12, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Dark-subtracted signal
    ax = axes[1, 0]
    ax.plot(data['signal_time'], data['signal_corrected'], 'o-', markersize=2, 
            linewidth=1, color='green', alpha=0.8)
    ax.set_xlabel('Time (ns)', fontsize=11, weight='bold')
    ax.set_ylabel('Counts (dark-subtracted)', fontsize=11, weight='bold')
    ax.set_title(f'Dark-Subtracted n=1 Spectrum', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Normalized spectrum
    ax = axes[1, 1]
    ax.plot(data['signal_time'], data['signal_normalized'], 'o-', markersize=2, 
            linewidth=1.5, color='purple', alpha=0.8)
    ax.set_xlabel('Time (ns)', fontsize=11, weight='bold')
    ax.set_ylabel('Normalized Intensity', fontsize=11, weight='bold')
    ax.set_title(f'Normalized n=1 Spectrum', fontsize=12, weight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / f'block_{block_id:03d}_n1_analysis.png'
    fig.savefig(plot_file, dpi=120, bbox_inches='tight')
    plt.close(fig)

print(f"✓ Saved {len(results)} individual plots")

# ============================================================================
# PLOT 2: Combined normalized comparison
# ============================================================================

print("\nGenerating combined comparison plot...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Sort by block ID
sorted_blocks = sorted(results.items(), key=lambda x: x[0])

# Color map
colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_blocks)))

# Plot 1: All normalized spectra overlaid
ax = axes[0, 0]
for i, (block_id, data) in enumerate(sorted_blocks):
    ax.plot(data['signal_time'], data['signal_normalized'], '-', 
            linewidth=1.5, alpha=0.7, color=colors[i],
            label=f"Block {block_id} ({data['power_uw']:.4f} µW)")
ax.set_xlabel('Time (ns)', fontsize=12, weight='bold')
ax.set_ylabel('Normalized Intensity', fontsize=12, weight='bold')
ax.set_title('Normalized n=1 Spectra - All Blocks > 190', fontsize=13, weight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='upper right', ncol=2)

# Plot 2: Log scale
ax = axes[0, 1]
for i, (block_id, data) in enumerate(sorted_blocks):
    # Avoid log(0)
    signal_log = np.maximum(data['signal_normalized'], 1e-4)
    ax.semilogy(data['signal_time'], signal_log, '-', 
                linewidth=1.5, alpha=0.7, color=colors[i],
                label=f"Block {block_id}")
ax.set_xlabel('Time (ns)', fontsize=12, weight='bold')
ax.set_ylabel('Normalized Intensity (log)', fontsize=12, weight='bold')
ax.set_title('Normalized n=1 Spectra (Log Scale)', fontsize=13, weight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='upper right', ncol=2)

# Plot 3: Dark count statistics
ax = axes[1, 0]
block_ids = [bid for bid, _ in sorted_blocks]
dark_rates = [data['dark_counts_scaled'] for _, data in sorted_blocks]
ax.bar(range(len(block_ids)), dark_rates, color='darkred', alpha=0.7)
ax.set_xlabel('Block ID', fontsize=12, weight='bold')
ax.set_ylabel('Dark Counts (scaled to signal window)', fontsize=12, weight='bold')
ax.set_title('Dark Count Distribution by Block', fontsize=13, weight='bold')
ax.set_xticks(range(len(block_ids)))
ax.set_xticklabels(block_ids, rotation=45)
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Peak position analysis
ax = axes[1, 1]
peak_positions = []
peak_powers = []
for block_id, data in sorted_blocks:
    if np.max(data['signal_normalized']) > 0.5:
        peak_idx = np.argmax(data['signal_normalized'])
        peak_time = data['signal_time'][peak_idx]
        peak_positions.append(peak_time)
        peak_powers.append(data['power_uw'])
    else:
        peak_positions.append(np.nan)
        peak_powers.append(data['power_uw'])

ax.plot(peak_powers, peak_positions, 'o-', markersize=8, linewidth=2, color='darkblue')
ax.set_xlabel('Optical Power (µW)', fontsize=12, weight='bold')
ax.set_ylabel('Peak Position (ns)', fontsize=12, weight='bold')
ax.set_title('n=1 Peak Position vs Power', fontsize=13, weight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
combined_file = output_dir / 'combined_n1_comparison.png'
fig.savefig(combined_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"✓ Saved combined comparison plot: {combined_file}")

# ============================================================================
# SAVE DATA
# ============================================================================

print("\nSaving analysis data...")

# Save summary JSON
summary = {}
for block_id, data in results.items():
    summary[str(block_id)] = {
        'power_uw': float(data['power_uw']),
        'total_signal_counts': int(data['total_signal_counts']),
        'total_oot_counts': int(data['total_oot_counts']),
        'dark_counts_scaled': float(data['dark_counts_scaled']),
        'avg_dark_per_bin': float(data['avg_dark_per_bin']),
        'signal_time': data['signal_time'].tolist(),
        'signal_normalized': data['signal_normalized'].tolist()
    }

summary_file = output_dir / 'n1_spectrum_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Saved summary: {summary_file}")

# Save text report
report_file = output_dir / 'n1_spectrum_report.txt'
with open(report_file, 'w') as f:
    f.write("# n=1 Spectrum Analysis for Blocks > 190\n")
    f.write("# Dark count subtraction using OOT region (80-100 ns)\n\n")
    f.write(f"{'Block':<8} {'Power(µW)':<12} {'Signal':<10} {'OOT':<10} {'Dark(scaled)':<12} {'SNR':<10}\n")
    f.write("-" * 70 + "\n")
    for block_id, data in sorted(results.items()):
        snr = data['total_signal_counts'] / data['dark_counts_scaled'] if data['dark_counts_scaled'] > 0 else 0
        f.write(f"{block_id:<8} {data['power_uw']:<12.6f} {data['total_signal_counts']:<10.0f} "
               f"{data['total_oot_counts']:<10.0f} {data['dark_counts_scaled']:<12.1f} {snr:<10.2f}\n")

print(f"✓ Saved report: {report_file}")

# ============================================================================
# CREATE SUMMED TEMPLATE AND FIT ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CREATING SUMMED TEMPLATE AND FITTING")
print("=" * 80)

# Sum all dark-subtracted spectra to create template
template_sum = None
for block_id, data in results.items():
    if template_sum is None:
        template_sum = data['signal_corrected'].copy()
    else:
        template_sum += data['signal_corrected']

# Normalize template
template_normalized = template_sum / np.max(template_sum)

# Use time axis from first block (all should be the same)
template_time = results[sorted(results.keys())[0]]['signal_time']

print(f"✓ Created summed template from {len(results)} blocks")
print(f"  Template peak: {np.max(template_sum):.1f} counts")

# Save template to file
template_file = output_dir / 'n1_template.npz'
np.savez(template_file, 
         time=template_time, 
         template=template_normalized,
         template_sum=template_sum,
         signal_start_ns=SIGNAL_START_NS,
         signal_end_ns=SIGNAL_END_NS,
         resolution_s=resolution_s)
print(f"✓ Saved template: {template_file}")

# Fit each block with the template
from scipy.optimize import curve_fit

def template_fit_func(x, amplitude, offset):
    """Template fitting function: amplitude * template + offset"""
    # Interpolate template to match x values
    template_interp = np.interp(x, template_time, template_normalized)
    return amplitude * template_interp + offset

fit_results = {}
for block_id, data in sorted(results.items()):
    try:
        # Fit dark-subtracted data with template
        popt, pcov = curve_fit(
            template_fit_func,
            data['signal_time'],
            data['signal_corrected'],
            p0=[np.max(data['signal_corrected']), 0],
            bounds=([0, -100], [1e6, 1000])
        )
        
        amplitude, offset = popt
        amplitude_err = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else 0
        
        # Calculate R²
        y_pred = template_fit_func(data['signal_time'], *popt)
        ss_res = np.sum((data['signal_corrected'] - y_pred) ** 2)
        ss_tot = np.sum((data['signal_corrected'] - np.mean(data['signal_corrected'])) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        fit_results[block_id] = {
            'power_uw': data['power_uw'],
            'amplitude': amplitude,
            'amplitude_err': amplitude_err,
            'offset': offset,
            'r_squared': r_squared,
            'y_fit': y_pred
        }
        
        print(f"Block {block_id:3d} @ {data['power_uw']:7.4f} µW | "
              f"Amplitude: {amplitude:8.2f} ± {amplitude_err:6.2f} | "
              f"R²: {r_squared:.4f}")
        
    except Exception as e:
        print(f"Block {block_id:3d} @ {data['power_uw']:7.4f} µW | FIT FAILED: {str(e)[:40]}")

print(f"\n✓ Fitted {len(fit_results)}/{len(results)} blocks with template")

# ============================================================================
# PLOT TEMPLATE FIT RESULTS
# ============================================================================

print("\nGenerating template fit analysis plots...")

# Plot 1: Template and example fits
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Show template
ax = axes[0, 0]
ax.plot(template_time, template_sum, 'k-', linewidth=2, label='Summed template')
ax.set_xlabel('Time (ns)', fontsize=11, weight='bold')
ax.set_ylabel('Counts', fontsize=11, weight='bold')
ax.set_title('Summed n=1 Template (all blocks)', fontsize=12, weight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[0, 1]
ax.plot(template_time, template_normalized, 'k-', linewidth=2, label='Normalized template')
ax.set_xlabel('Time (ns)', fontsize=11, weight='bold')
ax.set_ylabel('Normalized intensity', fontsize=11, weight='bold')
ax.set_title('Normalized Template', fontsize=12, weight='bold')
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3)
ax.legend()

# Show 4 example fits
example_blocks = sorted(fit_results.keys())
example_indices = [0, len(example_blocks)//3, 2*len(example_blocks)//3, -1]
axes_flat = [axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]

for idx, ax in zip(example_indices, axes_flat):
    block_id = example_blocks[idx]
    data = results[block_id]
    fit_data = fit_results[block_id]
    
    ax.plot(data['signal_time'], data['signal_corrected'], 'o', 
            markersize=3, color='steelblue', alpha=0.7, label='Data')
    ax.plot(data['signal_time'], fit_data['y_fit'], '-', 
            linewidth=2, color='red', label='Template fit')
    ax.set_xlabel('Time (ns)', fontsize=10, weight='bold')
    ax.set_ylabel('Counts', fontsize=10, weight='bold')
    ax.set_title(f"Block {block_id} ({fit_data['power_uw']:.4f} µW)\n"
                f"A={fit_data['amplitude']:.1f}, R²={fit_data['r_squared']:.3f}", 
                fontsize=11, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
template_fit_file = output_dir / 'template_fit_examples.png'
fig.savefig(template_fit_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"✓ Saved template fit examples: {template_fit_file}")

# Plot 2: Amplitude vs Power
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sorted_fit_blocks = sorted(fit_results.items(), key=lambda x: x[1]['power_uw'])
powers = [data['power_uw'] for _, data in sorted_fit_blocks]
amplitudes = [data['amplitude'] for _, data in sorted_fit_blocks]
amp_errors = [data['amplitude_err'] for _, data in sorted_fit_blocks]

# Linear scale
ax = axes[0, 0]
ax.errorbar(powers, amplitudes, yerr=amp_errors, fmt='o-', 
            markersize=6, capsize=3, linewidth=1.5, color='darkblue', alpha=0.7)
ax.set_xlabel('Optical Power (µW)', fontsize=12, weight='bold')
ax.set_ylabel('Fitted Amplitude', fontsize=12, weight='bold')
ax.set_title('Template Fit Amplitude vs Power', fontsize=13, weight='bold')
ax.grid(True, alpha=0.3)

# Log-log scale
ax = axes[0, 1]
ax.errorbar(powers, amplitudes, yerr=amp_errors, fmt='o', 
            markersize=6, capsize=3, linewidth=1.5, color='darkblue', alpha=0.7)
ax.set_xlabel('Optical Power (µW)', fontsize=12, weight='bold')
ax.set_ylabel('Fitted Amplitude', fontsize=12, weight='bold')
ax.set_title('Template Fit Amplitude vs Power (Log-Log)', fontsize=13, weight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# Power-law fit on log-log
from scipy.stats import linregress
log_powers = np.log10(powers)
log_amps = np.log10(amplitudes)
slope, intercept, r_value, p_value, std_err = linregress(log_powers, log_amps)

fit_powers = np.logspace(np.log10(min(powers)), np.log10(max(powers)), 100)
fit_amps = 10**(slope * np.log10(fit_powers) + intercept)
ax.plot(fit_powers, fit_amps, '--', linewidth=2, color='red', alpha=0.8,
        label=f'Power law: A ∝ P^{slope:.3f}\nR² = {r_value**2:.4f}')
ax.legend(fontsize=11)

print(f"\nPower-law fit: Amplitude ∝ Power^{slope:.3f} (R² = {r_value**2:.4f})")

# R² vs Power
ax = axes[1, 0]
r_squareds = [data['r_squared'] for _, data in sorted_fit_blocks]
ax.plot(powers, r_squareds, 'o-', markersize=6, linewidth=1.5, color='green', alpha=0.7)
ax.set_xlabel('Optical Power (µW)', fontsize=12, weight='bold')
ax.set_ylabel('R² (fit quality)', fontsize=12, weight='bold')
ax.set_title('Template Fit Quality vs Power', fontsize=13, weight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

# Relative error vs Power
ax = axes[1, 1]
rel_errors = [data['amplitude_err'] / data['amplitude'] * 100 
              for _, data in sorted_fit_blocks if data['amplitude'] > 0]
powers_valid = [data['power_uw'] for _, data in sorted_fit_blocks if data['amplitude'] > 0]
ax.plot(powers_valid, rel_errors, 'o-', markersize=6, linewidth=1.5, color='orange', alpha=0.7)
ax.set_xlabel('Optical Power (µW)', fontsize=12, weight='bold')
ax.set_ylabel('Relative Error (%)', fontsize=12, weight='bold')
ax.set_title('Template Fit Amplitude Error vs Power', fontsize=13, weight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
amplitude_plot_file = output_dir / 'template_amplitude_vs_power.png'
fig.savefig(amplitude_plot_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"✓ Saved amplitude analysis: {amplitude_plot_file}")

# Save template fit data
template_fit_summary = {
    'template_time': template_time.tolist(),
    'template_sum': template_sum.tolist(),
    'template_normalized': template_normalized.tolist(),
    'fit_results': {}
}

for block_id, fit_data in fit_results.items():
    template_fit_summary['fit_results'][str(block_id)] = {
        'power_uw': float(fit_data['power_uw']),
        'amplitude': float(fit_data['amplitude']),
        'amplitude_err': float(fit_data['amplitude_err']),
        'offset': float(fit_data['offset']),
        'r_squared': float(fit_data['r_squared'])
    }

template_fit_summary['power_law_fit'] = {
    'slope': float(slope),
    'intercept': float(intercept),
    'r_squared': float(r_value**2)
}

template_summary_file = output_dir / 'template_fit_summary.json'
with open(template_summary_file, 'w') as f:
    json.dump(template_fit_summary, f, indent=2)
print(f"✓ Saved template fit summary: {template_summary_file}")

# Save text report
template_report_file = output_dir / 'template_fit_report.txt'
with open(template_report_file, 'w') as f:
    f.write("# Template Fit Analysis for n=1 Spectrum\n")
    f.write("# Fitting dark-subtracted spectra with normalized summed template\n\n")
    f.write(f"Power-law fit: Amplitude ∝ Power^{slope:.3f} (R² = {r_value**2:.4f})\n\n")
    f.write(f"{'Block':<8} {'Power(µW)':<12} {'Amplitude':<12} {'Error':<10} {'Offset':<10} {'R²':<10}\n")
    f.write("-" * 70 + "\n")
    for block_id, fit_data in sorted(fit_results.items()):
        f.write(f"{block_id:<8} {fit_data['power_uw']:<12.6f} {fit_data['amplitude']:<12.2f} "
               f"{fit_data['amplitude_err']:<10.2f} {fit_data['offset']:<10.2f} {fit_data['r_squared']:<10.4f}\n")

print(f"✓ Saved template fit report: {template_report_file}")

print("\n✓ n=1 spectrum analysis complete!")
print(f"  Output directory: {output_dir}")
print(f"  Analyzed {len(results)} blocks (> 190)")
print(f"  Template fit successful for {len(fit_results)} blocks")
