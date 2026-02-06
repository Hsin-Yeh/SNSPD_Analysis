#!/usr/bin/env python3
"""
Create combined plot comparing SNSPD response at different bias voltages.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from read_phu import read_phu_file, load_power_data
from tcspc_config import (
    T_MIN_NS, T_MAX_NS, SIGNAL_WIDTH_NS, FIT_MAX_UW,
    POWER_DATA_FILE, OUTPUT_DIR_COMBINED, BIAS_SETTINGS, BIAS_FILES
)
from tcspc_analysis import extract_oot_pre_dark_counts, subtract_dark_counts, fit_power_law

# Data files
data_files = BIAS_FILES

output_dir = OUTPUT_DIR_COMBINED
output_dir.mkdir(parents=True, exist_ok=True)

# Load power data
if not POWER_DATA_FILE.exists():
    print(f"Error: Power data file not found: {POWER_DATA_FILE}")
    sys.exit(1)

power_data = load_power_data(POWER_DATA_FILE)
print(f"Loaded power data for {len(power_data)} block IDs")

# Signal window - from config
t_min_ns, t_max_ns = T_MIN_NS, T_MAX_NS
signal_width_ns = SIGNAL_WIDTH_NS

# Colors and markers for each bias - from config
colors = {bias: BIAS_SETTINGS[bias]['color'] for bias in BIAS_SETTINGS}
markers = {bias: BIAS_SETTINGS[bias]['marker'] for bias in BIAS_SETTINGS}

# Collect data from all files
all_data = {}

for bias, filepath in data_files.items():
    print(f"\nProcessing {bias}...")
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"Warning: File not found: {filepath}")
        continue
    
    header, histograms = read_phu_file(filepath, verbose=False)
    
    curve_indices = header.get('HistResDscr_CurveIndex', {})
    acq_time_ms = header.get('MeasDesc_AcquisitionTime', 10000)
    acq_time_s = acq_time_ms / 1000.0
    resolution_s = header.get('MeasDesc_Resolution', 4e-12)
    
    # Convert time window to bins
    bin_min = int(t_min_ns * 1e-9 / resolution_s)
    bin_max = int(t_max_ns * 1e-9 / resolution_s)
    
    powers = []
    counts = []
    oot_pre_darks = []  # Store OOT_pre dark counts for each point
    
    for i, hist in enumerate(histograms):
        block_id = curve_indices.get(i, None)
        
        if block_id is None or block_id not in power_data:
            continue
        
        counts_in_window = int(np.sum(hist[bin_min:bin_max]))
        count_rate = counts_in_window / acq_time_s
        
        # Calculate OOT_pre dark count using shared function
        oot_pre_dark = extract_oot_pre_dark_counts(hist, resolution_s, SIGNAL_WIDTH_NS, acq_time_s)
        
        if block_id != 0:
            powers.append(power_data[block_id])
            counts.append(count_rate)
            oot_pre_darks.append(oot_pre_dark)
    
    # Subtract dark count using shared function
    powers_arr = np.array(powers)
    counts_arr = np.array(counts)
    oot_pre_darks_arr = np.array(oot_pre_darks)
    
    # Use OOT_pre as dark count for each measurement
    if len(oot_pre_darks_arr) > 0:
        counts_corrected, dark_count_rate = subtract_dark_counts(
            counts_arr, oot_pre_darks_arr, method='per_measurement'
        )
        print(f"  Using OOT_pre (0-60 ns) dark count method")
    else:
        counts_corrected = counts_arr
        dark_count_rate = None
    
    # Fit power law in low-power region using shared function
    # Try adaptive fit range for biases with insufficient low-power data
    try:
        fit_results = fit_power_law(powers_arr, counts_corrected, FIT_MAX_UW)
        fit_max_used = FIT_MAX_UW
    except ValueError as e:
        # If default fit range has insufficient points, try expanding
        print(f"  Warning: {e}")
        print(f"  Attempting adaptive fit range...")
        
        # Find minimum power that gives at least 5 data points
        sorted_powers = np.sort(powers_arr[powers_arr > 0])
        if len(sorted_powers) >= 5:
            adaptive_fit_max = sorted_powers[4]  # 5th smallest power (0-indexed)
            print(f"  Using adaptive fit max: {adaptive_fit_max:.3f} µW (includes {np.sum(powers_arr <= adaptive_fit_max)} points)")
            fit_results = fit_power_law(powers_arr, counts_corrected, adaptive_fit_max)
            fit_max_used = adaptive_fit_max
        else:
            print(f"  Error: Not enough data points for adaptive fit")
            continue
    
    slope = fit_results['slope']
    intercept = fit_results['intercept']
    std_err = fit_results['std_err']
    chi2_ndf = fit_results['chi2_ndf']
    fit_mask = fit_results['fit_mask']
    fit_powers = fit_results['fit_powers']
    
    all_data[bias] = {
        'powers': powers_arr,
        'counts': counts_corrected,
        'dark': dark_count_rate,
        'slope': slope,
        'intercept': intercept,
        'std_err': std_err,
        'fit_powers': fit_powers,
        'chi2_ndf': chi2_ndf,
        'fit_max_used': fit_max_used,
    }
    
    if dark_count_rate is not None:
        print(f"  Dark count: {dark_count_rate:.2f} cts/s")
    print(f"  Power law exponent: n = {slope:.4f} ± {std_err:.4f}")
    print(f"  Chi²/ndf: {chi2_ndf:.4f}")
    if fit_max_used != FIT_MAX_UW:
        print(f"  Note: Used adaptive fit range up to {fit_max_used:.3f} µW")

print(f"\n{'='*80}")
print(f"Successfully loaded biases: {sorted(all_data.keys())}")
print(f"Failed/skipped biases: {[b for b in data_files.keys() if b not in all_data]}")
print(f"{'='*80}\n")

# Create combined plot
fig, ax = plt.subplots(figsize=(12, 8))

for bias in sorted(all_data.keys()):
    data = all_data[bias]
    color = colors[bias]
    marker = markers[bias]
    
    # Plot data points
    ax.scatter(data['powers'], data['counts'], s=100, alpha=0.7,
               color=color, edgecolors='black', linewidth=1.5,
               marker=marker, label=f'{bias} data', zorder=5)
    
    # Plot fit line
    fit_line = 10**(data['slope'] * np.log10(data['fit_powers']) + data['intercept'])
    ax.plot(data['fit_powers'], fit_line, 
            color=color, linewidth=3, linestyle='-', alpha=0.85,
            label=f'{bias} fit: n={data["slope"]:.3f}±{data["std_err"]:.3f}, χ²/ndf={data["chi2_ndf"]:.4f}', zorder=3)

ax.set_xlabel('Laser Power (µW)', fontsize=14, weight='bold')
ax.set_ylabel('Count Rate (cts/s)', fontsize=14, weight='bold')
ax.set_title('SNSPD Output vs Power: Bias Voltage Comparison\n(Dark-corrected, Signal window: 75.0-79.0 ns)', 
             fontsize=15, weight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='upper left', fontsize=11, framealpha=0.95, ncol=2)
ax.tick_params(labelsize=11)

plt.tight_layout()

# Save multiple plot versions
output_path_loglog = output_dir / 'combined_power_sweep_loglog.png'
fig.savefig(output_path_loglog, dpi=300, bbox_inches='tight')
print(f"\n✓ Combined plot saved: {output_path_loglog}")

# Create linear scale version
fig2, ax2 = plt.subplots(figsize=(12, 8))

for bias in sorted(all_data.keys()):
    data = all_data[bias]
    color = colors[bias]
    marker = markers[bias]
    
    ax2.scatter(data['powers'], data['counts'], s=100, alpha=0.7,
               color=color, edgecolors='black', linewidth=1.5,
               marker=marker, label=f'{bias}', zorder=5)

ax2.set_xlabel('Laser Power (µW)', fontsize=14, weight='bold')
ax2.set_ylabel('Count Rate (cts/s)', fontsize=14, weight='bold')
ax2.set_title('SNSPD Output vs Power: Bias Voltage Comparison (Linear Scale)\n(Dark-corrected, Signal window: 75.0-79.0 ns)', 
             fontsize=15, weight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax2.tick_params(labelsize=11)
plt.tight_layout()

output_path_linear = output_dir / 'combined_power_sweep_linear.png'
fig2.savefig(output_path_linear, dpi=300, bbox_inches='tight')
print(f"✓ Linear plot saved: {output_path_linear}")
plt.close(fig2)

# Create saturation zoom version
fig3, ax3 = plt.subplots(figsize=(12, 8))

for bias in sorted(all_data.keys()):
    data = all_data[bias]
    color = colors[bias]
    marker = markers[bias]
    
    ax3.scatter(data['powers'], data['counts'], s=100, alpha=0.7,
               color=color, edgecolors='black', linewidth=1.5,
               marker=marker, label=f'{bias}', zorder=5)

ax3.set_xlabel('Laser Power (µW)', fontsize=14, weight='bold')
ax3.set_ylabel('Count Rate (cts/s)', fontsize=14, weight='bold')
ax3.set_title('SNSPD Output vs Power: Saturation Region\n(Dark-corrected, Signal window: 75.0-79.0 ns)', 
             fontsize=15, weight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 0.3)  # Focus on low-power saturation region
ax3.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax3.tick_params(labelsize=11)
plt.tight_layout()

output_path_sat = output_dir / 'combined_power_sweep_saturation.png'
fig3.savefig(output_path_sat, dpi=300, bbox_inches='tight')
print(f"✓ Saturation plot saved: {output_path_sat}")
plt.close(fig3)

plt.close(fig)

# Create summary table
print(f"\n{'='*80}")
print("SUMMARY: Power Law Fits for Different Bias Voltages")
print(f"{'='*80}")
print(f"{'Bias':<10} {'Dark (cts/s)':<15} {'Exponent (n)':<25} {'Chi^2/ndf':<15}")
print(f"{'-'*80}")
for bias in sorted(all_data.keys()):
    data = all_data[bias]
    print(f"{bias:<10} {data['dark']:<15.2f} {data['slope']:.4f} ± {data['std_err']:.4f}       {data['chi2_ndf']:<15.4f}")
print(f"{'='*80}\n")

print(f"All outputs saved to: {output_dir.parent}")
