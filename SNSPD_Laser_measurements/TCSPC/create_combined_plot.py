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

# Data files - can be customized via command line or modified here
data_files = {
    '70mV': '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_70mV_20260205_0122.phu',
    '74mV': '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_74mV_20260205_0102.phu',
    '78mV': '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_78mV_20260205_0230.phu',
    # '66mV': '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_66mV_20260205_0246.phu',  # Optional
}

output_dir = Path('/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/combined')
output_dir.mkdir(parents=True, exist_ok=True)

# Load power data
workspace_attenuation = Path(__file__).parent.parent / "Attenuation" / "Rotation_10MHz_5degrees_data_20260205.txt"
if not workspace_attenuation.exists():
    print(f"Error: Power data file not found: {workspace_attenuation}")
    sys.exit(1)

power_data = load_power_data(workspace_attenuation)
print(f"Loaded power data for {len(power_data)} block IDs")

# Signal window
t_min_ns, t_max_ns = 75.0, 79.0
signal_width_ns = t_max_ns - t_min_ns

# Colors for each bias
colors = {
    '70mV': 'blue',
    '74mV': 'green',
    '78mV': 'red',
}

markers = {
    '70mV': 'o',
    '74mV': 's',
    '78mV': '^',
}

# Collect data from all files
all_data = {}

for bias, filepath in data_files.items():
    print(f"\nProcessing {bias}...")
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"Warning: File not found: {filepath}")
        continue
    
    header, histograms = read_phu_file(filepath)
    
    curve_indices = header.get('HistResDscr_CurveIndex', {})
    acq_time_ms = header.get('MeasDesc_AcquisitionTime', 10000)
    acq_time_s = acq_time_ms / 1000.0
    resolution_s = header.get('MeasDesc_Resolution', 4e-12)
    
    # Convert time window to bins
    bin_min = int(t_min_ns * 1e-9 / resolution_s)
    bin_max = int(t_max_ns * 1e-9 / resolution_s)
    
    powers = []
    counts = []
    dark_count_rate = None
    
    for i, hist in enumerate(histograms):
        block_id = curve_indices.get(i, None)
        
        if block_id is None or block_id not in power_data:
            continue
        
        counts_in_window = int(np.sum(hist[bin_min:bin_max]))
        count_rate = counts_in_window / acq_time_s
        
        if block_id == 0:
            dark_count_rate = count_rate
        else:
            powers.append(power_data[block_id])
            counts.append(count_rate)
    
    # Subtract dark count
    powers_arr = np.array(powers)
    counts_arr = np.array(counts)
    
    if dark_count_rate is not None:
        counts_corrected = counts_arr - dark_count_rate
    else:
        counts_corrected = counts_arr
    
    # Fit power law in low-power region
    fit_max = 2e-1  # < 0.2 µW
    fit_mask = powers_arr <= fit_max
    
    if np.sum(fit_mask) >= 2:
        from scipy import stats
        log_powers_fit = np.log10(powers_arr[fit_mask])
        log_counts_fit = np.log10(counts_corrected[fit_mask])
        slope, intercept, r_value, _, std_err = stats.linregress(log_powers_fit, log_counts_fit)
        
        all_data[bias] = {
            'powers': powers_arr,
            'counts': counts_corrected,
            'dark': dark_count_rate,
            'slope': slope,
            'intercept': intercept,
            'std_err': std_err,
            'fit_powers': powers_arr[fit_mask],
        }
        
        print(f"  Dark count: {dark_count_rate:.2f} cts/s")
        print(f"  Power law exponent: n = {slope:.4f} ± {std_err:.4f}")
    else:
        print(f"  Not enough data points for fit")

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
            label=f'{bias} fit: n={data["slope"]:.3f}±{data["std_err"]:.3f}', zorder=3)

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
print(f"{'Bias':<10} {'Dark (cts/s)':<15} {'Exponent (n)':<20} {'Chi^2/ndf':<15}")
print(f"{'-'*80}")
for bias in sorted(all_data.keys()):
    data = all_data[bias]
    print(f"{bias:<10} {data['dark']:<15.2f} {data['slope']:.4f} ± {data['std_err']:.4f}")
print(f"{'='*80}\n")

print(f"All outputs saved to: {output_dir.parent}")
