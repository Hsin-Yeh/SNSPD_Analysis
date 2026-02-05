#!/usr/bin/env python3
"""
Extract power sweep data from the output and create combined comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import re


def extract_power_sweep_data(filepath):
    """
    Read a power sweep PHU file and extract power vs count rate.
    Uses the exact same logic as read_phu.py for consistency.
    Returns dict with power_uW and net_rate_cts_s.
    """
    from read_phu import read_phu_file, load_power_data
    
    print(f"Reading: {filepath}")
    header, histograms = read_phu_file(filepath)
    
    # Load power data
    power_data_file = Path(__file__).parent.parent / "Attenuation" / "Rotation_10MHz_5degrees_data_20260205.txt"
    power_data = load_power_data(power_data_file)
    
    # Extract parameters
    curve_indices = header.get('HistResDscr_CurveIndex', {})
    acq_time_ms = header.get('MeasDesc_AcquisitionTime', 10000)
    acq_time_s = acq_time_ms / 1000.0
    resolution_s = header.get('MeasDesc_Resolution', 4e-12)
    
    # Time window
    t_min_ns, t_max_ns = 75.0, 80.0
    bin_min = int(t_min_ns * 1e-9 / resolution_s)
    bin_max = int(t_max_ns * 1e-9 / resolution_s)
    signal_width_ns = t_max_ns - t_min_ns
    
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
            print(f"Using Block 0 dark count: {dark_count_rate:.2f} cts/s")
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
    
    return {
        'power_uW': list(powers_arr),
        'signal_rate_cts_s': list(counts_arr),
        'net_rate_cts_s': list(counts_corrected),
        'dark_count_rate': dark_count_rate,
        'signal_width_ns': signal_width_ns
    }


def main():
    """Generate combined power sweep plots."""
    
    base_path = Path("/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep")
    output_dir = base_path / "combined"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data from all bias voltages
    bias_dirs = [
        # ("66mV", 66),  # Excluded from comparison
        ("70mV", 70),
        ("74mV", 74),
        ("78mV", 78),
    ]
    
    datasets = []
    phu_files = {
        # "66mV": "/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_66mV_20260205_0754.phu",
        "70mV": "/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_70mV_20260205_0122.phu",
        "74mV": "/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_74mV_20260205_0102.phu",
        "78mV": "/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_78mV_20260205_0230.phu",
    }
    
    for dir_name, bias_mV in bias_dirs:
        phu_file = phu_files.get(dir_name)
        if phu_file and Path(phu_file).exists():
            try:
                data = extract_power_sweep_data(phu_file)
                data['bias_mV'] = bias_mV
                datasets.append(data)
                print(f"✓ Loaded {dir_name}: {len(data['power_uW'])} points")
            except Exception as e:
                print(f"Warning: Could not load {dir_name}: {e}")
    
    if len(datasets) == 0:
        print("ERROR: No datasets loaded")
        return
    
    # Sort by bias voltage
    datasets.sort(key=lambda x: x['bias_mV'])
    
    # ========== PLOT 1: Log-Log Power Law ==========
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    colors = ['blue', 'red', 'green']
    markers = ['s', '^', 'd']
    
    for idx, data in enumerate(datasets):
        bias_mV = data['bias_mV']
        power_uW = np.array(data['power_uW'])
        signal_rate = np.array(data['signal_rate_cts_s'])  # Original
        net_rate = np.array(data['net_rate_cts_s'])  # Dark-corrected
        
        # Filter positive rates for dark-corrected data
        valid_mask = net_rate > 0
        power_valid = power_uW[valid_mask]
        rate_valid = net_rate[valid_mask]
        
        if len(power_valid) > 0:
            label = f'{bias_mV} mV'
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            
            ax1.loglog(power_valid, rate_valid, marker=marker, color=color, markersize=8, 
                      linewidth=2, alpha=0.8, label=label, zorder=5)
    
    ax1.set_xlabel('Laser Power (µW)', fontsize=12)
    ax1.set_ylabel('Detection Rate (cts/s)', fontsize=12)
    ax1.set_title('SNSPD Power Response at Fixed Bias Voltages\n(75-80 ns window)', fontsize=13, weight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')
    
    plot1_file = output_dir / "combined_power_sweep_loglog.png"
    plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot 1 saved: {plot1_file}")
    plt.close()
    
    # ========== PLOT 2: Linear Scale ==========
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    for idx, data in enumerate(datasets):
        bias_mV = data['bias_mV']
        power_uW = np.array(data['power_uW'])
        net_rate = np.array(data['net_rate_cts_s'])
        
        label = f'{bias_mV} mV'
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        ax2.plot(power_uW, net_rate, marker=marker, color=color, markersize=8, 
                linewidth=2, alpha=0.8, label=label, zorder=5)
    
    ax2.set_xlabel('Laser Power (µW)', fontsize=12)
    ax2.set_ylabel('Detection Rate (cts/s)', fontsize=12)
    ax2.set_title('SNSPD Power Response - Linear Scale\n(75-80 ns window)', fontsize=13, weight='bold')
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plot2_file = output_dir / "combined_power_sweep_linear.png"
    plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot 2 saved: {plot2_file}")
    plt.close()
    
    # ========== PLOT 3: Saturation ==========
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    
    for idx, data in enumerate(datasets):
        bias_mV = data['bias_mV']
        power_uW = np.array(data['power_uW'])
        net_rate = np.array(data['net_rate_cts_s'])
        
        # Normalize
        max_rate = np.max(net_rate[net_rate > 0]) if np.any(net_rate > 0) else 1
        norm_rate = net_rate / max_rate
        
        label = f'{bias_mV} mV (max: {max_rate:.1f} cts/s)'
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        ax3.semilogx(power_uW, norm_rate, marker=marker, color=color, markersize=8, 
                    linewidth=2, alpha=0.8, label=label, zorder=5)
    
    ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.axhline(y=0.95, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    
    ax3.set_xlabel('Laser Power (µW)', fontsize=12)
    ax3.set_ylabel('Normalized Detection Rate', fontsize=12)
    ax3.set_title('SNSPD Saturation Behavior\n(75-80 ns window)', fontsize=13, weight='bold')
    ax3.legend(fontsize=11, loc='lower right')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_ylim([0, 1.15])
    
    plot3_file = output_dir / "combined_power_sweep_saturation.png"
    plt.savefig(plot3_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot 3 saved: {plot3_file}")
    plt.close()
    
    print(f"\n✓ All combined power sweep plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
