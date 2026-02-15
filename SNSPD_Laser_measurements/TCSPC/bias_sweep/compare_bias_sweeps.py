#!/usr/bin/env python3
"""
Compare multiple bias sweep measurements with different laser powers.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys


def load_bias_sweep_data(summary_file):
    """Load data from a bias sweep summary JSON file."""
    with open(summary_file, 'r') as f:
        data = json.load(f)
    return data


def plot_combined_bias_sweeps(data_files, output_dir=None):
    """
    Create combined plots comparing multiple bias sweeps.
    
    Parameters:
    -----------
    data_files : list of str
        Paths to bias_sweep_summary.json files
    output_dir : str or Path
        Output directory for combined plots
    """
    
    if output_dir is None:
        output_dir = Path.cwd() / "bias_sweep_comparison"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all datasets
    datasets = []
    for f in data_files:
        try:
            data = load_bias_sweep_data(f)
            datasets.append(data)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    if len(datasets) == 0:
        print("ERROR: No valid datasets loaded")
        return
    
    # Sort by laser power
    datasets.sort(key=lambda x: x.get('laser_power_nW', 0))
    
    # ========== PLOT 1: Normalized comparison with dark on right axis ==========
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1_right = ax1.twinx()  # Create second y-axis for dark counts
    ax1_top = ax1.twiny()    # Create second x-axis for Ib/Isw ratio
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['s', '^', 'd', 'v', '<']
    
    # Separate dark and signal datasets
    dark_data = None
    signal_datasets = []
    
    for data in datasets:
        if data.get('laser_power_nW', 0) == 0:
            dark_data = data
        else:
            signal_datasets.append(data)
    
    # Plot signal datasets on left axis (normalized)
    for idx, data in enumerate(signal_datasets):
        power_nW = data.get('laser_power_nW', 0)
        bias = np.array(data['bias_voltages_mV'])
        counts_norm = np.array(data['signal_rates_normalized'])
        
        label = f'{power_nW:.0f} nW'
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        ax1.plot(bias, counts_norm, marker=marker, color=color, markersize=8, 
                linewidth=2, alpha=0.8, label=label, zorder=5)
    
    # Plot dark data on right axis (absolute counts)
    if dark_data is not None:
        bias_dark = np.array(dark_data['bias_voltages_mV'])
        counts_dark = np.array(dark_data['signal_rates_cts_s'])
        
        # Plot individual dark count points for each bias
        ax1_right.plot(bias_dark, counts_dark, marker='o', color='gray', markersize=6, 
                      linewidth=1.5, alpha=0.6, label='Dark counts', zorder=5)
    
    
    ax1.set_xlabel('Bias Voltage Ib (mV)', fontsize=12)
    ax1.set_ylabel('Normalized Detection Efficiency', fontsize=12, color='blue')
    ax1_right.set_ylabel('Dark Count Rate (cts/s)', fontsize=12, color='gray')
    
    # Set left axis limits
    ax1.set_ylim([-0.05, 1.15])
    left_min, left_max = ax1.get_ylim()
    
    # Set right y-axis limits to align zero points
    if dark_data is not None:
        counts_dark = np.array(dark_data['signal_rates_cts_s'])
        max_dark = np.max(counts_dark)
        # Calculate right axis limits so 0 aligns with left axis 0
        # Position of 0 on left axis: -0.05 / (1.15 - (-0.05)) = -0.05 / 1.2
        # We want right axis 0 at same position: right_min / (right_max - right_min) = -0.05 / 1.2
        # If right_min = 0, then: 0 / (right_max - 0) ≠ -0.05 / 1.2
        # Need to set: right_min / (right_max - right_min) = left_min / (left_max - left_min)
        left_ratio = left_min / (left_max - left_min)  # Position of left 0
        # For right axis with 0 at same position: right_min = left_ratio * (right_max - right_min)
        # If max_dark * 1.5 is the top: right_max = max_dark * 1.5
        # right_min = left_ratio * max_dark * 1.5 / (1 - left_ratio)
        right_max = max_dark * 1.5
        right_min = left_ratio * right_max / (1 - left_ratio)
        ax1_right.set_ylim([right_min, right_max])
    
    # Set up top axis for Ib/Isw ratio
    bias_min, bias_max = ax1.get_xlim()
    ax1_top.set_xlim(bias_min/90, bias_max/90)  # Convert to ratio
    ax1_top.set_xlabel('Normalized Bias Ib/Isw', fontsize=12)
    
    ax1.set_title('SNSPD Response vs Bias Voltage\n(Efficiency & Dark Counts)', fontsize=13, weight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Format y-axis colors
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_right.tick_params(axis='y', labelcolor='gray')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_right.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
    
    plot1_file = output_dir / "combined_bias_sweep_efficiency_dark.png"
    plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot 1 saved: {plot1_file}")
    plt.close()
    
    # ========== PLOT 2: Log-scale comparison ==========
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    for idx, data in enumerate(datasets):
        power_nW = data.get('laser_power_nW', 0)
        bias = np.array(data['bias_voltages_mV'])
        counts_norm = np.array(data['signal_rates_normalized'])
        
        # Filter out zeros for log plot
        valid_mask = counts_norm > 0
        bias_valid = bias[valid_mask]
        counts_valid = counts_norm[valid_mask]
        
        if power_nW == 0:
            label = 'Dark (0 nW)'
            color = 'black'
            marker = 'x'
            alpha = 0.8
            zorder = 10
        else:
            label = f'{power_nW:.0f} nW'
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            alpha = 0.7
            zorder = 5
        
        ax2.semilogy(bias_valid, counts_valid, marker=marker, color=color, markersize=8,
                    linewidth=2, alpha=alpha, label=label, zorder=zorder)
    
    ax2.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax2.set_ylabel('Normalized Count Rate [log scale]', fontsize=12)
    ax2.set_title('SNSPD Response vs Bias - Log Scale\n(Multiple Laser Powers)', fontsize=13, weight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10, loc='best')
    
    plot2_file = output_dir / "combined_bias_sweep_log.png"
    plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot 2 saved: {plot2_file}")
    plt.close()
    
    # ========== PLOT 3: Absolute count rates ==========
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    for idx, data in enumerate(datasets):
        power_nW = data.get('laser_power_nW', 0)
        bias = np.array(data['bias_voltages_mV'])
        counts_abs = np.array(data['signal_rates_cts_s'])
        
        if power_nW == 0:
            label = 'Dark (0 nW)'
            color = 'black'
            marker = 'x'
            alpha = 0.8
            zorder = 10
        else:
            label = f'{power_nW:.0f} nW'
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            alpha = 0.7
            zorder = 5
        
        ax3.plot(bias, counts_abs, marker=marker, color=color, markersize=8,
                linewidth=2, alpha=alpha, label=label, zorder=zorder)
    
    ax3.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax3.set_ylabel('Count Rate (cts/s)', fontsize=12)
    ax3.set_title('SNSPD Absolute Count Rates vs Bias\n(Multiple Laser Powers)', fontsize=13, weight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10, loc='best')
    
    plot3_file = output_dir / "combined_bias_sweep_absolute.png"
    plt.savefig(plot3_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot 3 saved: {plot3_file}")
    plt.close()
    
    # ========== PLOT 4: Threshold comparison ==========
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    # Focus on threshold region (60-80 mV)
    for idx, data in enumerate(datasets):
        power_nW = data.get('laser_power_nW', 0)
        bias = np.array(data['bias_voltages_mV'])
        counts_norm = np.array(data['signal_rates_normalized'])
        
        # Filter threshold region
        threshold_mask = (bias >= 60) & (bias <= 80)
        bias_thresh = bias[threshold_mask]
        counts_thresh = counts_norm[threshold_mask]
        
        if power_nW == 0:
            label = 'Dark (0 nW)'
            color = 'black'
            marker = 'x'
            alpha = 0.8
            linewidth = 2.5
            markersize = 10
            zorder = 10
        else:
            label = f'{power_nW:.0f} nW'
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            alpha = 0.7
            linewidth = 2
            markersize = 8
            zorder = 5
        
        ax4.plot(bias_thresh, counts_thresh, marker=marker, color=color, 
                markersize=markersize, linewidth=linewidth, alpha=alpha, 
                label=label, zorder=zorder)
    
    ax4.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='50% level')
    ax4.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax4.set_ylabel('Normalized Count Rate', fontsize=12)
    ax4.set_title('Threshold Region Detail (60-80 mV)\n(Multiple Laser Powers)', fontsize=13, weight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10, loc='best')
    ax4.set_xlim([60, 80])
    
    plot4_file = output_dir / "combined_bias_sweep_threshold.png"
    plt.savefig(plot4_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot 4 saved: {plot4_file}")
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("BIAS SWEEP COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Power (nW)':<12} {'Max Rate (cts/s)':<18} {'Saturation (mV)':<18} {'Notes':<20}")
    print("-"*80)
    
    for data in datasets:
        power = data.get('laser_power_nW', 0)
        max_rate = data.get('saturation_count_rate_cts_s', data.get('max_output_rate_cts_s', 0))
        
        # Estimate saturation bias (where normalized rate > 0.95)
        bias = np.array(data['bias_voltages_mV'])
        counts_norm = np.array(data['signal_rates_normalized'])
        sat_indices = np.where(counts_norm >= 0.95)[0]
        if len(sat_indices) > 0:
            sat_bias = bias[sat_indices[0]]
        else:
            sat_bias = np.nan
        
        if power == 0:
            notes = "Dark baseline"
        else:
            notes = ""
        
        print(f"{power:<12.0f} {max_rate:<18.1f} {sat_bias:<18.0f} {notes:<20}")
    
    print("\n✓ Comparison complete. Output saved to:", output_dir)


if __name__ == '__main__':
    # Default: look for summary files in standard locations
    base_output = Path("/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3")
    
    data_files = [
        base_output / "bias_sweep" / "0nW" / "bias_sweep_summary.json",
        base_output / "bias_sweep" / "99nW" / "bias_sweep_summary.json",
        base_output / "bias_sweep" / "1442nW" / "bias_sweep_summary.json",
    ]
    
    # Check if files exist
    valid_files = [str(f) for f in data_files if f.exists()]
    
    if len(valid_files) == 0:
        print("ERROR: No summary files found. Run analyze_bias_sweep.py first.")
        sys.exit(1)
    
    print(f"Found {len(valid_files)} bias sweep datasets")
    
    output_dir = base_output / "bias_sweep" / "combined"
    plot_combined_bias_sweeps(valid_files, output_dir=output_dir)
