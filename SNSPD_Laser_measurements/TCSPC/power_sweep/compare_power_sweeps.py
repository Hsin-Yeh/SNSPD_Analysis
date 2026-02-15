#!/usr/bin/env python3
"""
Compare multiple power sweep measurements at different fixed bias voltages.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys


def load_power_sweep_data(summary_file):
    """Load data from a power sweep summary JSON file."""
    with open(summary_file, 'r') as f:
        data = json.load(f)
    return data


def plot_combined_power_sweeps(data_files, output_dir=None):
    """
    Create combined plots comparing multiple power sweeps.
    
    Parameters:
    -----------
    data_files : list of str
        Paths to power_sweep_summary.json files (or similar)
    output_dir : str or Path
        Output directory for combined plots
    """
    
    if output_dir is None:
        output_dir = Path.cwd() / "power_sweep_combined"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hardcoded paths for the power sweep data (need to reconstruct from histogram files)
    base_path = Path("/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep")
    
    data_files = [
        base_path / "70mV" / "power_sweep_summary.json",
        base_path / "74mV" / "power_sweep_summary.json",
        base_path / "78mV" / "power_sweep_summary.json",
    ]
    
    # Load all datasets
    datasets = []
    bias_voltages = []
    
    for f in data_files:
        try:
            if f.exists():
                data = load_power_sweep_data(f)
                datasets.append(data)
                # Extract bias voltage from filename or data
                bias_mV = int(f.parent.name.replace("mV", ""))
                bias_voltages.append(bias_mV)
                print(f"✓ Loaded {f.parent.name}: {len(data.get('power_uW', []))} power points")
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    if len(datasets) == 0:
        print("ERROR: No valid datasets loaded")
        return
    
    # Sort by bias voltage
    sorted_data = sorted(zip(bias_voltages, datasets), key=lambda x: x[0])
    bias_voltages = [x[0] for x in sorted_data]
    datasets = [x[1] for x in sorted_data]
    
    # ========== PLOT 1: Power Law Comparison ==========
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    colors = ['blue', 'red', 'green']
    markers = ['s', '^', 'd']
    
    for idx, (bias_mV, data) in enumerate(zip(bias_voltages, datasets)):
        power_uW = np.array(data.get('power_uW', []))
        net_rate = np.array(data.get('net_rate_cts_s', []))
        
        # Filter positive rates for log plot
        valid_mask = net_rate > 0
        power_valid = power_uW[valid_mask]
        rate_valid = net_rate[valid_mask]
        
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
    
    for idx, (bias_mV, data) in enumerate(zip(bias_voltages, datasets)):
        power_uW = np.array(data.get('power_uW', []))
        net_rate = np.array(data.get('net_rate_cts_s', []))
        
        # Use all points
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
    
    # ========== PLOT 3: Saturation Comparison ==========
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    
    for idx, (bias_mV, data) in enumerate(zip(bias_voltages, datasets)):
        power_uW = np.array(data.get('power_uW', []))
        net_rate = np.array(data.get('net_rate_cts_s', []))
        
        # Normalize to maximum
        max_rate = np.max(net_rate)
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
    ax3.set_title('SNSPD Saturation Behavior - Normalized\n(75-80 ns window)', fontsize=13, weight='bold')
    ax3.legend(fontsize=11, loc='lower right')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_ylim([0, 1.15])
    
    plot3_file = output_dir / "combined_power_sweep_saturation.png"
    plt.savefig(plot3_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot 3 saved: {plot3_file}")
    plt.close()
    
    print(f"\n✓ Combined power sweep plots saved to: {output_dir}")


if __name__ == "__main__":
    output_dir = Path("/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/combined")
    
    plot_combined_power_sweeps([], output_dir=output_dir)
