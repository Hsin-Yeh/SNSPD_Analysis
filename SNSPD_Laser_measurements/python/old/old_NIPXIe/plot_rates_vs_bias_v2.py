#!/usr/bin/env python3
"""
Plot SNSPD rates vs bias voltage for different power levels
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
import plot_utils
from plot_utils import read_analysis_files, group_by_power, print_data_summary

parser = argparse.ArgumentParser(description='Plot count rates vs bias voltage from analysis.json files')
parser.add_argument('--input_dir', '-i', nargs='+', default=['./plots/test'], 
                   help='Directory or directories containing analysis.json files (supports multiple)')
parser.add_argument('--output_dir', '-d', default='.', help='Output directory for plots')
parser.add_argument('--pattern', '-p', default='*_analysis.json', help='File pattern to match')
parser.add_argument('--plot_individual', action='store_true', help='Plot individual power levels')
parser.add_argument('--plot_combined', action='store_true', help='Plot combined multi-power comparison')
parser.add_argument('--recursive', '-r', action='store_true', default=True, help='Search recursively in subdirectories')

def plot_single_power_vs_bias(power, power_data, output_dir):
    """Plot count rates vs bias voltage for a single power level"""
    
    bias_voltages = [d['bias_voltage'] for d in power_data]
    count_rates = [d['count_rate'] for d in power_data]
    signal_rates = [d['signal_rate'] for d in power_data]
    dark_count_rates = [d['dark_count_rate'] for d in power_data]
    efficiencies = [d['efficiency'] for d in power_data]
    
    count_rate_errors = [d['count_rate_error'] for d in power_data]
    signal_rate_errors = [d['signal_rate_error'] for d in power_data]
    dark_count_rate_errors = [d['dark_count_rate_error'] for d in power_data]
    efficiency_errors = [d['efficiency_error'] for d in power_data]
    
    resistances = [d['resistance'] for d in power_data if d['resistance'] is not None]
    resistance_voltages = [d['bias_voltage'] for d in power_data if d['resistance'] is not None]
    
    power_label = f'{power} nW' if isinstance(power, int) else str(power)
    
    # Create figure with 3 stacked subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    
    # Plot rates with dual y-axis
    color_signal = 'tab:blue'
    ax1.errorbar(bias_voltages, signal_rates, yerr=signal_rate_errors, marker='s', color=color_signal, 
                 label='Signal Rate', linewidth=2, capsize=4, capthick=1.5)
    ax1.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax1.set_ylabel('Signal Rate (Hz)', fontsize=12, color=color_signal)
    ax1.tick_params(axis='y', labelcolor=color_signal)
    ax1.grid(True, alpha=0.3)
    
    # Right axis for count rate and dark count rate
    ax1_right = ax1.twinx()
    color_count = 'tab:orange'
    color_dark = 'tab:red'
    ax1_right.errorbar(bias_voltages, count_rates, yerr=count_rate_errors, marker='o', color=color_count, 
                       label='Count Rate', linewidth=2, capsize=4, capthick=1.5)
    ax1_right.errorbar(bias_voltages, dark_count_rates, yerr=dark_count_rate_errors, marker='^', color=color_dark, 
                       label='Dark Count Rate', linewidth=2, capsize=4, capthick=1.5)
    ax1_right.set_ylabel('Count Rate / Dark Count Rate (Hz)', fontsize=12)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_right.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
    ax1.set_title(f'Count Rates vs Bias Voltage ({power_label})', fontsize=14, fontweight='bold')
    
    # Plot efficiency
    ax2.errorbar(bias_voltages, efficiencies, yerr=efficiency_errors, marker='D', color='red', 
                 linewidth=2, label='Efficiency', capsize=4, capthick=1.5)
    ax2.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax2.set_ylabel('Efficiency', fontsize=12)
    ax2.set_title(f'Efficiency vs Bias Voltage ({power_label})', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot resistance
    if resistances:
        ax3.plot(resistance_voltages, resistances, marker='o', color='purple', linewidth=2, markersize=8)
        ax3.set_xlabel('Bias Voltage (mV)', fontsize=12)
        ax3.set_ylabel('Resistance (Ω)', fontsize=12)
        ax3.set_title(f'Resistance vs Bias Voltage ({power_label})', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No resistance data available', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax3.transAxes, fontsize=14)
        ax3.set_title(f'Resistance vs Bias Voltage ({power_label})', fontsize=14, fontweight='bold')
    
    # Set same x-axis limits
    x_min, x_max = min(bias_voltages), max(bias_voltages)
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    ax3.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    
    # Save
    power_str = f'{power}nW' if isinstance(power, int) else str(power)
    output_path = os.path.join(output_dir, f'rates_vs_bias_{power_str}.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    plt.close()

def plot_multi_power_vs_bias(power_groups, output_dir):
    """Plot count rates vs bias voltage for multiple powers on same plot"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(power_groups)))
    
    for idx, (power, power_data) in enumerate(sorted(power_groups.items())):
        bias_voltages = [d['bias_voltage'] for d in power_data]
        count_rates = [d['count_rate'] for d in power_data]
        signal_rates = [d['signal_rate'] for d in power_data]
        dark_count_rates = [d['dark_count_rate'] for d in power_data]
        efficiencies = [d['efficiency'] for d in power_data]
        
        count_rate_errors = [d['count_rate_error'] for d in power_data]
        signal_rate_errors = [d['signal_rate_error'] for d in power_data]
        dark_count_rate_errors = [d['dark_count_rate_error'] for d in power_data]
        efficiency_errors = [d['efficiency_error'] for d in power_data]
        
        color = colors[idx]
        label = f'{power} nW' if isinstance(power, int) else str(power)
        
        # Plot each rate type
        ax1.errorbar(bias_voltages, count_rates, yerr=count_rate_errors, marker='o', 
                    color=color, label=label, linewidth=2, capsize=3)
        ax2.errorbar(bias_voltages, signal_rates, yerr=signal_rate_errors, marker='s', 
                    color=color, label=label, linewidth=2, capsize=3)
        ax3.errorbar(bias_voltages, dark_count_rates, yerr=dark_count_rate_errors, marker='^', 
                    color=color, label=label, linewidth=2, capsize=3)
        ax4.errorbar(bias_voltages, efficiencies, yerr=efficiency_errors, marker='D', 
                    color=color, label=label, linewidth=2, capsize=3)
    
    # Configure subplots
    for ax, ylabel, title in [
        (ax1, 'Count Rate (Hz)', 'Count Rate vs Bias Voltage'),
        (ax2, 'Signal Rate (Hz)', 'Signal Rate vs Bias Voltage'),
        (ax3, 'Dark Count Rate (Hz)', 'Dark Count Rate vs Bias Voltage'),
        (ax4, 'Efficiency', 'Efficiency vs Bias Voltage')
    ]:
        ax.set_xlabel('Bias Voltage (mV)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'multi_power_rates_vs_bias.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    plt.close()

def main():
    args = parser.parse_args()
    input_dirs = args.input_dir if isinstance(args.input_dir, list) else [args.input_dir]
    print(f"Reading analysis files from: {', '.join(input_dirs)}")
    print(f"File pattern: {args.pattern}")
    print(f"Recursive search: {args.recursive}")
    
    data = read_analysis_files(args.input_dir, args.pattern, args.recursive)
    
    if not data:
        print("No valid data found!")
        return
    
    print_data_summary(data)
    
    # Group data by power
    power_groups = group_by_power(data)
    
    # Default: plot both if no flags specified
    if not args.plot_individual and not args.plot_combined:
        args.plot_individual = True
        args.plot_combined = True
    
    # Plot individual power datasets
    if args.plot_individual:
        print("\n=== Plotting individual power levels ===")
        for power, power_data in sorted(power_groups.items()):
            plot_single_power_vs_bias(power, power_data, args.output_dir)
    
    # Plot combined multi-power comparison
    if args.plot_combined and len(power_groups) > 1:
        print("\n=== Plotting combined multi-power comparison ===")
        plot_multi_power_vs_bias(power_groups, args.output_dir)

if __name__ == "__main__":
    main()
