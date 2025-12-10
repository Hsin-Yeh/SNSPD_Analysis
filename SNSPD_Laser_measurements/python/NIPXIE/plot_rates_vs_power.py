#!/usr/bin/env python3
"""
Plot SNSPD rates vs power for different bias voltages
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
from plot_utils import read_analysis_files, group_by_bias, print_data_summary

parser = argparse.ArgumentParser(description='Plot count rates vs power from analysis.json files')
parser.add_argument('--input_dir', '-i', nargs='+', default=['./plots/test'], 
                   help='Directory or directories containing analysis.json files (supports multiple)')
parser.add_argument('--output_dir', '-d', default='.', help='Output directory for plots')
parser.add_argument('--pattern', '-p', default='*_analysis.json', help='File pattern to match')
parser.add_argument('--plot_individual', action='store_true', help='Plot individual bias voltages')
parser.add_argument('--plot_combined', action='store_true', help='Plot combined multi-bias comparison')
parser.add_argument('--log_scale', action='store_true', help='Use log scale for y-axis')
parser.add_argument('--recursive', '-r', action='store_true', default=True, help='Search recursively in subdirectories')

def plot_single_bias_vs_power(bias, bias_data, output_dir, log_scale=False):
    """Plot count rates vs power for a single bias voltage"""
    
    # Sort by power
    bias_data = sorted(bias_data, key=lambda x: x['power'] if x['power'] is not None else -1)
    
    powers = [d['power'] for d in bias_data if d['power'] is not None]
    count_rates = [d['count_rate'] for d in bias_data if d['power'] is not None]
    signal_rates = [d['signal_rate'] for d in bias_data if d['power'] is not None]
    dark_count_rates = [d['dark_count_rate'] for d in bias_data if d['power'] is not None]
    efficiencies = [d['efficiency'] for d in bias_data if d['power'] is not None]
    
    count_rate_errors = [d['count_rate_error'] for d in bias_data if d['power'] is not None]
    signal_rate_errors = [d['signal_rate_error'] for d in bias_data if d['power'] is not None]
    dark_count_rate_errors = [d['dark_count_rate_error'] for d in bias_data if d['power'] is not None]
    efficiency_errors = [d['efficiency_error'] for d in bias_data if d['power'] is not None]
    
    if not powers:
        print(f"Warning: No power data for bias {bias} mV")
        return
    
    bias_label = f'{bias} mV'
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot count rate
    ax1.errorbar(powers, count_rates, yerr=count_rate_errors, marker='o', 
                color='tab:orange', linewidth=2, capsize=4, capthick=1.5, label='Count Rate')
    ax1.set_xlabel('Power (nW)', fontsize=12)
    ax1.set_ylabel('Count Rate (Hz)', fontsize=12)
    ax1.set_title(f'Count Rate vs Power ({bias_label})', fontsize=14, fontweight='bold')
    if log_scale:
        ax1.set_yscale('log')
        ax1.set_xscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot signal rate
    ax2.errorbar(powers, signal_rates, yerr=signal_rate_errors, marker='s', 
                color='tab:blue', linewidth=2, capsize=4, capthick=1.5, label='Signal Rate')
    ax2.set_xlabel('Power (nW)', fontsize=12)
    ax2.set_ylabel('Signal Rate (Hz)', fontsize=12)
    ax2.set_title(f'Signal Rate vs Power ({bias_label})', fontsize=14, fontweight='bold')
    if log_scale:
        ax2.set_yscale('log')
        ax2.set_xscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot dark count rate
    ax3.errorbar(powers, dark_count_rates, yerr=dark_count_rate_errors, marker='^', 
                color='tab:red', linewidth=2, capsize=4, capthick=1.5, label='Dark Count Rate')
    ax3.set_xlabel('Power (nW)', fontsize=12)
    ax3.set_ylabel('Dark Count Rate (Hz)', fontsize=12)
    ax3.set_title(f'Dark Count Rate vs Power ({bias_label})', fontsize=14, fontweight='bold')
    if log_scale:
        ax3.set_yscale('log')
        ax3.set_xscale('log')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot efficiency
    ax4.errorbar(powers, efficiencies, yerr=efficiency_errors, marker='D', 
                color='tab:purple', linewidth=2, capsize=4, capthick=1.5, label='Efficiency')
    ax4.set_xlabel('Power (nW)', fontsize=12)
    ax4.set_ylabel('Efficiency', fontsize=12)
    ax4.set_title(f'Efficiency vs Power ({bias_label})', fontsize=14, fontweight='bold')
    if log_scale:
        ax4.set_xscale('log')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    scale_suffix = '_logscale' if log_scale else ''
    output_path = os.path.join(output_dir, f'rates_vs_power_{bias}mV{scale_suffix}.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    plt.close()

def plot_multi_bias_vs_power(bias_groups, output_dir, log_scale=False):
    """Plot count rates vs power for multiple bias voltages on same plot"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define colors - use unique bias voltages only
    unique_biases = sorted(set(bias_groups.keys()))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_biases)))
    color_map = {bias: colors[i] for i, bias in enumerate(unique_biases)}
    
    # Debug: print what we're plotting
    print("\n=== Multi-bias plot data ===")
    for bias, bias_data in sorted(bias_groups.items()):
        powers = [d['power'] for d in bias_data if d['power'] is not None]
        print(f"Bias {bias} mV: {len(bias_data)} data points, powers = {sorted(set(powers))}")
    
    for bias, bias_data in sorted(bias_groups.items()):
        # Sort by power
        bias_data = sorted(bias_data, key=lambda x: x['power'] if x['power'] is not None else -1)
        
        # Group by power to handle any duplicates at same (bias, power)
        from collections import defaultdict
        power_dict = defaultdict(list)
        for d in bias_data:
            if d['power'] is not None:
                power_dict[d['power']].append(d)
        
        # For each power, average if there are duplicates
        powers = []
        count_rates = []
        signal_rates = []
        dark_count_rates = []
        efficiencies = []
        count_rate_errors = []
        signal_rate_errors = []
        dark_count_rate_errors = []
        efficiency_errors = []
        
        for power in sorted(power_dict.keys()):
            items = power_dict[power]
            if len(items) > 1:
                print(f"  Warning: {len(items)} measurements at {bias}mV, {power}nW - averaging")
            
            powers.append(power)
            count_rates.append(np.mean([d['count_rate'] for d in items]))
            signal_rates.append(np.mean([d['signal_rate'] for d in items]))
            dark_count_rates.append(np.mean([d['dark_count_rate'] for d in items]))
            efficiencies.append(np.mean([d['efficiency'] for d in items]))
            count_rate_errors.append(np.mean([d['count_rate_error'] for d in items]))
            signal_rate_errors.append(np.mean([d['signal_rate_error'] for d in items]))
            dark_count_rate_errors.append(np.mean([d['dark_count_rate_error'] for d in items]))
            efficiency_errors.append(np.mean([d['efficiency_error'] for d in items]))
        
        if not powers:
            continue
        
        color = color_map[bias]
        label = f'{bias} mV'
        
        # Plot each rate type
        ax1.errorbar(powers, count_rates, yerr=count_rate_errors, marker='o', 
                    color=color, label=label, linewidth=2, capsize=3)
        ax2.errorbar(powers, signal_rates, yerr=signal_rate_errors, marker='s', 
                    color=color, label=label, linewidth=2, capsize=3)
        ax3.errorbar(powers, dark_count_rates, yerr=dark_count_rate_errors, marker='^', 
                    color=color, label=label, linewidth=2, capsize=3)
        ax4.errorbar(powers, efficiencies, yerr=efficiency_errors, marker='D', 
                    color=color, label=label, linewidth=2, capsize=3)
    
    # Configure subplots
    for ax, ylabel, title in [
        (ax1, 'Count Rate (Hz)', 'Count Rate vs Power'),
        (ax2, 'Signal Rate (Hz)', 'Signal Rate vs Power'),
        (ax3, 'Dark Count Rate (Hz)', 'Dark Count Rate vs Power'),
        (ax4, 'Efficiency', 'Efficiency vs Power')
    ]:
        ax.set_xlabel('Power (nW)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        if log_scale:
            ax.set_yscale('log')
            ax.set_xscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    scale_suffix = '_logscale' if log_scale else ''
    output_path = os.path.join(output_dir, f'multi_bias_rates_vs_power{scale_suffix}.png')
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
    
    # Group data by bias voltage
    bias_groups = group_by_bias(data)
    
    # Default: plot both if no flags specified
    if not args.plot_individual and not args.plot_combined:
        args.plot_individual = True
        args.plot_combined = True
    
    # Plot individual bias voltages
    if args.plot_individual:
        print("\n=== Plotting individual bias voltages ===")
        for bias, bias_data in sorted(bias_groups.items()):
            plot_single_bias_vs_power(bias, bias_data, args.output_dir, args.log_scale)
    
    # Plot combined multi-bias comparison
    if args.plot_combined and len(bias_groups) > 1:
        print("\n=== Plotting combined multi-bias comparison ===")
        plot_multi_bias_vs_power(bias_groups, args.output_dir, args.log_scale)

if __name__ == "__main__":
    main()
