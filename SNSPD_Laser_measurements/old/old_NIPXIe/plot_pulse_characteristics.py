#!/usr/bin/env python3
"""
Plot SNSPD pulse characteristics (pulse_fall_range_ptp) vs bias and power
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
from plot_utils import read_analysis_files, group_by_power, group_by_bias, print_data_summary

parser = argparse.ArgumentParser(description='Plot pulse characteristics from analysis.json files')
parser.add_argument('--input_dir', '-i', nargs='+', default=['./plots/test'], 
                   help='Directory or directories containing analysis.json files (supports multiple)')
parser.add_argument('--output_dir', '-d', default='.', help='Output directory for plots')
parser.add_argument('--pattern', '-p', default='*_analysis.json', help='File pattern to match')
parser.add_argument('--recursive', '-r', action='store_true', default=True, help='Search recursively in subdirectories')
parser.add_argument('--mode', '-m', choices=['all', 'vs_bias', 'vs_power', 'ptp', 'amplitude'], default='all',
                   help='Plot mode: all, vs_bias, vs_power, ptp (ptp only), amplitude (amplitude only)')
parser.add_argument('--variable', '-v', choices=['ptp', 'rise_amplitude', 'both'], default='both',
                   help='Which variable to plot: ptp, rise_amplitude, or both')

def plot_pulse_ptp_vs_bias(power_groups, output_dir):
    """Plot pulse_fall_range_ptp vs bias voltage for multiple powers"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(power_groups)))
    
    print("\n=== Pulse PTP vs Bias data ===")
    for idx, (power, power_data) in enumerate(sorted(power_groups.items())):
        # Sort by bias
        power_data = sorted(power_data, key=lambda x: x['bias_voltage'])
        
        # Group by bias to handle duplicates
        from collections import defaultdict
        bias_dict = defaultdict(list)
        for d in power_data:
            if d['pulse_fall_range_ptp_mean'] is not None:
                bias_dict[d['bias_voltage']].append(d)
        
        bias_voltages = []
        pulse_ptp_means = []
        pulse_ptp_errors = []
        
        for bias in sorted(bias_dict.keys()):
            items = bias_dict[bias]
            if len(items) > 1:
                print(f"  Warning: {len(items)} measurements at {power}nW, {bias}mV - averaging")
            
            bias_voltages.append(bias)
            pulse_ptp_means.append(np.mean([d['pulse_fall_range_ptp_mean'] for d in items]))
            pulse_ptp_errors.append(np.mean([d['pulse_fall_range_ptp_std'] for d in items if d['pulse_fall_range_ptp_std'] is not None]))
        
        if not bias_voltages:
            print(f"Power {power} nW: No pulse PTP data")
            continue
        
        print(f"Power {power} nW: {len(bias_voltages)} bias points")
        
        color = colors[idx]
        label = f'{power} nW' if isinstance(power, int) else str(power)
        
        ax.errorbar(bias_voltages, pulse_ptp_means, yerr=pulse_ptp_errors, marker='o', 
                   color=color, label=label, linewidth=2, capsize=4, markersize=6)
    
    ax.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax.set_ylabel('Pulse Fall Range PTP (V)', fontsize=12)
    ax.set_title('Pulse Fall Range (Peak-to-Peak) vs Bias Voltage', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'pulse_ptp_vs_bias.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    plt.close()

def plot_pulse_ptp_vs_power(bias_groups, output_dir):
    """Plot pulse_fall_range_ptp vs power for multiple bias voltages"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors - use unique bias voltages only
    unique_biases = sorted(set(bias_groups.keys()))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_biases)))
    color_map = {bias: colors[i] for i, bias in enumerate(unique_biases)}
    
    print("\n=== Pulse PTP vs Power data ===")
    for bias, bias_data in sorted(bias_groups.items()):
        # Sort by power
        bias_data = sorted(bias_data, key=lambda x: x['power'] if x['power'] is not None else -1)
        
        # Group by power to handle duplicates
        from collections import defaultdict
        power_dict = defaultdict(list)
        for d in bias_data:
            if d['power'] is not None and d['pulse_fall_range_ptp_mean'] is not None:
                power_dict[d['power']].append(d)
        
        powers = []
        pulse_ptp_means = []
        pulse_ptp_errors = []
        
        for power in sorted(power_dict.keys()):
            items = power_dict[power]
            if len(items) > 1:
                print(f"  Warning: {len(items)} measurements at {bias}mV, {power}nW - averaging")
            
            powers.append(power)
            pulse_ptp_means.append(np.mean([d['pulse_fall_range_ptp_mean'] for d in items]))
            pulse_ptp_errors.append(np.mean([d['pulse_fall_range_ptp_std'] for d in items if d['pulse_fall_range_ptp_std'] is not None]))
        
        if not powers:
            print(f"Bias {bias} mV: No pulse PTP data")
            continue
        
        print(f"Bias {bias} mV: {len(powers)} power points")
        
        color = color_map[bias]
        label = f'{bias} mV'
        
        ax.errorbar(powers, pulse_ptp_means, yerr=pulse_ptp_errors, marker='s', 
                   color=color, label=label, linewidth=2, capsize=4, markersize=6)
    
    ax.set_xlabel('Power (nW)', fontsize=12)
    ax.set_ylabel('Pulse Fall Range PTP (V)', fontsize=12)
    ax.set_title('Pulse Fall Range (Peak-to-Peak) vs Power', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'pulse_ptp_vs_power.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    plt.close()

def plot_rise_amplitude_vs_bias(power_groups, output_dir):
    """Plot rise_amplitude vs bias voltage for multiple powers"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(power_groups)))
    
    print("\n=== Rise Amplitude vs Bias data ===")
    for idx, (power, power_data) in enumerate(sorted(power_groups.items())):
        # Sort by bias
        power_data = sorted(power_data, key=lambda x: x['bias_voltage'])
        
        # Group by bias to handle duplicates
        from collections import defaultdict
        bias_dict = defaultdict(list)
        for d in power_data:
            if d.get('rise_amplitude_mean') is not None:
                bias_dict[d['bias_voltage']].append(d)
        
        bias_voltages = []
        rise_amp_means = []
        rise_amp_errors = []
        
        for bias in sorted(bias_dict.keys()):
            items = bias_dict[bias]
            if len(items) > 1:
                print(f"  Warning: {len(items)} measurements at {power}nW, {bias}mV - averaging")
            
            bias_voltages.append(bias)
            rise_amp_means.append(np.mean([d['rise_amplitude_mean'] for d in items]))
            rise_amp_errors.append(np.mean([d['rise_amplitude_std'] for d in items if d.get('rise_amplitude_std') is not None]))
        
        if not bias_voltages:
            print(f"Power {power} nW: No rise amplitude data")
            continue
        
        print(f"Power {power} nW: {len(bias_voltages)} bias points")
        
        color = colors[idx]
        label = f'{power} nW' if isinstance(power, int) else str(power)
        
        ax.errorbar(bias_voltages, rise_amp_means, yerr=rise_amp_errors, marker='o', 
                   color=color, label=label, linewidth=2, capsize=4, markersize=6)
    
    ax.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax.set_ylabel('Rise Amplitude (V)', fontsize=12)
    ax.set_title('Pulse Rise Amplitude vs Bias Voltage', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'rise_amplitude_vs_bias.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    plt.close()

def plot_rise_amplitude_vs_power(bias_groups, output_dir):
    """Plot rise_amplitude vs power for multiple bias voltages"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors - use unique bias voltages only
    unique_biases = sorted(set(bias_groups.keys()))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_biases)))
    color_map = {bias: colors[i] for i, bias in enumerate(unique_biases)}
    
    print("\n=== Rise Amplitude vs Power data ===")
    for bias, bias_data in sorted(bias_groups.items()):
        # Sort by power
        bias_data = sorted(bias_data, key=lambda x: x['power'] if x['power'] is not None else -1)
        
        # Group by power to handle duplicates
        from collections import defaultdict
        power_dict = defaultdict(list)
        for d in bias_data:
            if d['power'] is not None and d.get('rise_amplitude_mean') is not None:
                power_dict[d['power']].append(d)
        
        powers = []
        rise_amp_means = []
        rise_amp_errors = []
        
        for power in sorted(power_dict.keys()):
            items = power_dict[power]
            if len(items) > 1:
                print(f"  Warning: {len(items)} measurements at {bias}mV, {power}nW - averaging")
            
            powers.append(power)
            rise_amp_means.append(np.mean([d['rise_amplitude_mean'] for d in items]))
            rise_amp_errors.append(np.mean([d['rise_amplitude_std'] for d in items if d.get('rise_amplitude_std') is not None]))
        
        if not powers:
            print(f"Bias {bias} mV: No rise amplitude data")
            continue
        
        print(f"Bias {bias} mV: {len(powers)} power points")
        
        color = color_map[bias]
        label = f'{bias} mV'
        
        ax.errorbar(powers, rise_amp_means, yerr=rise_amp_errors, marker='s', 
                   color=color, label=label, linewidth=2, capsize=4, markersize=6)
    
    ax.set_xlabel('Power (nW)', fontsize=12)
    ax.set_ylabel('Rise Amplitude (V)', fontsize=12)
    ax.set_title('Pulse Rise Amplitude vs Power', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'rise_amplitude_vs_power.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    plt.close()

def main():
    args = parser.parse_args()
    print("="*60)
    print("SNSPD Pulse Characteristics Plotting")
    print("="*60)
    input_dirs = args.input_dir if isinstance(args.input_dir, list) else [args.input_dir]
    print(f"\nReading analysis files from: {', '.join(input_dirs)}")
    print(f"File pattern: {args.pattern}")
    print(f"Recursive search: {args.recursive}")
    print(f"Mode: {args.mode}")
    
    # Read data
    data = read_analysis_files(args.input_dir, args.pattern, args.recursive)
    
    if not data:
        print("\nNo valid data found!")
        return
    
    print_data_summary(data)
    
    # Group data
    power_groups = group_by_power(data)
    bias_groups = group_by_bias(data)
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which variables to plot based on mode and variable args
    # If mode is 'ptp' or 'amplitude', override variable selection
    if args.mode == 'ptp':
        plot_ptp = True
        plot_amplitude = False
    elif args.mode == 'amplitude':
        plot_ptp = False
        plot_amplitude = True
    else:
        # For 'all', 'vs_bias', 'vs_power' modes, use variable argument
        plot_ptp = args.variable in ['ptp', 'both']
        plot_amplitude = args.variable in ['rise_amplitude', 'both']
    
    # Generate plots based on mode and variable selection
    if args.mode in ['all', 'vs_bias'] and plot_ptp:
        print("\n" + "="*60)
        print("GENERATING PULSE PTP VS BIAS VOLTAGE PLOT")
        print("="*60)
        plot_pulse_ptp_vs_bias(power_groups, args.output_dir)
    
    if args.mode in ['all', 'vs_power'] and plot_ptp:
        print("\n" + "="*60)
        print("GENERATING PULSE PTP VS POWER PLOT")
        print("="*60)
        plot_pulse_ptp_vs_power(bias_groups, args.output_dir)
    
    if args.mode in ['all', 'vs_bias'] and plot_amplitude:
        print("\n" + "="*60)
        print("GENERATING RISE AMPLITUDE VS BIAS VOLTAGE PLOT")
        print("="*60)
        plot_rise_amplitude_vs_bias(power_groups, args.output_dir)
    
    if args.mode in ['all', 'vs_power'] and plot_amplitude:
        print("\n" + "="*60)
        print("GENERATING RISE AMPLITUDE VS POWER PLOT")
        print("="*60)
        plot_rise_amplitude_vs_power(bias_groups, args.output_dir)
    
    print("\n" + "="*60)
    print("PLOTTING COMPLETE")
    print("="*60)
    print(f"\nAll plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
