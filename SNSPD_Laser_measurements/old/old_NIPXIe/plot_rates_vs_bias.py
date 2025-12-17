#!/usr/bin/env python3

import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import glob
import re
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot count rates vs bias voltage from analysis.json files')
parser.add_argument('--input_dir', '-i', default='./plots/test', help='Directory containing analysis.json files')
parser.add_argument('--output_dir', '-d', default='.', help='Output directory for plots')
parser.add_argument('--pattern', '-p', default='*_analysis.json', help='File pattern to match')
args = parser.parse_args()

def extract_bias_voltage(filename):
    """Extract bias voltage from filename (e.g., 65mV -> 65)"""
    match = re.search(r'_(\d+)mV_', filename)
    if match:
        return int(match.group(1))
    return None

def extract_power(filename):
    """Extract power from filename (e.g., 363nW -> 363)"""
    match = re.search(r'_(\d+)nW_', filename)
    if match:
        return int(match.group(1))
    return None

def read_analysis_files(input_dir, pattern):
    """Read all analysis.json files and extract rates and bias voltages"""
    files = glob.glob(os.path.join(input_dir, pattern))
    
    data = []
    for filepath in files:
        filename = os.path.basename(filepath)
        bias_voltage = extract_bias_voltage(filename)
        power = extract_power(filename)
        
        if bias_voltage is None:
            print(f"Warning: Could not extract bias voltage from {filename}")
            continue
        
        try:
            with open(filepath, 'r') as f:
                analysis = json.load(f)
            
            summary = analysis.get('summary_statistics', {})
            
            # Extract rates and errors
            count_rate = summary.get('count_rate', 0)
            signal_rate = summary.get('signal_rate', 0)
            dark_count_rate = summary.get('dark_count_rate', 0)
            efficiency = summary.get('efficiency', 0)
            
            count_rate_error = summary.get('count_rate_error', 0)
            signal_rate_error = summary.get('signal_rate_error', 0)
            dark_count_rate_error = summary.get('dark_count_rate_error', 0)
            efficiency_error = summary.get('efficiency_error', 0)
            resistance = summary.get('resistance_ohm', None)
            
            data.append({
                'bias_voltage': bias_voltage,
                'power': power,
                'count_rate': count_rate,
                'signal_rate': signal_rate,
                'dark_count_rate': dark_count_rate,
                'efficiency': efficiency,
                'count_rate_error': count_rate_error,
                'signal_rate_error': signal_rate_error,
                'dark_count_rate_error': dark_count_rate_error,
                'efficiency_error': efficiency_error,
                'resistance': resistance,
                'filename': filename
            })
            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
    
    # Sort by power then bias voltage
    data.sort(key=lambda x: (x['power'] if x['power'] is not None else -1, x['bias_voltage']))
    
    return data

def plot_individual_power_rates(power, power_data, output_dir):
    """Plot count rates for a single power level"""
    
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
    
    # Plot rates with dual y-axis and error bars
    # Left axis for signal rate
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
    
    # Plot efficiency with error bars
    ax2.errorbar(bias_voltages, efficiencies, yerr=efficiency_errors, marker='D', color='red', 
                 linewidth=2, label='Efficiency', capsize=4, capthick=1.5)
    ax2.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax2.set_ylabel('Efficiency', fontsize=12)
    ax2.set_title(f'Efficiency vs Bias Voltage ({power_label})', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot resistance vs bias voltage
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
    
    # Set same x-axis limits for all subplots
    x_min, x_max = min(bias_voltages), max(bias_voltages)
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    ax3.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    
    # Save with power in filename
    power_str = f'{power}nW' if isinstance(power, int) else str(power)
    output_path = os.path.join(output_dir, f'rates_vs_bias_{power_str}.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved individual power plot to {output_path}")
    plt.close()

def plot_multi_power_rates(data, output_dir):
    """Plot count rates vs bias voltage for multiple powers on the same plot"""
    
    # Group data by power
    from collections import defaultdict
    power_groups = defaultdict(list)
    for d in data:
        power = d['power'] if d['power'] is not None else 'Unknown'
        power_groups[power].append(d)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define colors for different powers
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
        
        # Plot count rate
        ax1.errorbar(bias_voltages, count_rates, yerr=count_rate_errors, marker='o', 
                    color=color, label=label, linewidth=2, capsize=3)
        
        # Plot signal rate
        ax2.errorbar(bias_voltages, signal_rates, yerr=signal_rate_errors, marker='s', 
                    color=color, label=label, linewidth=2, capsize=3)
        
        # Plot dark count rate
        ax3.errorbar(bias_voltages, dark_count_rates, yerr=dark_count_rate_errors, marker='^', 
                    color=color, label=label, linewidth=2, capsize=3)
        
        # Plot efficiency
        ax4.errorbar(bias_voltages, efficiencies, yerr=efficiency_errors, marker='D', 
                    color=color, label=label, linewidth=2, capsize=3)
    
    # Configure subplots
    ax1.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax1.set_ylabel('Count Rate (Hz)', fontsize=12)
    ax1.set_title('Count Rate vs Bias Voltage', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax2.set_ylabel('Signal Rate (Hz)', fontsize=12)
    ax2.set_title('Signal Rate vs Bias Voltage', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax3.set_ylabel('Dark Count Rate (Hz)', fontsize=12)
    ax3.set_title('Dark Count Rate vs Bias Voltage', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax4.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax4.set_ylabel('Efficiency', fontsize=12)
    ax4.set_title('Efficiency vs Bias Voltage', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'multi_power_rates_vs_bias.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved multi-power plot to {output_path}")
    plt.close()

def plot_rates_vs_bias(data, output_dir):
    """Plot count rate, signal rate, and dark count rate vs bias voltage"""
    
    if not data:
        print("No data to plot!")
        return
    
    bias_voltages = [d['bias_voltage'] for d in data]
    count_rates = [d['count_rate'] for d in data]
    signal_rates = [d['signal_rate'] for d in data]
    dark_count_rates = [d['dark_count_rate'] for d in data]
    efficiencies = [d['efficiency'] for d in data]
    
    count_rate_errors = [d['count_rate_error'] for d in data]
    signal_rate_errors = [d['signal_rate_error'] for d in data]
    dark_count_rate_errors = [d['dark_count_rate_error'] for d in data]
    efficiency_errors = [d['efficiency_error'] for d in data]
    
    # Get resistance data
    resistances = [d['resistance'] for d in data if d['resistance'] is not None]
    resistance_voltages = [d['bias_voltage'] for d in data if d['resistance'] is not None]
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    
    # Plot rates with dual y-axis and error bars
    # Left axis for signal rate
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
    
    ax1.set_title('Count Rates vs Bias Voltage', fontsize=14, fontweight='bold')
    
    # Plot efficiency with error bars
    ax2.errorbar(bias_voltages, efficiencies, yerr=efficiency_errors, marker='D', color='red', 
                 linewidth=2, label='Efficiency', capsize=4, capthick=1.5)
    ax2.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax2.set_ylabel('Efficiency', fontsize=12)
    ax2.set_title('Efficiency vs Bias Voltage', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot resistance vs bias voltage
    if resistances:
        ax3.plot(resistance_voltages, resistances, marker='o', color='purple', linewidth=2, markersize=8)
        ax3.set_xlabel('Bias Voltage (mV)', fontsize=12)
        ax3.set_ylabel('Resistance (Ω)', fontsize=12)
        ax3.set_title('Resistance vs Bias Voltage', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No resistance data available', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Resistance vs Bias Voltage', fontsize=14, fontweight='bold')
    
    # Set same x-axis limits for all subplots
    x_min, x_max = min(bias_voltages), max(bias_voltages)
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    ax3.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'rates_vs_bias_voltage.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()
    
    # Also create a log scale version for rates with error bars
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.errorbar(bias_voltages, count_rates, yerr=count_rate_errors, marker='o', label='Count Rate', 
                linewidth=2, capsize=4, capthick=1.5)
    ax.errorbar(bias_voltages, signal_rates, yerr=signal_rate_errors, marker='s', label='Signal Rate', 
                linewidth=2, capsize=4, capthick=1.5)
    ax.errorbar(bias_voltages, dark_count_rates, yerr=dark_count_rate_errors, marker='^', label='Dark Count Rate', 
                linewidth=2, capsize=4, capthick=1.5)
    ax.set_yscale('log')
    ax.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax.set_ylabel('Rate (Hz, log scale)', fontsize=12)
    ax.set_title('Count Rates vs Bias Voltage (Log Scale)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    output_path_log = os.path.join(output_dir, 'rates_vs_bias_voltage_logscale.png')
    plt.savefig(output_path_log, dpi=300)
    print(f"Saved log-scale plot to {output_path_log}")
    plt.close()

def main():
    print(f"Reading analysis files from: {args.input_dir}")
    print(f"File pattern: {args.pattern}")
    
    data = read_analysis_files(args.input_dir, args.pattern)
    
    print(f"\nFound {len(data)} files with valid data:")
    for d in data:
        power_str = f"{d['power']} nW" if d['power'] is not None else "Unknown power"
        print(f"  {d['bias_voltage']} mV, {power_str}: count_rate={d['count_rate']:.3f} Hz, "
              f"signal_rate={d['signal_rate']:.3f} Hz, "
              f"dark_count_rate={d['dark_count_rate']:.3f} Hz, "
              f"efficiency={d['efficiency']:.3e}")
    
    if data:
        # Group data by power
        from collections import defaultdict
        power_groups = defaultdict(list)
        for d in data:
            power = d['power'] if d['power'] is not None else 'Unknown'
            power_groups[power].append(d)
        
        # Plot individual power datasets
        for power, power_data in sorted(power_groups.items()):
            plot_individual_power_rates(power, power_data, args.output_dir)
        
        # Plot all data together (if only one power)
        if len(power_groups) == 1:
            plot_rates_vs_bias(data, args.output_dir)
        
        # Plot multiple powers together (if more than one power)
        if len(power_groups) > 1:
            plot_multi_power_rates(data, args.output_dir)
    else:
        print("\nNo valid data found to plot!")

if __name__ == "__main__":
    main()
