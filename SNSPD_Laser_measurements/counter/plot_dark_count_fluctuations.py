#!/usr/bin/env python3
"""
Plot dark count rate fluctuations over time
Shows how dark count rates vary across different bias voltages and measurement times
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import re

def parse_filename(filename):
    """Extract power and timestamp from filename"""
    # Example: SMSPD_3_2-7_0nW_20251212_1749.txt
    match = re.search(r'(\d+)nW_(\d{8}_\d{4})\.txt', filename)
    if match:
        power_nw = int(match.group(1))
        timestamp_str = match.group(2)
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M')
        return power_nw, timestamp
    return None, None

def read_counter_file(filepath):
    """Read counter data file and extract bias voltage and count rates"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header
    data_lines = lines[1:]
    
    bias_voltages = []
    count_rates = []
    count_stds = []
    
    for line in data_lines:
        parts = line.strip().split('\t')
        if len(parts) < 7:
            continue
            
        bias_voltage = float(parts[0])
        # measurements start from index 6 (7th column)
        measurements = [float(x) for x in parts[6:]]
        
        # Calculate mean and std of count rate
        mean_count_rate = np.mean(measurements)
        std_count_rate = np.std(measurements)
        
        bias_voltages.append(bias_voltage)
        count_rates.append(mean_count_rate)
        count_stds.append(std_count_rate)
    
    return np.array(bias_voltages), np.array(count_rates), np.array(count_stds)

def main():
    # Data directory
    data_dir = Path('/Users/ya/Documents/Projects/SNSPD/SNSPD_data/SMSPD_3/Counter_sweep_power_3')
    
    # Find all text files
    all_files = list(data_dir.rglob('*.txt'))
    
    # Find dark count files (0nW)
    dark_files = []
    
    for filepath in all_files:
        power, timestamp = parse_filename(filepath.name)
        if power is not None and timestamp is not None and power == 0:
            dark_files.append((filepath, timestamp))
    
    # Sort by timestamp
    dark_files.sort(key=lambda x: x[1])
    
    print(f"Found {len(dark_files)} dark count files")
    
    if len(dark_files) == 0:
        print("No dark count files found!")
        return
    
    # Create output directory
    output_dir = Path('output/SMSPD_3')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Dark count rate vs bias voltage for each measurement time
    colors = plt.cm.plasma(np.linspace(0, 1, len(dark_files)))
    
    for idx, (filepath, timestamp) in enumerate(dark_files):
        print(f"Processing: {filepath.name} at {timestamp.strftime('%H:%M')}")
        
        bias_voltages, count_rates, count_stds = read_counter_file(filepath)
        
        time_label = timestamp.strftime('%H:%M')
        axes[0].errorbar(bias_voltages * 1000, count_rates, yerr=count_stds,
                        fmt='o-', label=time_label, color=colors[idx],
                        linewidth=2, markersize=5, alpha=0.7, capsize=3)
    
    axes[0].set_xlabel('Bias Voltage (mV)', fontsize=12)
    axes[0].set_ylabel('Dark Count Rate (counts/s)', fontsize=12)
    axes[0].set_title('Dark Count Rate vs Bias Voltage at Different Times', fontsize=14)
    axes[0].legend(fontsize=10, loc='best', title='Time')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Dark count rate fluctuations over time at selected bias voltages
    # Read first file to get available bias voltages
    first_bias, _, _ = read_counter_file(dark_files[0][0])
    
    # Select a few representative bias voltages
    num_bias_to_plot = min(5, len(first_bias))
    bias_indices = np.linspace(0, len(first_bias)-1, num_bias_to_plot, dtype=int)
    selected_biases = first_bias[bias_indices]
    
    colors2 = plt.cm.viridis(np.linspace(0, 1, num_bias_to_plot))
    
    for bias_idx, bias_val in zip(bias_indices, selected_biases):
        timestamps = []
        rates_at_bias = []
        stds_at_bias = []
        
        for filepath, timestamp in dark_files:
            bias_voltages, count_rates, count_stds = read_counter_file(filepath)
            
            # Find the closest bias voltage
            closest_idx = np.argmin(np.abs(bias_voltages - bias_val))
            
            timestamps.append(timestamp)
            rates_at_bias.append(count_rates[closest_idx])
            stds_at_bias.append(count_stds[closest_idx])
        
        # Convert timestamps to minutes from first measurement
        time_minutes = [(t - timestamps[0]).total_seconds() / 60 for t in timestamps]
        
        axes[1].errorbar(time_minutes, rates_at_bias, yerr=stds_at_bias,
                        fmt='o-', label=f'{bias_val*1000:.1f} mV',
                        color=colors2[bias_idx % num_bias_to_plot],
                        linewidth=2, markersize=6, alpha=0.7, capsize=3)
    
    axes[1].set_xlabel('Time (minutes from first measurement)', fontsize=12)
    axes[1].set_ylabel('Dark Count Rate (counts/s)', fontsize=12)
    axes[1].set_title('Dark Count Rate Fluctuations Over Time', fontsize=14)
    axes[1].legend(fontsize=10, loc='best', title='Bias Voltage')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / 'dark_count_fluctuations.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Dark Count Statistics Summary")
    print("="*60)
    
    for bias_idx, bias_val in zip(bias_indices, selected_biases):
        rates = []
        for filepath, timestamp in dark_files:
            bias_voltages, count_rates, _ = read_counter_file(filepath)
            closest_idx = np.argmin(np.abs(bias_voltages - bias_val))
            rates.append(count_rates[closest_idx])
        
        rates = np.array(rates)
        print(f"\nBias: {bias_val*1000:.1f} mV")
        print(f"  Mean: {np.mean(rates):.1f} counts/s")
        print(f"  Std:  {np.std(rates):.1f} counts/s")
        print(f"  Min:  {np.min(rates):.1f} counts/s")
        print(f"  Max:  {np.max(rates):.1f} counts/s")
        print(f"  Variation: {(np.std(rates)/np.mean(rates)*100):.2f}%")
    
    plt.show()

if __name__ == "__main__":
    main()
