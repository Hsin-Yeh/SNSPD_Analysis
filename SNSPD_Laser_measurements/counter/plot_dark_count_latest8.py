#!/usr/bin/env python3
"""
Plot dark count rate fluctuations for the latest 8 measurements
Shows temporal stability of dark counts
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import re

def parse_filename(filename):
    """Extract power and timestamp from filename"""
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
    
    data_lines = lines[1:]
    
    bias_voltages = []
    count_rates = []
    count_errors = []  # standard error
    
    for line in data_lines:
        parts = line.strip().split('\t')
        if len(parts) < 7:
            continue
            
        bias_voltage = float(parts[0])
        measurements = [float(x) for x in parts[6:]]
        
        median_count_rate = np.median(measurements)
        # Standard error = std / sqrt(n)
        std_error = np.std(measurements) / np.sqrt(len(measurements))
        
        bias_voltages.append(bias_voltage)
        count_rates.append(median_count_rate)
        count_errors.append(std_error)
    
    return np.array(bias_voltages), np.array(count_rates), np.array(count_errors)

def main():
    # Data directory
    data_dir = Path('/Users/ya/Documents/Projects/SNSPD/SNSPD_data/SMSPD_3/Counter_sweep_power_3/2-7/6K/0nW')
    
    # Get all dark count files and sort by modification time (newest first)
    all_files = sorted(data_dir.glob('*.txt'), key=os.path.getmtime, reverse=True)
    
    # Take the latest 8
    latest_files = all_files[:8]
    
    print(f"Analyzing the latest {len(latest_files)} dark count files:")
    
    # Parse and sort by timestamp
    dark_files = []
    for filepath in latest_files:
        power, timestamp = parse_filename(filepath.name)
        if power == 0 and timestamp is not None:
            dark_files.append((filepath, timestamp))
    
    # Sort by timestamp (oldest first for plotting)
    dark_files.sort(key=lambda x: x[1])
    
    # Create output directory
    output_dir = Path('output/SMSPD_3')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Dark count rate vs bias voltage for each measurement
    colors = plt.cm.plasma(np.linspace(0, 1, len(dark_files)))
    
    for idx, (filepath, timestamp) in enumerate(dark_files):
        print(f"  {timestamp.strftime('%Y-%m-%d %H:%M')}: {filepath.name}")
        
        bias_voltages, count_rates, count_errors = read_counter_file(filepath)
        
        time_label = timestamp.strftime('%m/%d %H:%M')
        axes[0].errorbar(bias_voltages * 1000, count_rates, yerr=count_errors,
                        fmt='o-', label=time_label, color=colors[idx],
                        linewidth=2, markersize=5, alpha=0.7, capsize=3)
    
    axes[0].set_xlabel('Bias Voltage (mV)', fontsize=12)
    axes[0].set_ylabel('Median Dark Count Rate (counts/s)', fontsize=12)
    axes[0].set_title('Dark Count Rate vs Bias Voltage (Latest 8 Measurements)', fontsize=14)
    axes[0].legend(fontsize=9, loc='best', title='Time', ncol=2)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Dark count rate fluctuations over time at selected bias voltages
    first_bias, _, _ = read_counter_file(dark_files[0][0])
    
    # Select representative bias voltages
    num_bias_to_plot = min(5, len(first_bias))
    bias_indices = np.linspace(0, len(first_bias)-1, num_bias_to_plot, dtype=int)
    selected_biases = first_bias[bias_indices]
    
    colors2 = plt.cm.viridis(np.linspace(0, 1, num_bias_to_plot))
    
    for bias_idx, bias_val in zip(bias_indices, selected_biases):
        timestamps = []
        rates_at_bias = []
        errors_at_bias = []
        
        for filepath, timestamp in dark_files:
            bias_voltages, count_rates, count_errors = read_counter_file(filepath)
            
            closest_idx = np.argmin(np.abs(bias_voltages - bias_val))
            
            timestamps.append(timestamp)
            rates_at_bias.append(count_rates[closest_idx])
            errors_at_bias.append(count_errors[closest_idx])
        
        # Convert to hours from first measurement
        time_hours = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]
        
        axes[1].errorbar(time_hours, rates_at_bias, yerr=errors_at_bias,
                        fmt='o-', label=f'{bias_val*1000:.1f} mV',
                        color=colors2[bias_idx % num_bias_to_plot],
                        linewidth=2, markersize=6, alpha=0.7, capsize=3)
    
    axes[1].set_xlabel('Time (hours from first measurement)', fontsize=12)
    axes[1].set_ylabel('Median Dark Count Rate (counts/s)', fontsize=12)
    axes[1].set_title('Dark Count Rate Stability Over Time (Error bars = SE)', fontsize=14)
    axes[1].legend(fontsize=10, loc='best', title='Bias Voltage')
    axes[1].grid(True, alpha=0.3)
    
    # Create third plot for drift analysis
    fig2, ax3 = plt.subplots(1, 1, figsize=(14, 6))
    
    # Plot normalized drift for each bias voltage
    for bias_idx, bias_val in zip(bias_indices, selected_biases):
        timestamps = []
        rates_at_bias = []
        errors_at_bias = []
        
        for filepath, timestamp in dark_files:
            bias_voltages, count_rates, count_errors = read_counter_file(filepath)
            closest_idx = np.argmin(np.abs(bias_voltages - bias_val))
            timestamps.append(timestamp)
            rates_at_bias.append(count_rates[closest_idx])
            errors_at_bias.append(count_errors[closest_idx])
        
        time_hours = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]
        rates_array = np.array(rates_at_bias)
        
        # Normalize to first measurement (show relative change)
        if rates_array[0] > 0:
            normalized_rates = (rates_array - rates_array[0]) / rates_array[0] * 100
            
            # Linear fit to detect drift
            if len(time_hours) > 2:
                coeffs = np.polyfit(time_hours, rates_array, 1)
                slope = coeffs[0]
                drift_per_hour = slope / rates_array[0] * 100 if rates_array[0] > 0 else 0
                
                # Plot fit line
                fit_line = np.poly1d(coeffs)
                fit_normalized = (fit_line(time_hours) - rates_array[0]) / rates_array[0] * 100
                
                ax3.plot(time_hours, normalized_rates, 'o-', 
                        label=f'{bias_val*1000:.1f} mV ({drift_per_hour:+.2f}%/hr)',
                        color=colors2[bias_idx % num_bias_to_plot],
                        linewidth=2, markersize=6, alpha=0.7)
                ax3.plot(time_hours, fit_normalized, '--', 
                        color=colors2[bias_idx % num_bias_to_plot],
                        linewidth=1.5, alpha=0.5)
    
    ax3.set_xlabel('Time (hours from first measurement)', fontsize=12)
    ax3.set_ylabel('Relative Change from Initial (%)', fontsize=12)
    ax3.set_title('Dark Count Rate Drift Analysis (Normalized to t=0)', fontsize=14)
    ax3.legend(fontsize=10, loc='best', title='Bias Voltage (drift rate)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / 'dark_count_latest8_fluctuations.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Save drift analysis plot
    output_file2 = output_dir / 'dark_count_drift_analysis.png'
    fig2.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Drift analysis saved to: {output_file2}")
    
    # Print statistics
    print("\n" + "="*70)
    print("Dark Count Statistics Summary (Latest 8 Measurements)")
    print("="*70)
    print(f"Time span: {dark_files[0][1].strftime('%Y-%m-%d %H:%M')} to {dark_files[-1][1].strftime('%Y-%m-%d %H:%M')}")
    print(f"Duration: {(dark_files[-1][1] - dark_files[0][1]).total_seconds()/3600:.2f} hours")
    
    print("\n" + "="*70)
    print("DRIFT ANALYSIS")
    print("="*70)
    
    for bias_idx, bias_val in zip(bias_indices, selected_biases):
        rates = []
        stat_errors = []
        timestamps_list = []
        for filepath, timestamp in dark_files:
            bias_voltages, count_rates, count_errors = read_counter_file(filepath)
            closest_idx = np.argmin(np.abs(bias_voltages - bias_val))
            rates.append(count_rates[closest_idx])
            stat_errors.append(count_errors[closest_idx])
            timestamps_list.append(timestamp)
        
        rates = np.array(rates)
        stat_errors = np.array(stat_errors)
        
        # Calculate drift rate using linear regression
        time_hours = [(t - timestamps_list[0]).total_seconds() / 3600 for t in timestamps_list]
        coeffs = np.polyfit(time_hours, rates, 1)
        slope = coeffs[0]  # counts/s per hour
        intercept = coeffs[1]
        
        # Calculate R-squared for goodness of fit
        fit_rates = slope * np.array(time_hours) + intercept
        ss_res = np.sum((rates - fit_rates)**2)
        ss_tot = np.sum((rates - np.mean(rates))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Temporal fluctuation metrics
        median_rate = np.median(rates)
        temporal_std = np.std(rates)
        mean_stat_error = np.mean(stat_errors)
        
        # Ratio: how much larger is temporal fluctuation vs statistical error?
        fluctuation_ratio = temporal_std / mean_stat_error if mean_stat_error > 0 else np.inf
        
        # Drift metrics
        drift_per_hour_abs = slope  # counts/s per hour
        drift_per_hour_pct = (slope / median_rate * 100) if median_rate > 0 else 0  # % per hour
        total_drift_pct = (slope * time_hours[-1] / median_rate * 100) if median_rate > 0 else 0
        
        print(f"\nBias: {bias_val*1000:.1f} mV")
        print(f"  Median:              {median_rate:8.1f} counts/s")
        print(f"  Temporal Std:        {temporal_std:8.1f} counts/s")
        print(f"  Mean Stat. Error:    {mean_stat_error:8.1f} counts/s")
        print(f"  Temporal/Stat ratio: {fluctuation_ratio:8.2f}x")
        print(f"  ---")
        print(f"  Drift rate:          {drift_per_hour_abs:8.2f} counts/s/hr ({drift_per_hour_pct:+.3f}%/hr)")
        print(f"  Total drift (7hr):   {total_drift_pct:+.2f}%")
        print(f"  R² (linear fit):     {r_squared:8.4f}")
        print(f"  Min:                 {np.min(rates):8.1f} counts/s")
        print(f"  Max:                 {np.max(rates):8.1f} counts/s")
        print(f"  Range:               {np.max(rates)-np.min(rates):8.1f} counts/s")
        if median_rate > 0:
            print(f"  CV (%):               {(temporal_std/median_rate*100):8.2f}%")
        
        # Interpretation
        if abs(drift_per_hour_pct) > 0.5 and r_squared > 0.5:
            print(f"  ⚠️  Strong drift detected (R²={r_squared:.3f})")
        elif abs(drift_per_hour_pct) > 0.2 and r_squared > 0.3:
            print(f"  ⚠️  Moderate drift detected (R²={r_squared:.3f})")
        elif fluctuation_ratio > 3:
            print(f"  ⚠️  High fluctuation but no clear drift (likely noise/oscillation)")
        elif fluctuation_ratio > 1.5:
            print(f"  ⚠️  Some fluctuation (drift uncertain)")
        else:
            print(f"  ✓  Stable (fluctuation ~ statistical)")
    
    plt.show()

if __name__ == "__main__":
    main()
