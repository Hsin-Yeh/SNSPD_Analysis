#!/usr/bin/env python3
"""
Plot dark count rate fluctuations over time for a target bias voltage.
Uses filename timestamps as start times and per-sample totalize time from files
to build a continuous time series across multiple measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
import re
import argparse

def parse_filename_timestamp(filename):
    """Extract timestamp from filename. Example: *_20251212_1749.txt"""
    match = re.search(r'(\d{8}_\d{4})\.txt', filename)
    if match:
        timestamp_str = match.group(1)
        return datetime.strptime(timestamp_str, '%Y%m%d_%H%M')
    return None

def read_counter_file(filepath):
    """Read counter data file and extract bias voltage and per-sample count rates."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header
    data_lines = lines[1:]
    
    bias_voltages = []
    count_rates = []
    count_stds = []
    time_totalize = None
    
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
            
        bias_voltage = float(parts[0])
        # Integration time per measurement (column index 4)
        try:
            time_totalize = float(parts[4])
        except (ValueError, IndexError):
            time_totalize = None
        # measurements start from index 6 (7th column)
        measurements = [float(x) for x in parts[6:]]
        if time_totalize and time_totalize > 0:
            measurements = [m / time_totalize for m in measurements]
        
        # Calculate mean and std of count rate
        mean_count_rate = np.mean(measurements)
        std_count_rate = np.std(measurements)
        
        bias_voltages.append(bias_voltage)
        count_rates.append(mean_count_rate)
        count_stds.append(std_count_rate)
    
    return np.array(bias_voltages), np.array(count_rates), np.array(count_stds), time_totalize


def extract_time_series_for_bias(filepath, target_bias_mv, bias_tolerance_mv=1.0):
    """Extract per-sample time series for the closest bias voltage."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data_lines = lines[1:]
    closest_series = None
    closest_bias = None
    closest_delta = None
    time_totalize = None
    
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        
        try:
            bias_voltage = float(parts[0])
            time_totalize = float(parts[4])
        except (ValueError, IndexError):
            continue
        
        measurements = np.array([float(x) for x in parts[6:]])
        if time_totalize > 0:
            rates = measurements / time_totalize
        else:
            rates = measurements
        
        bias_mv = bias_voltage * 1000
        delta = abs(bias_mv - target_bias_mv)
        if closest_delta is None or delta < closest_delta:
            closest_delta = delta
            closest_series = rates
            closest_bias = bias_mv
    
    if closest_series is None:
        return None
    
    if closest_delta is not None and closest_delta > bias_tolerance_mv:
        print(f"  Warning: Closest bias is {closest_bias:.2f} mV (Δ={closest_delta:.2f} mV)")
    
    return {
        'bias_mv': closest_bias,
        'rates': closest_series,
        'time_totalize': time_totalize
    }

def main():
    parser = argparse.ArgumentParser(description='Plot dark count time series at a target bias voltage')
    parser.add_argument('data_dir', type=str, help='Path to dark count folder (e.g., /.../2-7/6K/0nW)')
    parser.add_argument('--bias-mv', type=float, required=True, help='Target bias voltage in mV')
    parser.add_argument('--output', type=str, default='output/dark_count_time_series.png',
                        help='Output plot file path')
    parser.add_argument('--bias-tolerance-mv', type=float, default=1.0,
                        help='Allowed bias voltage mismatch in mV (default: 1.0)')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1
    
    # Find all text files in the dark count folder
    all_files = list(data_dir.rglob('*.txt'))
    
    # Extract timestamps
    dark_files = []
    for filepath in all_files:
        timestamp = parse_filename_timestamp(filepath.name)
        if timestamp is not None:
            dark_files.append((filepath, timestamp))
    
    # Sort by timestamp
    dark_files.sort(key=lambda x: x[1])
    
    print(f"Found {len(dark_files)} dark count files")
    if len(dark_files) == 0:
        print("No dark count files found!")
        return 1
    
    # Build time series
    time_points = []
    rate_points = []
    
    for filepath, timestamp in dark_files:
        print(f"Processing: {filepath.name} at {timestamp.strftime('%Y-%m-%d %H:%M')}")
        series = extract_time_series_for_bias(filepath, args.bias_mv, args.bias_tolerance_mv)
        if series is None:
            continue
        
        rates = series['rates']
        time_totalize = series['time_totalize']
        if time_totalize is None or time_totalize <= 0:
            print(f"  Warning: Missing/invalid totalize time in {filepath.name}, skipping")
            continue
        
        # Build timestamps for each sample
        for i, rate in enumerate(rates):
            sample_time = timestamp + timedelta(seconds=time_totalize * i)
            time_points.append(sample_time)
            rate_points.append(rate)
    
    if not time_points:
        print("No valid data points collected.")
        return 1
    
    # Plot with absolute timestamps
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_points, rate_points, 'o', markersize=4, alpha=0.7)
    ax.set_xlabel('Time (absolute)', fontsize=12)
    ax.set_ylabel('Dark Count Rate (counts/s)', fontsize=12)
    ax.set_title(f'Dark Count Fluctuations at {args.bias_mv:.1f} mV', fontsize=14, weight='bold')
    ax.set_ylim(0, 4000)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    
    # Save plot
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Print summary
    rates = np.array(rate_points)
    print("\n" + "="*60)
    print(f"Dark Count Summary at {args.bias_mv:.1f} mV")
    print("="*60)
    print(f"Mean: {np.mean(rates):.1f} counts/s")
    print(f"Std:  {np.std(rates):.1f} counts/s")
    print(f"Min:  {np.min(rates):.1f} counts/s")
    print(f"Max:  {np.max(rates):.1f} counts/s")
    print(f"Variation: {(np.std(rates)/np.mean(rates)*100):.2f}%")
    
    plt.show()
    return 0

if __name__ == "__main__":
    exit(main())
