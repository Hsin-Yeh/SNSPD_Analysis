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
        parts = line.strip().split('\t')
        if len(parts) < 8:
            continue
            
        try:
            target_voltage = float(parts[0])
            time_totalize = float(parts[5])
            # measurements start from index 7 (8th column) after sample_count
            measurements = np.array([float(x) for x in parts[7:]])
            # Divide each measurement by time_totalize to get counts/s
            rates = measurements / time_totalize
        except (ValueError, IndexError):
            continue
        
        # Calculate mean and std of count rate
        mean_count_rate = np.mean(rates)
        std_count_rate = np.std(rates)
        
        bias_voltages.append(target_voltage)
        count_rates.append(mean_count_rate)
        count_stds.append(std_count_rate)
    
    return np.array(bias_voltages), np.array(count_rates), np.array(count_stds), time_totalize


def extract_time_series_for_bias(filepath, target_bias_mv, bias_tolerance_mv=1.0, exact_only=False):
    """Extract per-sample time series for the requested bias voltage within tolerance.
    
    If exact_only=True, only return data if exact match found within tolerance.
    If exact_only=False, return closest match and warn if outside tolerance.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data_lines = lines[1:]
    candidates = []  # List of (delta, bias_mv, rates, time_totalize)
    
    for line in data_lines:
        parts = line.strip().split('\t')
        if len(parts) < 8:
            continue
        
        try:
            target_voltage = float(parts[0])
            time_totalize = float(parts[5])
            # measurements start from index 7
            measurements = np.array([float(x) for x in parts[7:]])
            # Convert to count rates by dividing by time_totalize (time per measurement)
            rates = measurements / time_totalize
        except (ValueError, IndexError):
            continue
        
        bias_mv = target_voltage * 1000
        delta = abs(bias_mv - target_bias_mv)
        candidates.append((delta, bias_mv, rates, time_totalize))
    
    if not candidates:
        return None
    
    # Sort by delta (closest first)
    candidates.sort(key=lambda x: x[0])
    delta, closest_bias, closest_series, time_totalize = candidates[0]
    
    # If exact_only, skip files that don't have exact match within tolerance
    if exact_only and delta > bias_tolerance_mv:
        return None
    
    if delta > bias_tolerance_mv:
        print(f"  Warning: No data at {target_bias_mv:.1f} mV. Closest is {closest_bias:.2f} mV (Δ={delta:.2f} mV, exceeds {bias_tolerance_mv:.1f} mV tolerance)")
    
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
    parser.add_argument('--allow-fallback', action='store_false', dest='exact_only',
                        help='Allow falling back to closest bias if exact match not found (default: only use exact matches)')
    args = parser.parse_args()
    args.exact_only = True  # Set default to True (exact matching only)
    
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
        series = extract_time_series_for_bias(filepath, args.bias_mv, args.bias_tolerance_mv, args.exact_only)
        if series is None:
            if args.exact_only:
                print(f"  Skipped: No data at {args.bias_mv:.1f} mV")
            continue
        
        rates = series['rates']
        time_totalize = series['time_totalize']
        if time_totalize is None or time_totalize <= 0:
            print(f"  Warning: Missing/invalid time_totalize in {filepath.name}, skipping")
            continue
        
        # Build timestamps for each sample
        # time_totalize is the duration of each measurement
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
