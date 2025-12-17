#!/usr/bin/env python3
"""
Utility functions for reading and organizing SNSPD analysis data
"""

import json
import glob
import re
import os
import numpy as np
from collections import defaultdict

# Constants
SOURCE_RATE = 10e6  # 10 MHz laser pulse rate

def calculate_errors_from_events(events, signal_window_ns=20):
    """
    Calculate rate errors using time binning method
    
    Args:
        events: List of event dictionaries with 'pulse_time' field
        signal_window_ns: Time window for signal events (default 20ns)
    
    Returns:
        dict: Dictionary with calculated errors
    """
    if not events or len(events) == 0:
        return {
            'count_rate_error': 0.0,
            'signal_rate_error': 0.0,
            'dark_count_rate_error': 0.0
        }
    
    # Extract timing information
    time_values = [e['pulse_time'] for e in events if 'pulse_time' in e]
    
    if not time_values:
        # No timing information, use Poisson errors
        total_time = 1.0  # Assume 1 second if no timing
        return {
            'count_rate_error': np.sqrt(len(events)) / total_time,
            'signal_rate_error': 0.0,
            'dark_count_rate_error': 0.0
        }
    
    total_time = max(time_values) - min(time_values)
    total_events = len(events)
    
    # Separate signal and dark events
    signal_events = [e for e in events if e.get('is_signal', False)]
    dark_events = [e for e in events if not e.get('is_signal', False)]
    
    signal_count = len(signal_events)
    dark_count = len(dark_events)
    
    # Calculate standard errors using binning approach
    n_bins = min(10, total_events // 10) if total_events >= 100 else max(2, total_events // 5)
    
    if total_time > 0 and n_bins >= 2:
        time_min = min(time_values)
        time_max = max(time_values)
        bin_edges = np.linspace(time_min, time_max, n_bins + 1)
        
        # Calculate rate in each bin
        count_rates_binned = []
        signal_rates_binned = []
        dark_rates_binned = []
        
        for i in range(n_bins):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bin_duration = bin_end - bin_start
            
            if bin_duration > 0:
                # Count events in this bin
                events_in_bin = [e for e in events if bin_start <= e['pulse_time'] < bin_end]
                signal_in_bin = [e for e in signal_events if bin_start <= e['pulse_time'] < bin_end]
                dark_in_bin = [e for e in dark_events if bin_start <= e['pulse_time'] < bin_end]
                
                count_rates_binned.append(len(events_in_bin) / bin_duration)
                signal_rates_binned.append(len(signal_in_bin) / bin_duration)
                dark_rates_binned.append(len(dark_in_bin) / bin_duration)
        
        # Standard error = std / sqrt(n)
        count_rate_error = np.std(count_rates_binned, ddof=1) / np.sqrt(len(count_rates_binned)) if len(count_rates_binned) > 1 else 0
        signal_rate_error = np.std(signal_rates_binned, ddof=1) / np.sqrt(len(signal_rates_binned)) if len(signal_rates_binned) > 1 and signal_count > 0 else 0
        dark_count_rate_error = np.std(dark_rates_binned, ddof=1) / np.sqrt(len(dark_rates_binned)) if len(dark_rates_binned) > 1 and dark_count > 0 else 0
    else:
        # Fallback to Poisson errors for small datasets
        count_rate_error = np.sqrt(total_events) / total_time if total_time > 0 else 0
        signal_rate_error = np.sqrt(signal_count) / total_time if total_time > 0 and signal_count > 0 else 0
        dark_count_rate_error = np.sqrt(dark_count) / total_time if total_time > 0 and dark_count > 0 else 0
    
    return {
        'count_rate_error': float(count_rate_error),
        'signal_rate_error': float(signal_rate_error),
        'dark_count_rate_error': float(dark_count_rate_error)
    }

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

def extract_timestamp(filename):
    """Extract timestamp from filename (e.g., 20250611_025719)"""
    match = re.search(r'_(\d{8}_\d{6})', filename)
    if match:
        return match.group(1)
    return None

def read_analysis_files(input_dir, pattern='*_analysis.json', recursive=True):
    """
    Read all analysis.json files and extract rates and bias voltages
    
    Args:
        input_dir: Directory or list of directories to search
        pattern: File pattern to match
        recursive: If True, search recursively in subdirectories
    
    Returns:
        list: List of dictionaries containing data for each file
    """
    # Handle single directory or list of directories
    if isinstance(input_dir, str):
        input_dirs = [input_dir]
    else:
        input_dirs = input_dir
    
    files = []
    for directory in input_dirs:
        if recursive:
            # Recursively search for files
            search_pattern = os.path.join(directory, '**', pattern)
            files.extend(glob.glob(search_pattern, recursive=True))
        else:
            # Only search in the specified directory
            search_pattern = os.path.join(directory, pattern)
            files.extend(glob.glob(search_pattern))
    
    data = []
    for filepath in files:
        filename = os.path.basename(filepath)
        bias_voltage = extract_bias_voltage(filename)
        power = extract_power(filename)
        timestamp = extract_timestamp(filename)
        
        if bias_voltage is None:
            print(f"Warning: Could not extract bias voltage from {filename}")
            continue
        
        try:
            with open(filepath, 'r') as f:
                analysis = json.load(f)
            
            # Handle both Stage 1 (*_analysis.json) and Stage 2 (statistics_*.json) formats
            summary = analysis.get('summary_statistics', analysis.get('summary_from_stage1', {}))
            
            # Extract rates from summary
            count_rate = summary.get('count_rate', 0)
            signal_rate = summary.get('signal_rate', 0)
            dark_count_rate = summary.get('dark_count_rate', 0)
            efficiency = summary.get('efficiency', 0)
            resistance = summary.get('resistance_ohm', None)
            
            # Calculate errors from event-by-event data
            events = analysis.get('event_by_event_data', [])
            calculated_errors = calculate_errors_from_events(events)
            
            count_rate_error = calculated_errors['count_rate_error']
            signal_rate_error = calculated_errors['signal_rate_error']
            dark_count_rate_error = calculated_errors['dark_count_rate_error']
            
            # Calculate efficiency error from binomial statistics
            if efficiency > 0 and efficiency < 1 and events:
                time_values = [e['pulse_time'] for e in events if 'pulse_time' in e]
                if time_values:
                    total_time = max(time_values) - min(time_values)
                    n_pulses = SOURCE_RATE * total_time
                    efficiency_error = np.sqrt(efficiency * (1 - efficiency) / n_pulses) if n_pulses > 0 else 0
                else:
                    efficiency_error = 0
            else:
                efficiency_error = 0
            
            # Extract pulse characteristics
            pulse_fall_range_ptp_mean = summary.get('pulse_fall_range_ptp_mean', None)
            pulse_fall_range_ptp_std = summary.get('pulse_fall_range_ptp_std', None)
            
            # Extract rise amplitude characteristics
            rise_amplitude_mean = summary.get('rise_amplitude_mean', None)
            rise_amplitude_std = summary.get('rise_amplitude_std', None)
            
            data.append({
                'bias_voltage': bias_voltage,
                'power': power,
                'timestamp': timestamp,
                'count_rate': count_rate,
                'signal_rate': signal_rate,
                'dark_count_rate': dark_count_rate,
                'efficiency': efficiency,
                'count_rate_error': count_rate_error,
                'signal_rate_error': signal_rate_error,
                'dark_count_rate_error': dark_count_rate_error,
                'efficiency_error': efficiency_error,
                'resistance': resistance,
                'pulse_fall_range_ptp_mean': pulse_fall_range_ptp_mean,
                'pulse_fall_range_ptp_std': pulse_fall_range_ptp_std,
                'rise_amplitude_mean': rise_amplitude_mean,
                'rise_amplitude_std': rise_amplitude_std,
                'filename': filename
            })
            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
    
    # Remove duplicates: keep only the most recent file for each (bias_voltage, power) pair
    data = deduplicate_by_timestamp(data)
    
    # Sort by power then bias voltage
    data.sort(key=lambda x: (x['power'] if x['power'] is not None else -1, x['bias_voltage']))
    
    return data

def deduplicate_by_timestamp(data):
    """
    Remove duplicate measurements, keeping only the most recent one.
    Groups by (bias_voltage, power) and selects the entry with the latest timestamp.
    """
    from collections import defaultdict
    
    # Group by (bias_voltage, power)
    groups = defaultdict(list)
    for entry in data:
        key = (entry['bias_voltage'], entry['power'])
        groups[key].append(entry)
    
    # Keep only the most recent entry from each group
    deduplicated = []
    for key, entries in groups.items():
        if len(entries) > 1:
            # Sort by timestamp (newest first)
            entries_sorted = sorted(entries, key=lambda x: x.get('timestamp', ''), reverse=True)
            print(f"  Multiple files for {key[0]}mV, {key[1]}nW: keeping most recent ({entries_sorted[0]['filename']})")
            deduplicated.append(entries_sorted[0])
        else:
            deduplicated.append(entries[0])
    
    return deduplicated

def group_by_power(data):
    """Group data by power level"""
    power_groups = defaultdict(list)
    for d in data:
        power = d['power'] if d['power'] is not None else 'Unknown'
        power_groups[power].append(d)
    return power_groups

def group_by_bias(data):
    """Group data by bias voltage"""
    bias_groups = defaultdict(list)
    for d in data:
        bias = d['bias_voltage']
        bias_groups[bias].append(d)
    return bias_groups

def print_data_summary(data):
    """Print summary of loaded data"""
    print(f"\nFound {len(data)} files with valid data:")
    for d in data:
        power_str = f"{d['power']} nW" if d['power'] is not None else "Unknown power"
        print(f"  {d['bias_voltage']} mV, {power_str}: "
              f"count_rate={d['count_rate']:.3f} Hz, "
              f"signal_rate={d['signal_rate']:.3f} Hz, "
              f"dark_count_rate={d['dark_count_rate']:.3f} Hz, "
              f"efficiency={d['efficiency']:.3e}")
