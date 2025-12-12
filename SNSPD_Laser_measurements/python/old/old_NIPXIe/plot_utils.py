#!/usr/bin/env python3
"""
Utility functions for reading and organizing SNSPD analysis data
"""

import json
import glob
import re
import os
from collections import defaultdict

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
            
            # Extract pulse characteristics
            pulse_fall_range_ptp_mean = summary.get('pulse_fall_range_ptp_mean', None)
            pulse_fall_range_ptp_std = summary.get('pulse_fall_range_ptp_std', None)
            
            # Extract rise amplitude characteristics
            rise_amplitude_mean = summary.get('rise_amplitude_mean', None)
            rise_amplitude_std = summary.get('rise_amplitude_std', None)
            
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
                'pulse_fall_range_ptp_mean': pulse_fall_range_ptp_mean,
                'pulse_fall_range_ptp_std': pulse_fall_range_ptp_std,
                'rise_amplitude_mean': rise_amplitude_mean,
                'rise_amplitude_std': rise_amplitude_std,
                'filename': filename
            })
            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
    
    # Sort by power then bias voltage
    data.sort(key=lambda x: (x['power'] if x['power'] is not None else -1, x['bias_voltage']))
    
    return data

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
