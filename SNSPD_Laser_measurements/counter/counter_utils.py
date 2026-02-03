#!/usr/bin/env python3
"""
Shared utilities for counter data analysis.
"""

from pathlib import Path
from datetime import datetime
import re
import numpy as np


def parse_power_timestamp(filename):
    """Extract power (nW) and timestamp from filename."""
    match = re.search(r'(\d+)nW_(\d{8}_\d{4})\.txt', filename)
    if match:
        power_nw = int(match.group(1))
        timestamp_str = match.group(2)
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M')
        return power_nw, timestamp
    return None, None


def read_counter_file_median(filepath):
    """Read counter data file and return target voltages and median count rates."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_lines = lines[1:]

    target_voltages = []
    count_rates = []

    for line in data_lines:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        target_voltage = float(parts[0])
        measurements = [float(x) for x in parts[7:]]
        median_count_rate = np.median(measurements)
        target_voltages.append(target_voltage)
        count_rates.append(median_count_rate)

    return np.array(target_voltages), np.array(count_rates)


def find_latest_files(data_dir):
    """Find the latest file for each power level in the directory."""
    data_dir = Path(data_dir)

    power_dirs = [d for d in data_dir.iterdir() if d.is_dir() and 'nW' in d.name]

    power_files = {}
    dark_files = []

    for power_dir in power_dirs:
        power_match = re.search(r'(\d+)nW', power_dir.name)
        if not power_match:
            continue

        power_nw = int(power_match.group(1))

        txt_files = list(power_dir.glob('*.txt'))
        if not txt_files:
            continue

        files_with_time = []
        for f in txt_files:
            _, timestamp = parse_power_timestamp(f.name)
            if timestamp:
                files_with_time.append((f, timestamp))

        if files_with_time:
            latest_file = max(files_with_time, key=lambda x: x[1])[0]
            if power_nw == 0:
                dark_files.extend(files_with_time)
            else:
                power_files[power_nw] = latest_file

    return power_files, dark_files


def find_closest_dark_file(signal_timestamp, dark_files):
    """Find the dark count file with timestamp earlier and closest to signal file."""
    best_dark = None
    min_time_diff = None

    for dark_file, dark_time in dark_files:
        if dark_time <= signal_timestamp:
            time_diff = (signal_timestamp - dark_time).total_seconds()
            if min_time_diff is None or time_diff < min_time_diff:
                min_time_diff = time_diff
                best_dark = dark_file

    return best_dark


def get_available_bias_voltages_from_file(filepath):
    """Extract all unique target voltages from a data file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    target_voltages = set()
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            try:
                target_mv = int(round(float(parts[1]) * 1000))
                if target_mv > 0:
                    target_voltages.add(target_mv)
            except ValueError:
                continue
    return sorted(list(target_voltages))
