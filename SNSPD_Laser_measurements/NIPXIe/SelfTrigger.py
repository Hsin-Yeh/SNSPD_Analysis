#!/usr/bin/env python3
"""
SNSPD TDMS Data Analysis Tool

Analyzes TDMS files from SNSPD measurements, extracting pulse characteristics,
calculating detection rates, and computing efficiency statistics.
"""

import argparse
import datetime
import json
import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

# User defined functions
from utils.Timing_Analyzer import *
from utils.tdmsUtils import *
from utils.osUtils import *
from utils.plotUtilscopy import *


# =============================================================================
# CONSTANTS
# =============================================================================

# Global state
DEBUG = False
DISPLAY = False
event_statistics = []
metadata_dict = {}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def debug_print(message: str):
    """Print debug message if DEBUG is enabled."""
    if DEBUG:
        print(message)


def determine_output_directory(input_filename: str, default_dir: str) -> str:
    """
    Determine output directory based on input file location.
    
    Creates flat 3-stage analysis folder structure:
    - SNSPD_rawdata/path/to/data/*.tdms
    - SNSPD_analyzed_output/stage1_events/path/to/data/*.json
    - SNSPD_analyzed_output/stage2_statistics/path/to/data/*.json
    - SNSPD_analyzed_output/stage3_plots/common_path/*.png
    """
    abs_path = os.path.abspath(input_filename)
    path_parts = abs_path.split(os.sep)
    
    for i, part in enumerate(path_parts):
        if part in ['SNSPD_rawdata', 'SNSPD_data']:
            base_path = os.sep.join(path_parts[:i])
            output_base_dir = os.path.join(base_path, 'SNSPD_analyzed_output', 'stage1_events')
            subdirs = path_parts[i+1:-1]  # Get subdirectory structure
            
            if subdirs:
                return os.path.join(output_base_dir, *subdirs) + '/'
            return output_base_dir + '/'
    
    return default_dir + '/'


def extract_bias_parameters(filename: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """Extract voltage, current from filename and calculate resistance."""
    voltage_match = re.search(r'_(\d+)mV_', filename)
    current_match = re.search(r'_(\d+)uA_', filename)
    
    voltage = int(voltage_match.group(1)) if voltage_match else None
    current = int(current_match.group(1)) if current_match else None
    
    if voltage is not None and current is not None and current != 0:
        resistance = (voltage / 1000.0) / (current / 1e6)
        return voltage, current, resistance
    
    return voltage, current, None


# =============================================================================
# PULSE ANALYSIS FUNCTIONS
# =============================================================================

def calculate_rise_fall_times(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate rise/fall time constants using 10%-90% method and slew rate.
    
    Returns dict with rise_time_10_90, fall_time_90_10, rise_slew_rate, fall_slew_rate, rise_amplitude.
    """
    # Find pulse baseline and extrema
    baseline = np.mean(data[:10]) if len(data) >= 10 else data[0]
    pulse_min = np.min(data)
    idx_min = np.argmin(data)
    
    # Calculate rising amplitude (baseline to minimum)
    rise_amplitude = abs(baseline - pulse_min)
    
    # Calculate threshold levels
    level_10 = baseline - 0.1 * (baseline - pulse_min)
    level_90 = baseline - 0.9 * (baseline - pulse_min)
    
    # Rising edge analysis
    rise_data = data[:idx_min+1]
    try:
        idx_10 = np.where(rise_data <= level_10)[0][0] if len(np.where(rise_data <= level_10)[0]) > 0 else 0
        idx_90 = np.where(rise_data <= level_90)[0][0] if len(np.where(rise_data <= level_90)[0]) > 0 else len(rise_data)-1
        rise_time_10_90 = abs(idx_90 - idx_10)
    except:
        rise_time_10_90 = 0
    
    # Falling edge analysis
    fall_data = data[idx_min:]
    try:
        idx_90 = np.where(fall_data >= level_90)[0][0] if len(np.where(fall_data >= level_90)[0]) > 0 else 0
        idx_10 = np.where(fall_data >= level_10)[0][0] if len(np.where(fall_data >= level_10)[0]) > 0 else len(fall_data)-1
        fall_time_90_10 = abs(idx_10 - idx_90)
    except:
        fall_time_90_10 = 0
    
    # Slew rates
    rise_slew_rate = abs(np.min(np.diff(rise_data))) if len(rise_data) > 1 else 0
    fall_slew_rate = abs(np.max(np.diff(fall_data))) if len(fall_data) > 1 else 0
    
    return {
        'rise_amplitude': rise_amplitude,
        'rise_time_10_90': rise_time_10_90,
        'fall_time_90_10': fall_time_90_10,
        'rise_slew_rate': rise_slew_rate,
        'fall_slew_rate': fall_slew_rate
    }

def analyze_single_event(data: np.ndarray, trig: np.ndarray, time: float, 
                        time_previous: float, event_num: int, find_sync_method: str):
    """Analyze a single event and store statistics."""
    # Calculate basic pulse parameters
    baseline = np.mean(data[:10]) if len(data) >= 10 else data[0]
    pulse_max = np.max(data)
    pulse_min = np.min(data)
    pulse_ptp = np.ptp(data)
    
    # Find laser sync time
    if find_sync_method == 'simple':
        laser_sync_arrival = Find_sync_arrival_simple(trig)
    else:
        laser_sync_arrival = Find_sync_arrival_splineFit(trig)
    
    # Calculate timing parameters
    timing_params = calculate_rise_fall_times(data)
    
    # Find signal pulse arrival times at different thresholds
    arrival_times = Find_signal_arrival(data)
    
    debug_print(f'Event {event_num}: Rise={timing_params["rise_time_10_90"]:.2f}, '
               f'Fall={timing_params["fall_time_90_10"]:.2f} samples, '
               f'Arrival@50%={arrival_times["arrival_time_50"]:.2f}')
    
    # Store event statistics
    event_stats = {
        "event_number": event_num,
        "event_time": float(time),
        "event_interval": float(time_previous - time),
        "signal_baseline": float(baseline),
        "signal_max": float(pulse_max),
        "signal_min": float(pulse_min),
        "signal_fall_amplitude": float(pulse_ptp),
        "signal_rise_amplitude": float(timing_params['rise_amplitude']),
        "signal_rise_time_10_90": float(timing_params['rise_time_10_90']),
        "signal_fall_time_90_10": float(timing_params['fall_time_90_10']),
        "signal_rise_slew_rate": float(timing_params['rise_slew_rate']),
        "signal_fall_slew_rate": float(timing_params['fall_slew_rate']),
        "laser_sync_arrival": float(laser_sync_arrival)
    }
    
    # Add arrival times to event statistics
    event_stats.update(arrival_times)
    
    event_statistics.append(event_stats)

def Find_sync_arrival_splineFit(chTrig: np.ndarray) -> float:
    """
    Find trigger arrival time using spline interpolation and derivative analysis.
    
    Fits cubic spline, finds turning points via derivative roots, calculates 50% crossing.
    Returns trigger time in samples, or -1 if not found.
    """
    # Early return for empty data
    if len(chTrig) == 0:
        return -1
    
    # Create spline interpolation
    x_index = np.arange(len(chTrig))
    chTrig_spline = CubicSpline(x_index, chTrig)
    
    # Get derivative roots (turning points)
    try:
        roots = chTrig_spline.derivative().roots()
    except:
        return -1
    
    if len(roots) < 2:
        return -1
    
    # Find falling edges with sufficient amplitude
    range_threshold = 0.1
    
    for i in range(1, len(roots)):
        prev_root, curr_root = roots[i-1], roots[i]
        
        # Validate roots
        if prev_root < 0 or curr_root < 0 or curr_root >= len(chTrig):
            continue
        
        prev_val = chTrig_spline(prev_root)
        curr_val = chTrig_spline(curr_root)
        
        # Check for falling edge
        if prev_val - curr_val > range_threshold:
            try:
                target_value = (prev_val + curr_val) / 2.0
                
                def crossing_func(x):
                    return chTrig_spline(x) - target_value
                
                if crossing_func(curr_root) * crossing_func(prev_root) < 0:
                    return brentq(crossing_func, prev_root, curr_root)
            except:
                continue
    
    return -1


def Find_sync_arrival_simple(chTrig: np.ndarray, threshold_fraction: float = 0.5) -> float:
    """
    Fast threshold crossing method (10-50x faster than spline).
    
    Uses simple threshold crossing with linear interpolation.
    Returns trigger time in samples, or -1 if not found.
    """
    if len(chTrig) < 2:
        return -1
    
    baseline = np.mean(chTrig[:10]) if len(chTrig) >= 10 else chTrig[0]
    minimum = np.min(chTrig)
    threshold = baseline - threshold_fraction * (baseline - minimum)
    
    # Find first crossing
    for i in range(len(chTrig) - 1):
        if chTrig[i] >= threshold and chTrig[i+1] < threshold:
            frac = (threshold - chTrig[i]) / (chTrig[i+1] - chTrig[i])
            return i + frac
    
    return -1


def Find_signal_arrival(data: np.ndarray, threshold_fractions: List[float] = None) -> Dict:
    """
    Find signal pulse arrival time on rising edge using multiple threshold levels.
    
    Detects when the signal pulse crosses various threshold levels on the rising edge,
    providing timing information at different points of the pulse rise.
    
    Args:
        data: Signal waveform data (numpy array)
        threshold_fractions: List of threshold levels (0-1) to detect.
                           Default: [0.1, 0.2, 0.5, 0.8, 0.9]
    
    Returns:
        Dictionary with arrival times at each threshold:
        {
            'arrival_time_10': time at 10% threshold (samples),
            'arrival_time_20': time at 20% threshold (samples),
            'arrival_time_50': time at 50% threshold (samples),
            'arrival_time_80': time at 80% threshold (samples),
            'arrival_time_90': time at 90% threshold (samples)
        }
        Returns -1 for any threshold not found.
    """
    if threshold_fractions is None:
        threshold_fractions = [0.1, 0.2, 0.5, 0.8, 0.9]
    
    if len(data) < 2:
        return {f'arrival_time_{int(f*100)}': -1.0 for f in threshold_fractions}
    
    # Calculate baseline and minimum
    baseline = np.mean(data[:10]) if len(data) >= 10 else data[0]
    pulse_min = np.min(data)
    
    # If no significant pulse detected, return -1 for all
    if abs(baseline - pulse_min) < 0.01:  # Less than 10mV change
        return {f'arrival_time_{int(f*100)}': -1.0 for f in threshold_fractions}
    
    arrival_times = {}
    
    # Find arrival time for each threshold
    for frac in threshold_fractions:
        threshold = baseline - frac * (baseline - pulse_min)
        arrival_time = -1.0
        
        # Find first crossing on falling edge (signal going negative)
        for i in range(len(data) - 1):
            if data[i] >= threshold and data[i+1] < threshold:
                # Linear interpolation for sub-sample precision
                if data[i+1] != data[i]:  # Avoid division by zero
                    frac_interp = (threshold - data[i]) / (data[i+1] - data[i])
                    arrival_time = i + frac_interp
                else:
                    arrival_time = float(i)
                break
        
        # Store with descriptive key
        key = f'arrival_time_{int(frac*100)}'
        arrival_times[key] = float(arrival_time)
    
    return arrival_times


# =============================================================================
# STATISTICS CALCULATION
# =============================================================================

def calculate_summary_statistics(events: List[Dict]) -> Dict:
    """Calculate summary statistics from event data."""
    summary = {}
    
    # Calculate mean, std, min, max for numeric fields
    for key in events[0].keys():
        if key not in ["event_number", "pass_selection"]:
            values = [event[key] for event in events]
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
            summary[f"{key}_min"] = float(np.min(values))
            summary[f"{key}_max"] = float(np.max(values))
    
    # Time and count statistics
    time_values = [event['event_time'] for event in events]
    total_time = max(time_values) - min(time_values) if time_values else 0    
    total_events = len(events)
      
    # Calculate rates
    count_rate = total_events / total_time if total_time > 0 else 0 
    count_rate_error = np.sqrt(total_events) / total_time if total_time > 0 else 0     
   
    summary.update({
        'total_time': float(total_time),
        'total_events': int(total_events),
        'count_rate': float(count_rate),
        'count_rate_error': float(count_rate_error),
    })
    
    return summary


# =============================================================================
# TDMS FILE PROCESSING
# =============================================================================

def process_tdms_file(filename: str, output_filename: str, args):
    """Process a single TDMS file and generate analysis JSON."""
    global event_statistics, metadata_dict
    event_statistics = []
    
    with TdmsFile.open(filename) as tdms_file:
        # Extract metadata
        metadata = tdms_file.properties
        metadata_df = pd.DataFrame(metadata.items(), columns=['metaKey', 'metaValue'])
        debug_print(metadata_df)
        
        metadata_dict = {row['metaKey']: row['metaValue'] 
                        for _, row in metadata_df.iterrows()}
        
        recordlength = int(metadata_df.loc[metadata_df['metaKey'] == 'record length', 'metaValue'].iloc[0])
        
        # Check for ADC Readout Channels
        group_names = [group.name for group in tdms_file.groups()]
        debug_print(f"Groups: {group_names}")
        
        if 'ADC Readout Channels' not in group_names:
            print("Error: No ADC Readout Channels - signal too low or detector latched")
            save_zero_results(output_filename)
            return
        
        # Get channel data
        chSig_total = tdms_file['ADC Readout Channels']['chSig']
        chTrig_total = tdms_file['ADC Readout Channels']['chTrig']
        chTime_total = tdms_file['ADC Readout Channels']['Time']
        
        totalEvents = int(len(chSig_total) / recordlength)
        debug_print(f"Total events: {totalEvents}")
        print(f"==========Start Processing {datetime.datetime.now()}==========")
        
        for event in range(totalEvents - 1):
            if args.subset > 0 and event >= args.subset:
                break
            if args.checkSingleEvent != -1 and event != args.checkSingleEvent:
                continue
            
            if event % args.report == 0:
                print(f"Processing {event}/{totalEvents}")
            
            idx = event * recordlength
            chSig = chSig_total[idx:idx+recordlength]
            chTrig = chTrig_total[idx:idx+recordlength]
            chTime = chTime_total[idx:idx+1][0]
            chTime_prev = chTime_total[idx+recordlength:idx+recordlength+1][0]
            
            if DISPLAY:
                event_display_2ch(chSig, chTrig, 'Waveform')
            
            analyze_single_event(chSig, chTrig, chTime, chTime_prev, 
                               event, args.find_sync_method)
        
        print(f"==========End Processing {datetime.datetime.now()}==========")
        print(f"Processed {len(event_statistics)} events")


def save_zero_results(output_filename: str):
    """Save zero-filled results for files with no valid data."""
    final_analysis = {
        "metadata": metadata_dict,
        "summary_statistics": {
            "total_time": 0.0, "total_events": 0, "signal_count": 0,
            "dark_count": 0, "efficiency": 0.0, "count_rate": 0.0,
            "signal_rate": 0.0, "dark_count_rate": 0.0
        },
        "total_events": 0,
        "event_by_event_data": []
    }
    
    with open(output_filename, "w") as f:
        json.dump(final_analysis, f, indent=2)
    print(f"Saved zero-filled results to {output_filename}")


def save_single_event_data(filename: str, output_filename: str, event_number: int):
    """
    Save a single event with metadata to JSON without full analysis.
    
    Args:
        filename: Input TDMS filename
        output_filename: Output JSON filename
        event_number: Event number to extract (0-indexed)
    """
    global metadata_dict
    
    with TdmsFile.open(filename) as tdms_file:
        # Extract metadata
        metadata = tdms_file.properties
        metadata_df = pd.DataFrame(metadata.items(), columns=['metaKey', 'metaValue'])
        
        metadata_dict = {row['metaKey']: row['metaValue'] 
                        for _, row in metadata_df.iterrows()}
        
        recordlength = int(metadata_df.loc[metadata_df['metaKey'] == 'record length', 'metaValue'].iloc[0])
        
        # Check for ADC Readout Channels
        group_names = [group.name for group in tdms_file.groups()]
        
        if 'ADC Readout Channels' not in group_names:
            print("Error: No ADC Readout Channels - signal too low or detector latched")
            return
        
        # Get channel data
        chSig_total = tdms_file['ADC Readout Channels']['chSig']
        chTrig_total = tdms_file['ADC Readout Channels']['chTrig']
        chTime_total = tdms_file['ADC Readout Channels']['Time']
        
        totalEvents = int(len(chSig_total) / recordlength)
        
        if event_number >= totalEvents or event_number < 0:
            print(f"Error: Event {event_number} out of range (0-{totalEvents-1})")
            return
        
        print(f"Extracting event {event_number} from {totalEvents} total events")
        
        # Extract single event
        idx = event_number * recordlength
        chSig = chSig_total[idx:idx+recordlength].tolist()
        chTrig = chTrig_total[idx:idx+recordlength].tolist()
        chTime = float(chTime_total[idx:idx+1][0])
        
        # Get timing info - calculate from sample rate
        sample_rate = float(metadata_dict.get('actual sample rate', 2.5e9))
        sample_interval = 1.0 / sample_rate if sample_rate > 0 else 0.0
        time_array = [i * sample_interval for i in range(len(chSig))]
        
        # Create output structure
        single_event_data = {
            "metadata": metadata_dict,
            "event_info": {
                "event_number": event_number,
                "total_events_in_file": totalEvents,
                "record_length": recordlength,
                "time_stamp": chTime,
                "sample_interval": sample_interval
            },
            "waveform_data": {
                "time": time_array,
                "signal_channel": chSig,
                "trigger_channel": chTrig
            }
        }
        
        # Save to JSON
        with open(output_filename, "w") as f:
            json.dump(single_event_data, f, indent=2)
        
        print(f"\n==========Single Event Saved==========")
        print(f"Event number: {event_number}")
        print(f"Time stamp: {chTime}")
        print(f"Record length: {recordlength}")
        print(f"Sample interval: {sample_interval}")
        print(f"Saved to: {output_filename}")


def save_analysis_results(output_filename: str, voltage: Optional[int], 
                         current: Optional[int], resistance: Optional[float]):
    """Save analysis results to JSON file."""
    if not event_statistics:
        final_analysis = {
            "metadata": metadata_dict,
            "summary_statistics": {},
            "total_events": 0,
            "event_by_event_data": []
        }
    else:
        summary_stats = calculate_summary_statistics(event_statistics)
        
        if voltage is not None:
            summary_stats['bias_voltage_mV'] = int(voltage)
        if current is not None:
            summary_stats['bias_current_uA'] = int(current)
        if resistance is not None:
            summary_stats['resistance_ohm'] = float(resistance)
        
        final_analysis = {
            "metadata": metadata_dict,
            "summary_statistics": summary_stats,
            "total_events": len(event_statistics),
            "event_by_event_data": event_statistics
        }
    
    with open(output_filename, "w") as f:
        json.dump(final_analysis, f, indent=2)
    
    print(f"\nSaved to: {output_filename}")


# =============================================================================
# ARGUMENT PARSER AND MAIN
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze SNSPD TDMS files')
    parser.add_argument('in_filenames', nargs="+", help='Input TDMS files')
    parser.add_argument('--outputDir', '-d', default="./Stats", help='Output directory')
    parser.add_argument('--report', '-r', default=1000, type=int, help='Report every N events')
    parser.add_argument('--checkSingleEvent', '-c', default=-1, type=int, help='Analyze single event')
    parser.add_argument('--debug_report', '-b', action="store_true", help='Debug output')
    parser.add_argument('--display_report', '-p', action="store_true", help='Display waveforms')
    parser.add_argument('--subset', '-s', default=-1, type=int, help='Process first N events')
    parser.add_argument('--find_sync_method', default='spline', choices=['spline', 'simple'],
                       help='Trigger calculation: spline (accurate) or simple (fast)')
    parser.add_argument('--save_single_event', '-e', default=-1, type=int, 
                       help='Save single event (by number) to JSON without full analysis')
    return parser.parse_args()


def main():
    """Main entry point."""
    global DEBUG, DISPLAY
    
    args = parse_arguments()
    DEBUG = args.debug_report
    DISPLAY = args.display_report
    
    # Collect all TDMS files (expand directories)
    tdms_files = []
    for path in args.in_filenames:
        if os.path.isdir(path):
            # Find all TDMS files in directory
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.tdms'):
                        tdms_files.append(os.path.join(root, file))
            if not tdms_files or not any(os.path.dirname(f).startswith(os.path.abspath(path)) or f.startswith(path) for f in tdms_files):
                print(f"Warning: No TDMS files found in directory: {path}")
        elif os.path.isfile(path):
            tdms_files.append(path)
        else:
            print(f"Error: Path not found: {path}")
    
    if not tdms_files:
        print("Error: No TDMS files to process")
        return
    
    print(f"Found {len(tdms_files)} TDMS file(s) to process\n")
    
    for index, in_filename in enumerate(tdms_files):
        print("\n" + "="*60)
        print(f"Processing: {in_filename} ({index+1}/{len(tdms_files)})")
        print("="*60)
        
        basename = os.path.basename(in_filename).replace('.tdms', '')
        output_dir = determine_output_directory(in_filename, args.outputDir)
        createDir(output_dir)
        
        # Check if single event save mode
        if args.save_single_event >= 0:
            output_filename = os.path.join(output_dir, basename + f"_event{args.save_single_event}.json")
            print(f"Single event mode - saving event {args.save_single_event}")
            print(f"Output: {output_filename}")
            save_single_event_data(in_filename, output_filename, args.save_single_event)
            print(f"\n==========Completed: {in_filename}==========\n")
            continue
        
        # Full analysis mode
        output_filename = os.path.join(output_dir, basename + "_analysis.json")
        debug_print(f"Output: {output_dir}")
        
        voltage, current, resistance = extract_bias_parameters(basename)
        if voltage and current:
            debug_print(f"Bias: {voltage} mV, {current} uA, R={resistance:.2f} Ω")
        
        process_tdms_file(in_filename, output_filename, args)
        save_analysis_results(output_filename, voltage, current, resistance)
        
        debug_print(f"\n==========Completed: {in_filename}==========\n")


if __name__ == "__main__":
    main()