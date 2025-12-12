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
from ..utils.Timing_Analyzer import *
from ..utils.tdmsUtils import *
from ..utils.osUtils import *
from ..utils.plotUtilscopy import *


# =============================================================================
# CONSTANTS
# =============================================================================

TRIGGER_CUT_MIN = 196
TRIGGER_CUT_MAX = 198
SOURCE_RATE = 1E7  # 10 MHz laser repetition rate

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
    
    If input is in SNSPD_rawdata or SNSPD_data, outputs to SNSPD_analyzed_json
    with the same subdirectory structure.
    """
    abs_path = os.path.abspath(input_filename)
    path_parts = abs_path.split(os.sep)
    
    for i, part in enumerate(path_parts):
        if part in ['SNSPD_rawdata', 'SNSPD_data']:
            base_path = os.sep.join(path_parts[:i])
            output_base_dir = os.path.join(base_path, 'SNSPD_analyzed_json')
            subdirs = path_parts[i+1:-1]
            
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

def Pulse_selection(trig_check):
    """Legacy function - kept for compatibility but not used."""
    if trig_check > 22 and trig_check < 24:
        return True
    return False

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
                        time_previous: float, event_num: int, trigger_method: str):
    """Analyze a single event and store statistics."""
    # Calculate basic pulse parameters
    baseline = np.mean(data[:10]) if len(data) >= 10 else data[0]
    pulse_max = np.max(data)
    pulse_min = np.min(data)
    pulse_ptp = np.ptp(data)
    
    # Find trigger time
    if trigger_method == 'simple':
        trig_check = Find_Trigger_time_simple(trig)
    else:
        trig_check = Find_Trigger_time_splineFit(trig)
    
    # Calculate timing parameters
    timing_params = calculate_rise_fall_times(data)
    
    debug_print(f'Event {event_num}: Rise={timing_params["rise_time_10_90"]:.2f}, '
               f'Fall={timing_params["fall_time_90_10"]:.2f} samples')
    
    # Store event statistics
    event_stats = {
        "event_number": event_num,
        "pre_mean": float(baseline),
        "pulse_max": float(pulse_max),
        "pulse_min": float(pulse_min),
        "pulse_fall_range_ptp": float(pulse_ptp),
        "pulse_time": float(time),
        "pulse_time_interval": float(time_previous - time),
        "trigger_check": float(trig_check),
        "pass_selection": TRIGGER_CUT_MIN <= trig_check <= TRIGGER_CUT_MAX,
        "rise_amplitude": float(timing_params['rise_amplitude']),
        "rise_time_10_90": float(timing_params['rise_time_10_90']),
        "fall_time_90_10": float(timing_params['fall_time_90_10']),
        "rise_slew_rate": float(timing_params['rise_slew_rate']),
        "fall_slew_rate": float(timing_params['fall_slew_rate'])
    }
    
    event_statistics.append(event_stats)

def Find_Trigger_time_splineFit(chTrig: np.ndarray) -> float:
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


def Find_Trigger_time_simple(chTrig: np.ndarray, threshold_fraction: float = 0.5) -> float:
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
    time_values = [event['pulse_time'] for event in events]
    total_time = max(time_values) - min(time_values) if time_values else 0
    
    # Separate signal and dark counts
    signal_events = [e for e in events 
                    if TRIGGER_CUT_MIN <= e.get('trigger_check', -999) <= TRIGGER_CUT_MAX]
    dark_events = [e for e in events 
                  if not (TRIGGER_CUT_MIN <= e.get('trigger_check', -999) <= TRIGGER_CUT_MAX)]
    
    signal_count = len(signal_events)
    dark_count = len(dark_events)
    total_events = len(events)
    
    # Signal and dark pulse characteristics
    if signal_events:
        signal_ptp = [e['pulse_fall_range_ptp'] for e in signal_events if 'pulse_fall_range_ptp' in e]
        if signal_ptp:
            summary['signal_pulse_fall_range_ptp_mean'] = float(np.mean(signal_ptp))
            summary['signal_pulse_fall_range_ptp_std'] = float(np.std(signal_ptp))
            summary['signal_pulse_fall_range_ptp_min'] = float(np.min(signal_ptp))
            summary['signal_pulse_fall_range_ptp_max'] = float(np.max(signal_ptp))
    
    if dark_events:
        dark_ptp = [e['pulse_fall_range_ptp'] for e in dark_events if 'pulse_fall_range_ptp' in e]
        if dark_ptp:
            summary['dark_pulse_fall_range_ptp_mean'] = float(np.mean(dark_ptp))
            summary['dark_pulse_fall_range_ptp_std'] = float(np.std(dark_ptp))
            summary['dark_pulse_fall_range_ptp_min'] = float(np.min(dark_ptp))
            summary['dark_pulse_fall_range_ptp_max'] = float(np.max(dark_ptp))
    
    # Calculate rates
    count_rate = total_events / total_time if total_time > 0 else 0
    signal_rate = signal_count / total_time if total_time > 0 else 0
    dark_count_rate = dark_count / total_time if total_time > 0 else 0
    
    # Poisson errors
    count_rate_error = np.sqrt(total_events) / total_time if total_time > 0 else 0
    signal_rate_error = np.sqrt(signal_count) / total_time if total_time > 0 and signal_count > 0 else 0
    dark_count_rate_error = np.sqrt(dark_count) / total_time if total_time > 0 and dark_count > 0 else 0
    
    # Efficiency and binomial error
    efficiency = (signal_rate / SOURCE_RATE) if signal_rate > 0 else 0
    
    if total_time > 0 and 0 < efficiency < 1:
        n_pulses = SOURCE_RATE * total_time
        efficiency_error = np.sqrt(efficiency * (1 - efficiency) / n_pulses)
    else:
        efficiency_error = 0
    
    summary.update({
        'total_time': float(total_time),
        'total_events': int(total_events),
        'signal_count': int(signal_count),
        'dark_count': int(dark_count),
        'efficiency': float(efficiency),
        'efficiency_error': float(efficiency_error),
        'count_rate': float(count_rate),
        'count_rate_error': float(count_rate_error),
        'signal_rate': float(signal_rate),
        'signal_rate_error': float(signal_rate_error),
        'dark_count_rate': float(dark_count_rate),
        'dark_count_rate_error': float(dark_count_rate_error)
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
        print(metadata_df)
        
        metadata_dict = {row['metaKey']: row['metaValue'] 
                        for _, row in metadata_df.iterrows()}
        
        recordlength = int(metadata_df.loc[metadata_df['metaKey'] == 'record length', 'metaValue'].iloc[0])
        
        # Check for ADC Readout Channels
        group_names = [group.name for group in tdms_file.groups()]
        print(f"Groups: {group_names}")
        
        if 'ADC Readout Channels' not in group_names:
            print("Error: No ADC Readout Channels - signal too low or detector latched")
            save_zero_results(output_filename)
            return
        
        # Get channel data
        chSig_total = tdms_file['ADC Readout Channels']['chSig']
        chTrig_total = tdms_file['ADC Readout Channels']['chTrig']
        chTime_total = tdms_file['ADC Readout Channels']['Time']
        
        totalEvents = int(len(chSig_total) / recordlength)
        print(f"Total events: {totalEvents}")
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
                               event, args.trigger_method)
        
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
    
    print(f"\n==========Analysis Summary==========")
    print(f"Total events: {final_analysis['total_events']}")
    
    if final_analysis['summary_statistics']:
        print("\n--- Summary Statistics ---")
        for key, value in final_analysis['summary_statistics'].items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.6f}")
    else:
        print("No statistics available")
    
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
    parser.add_argument('--trigger_method', default='spline', choices=['spline', 'simple'],
                       help='Trigger calculation: spline (accurate) or simple (fast)')
    return parser.parse_args()


def main():
    """Main entry point."""
    global DEBUG, DISPLAY
    
    args = parse_arguments()
    DEBUG = args.debug_report
    DISPLAY = args.display_report
    
    for in_filename in args.in_filenames:
        print("\n" + "="*60)
        print(f"Processing: {in_filename}")
        print("="*60)
        
        basename = os.path.basename(in_filename).replace('.tdms', '')
        output_dir = determine_output_directory(in_filename, args.outputDir)
        createDir(output_dir)
        
        output_filename = os.path.join(output_dir, basename + "_analysis.json")
        print(f"Output: {output_dir}")
        
        voltage, current, resistance = extract_bias_parameters(basename)
        if voltage and current:
            print(f"Bias: {voltage} mV, {current} uA, R={resistance:.2f} Ω")
        
        process_tdms_file(in_filename, output_filename, args)
        save_analysis_results(output_filename, voltage, current, resistance)
        
        print(f"\n==========Completed: {in_filename}==========\n")


if __name__ == "__main__":
    main()