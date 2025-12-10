#!/usr/bin/env python3

# python3 -m python.NIPXIE.simple -d plots/20240221/20231011_1/.
from nptdms import TdmsFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from array import array
from enum import Enum
import json
import math
import datetime

# User defined functions
from ..utils.Timing_Analyzer import *
from ..utils.tdmsUtils import *
from ..utils.osUtils import *
from ..utils.plotUtilscopy import *
from ..config import SMSPD_config as cf

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames', nargs="+", help='input filenames')
parser.add_argument('--outputDir','-d',default="./Stats",type=str,help='output directory')
parser.add_argument('--report','-r',default=1000,type=int,help='report every x events')
parser.add_argument('--checkSingleEvent','-c',default=-1,type=int,help='Check Single Event')
parser.add_argument('--debug_report','-b',action="store_true",help='report debug info')
parser.add_argument('--display_report','-p',action="store_true",help='display waveforms')
parser.add_argument('--doAdvanced',action="store_true",help='do single pulse analysis')
parser.add_argument('--doAverage',action="store_true",help='do average pulse analysis')
parser.add_argument('--dryRun',action="store_true",help='dry run')
parser.add_argument('--subset','-s',default=-1,type=int,help='Process a subset of data. -1 = all')
parser.add_argument('--trigger_method',default='spline',choices=['spline','simple'],help='Trigger time calculation method: spline (accurate) or simple (fast)')
args = parser.parse_args()

def debugPrint(string):
    if (cf.DEBUG):
        print(string)

def Sideband_selection():
    if pre_range[0] < cf.cut_preRange and pos_range[0] < cf.cut_posRange and pre_std[0] < cf.cut_preStd and pos_std[0] < cf.cut_posStd:
        return True
    else:
        return False

def Pulse_selection(trig_check):
    if trig_check > 22 and trig_check < 24:
        return True
    else:
        return False

def calculate_rise_fall_times(data, sample_rate=None):
    """
    Calculate rise/fall time constants using 10%-90% method and slew rate
    
    Args:
        data: Waveform data array
        sample_rate: Sample rate in Hz (optional, for time calculation)
    
    Returns:
        dict with rise_time_10_90, fall_time_90_10, rise_slew_rate, fall_slew_rate
    """
    # Find pulse baseline and peak
    pre_mean = np.mean(data[cf.prePulse_startT:cf.prePulse_endT])
    pulse_max = np.max(data[cf.Pulse_startT:cf.Pulse_endT])
    pulse_min = np.min(data[cf.Pulse_startT:cf.Pulse_endT])
    
    # Determine pulse polarity (negative-going pulse)
    pulse_amplitude = pulse_max - pulse_min
    
    # Calculate threshold levels (10%, 90% from baseline to peak)
    level_10 = pre_mean - 0.1 * pulse_amplitude
    level_90 = pre_mean - 0.9 * pulse_amplitude
    
    # Rising edge: find indices where signal crosses thresholds
    rise_data = data[cf.Pulse_startT:cf.Pulse_rise_endT]
    rise_indices = np.arange(len(rise_data))
    
    # Find 10% and 90% crossing points on rising edge
    try:
        # Rising edge goes from baseline towards minimum
        idx_10_rise = np.where(rise_data <= level_10)[0][0] if len(np.where(rise_data <= level_10)[0]) > 0 else 0
        idx_90_rise = np.where(rise_data <= level_90)[0][0] if len(np.where(rise_data <= level_90)[0]) > 0 else len(rise_data)-1
        rise_time_10_90 = abs(idx_90_rise - idx_10_rise)  # in sample points
    except:
        rise_time_10_90 = 0
    
    # Falling edge: find indices where signal returns from min to baseline
    fall_data = data[cf.Pulse_rise_endT:cf.Pulse_endT]
    
    # Find 90% and 10% crossing points on falling edge (recovery)
    try:
        # Falling edge goes from minimum back towards baseline
        idx_90_fall = np.where(fall_data >= level_90)[0][0] if len(np.where(fall_data >= level_90)[0]) > 0 else 0
        idx_10_fall = np.where(fall_data >= level_10)[0][0] if len(np.where(fall_data >= level_10)[0]) > 0 else len(fall_data)-1
        fall_time_90_10 = abs(idx_10_fall - idx_90_fall)  # in sample points
    except:
        fall_time_90_10 = 0
    
    # Calculate slew rates (dV/dt)
    # Rise slew rate: maximum negative slope during rising edge
    if len(rise_data) > 1:
        rise_derivative = np.diff(rise_data)
        rise_slew_rate = abs(np.min(rise_derivative))  # Maximum negative slope magnitude
    else:
        rise_slew_rate = 0
    
    # Fall slew rate: maximum positive slope during falling edge (recovery)
    if len(fall_data) > 1:
        fall_derivative = np.diff(fall_data)
        fall_slew_rate = abs(np.max(fall_derivative))  # Maximum positive slope magnitude
    else:
        fall_slew_rate = 0
    
    return {
        'rise_time_10_90': rise_time_10_90,
        'fall_time_90_10': fall_time_90_10,
        'rise_slew_rate': rise_slew_rate,
        'fall_slew_rate': fall_slew_rate
    }

def Simple_pulse_analysis(data, trig, time, time_previous, event):
    # pre_std_val = np.std(data[cf.prePulse_startT:cf.prePulse_endT])
    # pos_mean_val = np.mean(data[cf.prePulse_startT:cf.prePulse_endT])
    # pre_range_val = np.ptp(data[cf.prePulse_startT:cf.prePulse_endT])
    # pos_std_val = np.std(data[cf.postPulse_startT:cf.postPulse_endT])
    pre_mean_val = np.mean(data[cf.prePulse_startT:cf.prePulse_endT])
    # pos_range_val = np.ptp(data[cf.postPulse_startT:cf.postPulse_endT])
    # pre_max_val = np.max(data[cf.prePulse_startT:cf.prePulse_endT])
    # pos_max_val = np.max(data[cf.prePulse_startT:cf.prePulse_endT])
    pulse_max_val = np.max(data)
    pulse_min_val = np.min(data)
    # pulse_max_T_val = cf.Pulse_startT + np.argmax(data[cf.Pulse_startT:cf.Pulse_endT])
    # pulse_min_T_val = cf.Pulse_rise_endT + np.argmin(data[cf.Pulse_rise_endT:cf.Pulse_endT])
    # pulse_rise_range_val = data[cf.Pulse_rise_endT] - data[cf.Pulse_startT]
    # pulse_fall_range_val = data[cf.Pulse_rise_endT] - data[cf.Pulse_fall_endT]
    # pulse_rise_range_ptb_val = pulse_max_val - pre_mean_val
    pulse_fall_range_ptp_val = np.ptp(data)
    # Pulse_selection_val = pulse_rise_range_ptb_val > cf.cut_pulseRange
    pulse_time_val = time
    pulse_time_interval_val = time_previous - time
    
    # Use selected trigger time calculation method
    if args.trigger_method == 'simple':
        trig_check = Find_Trigger_time_simple(trig)
    else:
        trig_check = Find_Trigger_time_splineFit(trig)
    
    pass_selection = Pulse_selection(trig_check)
    
    # Calculate rise and fall time constants
    timing_params = calculate_rise_fall_times(data)
    
    debugPrint(f'Rise Range = {pulse_rise_range_val:.5f}, Fall Range = {pulse_fall_range_val:.5f}')
    debugPrint(f'Rise Time 10-90% = {timing_params["rise_time_10_90"]:.2f} samples, Fall Time 90-10% = {timing_params["fall_time_90_10"]:.2f} samples')
    
    # Store statistics for this event
    event_stats = {
        "event_number": event,
        # "pre_std": float(pre_std_val),
        # "pos_std": float(pos_std_val),
        "pre_mean": float(pre_mean_val),
        # "pos_mean": float(pos_mean_val),
        # "pre_range": float(pre_range_val),
        # "pos_range": float(pos_range_val),
        # "pre_max": float(pre_max_val),
        # "pos_max": float(pos_max_val),
        "pulse_max": float(pulse_max_val),
        "pulse_min": float(pulse_min_val),
        # "pulse_max_T": float(pulse_max_T_val),
        # "pulse_min_T": float(pulse_min_T_val),
        "pulse_rise_range": float(pulse_rise_range_val),
        "pulse_fall_range": float(pulse_fall_range_val),
        "pulse_rise_range_ptb": float(pulse_rise_range_ptb_val),
        "pulse_fall_range_ptp": float(pulse_fall_range_ptp_val),
        "pulse_time": float(pulse_time_val),
        "pulse_time_interval": float(pulse_time_interval_val),
        "trigger_check": float(trig_check),
        "pass_selection": pass_selection,
        "rise_time_10_90": float(timing_params['rise_time_10_90']),
        "fall_time_90_10": float(timing_params['fall_time_90_10']),
        "rise_slew_rate": float(timing_params['rise_slew_rate']),
        "fall_slew_rate": float(timing_params['fall_slew_rate'])
    }
    event_statistics.append(event_stats)

def Common_mode_analysis(chSig_average, data, event):
    chSig_average = np.add(chSig_average, data)
    if (cf.DISPLAY):
        event_display(chSig_average, f'Waveform{event}')
    return chSig_average

def Find_Trigger_time_predefined():
    return [58]

def Find_Trigger_time_splineFit(chTrig):
    """
    Find trigger arrival time using spline interpolation and derivative-based peak finding.
    Optimized version with early returns and simplified logic.
    
    Args:
        chTrig: Trigger channel waveform data
    
    Returns:
        float: Trigger arrival time, or -1 if not found
    """
    # Early return for empty data
    if len(chTrig) == 0:
        return -1
    
    # Create spline interpolation once
    x_index = np.arange(len(chTrig))
    chTrig_spline = CubicSpline(x_index, chTrig)
    
    # Get derivative roots (turning points)
    try:
        roots = chTrig_spline.derivative().roots()
    except:
        return -1
    
    if len(roots) < 2:
        return -1
    
    # Find falling edges (pedestal to peak transitions)
    # For falling edge: previous point (pedestal) should be higher than current point (peak)
    range_threshold = 0.1
    
    for i in range(1, len(roots)):
        prev_root = roots[i-1]
        curr_root = roots[i]
        
        # Check if both roots are in valid range
        if prev_root < 0 or curr_root < 0 or curr_root >= len(chTrig):
            continue
        
        # Check if this is a falling edge with sufficient range
        prev_val = chTrig_spline(prev_root)
        curr_val = chTrig_spline(curr_root)
        
        if prev_val - curr_val > range_threshold:
            # Found a falling edge - calculate 50% crossing time
            try:
                target_value = (prev_val + curr_val) / 2.0  # 50% level between peak and pedestal
                # Use brentq for fast root finding in the interval
                from scipy.optimize import brentq
                
                def crossing_func(x):
                    return chTrig_spline(x) - target_value
                
                # Check if signs are opposite at interval endpoints
                if crossing_func(curr_root) * crossing_func(prev_root) < 0:
                    arrival_time = brentq(crossing_func, prev_root, curr_root)
                    return arrival_time
            except:
                continue
    
    return -1

def Find_Trigger_time_simple(chTrig, threshold_fraction=0.5):
    """
    Faster alternative: Simple threshold crossing method without spline fitting.
    Typically 10-50x faster than spline method.
    
    Args:
        chTrig: Trigger channel waveform data
        threshold_fraction: Fraction of amplitude for threshold (default 0.5 for 50%)
    
    Returns:
        float: Trigger arrival time with linear interpolation, or -1 if not found
    """
    if len(chTrig) < 2:
        return -1
    
    # Find baseline (first few points) and minimum
    baseline = np.mean(chTrig[:10]) if len(chTrig) >= 10 else chTrig[0]
    minimum = np.min(chTrig)
    
    # Calculate threshold
    threshold = baseline - threshold_fraction * (baseline - minimum)
    
    # Find first crossing point
    for i in range(len(chTrig) - 1):
        if chTrig[i] >= threshold and chTrig[i+1] < threshold:
            # Linear interpolation for sub-sample precision
            frac = (threshold - chTrig[i]) / (chTrig[i+1] - chTrig[i])
            return i + frac
    
    return -1

def SingleTDMS_analysis():
    sideband_count = 0
    global metadata_dict  # Store metadata globally to access later
    with TdmsFile.open(in_filename) as tdms_file:
        metadata = tdms_file.properties
        metadata_df = pd.DataFrame(metadata.items(), columns=['metaKey', 'metaValue'])
        print(metadata_df)
        
        # Convert metadata to dictionary for JSON storage
        metadata_dict = {row['metaKey']: row['metaValue'] for _, row in metadata_df.iterrows()}
        
        # totalEvents = int(metadata_df.loc[metadata_df['metaKey'] == 'Total Events', 'metaValue'].iloc[0])
        recordlength = int(metadata_df.loc[metadata_df['metaKey'] == 'record length', 'metaValue'].iloc[0])
        vertical_range = float(metadata_df.loc[metadata_df['metaKey'] == 'vertical range Sig', 'metaValue'].iloc[0])
        SampleRate = float(metadata_df.loc[metadata_df['metaKey'] == 'actual sample rate', 'metaValue'].iloc[0])
        Read_Groups_and_Channels(tdms_file)
        # Get all group names
        group_names = [group.name for group in tdms_file.groups()]
        print(group_names)
        if 'ADC Readout Channels' not in group_names:
            print("Error: 'ADC Readout Channels' group not found in TDMS file.")
            print("This means 'signal rate too low or detector latches' - setting all final analysis numbers to zero")
            
            # Initialize empty event_statistics for this case
            event_statistics = []
            
            # Create final_analysis with all zeros
            final_analysis = {
                "summary_statistics": {
                    "total_time": 0.0,
                    "total_events": 0,
                    "signal_count": 0,
                    "dark_count": 0,
                    "efficiency": 0.0,
                    "count_rate": 0.0,
                    "signal_rate": 0.0,
                    "dark_count_rate": 0.0
                },
                "total_events": 0,
                "event_by_event_data": []
            }
            
            # Save the zero-filled results
            with open(analysisFileName, "w") as f:
                json.dump(final_analysis, f, indent=2)
            
            print(f"Saved zero-filled analysis to {analysisFileName}")
            return
        # Get channels
        chSig_total = tdms_file['ADC Readout Channels']['chSig']
        chTrig_total = tdms_file['ADC Readout Channels']['chTrig']
        chTime_total = tdms_file['ADC Readout Channels']['Time']
        totalEvents = int(len(chSig_total) / recordlength)
        chSig_average = np.zeros(recordlength)
        pulseCount, avgCount = 0, 0
        print(f"==========Start Looping at {datetime.datetime.now()}==========")
        for event in range(totalEvents-1):
            if event == args.subset:
                break
            if args.checkSingleEvent != -1 and event != args.checkSingleEvent:
                continue
            if event % args.report == 0:
                print(f"==========Processing {event}/{totalEvents} event==========")
            chSig = chSig_total[event*recordlength:(event+1)*recordlength]
            chTrig = chTrig_total[event*recordlength:(event+1)*recordlength]
            chTime = chTime_total[event*recordlength:event*recordlength+1][0]
            chTime_previous = chTime_total[(event+1)*recordlength:(event+1)*recordlength+1][0]
            if (cf.DISPLAY): event_display_2ch(chSig, chTrig, 'Waveform')
            pulseCount += 1
            Simple_pulse_analysis(chSig, chTrig, chTime, chTime_previous, event)
            # if Sideband_selection():
            #     sideband_count += 1
            #     debugPrint("pass sideband selection")
            #     if event < cf.avgMaxCount:
            #         avgCount += 1
            #         chSig_average = Common_mode_analysis(chSig_average, chSig, event)
            # else:
            #     debugPrint("fail sideband selection")
        print(f"==========End Looping at {datetime.datetime.now()}==========")
    print(f"TotalEvents:{totalEvents}, TriggerPulse_Count:{pulseCount}, PassSideband_Count:{sideband_count}")
    # if avgCount > 0:
    #     chSig_average = chSig_average / avgCount

if __name__ == "__main__":
    if args.debug_report:
        cf.DEBUG = True
    if args.display_report:
        cf.DISPLAY = True
    for in_filename in args.in_filenames:
        print("\n##############################")
        print(f"input file: {in_filename}")
        basename = in_filename.rsplit('/',1)[1].split('.tdms')[0]
        
        # Check if input file is from SNSPD_rawdata or SNSPD_data directory
        import os
        abs_path = os.path.abspath(in_filename)
        path_parts = abs_path.split(os.sep)
        
        # Find if SNSPD_rawdata or SNSPD_data is in the path
        output_base_dir = None
        subdirs = []
        
        for i, part in enumerate(path_parts):
            if part in ['SNSPD_rawdata', 'SNSPD_data']:
                # Found the mother directory
                # Get the path up to (but not including) the mother directory
                base_path = os.sep.join(path_parts[:i])
                # Create output directory: base_path/SNSPD_analyzed_json
                output_base_dir = os.path.join(base_path, 'SNSPD_analyzed_json')
                # Get subdirectories after the mother directory
                subdirs = path_parts[i+1:-1]  # exclude mother dir and filename
                break
        
        if output_base_dir is not None:
            # Build output directory with same subdirectory structure
            if subdirs:
                outDir = os.path.join(output_base_dir, *subdirs) + '/'
            else:
                outDir = output_base_dir + '/'
            print(f"Detected data directory structure, output will go to: {outDir}")
        else:
            # Use default output directory from args
            outDir = args.outputDir + '/'
            print(f"No SNSPD_rawdata/SNSPD_data found in path, using default: {outDir}")
        
        analysisFileName = outDir + basename + "_analysis.json"
        createDir(outDir)
        
        # Extract voltage and current from filename
        import re
        voltage_match = re.search(r'_(\d+)mV_', basename)
        current_match = re.search(r'_(\d+)uA_', basename)
        
        voltage = int(voltage_match.group(1)) if voltage_match else None
        current = int(current_match.group(1)) if current_match else None
        
        # Calculate resistance (R = V/I), convert mV to V and uA to A
        if voltage is not None and current is not None and current != 0:
            resistance = (voltage / 1000.0) / (current / 1e6)  # in Ohms
            print(f"Voltage: {voltage} mV, Current: {current} uA, Resistance: {resistance:.2f} Ω")
        else:
            resistance = None
            print(f"Could not extract voltage/current from filename")
        
        # Initialize storage for all event statistics
        event_statistics = []
        
        # Run analysis
        SingleTDMS_analysis()
        print(f"==========Analysis completed for {in_filename}==========")
        # Print metadata
        # Calculate summary statistics
        if event_statistics:
            summary_stats = {}
            for key in event_statistics[0].keys():
                if key != "event_number":
                    values = [event[key] for event in event_statistics]
                    summary_stats[f"{key}_mean"] = float(np.mean(values))
                    summary_stats[f"{key}_std"] = float(np.std(values))
                    summary_stats[f"{key}_min"] = float(np.min(values))
                    summary_stats[f"{key}_max"] = float(np.max(values))
            
            # Calculate total time, efficiency, and count rates
            time_cut_min = 196
            time_cut_max = 198
            
            # Get time information
            time_values = [event['pulse_time'] for event in event_statistics]
            total_time = max(time_values) - min(time_values) if time_values else 0
            
            # Calculate signal and dark counts based on trigger_check cut
            signal_events = [event for event in event_statistics if time_cut_min <= event.get('trigger_check', -999) <= time_cut_max]
            dark_events = [event for event in event_statistics if not (time_cut_min <= event.get('trigger_check', -999) <= time_cut_max)]
            
            signal_count = len(signal_events)
            dark_count = len(dark_events)
            total_events = len(event_statistics)
            
            # Calculate pulse_fall_range_ptp statistics for signal and dark subsets
            if signal_events:
                signal_ptp_values = [e['pulse_fall_range_ptp'] for e in signal_events if 'pulse_fall_range_ptp' in e]
                if signal_ptp_values:
                    summary_stats['signal_pulse_fall_range_ptp_mean'] = float(np.mean(signal_ptp_values))
                    summary_stats['signal_pulse_fall_range_ptp_std'] = float(np.std(signal_ptp_values))
                    summary_stats['signal_pulse_fall_range_ptp_min'] = float(np.min(signal_ptp_values))
                    summary_stats['signal_pulse_fall_range_ptp_max'] = float(np.max(signal_ptp_values))
            
            if dark_events:
                dark_ptp_values = [e['pulse_fall_range_ptp'] for e in dark_events if 'pulse_fall_range_ptp' in e]
                if dark_ptp_values:
                    summary_stats['dark_pulse_fall_range_ptp_mean'] = float(np.mean(dark_ptp_values))
                    summary_stats['dark_pulse_fall_range_ptp_std'] = float(np.std(dark_ptp_values))
                    summary_stats['dark_pulse_fall_range_ptp_min'] = float(np.min(dark_ptp_values))
                    summary_stats['dark_pulse_fall_range_ptp_max'] = float(np.max(dark_ptp_values))
            
            # Calculate rates
            count_rate = total_events / total_time if total_time > 0 else 0
            signal_rate = signal_count / total_time if total_time > 0 else 0
            dark_count_rate = dark_count / total_time if total_time > 0 else 0
            
            # Calculate rate errors using Poisson statistics (error = sqrt(N)/time)
            count_rate_error = np.sqrt(total_events) / total_time if total_time > 0 else 0
            signal_rate_error = np.sqrt(signal_count) / total_time if total_time > 0 and signal_count > 0 else 0
            dark_count_rate_error = np.sqrt(dark_count) / total_time if total_time > 0 and dark_count > 0 else 0
            
            # Calculate efficiency using signal_rate / 1E7 (assuming 10 MHz laser repetition rate)
            efficiency = (signal_rate / 1E7) if signal_rate > 0 else 0
            
            # Calculate efficiency error using binomial statistics
            # Number of laser pulses (trials) = source_rate * total_time = 1E7 * total_time
            # Binomial error: sigma = sqrt(p * (1-p) / N) where p = efficiency, N = number of pulses
            if total_time > 0 and efficiency > 0 and efficiency < 1:
                n_pulses = 1E7 * total_time
                efficiency_error = np.sqrt(efficiency * (1 - efficiency) / n_pulses)
            else:
                efficiency_error = 0
            
            # Add to summary statistics
            summary_stats['total_time'] = float(total_time)
            summary_stats['total_events'] = int(total_events)
            summary_stats['signal_count'] = int(signal_count)
            summary_stats['dark_count'] = int(dark_count)
            summary_stats['efficiency'] = float(efficiency)
            summary_stats['efficiency_error'] = float(efficiency_error)
            summary_stats['count_rate'] = float(count_rate)
            summary_stats['count_rate_error'] = float(count_rate_error)
            summary_stats['signal_rate'] = float(signal_rate)
            summary_stats['signal_rate_error'] = float(signal_rate_error)
            summary_stats['dark_count_rate'] = float(dark_count_rate)
            summary_stats['dark_count_rate_error'] = float(dark_count_rate_error)
            
            # Add voltage, current, and resistance
            if voltage is not None:
                summary_stats['bias_voltage_mV'] = int(voltage)
            if current is not None:
                summary_stats['bias_current_uA'] = int(current)
            if resistance is not None:
                summary_stats['resistance_ohm'] = float(resistance)

            # Store final results
            final_analysis = {
                "metadata": metadata_dict,
                "summary_statistics": summary_stats,
                "total_events": len(event_statistics),
                "event_by_event_data": event_statistics
            }
        else:
            final_analysis = {
                "metadata": metadata_dict if 'metadata_dict' in globals() else {},
                "summary_statistics": {},
                "total_events": 0,
                "event_by_event_data": []
            }
        # Print statistics
        print(f"\n==========Analysis Summary==========")
        print(f"Total events processed: {final_analysis['total_events']}")
        
        if final_analysis['summary_statistics']:
            print(f"\n--- Summary Statistics ---")
            for key, value in final_analysis['summary_statistics'].items():
                print(f"{key}: {value:.6f}")
        else:
            print("No statistics available - no events were processed")            
        with open(analysisFileName, "w") as f:
            json.dump(final_analysis, f, indent=2)
