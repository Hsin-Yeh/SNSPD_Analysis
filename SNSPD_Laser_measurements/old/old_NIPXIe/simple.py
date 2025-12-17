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
from ..config import SNSPD_5_Ch4 as cf

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
args = parser.parse_args()

def debugPrint(string):
    if (cf.DEBUG):
        print(string)

def Sideband_selection():
    if pre_range[0] < cf.cut_preRange and pos_range[0] < cf.cut_posRange and pre_std[0] < cf.cut_preStd and pos_std[0] < cf.cut_posStd:
        return True
    else:
        return False

def Pulse_selection():
    if pulse_fall_range[0] > cf.cut_pulseRange:
        return True
    else:
        return True

def Simple_pulse_analysis(data, event):
    # pre_std_val = np.std(data[cf.prePulse_startT:cf.prePulse_endT])
    # pos_mean_val = np.mean(data[cf.prePulse_startT:cf.prePulse_endT])
    # pre_range_val = np.ptp(data[cf.prePulse_startT:cf.prePulse_endT])
    # pos_std_val = np.std(data[cf.postPulse_startT:cf.postPulse_endT])
    pre_mean_val = np.mean(data[cf.postPulse_startT:cf.postPulse_endT])
    # pos_range_val = np.ptp(data[cf.postPulse_startT:cf.postPulse_endT])
    # pre_max_val = np.max(data[cf.prePulse_startT:cf.prePulse_endT])
    # pos_max_val = np.max(data[cf.prePulse_startT:cf.prePulse_endT])
    pulse_max_val = np.max(data[cf.Pulse_startT:cf.Pulse_endT])
    pulse_min_val = np.min(data[cf.Pulse_startT:cf.Pulse_endT])
    # pulse_max_T_val = cf.Pulse_startT + np.argmax(data[cf.Pulse_startT:cf.Pulse_endT])
    # pulse_min_T_val = cf.Pulse_rise_endT + np.argmin(data[cf.Pulse_rise_endT:cf.Pulse_endT])
    pulse_rise_range_val = data[cf.Pulse_rise_endT] - data[cf.Pulse_startT]
    pulse_fall_range_val = data[cf.Pulse_rise_endT] - data[cf.Pulse_fall_endT]
    pulse_rise_range_ptb_val = pulse_max_val - pre_mean_val
    pulse_fall_range_ptp_val = np.ptp(data[cf.Pulse_startT:cf.Pulse_endT])
    # Pulse_selection_val = pulse_rise_range_ptb_val > cf.cut_pulseRange
    debugPrint(f'Rise Range = {pulse_rise_range_val:.5f}, Fall Range = {pulse_fall_range_val:.5f}')
    
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
    }
    event_statistics.append(event_stats)

def Common_mode_analysis(chSig_average, data, event):
    chSig_average = np.add(chSig_average, data)
    if (cf.DISPLAY):
        event_display(chSig_average, f'Waveform{event}')
    return chSig_average

def Find_Trigger_time_predefined():
    return [58]

def SingleTDMS_analysis():
    sideband_count = 0
    with TdmsFile.open(in_filename) as tdms_file:
        metadata = tdms_file.properties
        metadata_df = pd.DataFrame(metadata.items(), columns=['metaKey', 'metaValue'])
        print(metadata_df)
        totalEvents = int(metadata_df.loc[metadata_df['metaKey'] == 'Total Events', 'metaValue'].iloc[0])
        recordlength = int(metadata_df.loc[metadata_df['metaKey'] == 'record length', 'metaValue'].iloc[0])
        vertical_range = float(metadata_df.loc[metadata_df['metaKey'] == 'vertical range Sig', 'metaValue'].iloc[0])
        SampleRate = float(metadata_df.loc[metadata_df['metaKey'] == 'actual sample rate', 'metaValue'].iloc[0])
        metadata_df.to_json(metaFileName, orient="records", lines=True)
        Read_Groups_and_Channels(tdms_file)
        chSig_total = tdms_file['ADC Readout Channels']['chSig']
        chTrig_total = tdms_file['ADC Readout Channels']['chTrig']
        chSig_average = np.zeros(recordlength)
        pulseCount, avgCount = 0, 0
        print(f"==========Start Looping at {datetime.datetime.now()}==========")
        for event in range(totalEvents):
            if event == args.subset:
                break
            if args.checkSingleEvent != -1 and event != args.checkSingleEvent:
                continue
            if event % args.report == 0:
                print(f"==========Processing {event}/{totalEvents} event==========")
            chSig = chSig_total[event*recordlength:(event+1)*recordlength]
            chTrig = chTrig_total[event*recordlength:(event+1)*recordlength]
            # event_display(chSig, 'Waveform')
            pulseCount += 1
            Simple_pulse_analysis(chSig, event)
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
        outDir = args.outputDir + '/'
        analysisFileName = outDir + basename + "_analysis.json"
        metaFileName = outDir + basename + "_meta.json"
        createDir(outDir)
        
        # Initialize storage for all event statistics
        event_statistics = []
        
        # Prepare arrays for current event values
        pre_std, pos_std, pre_mean, pos_mean = array('f',[0]),array('f',[0]),array('f',[0]),array('f',[0])
        pre_range, pos_range, pre_max, pos_max = array('f',[0]),array('f',[0]),array('f',[0]),array('f',[0])
        pulse_max, pulse_min, pulse_max_T, pulse_min_T = array('f',[0]),array('f',[0]),array('f',[0]),array('f',[0])
        pulse_rise_range, pulse_fall_range = array('f',[0]),array('f',[0])
        pulse_rise_range_ptb, pulse_fall_range_ptp = array('f',[0]),array('f',[0])

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


            # Store final results
            final_analysis = {
                "summary_statistics": summary_stats,
                "total_events": len(event_statistics),
                "event_by_event_data": event_statistics
            }
        else:
            final_analysis = {
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
