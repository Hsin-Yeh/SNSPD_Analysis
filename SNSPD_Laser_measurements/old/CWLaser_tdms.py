#!/usr/bin/env python3

from nptdms import TdmsFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from array import array
import os
import errno
from enum import Enum
import json
import math
from scipy.signal import find_peaks

# User defined functions
from .utils.Timing_Analyzer import *
from .utils.tdmsUtils import *
from .utils.plotUtils import *
from .config import CW_config as config

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-d',default="./plots",type=str,help='output directory')
parser.add_argument('--report','-r',default=100,type=int,help='report every x events')
parser.add_argument('--debug_report','-b',action="store_true",help='report every x events')
parser.add_argument('--display_report','-p',action="store_true",help='report every x events')
parser.add_argument('--subset','-s',default=-1,type=int,help='Process a subset of data. -1 = all')
args = parser.parse_args()

def debugPrint(string):
    if (args.debug_report): print(string)

def SingleTDMS_CW_analysis(in_filename):
    with TdmsFile.open(in_filename) as tdms_file:
        print(f"\n########## Processing {in_filename} ##########")
        # Directories
        basename = in_filename.rsplit('/',1)[1].split('.tdms')[0]
        baseDir = in_filename.split('SNSPD_rawdata/')[1].rsplit('/',1)[0]
        plotDir = args.outputDir + '/' + baseDir + '/' + basename
        metaFileName = plotDir + '/' + in_filename.rsplit('/',1)[1].split('.tdms')[0] + ".json"
        print(f"output plot Directory: {plotDir}")
        # make outputDir
        try:
            os.makedirs(plotDir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('output directory exists.')
            else:
                raise
        # Create root filen
        outfile = ROOT.TFile(f'{plotDir}/{basename}.root', 'RECREATE', f'analysis histograms of {basename} measurements' )
        # Create output tree
        outtree = ROOT.TTree("SNSPD_data", "SNSPD_data")
        pulseCount = array( 'f', [ 0 ] )
        pulseRange = array( 'f', [ 0 ])
        outtree.Branch('pulseCount',pulseCount,"pulseCount/F")
        outtree.Branch('pulseRange',pulseRange,"pulseRange/F")

        # Read Meta Data (Basic information)
        metadata = tdms_file.properties
        metadata_df = pd.DataFrame(metadata.items(), columns=['metaKey', 'metaValue'])
        print(metadata_df)
        totalEvents = metadata_df.loc[metadata_df['metaKey'] == 'Total Events', 'metaValue'].iloc[0]
        recordlength = float(metadata_df.loc[metadata_df['metaKey'] == 'record length', 'metaValue'].iloc[0])
        sampleRate = float(metadata_df.loc[metadata_df['metaKey'] == 'actual sample rate', 'metaValue'].iloc[0])
        timeWindow = recordlength / sampleRate
        metadata_df.to_json(metaFileName,orient="records",lines=True) # Write metadata to json file
        # Read Groups and Channels
        Read_Groups_and_Channels(tdms_file)
        # Initialize arrays
        counts, pulseRanges = np.array([]), np.array([])
        # Set threshold
        threshold = 0.35
        # Start Looping through events
        print("========== Start Loop ==========")
        for event, chunk in enumerate(tdms_file.data_chunks()):
            # Choose a subset of the whole data to do the analysis. -1 = run All
            if (event == args.subset ): break
            # Loop progress
            if ((event)%args.report==0): print (f"==========Processing {event}/{totalEvents} event==========")
            # Skip chunk larger than totalEvents
            if (event > int(totalEvents)-1): break
            # Read chSig into np array
            try:
                chSig = chunk['ADC Readout Channels']['chSig']._data()
            except KeyError:
                print ('No ADC Readout Channels key')
                continue
            # Initialize counts
            count=0
            # Find peaks
            peaks, _ = find_peaks(chSig, height=threshold, distance=100)
            # Count peaks
            pulseCount[0] = len(peaks)
            debugPrint(f"Event {event} Number of peaks above threshold: {pulseCount[0]}")
            counts = np.append(counts, pulseCount[0])
            # peak ranges
            for peak in peaks:
                pulseRange[0] = np.ptp(chSig[peak:peak+300])
                pulseRanges = np.append(pulseRanges, pulseRange[0])

            # Write Tree
            outtree.Fill()
            # Plot waveform with peaks
            if (args.display_report):
                # info = r'$T=4.6K,\quad V_{Bias}=1.7V,\quad 100 \mu W,\quad 532nm\, CW\, laser$'
                # plt.title(in_filename, fontsize = 13, loc='right')
                plt.plot(chSig, label='data')
                plt.plot(peaks, chSig[peaks], "x", label='Found peaks')
                plt.xlabel(r'Index [0.4ns], Gate width 20$\mu$s',fontsize=15)
                plt.ylabel('ADC [V]',fontsize=15)
                # plt.ylim(-0.06,0.06)
                plt.legend()
                plt.tight_layout()
                # plt.title('Waveform with Peaks')
                plt.show()
        print("========== End Loop ==========")
        outtree.Write()
        outfile.Close()
        if ( len(pulseRanges) == 0 or len(counts) == 0 ):
            print("No Signal Pulses")
            return
        # Results
        hist_counts  = plot_histo_root(counts      , 15, 0, np.max(counts), 'counts', 'counts', f'Photon Counts / {timeWindow}s', f'{plotDir}/{basename}_counts.png')
        hist_range   = plot_histo_root(pulseRanges , 15, 0.02, 2, 'signal_range', 'signal_range', 'Voltage [V]', f'{plotDir}/{basename}_signal_range.png')

        PC_mean = np.mean(counts)
        PC_meanE = np.std(counts)/math.sqrt(len(counts))
        PCR_mean = PC_mean/timeWindow
        PCR_meanE = PC_meanE/timeWindow

        with open(f'{plotDir}/{basename}_CW_counts.txt','a') as f:
            f.write(f'{basename} ; Window : {timeWindow}s ; Total Events : {len(counts)} ;')
            f.write(f'PCR mean={PCR_mean:.3f} pm {PCR_meanE:.3f}\n')
            f.write(f'Range Mean={np.mean(pulseRanges):.5f} pm {np.std(pulseRanges)/math.sqrt(len(pulseRanges)):.5f} ; Range std={np.std(pulseRanges):.5f}\n')

        print(f'{basename} ; Window : {timeWindow}s ; Total Events : {len(counts)} ;')
        print(f'PC mean={PC_mean:.3f} pm {PC_meanE:.3f} ; PCR mean={PCR_mean:.3f} pm {PCR_meanE:.3f}')
        print(f'Range Mean={np.mean(pulseRanges):.5f} pm {np.std(pulseRanges)/math.sqrt(len(pulseRanges)):.5f} ; Range std={np.std(pulseRanges):.5f}\n')

if __name__ == "__main__":

    if (args.debug_report==True): config.DEBUG = True
    if (args.display_report==True): config.DISPLAY = True

    for in_filename in args.in_filenames:
        SingleTDMS_CW_analysis(in_filename)
