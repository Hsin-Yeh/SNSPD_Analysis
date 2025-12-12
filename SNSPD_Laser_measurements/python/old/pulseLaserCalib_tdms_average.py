#!/usr/bin/env python

from nptdms import TdmsFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from array import array
import os
import errno
from enum import Enum
import json

# User defined functions
from Timing_Analyzer import *
from tdmsUtils import *
from plotUtils import *
import config

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-d',default="./plots",type=str,help='output directory')
parser.add_argument('--report','-r',default=100,type=int,help='report every x events')
parser.add_argument('--debug_report','-b',default=10000,type=int,help='report every x events')
parser.add_argument('--display_report','-p',default=10000,type=int,help='report every x events')
parser.add_argument('--subset','-s',default=-1,type=int,help='Process a subset of data. -1 = all')
args = parser.parse_args()

def debugPrint(string):
    if (config.DEBUG): print(string)

def SingleTDMS_analysis(in_filename):
    with TdmsFile.open(in_filename) as tdms_file:
        # Read Meta Data (Basic information)
        metadata = tdms_file.properties
        metadata_df = pd.DataFrame(metadata.items(), columns=['metaKey', 'metaValue'])
        print(metadata_df)
        totalEvents = metadata_df.loc[metadata_df['metaKey'] == 'Total Events', 'metaValue'].iloc[0]
        recordLength = int(metadata_df.loc[metadata_df['metaKey'] == 'record length', 'metaValue'].iloc[0])
        # Read Groups and Channels
        Read_Groups_and_Channels(tdms_file)

        channel_sum = 0.0
        channel_length = 0
        nPass=0
        ch1_average = np.zeros(recordLength)
        ch2_average = np.zeros(recordLength)

        # Start Looping through events
        for event, chunk in enumerate(tdms_file.data_chunks()):
            # Choose a subset of the whole data to do the analysis. -1 = run All
            if (event == args.subset ): break
            # Loop progress
            if ((event+1)%args.report==0): print (f"==========Processing {event}/{totalEvents} event==========")
            config.DEBUG = True if (event+1)%args.debug_report==0 else False
            config.DISPLAY = True if (event+1)%args.display_report==0 else False
            # Skip chunk larger than totalEvents
            if (event > int(totalEvents)-1): break
            # Read ch1 into np array
            ch1 = chunk['ADC Readout Channels']['ch1']._data()
            ch2 = chunk['ADC Readout Channels']['ch2']._data()
            if (config.DISPLAY): event_display_2ch(ch1,ch2,f'Waveform')
            ch1_average = np.add(ch1_average,ch1)
            ch2_average = np.add(ch2_average,ch2)

        ch1_average = ch1_average / int(totalEvents)
        ch2_average = ch2_average / int(totalEvents)
        config.DISPLAY = True
        event_display_2ch(ch1_average,ch2_average,f'Waveform')

        # Plots

        # Define plotDir
        if(in_filename.find('.txt')!=0):
            basename = in_filename.rsplit('/',1)[1].split('.txt')[0]
            plotDir = args.outputDir + '/' + basename
        else:
            in_filename.rsplit('/',1)[1].split('.tdms')[0]
            plotDir = args.outputDir + '/' + basename
        # make plotdir
        try:
            os.makedirs(plotDir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Plot directory exists.')
            else:
                raise

        # Create root filen
        hfile = ROOT.TFile(f'{plotDir}/{basename}.root', 'RECREATE', 'analysis histograms of {basename} measurements' )

if __name__ == "__main__":

    for in_filename in args.in_filenames:
        SingleTDMS_analysis(in_filename)
