#!/usr/bin/env python3

from nptdms import TdmsFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# User defined functions
from pulseLaserCalib_tdms import SingleTDMS_analysis
from Timing_Analyzer import *
from tdmsUtils import *
from plotUtils import *

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-d',default="./",type=str,help='output directory')
parser.add_argument('--report','-r',default=100,type=int,help='report every x events')
parser.add_argument('--debug_report','-b',default=-1,type=int,help='report every x events')
parser.add_argument('--display_report','-p',default=-1,type=int,help='report every x events')
args = parser.parse_args()

if __name__ == "__main__":

    sweep_voltages = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800]
    SDEs = [0.014, 0.048, 0.044, 0.063, 0.042, 0.039, 0.039, 0.040, 0.049, 0.064, 0.093, 0.206, 0.522, 0.848, 0.964, 0.895, 0.561, 0.410]
    prePulse_std=[0.0128,0.0150,0.0153,0.0149,0.0151,0.0155,0.0150,0.0145,0.0156,0.0152,0.0146,0.0142,0.0151,0.0150,0.0151,0.0129,0.0123,0.0128]
    prePulse_range=[0.0567, 0.0699, 0.0705, 0.0689, 0.0711, 0.0691, 0.0669, 0.0722, 0.0697, 0.0667, 0.0692, 0.0660, 0.0697, 0.0692, 0.0694, 0.0593, 0.0563, 0.0581]
    prePulse_stderr=[0.0101,0.0084,0.0100,0.0082,0.0086,0.0093,0.0088,0.0074,0.0113,0.0097,0.0079,0.0072,0.0095,0.0087,0.0103,0.0128,0.0115,0.0101]



    # for in_filename in args.in_filenames:
    #     sweep_voltage = in_filename.rsplit('/',1)[1].split('mV')[0].rsplit('_',1)[1]
    #     sweep_voltages.append(sweep_voltage)
    #     print(f'Bias voltage : {sweep_voltage}')
    #     SDE, Sideband_results, Signal_results = SingleTDMS_analysis(in_filename)
    #     SDEs.append(SDE)
    #     print(f'SDE={SDE}')
        # Signal_results_collection.append(Signal_results)

    plt.scatter(sweep_voltages, SDEs)
    plt.title('SDE')
    plt.xlabel('Voltage [mV]')
    plt.ylabel('Efficiency')
    plt.show()

    plt.scatter(sweep_voltages, prePulse_std)
    plt.title('PrePulse stdev')
    plt.xlabel('Voltage')
    plt.ylabel('prePulse stdev [V]')
    plt.show()

    plt.scatter(sweep_voltages, prePulse_range)
    plt.title('PrePulse range')
    plt.xlabel('Voltage')
    plt.ylabel('prePulse range [V]')
    plt.show()
