#!/usr/bin/env python

from nptdms import TdmsFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all='ignore')
from enum import Enum
import json
import math

# User defined functions
from Timing_Analyzer import *
from tdmsUtils import *
from plotUtils import *
import config

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-d',default="./plots/test",type=str,help='output directory')
parser.add_argument('--report','-r',default=100,type=int,help='report every x events')
parser.add_argument('--debug_report','-b',default=10000,type=int,help='report every x events')
parser.add_argument('--display_report','-p',default=10000,type=int,help='report every x events')
parser.add_argument('--subset','-s',default=-1,type=int,help='Process a subset of data. -1 = all')
parser.add_argument('--txtfilename','-t',default="test",type=str,help='results txt file')
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
        vertical_range = float(metadata_df.loc[metadata_df['metaKey'] == 'vertical range Sig', 'metaValue'].iloc[0])
        # Read Groups and Channels
        Read_Groups_and_Channels(tdms_file)

        channel_sum = 0.0
        channel_length = 0
        nPass=0
        ranges=[]
        # Sideband region numpy
        prePulse_mean, postPulse_mean, prePulse_stdev, postPulse_stdev, prePulse_range, postPulse_range, prePulse_integral, postPulse_integral = [], [], [], [], [], [], [], []
        # Signal region simple array
        pulseRanges = []
        # Signal region numpy
        chSig_ranges, chSig_diff_ranges, chSig_pulse_amplitudes, chSig_pulse_arrivalTs, chSig_pulse_arrivalTs_220, chSig_pulse_riseTs, chSig_pulse_spline_integrals = [], [], [], [], [], [], []
        data_splines = []
        # Start Looping through events
        for event, chunk in enumerate(tdms_file.data_chunks()):
            # Choose a subset of the whole data to do the analysis. -1 = run All
            if (event == args.subset ): break
            # Loop progress
            if ((event+1)%args.report==0): print ("==========Processing {}/{} event==========".format(event,totalEvents))
            config.DEBUG = True if (event+1)%args.debug_report==0 else False
            config.DISPLAY = True if (event+1)%args.display_report==0 else False
            # Skip chunk larger than totalEvents
            if (event > int(totalEvents)-1): break
            # Skip first event
            if (event == 0 ): continue
            # initialize parameter
            chSig_pulse_arrivalT_0 = -1
            # Read chSig into np array
            chSig = chunk['ADC Readout Channels']['chSig']._data()
            chTrig = chunk['ADC Readout Channels']['chTrig']._data()
            if (config.DISPLAY): event_display_2ch(chSig,chTrig,'Waveform', 0.02)

            chSig_diff = np.diff(chSig)
            chTrig_diff = np.diff(chTrig)
            # Create a spline interpolation function for the data
            x_index = np.arange(len(chSig))
            chSig_spline = CubicSpline(x_index, chSig)
            chTrig_spline = CubicSpline(x_index, chTrig)
            # Get chTrig (trigger) arrival times
            chTrig_turning_pedestals, chTrig_turning_peaks, chTrig_turning_ranges = Get_turning_times(chTrig_spline, 0.1, 0, len(chTrig), 'Fall', config.DEBUG)
            # chTrig_turning_pedestals, chTrig_turning_peaks = Get_turning_times(chTrig_spline, 0.4, 0, len(chTrig), 'Fall', config.DEBUG)
            for ipulse, (chTrig_turning_pedestal, chTrig_turning_peak) in enumerate(zip(chTrig_turning_pedestals, chTrig_turning_peaks)):
                debugPrint('==========Event{}_Pulse{}=========='%(event,ipulse))
                # Skip last pulse due to distortion of the oscilloscop at the boundary
                if (ipulse >= config.NpulsePerTrigger-1): continue
                # Skip unreasonable turning points
                if ( chTrig_turning_peak.x < 0 or chTrig_turning_pedestal.x < 0 ): continue
                # Define time of arrival at the 50% level of the falling slope
                # chTrig_arrivalT = Get_Function_Arrival(chTrig_spline, (chTrig_turning_pedestal.y+chTrig_turning_peak.y)/2, chTrig_turning_pedestal.x, chTrig_turning_peak.x)
                chTrig_arrivalT = Get_Function_Arrival(chTrig_spline, chTrig_turning_pedestal.y-0.1, chTrig_turning_pedestal.x, chTrig_turning_peak.x)
                if (chTrig_arrivalT<0) : continue
                # print(chTrig_arrivalT, int(chTrig_arrivalT))
                # Define signal pulse region
                Pulse_startT =  int(chTrig_arrivalT) + 205
                Pulse_endT =  int(chTrig_arrivalT) + 250
                # Define pre-pulse (sideband) region
                prePulse_startT =  int(chTrig_arrivalT) + 10
                prePulse_endT =  int(chTrig_arrivalT) + 180
                # Define post-pulse (sideband) region
                postPulse_startT =  int(chTrig_arrivalT) + 500
                postPulse_endT =  int(chTrig_arrivalT) + 800
                event_display(chSig[prePulse_startT:postPulse_endT], 'Waveform#{}_pulse{}'.format(event,ipulse))

                # Sideband characteristic
                prePulse_mean.append(np.mean(chSig[prePulse_startT:prePulse_endT])) # mean
                postPulse_mean.append(np.mean(chSig[postPulse_startT:postPulse_endT]))
                prePulse_stdev.append(np.std(chSig[prePulse_startT:prePulse_endT])) # stdev
                postPulse_stdev.append(np.std(chSig[postPulse_startT:postPulse_endT]))
                prePulse_range.append(np.ptp(chSig[prePulse_startT:prePulse_endT])) # max - min
                postPulse_range.append(np.ptp(chSig[postPulse_startT:postPulse_endT]))
                prePulse_integral.append(np.sum(chSig[prePulse_startT:prePulse_endT])) # max - min
                postPulse_integral.append(np.sum(chSig[postPulse_startT:postPulse_endT]))
                # pulseRanges.append(np.ptp(chSig[Pulse_startT:Pulse_endT]))
                # Pulse pre-selection using sideband region
                # if (prePulse_range[-1] < 0.2 and prePulse_stdev[-1] < 0.03 and postPulse_range[-1] < 0.2 and postPulse_stdev[-1] < 0.03):
                if (prePulse_range[-1] < 0.25 and prePulse_stdev[-1] < 0.06):
                    debugPrint('Event{}_Pulse{} pass preselection: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(event,ipulse,prePulse_range[-1],prePulse_stdev[-1],postPulse_range[-1],postPulse_stdev[-1]))
                    # Pulse region
                    chSig_pulse = chSig[Pulse_startT:Pulse_endT]
                    chSig_pulse_xIndex = np.arange(len(chSig_pulse))
                    chSig_range = np.ptp(chSig_pulse[10:30])
                    chSig_ranges.append(chSig_range)
                    # Derivative of pulse region
                    chSig_pulse_diff = chSig_diff[Pulse_startT:Pulse_endT]
                    chSig_pulse_diff_xIndex = np.arange(len(chSig_pulse_diff))
                    chSig_diff_range = np.ptp(chSig_pulse_diff)
                    chSig_diff_ranges.append(chSig_diff_range)
                    debugPrint('Raw data signal region range = %.4f, Diff range = %.4f'%(chSig_range,chSig_diff_range))
                    event_display_2ch(chSig_pulse, chSig_pulse_diff, 'Wavform#%d_pulse%d'%(event,ipulse),0.02)
                    # Event Selection
                    if (chSig_range > 0.1 and chSig_diff_range > 0.1):
                        debugPrint('Event%d_Pulse%d Pass event selection'%(event,ipulse))
                        # simple range
                        chSig_pulse_range = np.ptp(chSig_pulse[10:40])
                        pulseRanges.append(chSig_pulse_range)
                        # Cubic Spline Fit
                        chSig_pulse_spline = CubicSpline(chSig_pulse_xIndex, chSig_pulse)
                        # Find turning point
                        chSig_pulse_turning_pedestals, chSig_pulse_turning_peaks, chSig_pulse_turning_ranges = Get_turning_times(chSig_pulse_spline, 0.02, 10, 40, 'Rise', config.DEBUG)
                        if (len(chSig_pulse_turning_peaks)<1):
                            print('Abnormal Event{}_Pulse{}. Pass Event Selection, but can\'t find turning points'.format(event,ipulse))
                            continue
                        # Get pulse amplitude --> Defined as range between pulse rising turning points
                        chSig_pulse_amplitude = max(chSig_pulse_turning_ranges)
                        imax = chSig_pulse_turning_ranges.index(chSig_pulse_amplitude)
                        chSig_pulse_amplitudes.append(chSig_pulse_amplitude)

                        data_spline = abs(chSig_pulse_amplitude-chSig_pulse_range)/chSig_pulse_range
                        data_splines.append(data_spline)
                        debugPrint("data vs spline diff: {%.3f}".format(data_spline))
                        # Get 50% pulse amplitude level
                        chSig_pulse_10 = chSig_pulse_turning_peaks[imax].y*0.1 + chSig_pulse_turning_pedestals[imax].y*0.9
                        chSig_pulse_50 = chSig_pulse_turning_peaks[imax].y*0.5 + chSig_pulse_turning_pedestals[imax].y*0.5
                        chSig_pulse_90 = chSig_pulse_turning_peaks[imax].y*0.9 + chSig_pulse_turning_pedestals[imax].y*0.1
                        # Get Arrival time
                        chSig_pulse_arrivalT = Get_Function_Arrival(chSig_pulse_spline, chSig_pulse_50, chSig_pulse_turning_pedestals[imax].x, chSig_pulse_turning_peaks[imax].x) + Pulse_startT - chTrig_arrivalT  #int(chTrig_arrivalT) + 205
                        if (chSig_pulse_arrivalT > 0): chSig_pulse_arrivalTs.append(chSig_pulse_arrivalT)
                        # Get Rise time
                        chSig_pulse_riseT = Get_Function_RiseFall_Range(chSig_pulse_spline, chSig_pulse_10, chSig_pulse_90, chSig_pulse_turning_pedestals[imax].x, chSig_pulse_turning_peaks[imax].x)
                        if (chSig_pulse_riseT > 0): chSig_pulse_riseTs.append(chSig_pulse_riseT)

                        debugPrint('Pulse amplitude = %.4f, arrival Time = %.4f, rise Time = %.4f'%(chSig_pulse_amplitude,chSig_pulse_arrivalT,chSig_pulse_riseT))
                        display_spline_fit(chSig_pulse_spline, chSig_pulse_xIndex)
                    else:
                        debugPrint('Event{}_Pulse{} Fail event selection'.format(event,ipulse))
                        # event_display_2ch(chSig_pulse_diff, chSig_pulse, f'Wavform#{event}_pulse{ipulse}')
                else:
                    debugPrint('Event{}_Pulse{} pass preselection: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(event,ipulse,prePulse_range[-1],prePulse_stdev[-1],postPulse_range[-1],postPulse_stdev[-1]))


        #################### End Loop ####################

        # Make Directories
        if(in_filename.find('.txt')!=-1):
            basename = in_filename.rsplit('/',1)[1].split('.txt')[0]
        else:
            basename = in_filename.rsplit('/',1)[1].split('.tdms')[0]

        baseDir = in_filename.split('/')[-2]
        plotDir = args.outputDir + '/' + baseDir + '/' + basename
        createDir(baseDir)
        createDir(plotDir)

        # Create root filen
        hfile = ROOT.TFile('%s/%s.root'%(plotDir,basename), 'RECREATE', 'analysis histograms of {basename} measurements' )


        ###############################################
        #################### Plots ####################
        ###############################################

        # Plots parameters
        bits = 1024
        arrival_beg = 225
        arrival_end = 235

        # Sideband Region
        Sideband_results={}
        # Plot two histograms side-by-side
        Sideband_results['mean']     = plot_2histo(prePulse_mean, postPulse_mean, 50, -0.0025, 0.0025, 'prePulse', 'postPulse', 'Sideband mean', '%s/sideband_mean.png'%plotDir)
        Sideband_results['stdev']    = plot_2histo(prePulse_stdev, postPulse_stdev, 50, 0, 0.1, 'prePulse', 'postPulse', 'Sideband stdev', '%s/sideband_stdev.png'%plotDir)
        Sideband_results['range']    = plot_2histo(prePulse_range, postPulse_range, 100, 0, 0.3, 'prePulse', 'postPulse', 'Sideband range', '%s/sideband_range.png'%plotDir)
        Sideband_results['integral'] = plot_2histo(prePulse_integral, postPulse_integral, 50, -1, 1, 'prePulse', 'postPulse', 'Sideband integral', '%s/sideband_integral.png'%plotDir)
        # # Signal Region
        # Signal_results={}
        # plot_2histo(chSig_pulse_amplitudes, chSig_pulse_ranges, 50, 0, 0.2, 'Spline amplitude', 'Range', 'Pulse amplitude')
        # Signal_results['amplitude'] = plot_histo(chSig_pulse_amplitudes, 256, -0.25, 0.25, 'Voltage [V]', 'Pulse amplitude',f'{plotDir}/signal_amplitude.png')
        # Signal_results['range']     = plot_histo(chSig_pulse_diff_ranges, 50, 0, 0.1, 'Voltage [V]', 'Signal Region differentiate range',f'{plotDir}/signal_diff.png')
        # Signal_results['arrivalT']  = plot_histo(chSig_pulse_arrivalTs, 100, 217, 221, 'Time [index]', 'Pulse arrival time',f'{plotDir}/signal_arrivalT.png')
        # Signal_results['riseT']     = plot_histo(chSig_pulse_riseTs, 50, 0, 7, 'Time [index]', 'Pulse rise time',f'{plotDir}/signal_riseT.png')
        # Signal_results['integral']  = plot_histo(chSig_pulse_spline_integrals, 50, -1, 1, 'Voltage [V]', 'Signal Region Integral',f'{plotDir}/signal_integral.png')
        # plot_histo(chSig_pulse_amplitudes, 50, 0, 0.5, 'Voltage [V]', 'Pulse amplitude',f'{plotDir}/signal_amplitude_zoomin_largerbin.png')

        # # root histograms
        hist_prePulse_mean = plot_histo_root(prePulse_mean, 50, -0.0025, 0.0025, 'prePulse_mean', 'Voltage [V]', 'PrePulse mean', '%s/sideband_prepulse_mean.png'%plotDir)
        hist_postPulse_mean = plot_histo_root(postPulse_mean, 50, -0.01, 0.01, 'postPulse_mean', 'Voltage [V]', 'PostPulse mean', '%s/sideband_postpulse_mean.png'%plotDir)
        hist_prePulse_stdev = plot_histo_root(prePulse_stdev, 50, 0, 0.1, 'prePulse_stdev', 'Voltage [V]', 'PrePulse stdev', '%s/sideband_prepulse_stdev.png'%plotDir)
        hist_postPulse_stdev = plot_histo_root(postPulse_stdev, 50, 0, 0.1, 'postPulse_stdev', 'Voltage [V]', 'PostPulse stdev', '%s/sideband_postpulse_stdev.png'%plotDir)
        hist_prePulse_range = plot_histo_root(prePulse_range, 100, 0, 0.3, 'prePulse_range', 'Voltage [V]', 'PrePulse range', '%s/sideband_prepulse_range.png'%plotDir)
        hist_postPulse_range = plot_histo_root(postPulse_range, 100, 0, 0.3, 'postPulse_range', 'Voltage [V]', 'PostPulse range', '%s/sideband_postpulse_range.png'%plotDir)

        hist_amplitudes        = plot_histo_root(chSig_pulse_amplitudes       , bits*2, 0,   vertical_range*2,   'signal_amplitude', 'Voltage [V]', 'Pulse amplitude','%s/signal_amplitude_root.png'%plotDir)
        hist_range             = plot_histo_root(pulseRanges                  , bits*2, 0,   vertical_range*2,   'signal_range', 'Voltage [V]', 'Pulse range','%s/signal_range_root.png'%plotDir)
        hist_range_0p1mVbinned = plot_histo_root(pulseRanges                  , 10000, 0,   1,                   'signal_range_0p1mVbinned', 'Voltage [V]', 'Pulse range','%s/signal_range_root_0p1mVbinned.png'%plotDir)
        hist_diff_ranges       = plot_histo_root(chSig_diff_ranges            , 50,    0,   0.1,                 'signal_diff', 'Voltage [V]', 'Signal Region differentiate range','%s/signal_diff_root.png'%plotDir)
        hist_arrivalTs         = plot_histo_root(chSig_pulse_arrivalTs        , 25,    arrival_beg, arrival_end, 'signal_arrivalT', 'Time [index]', 'Pulse arrival time','%s/signal_arrivalT_root.png'%plotDir)
        hist_riseTs            = plot_histo_root(chSig_pulse_riseTs           , 50,    0,   7,                   'signal_riseT', 'Time [index]', 'Pulse rise time','%s/signal_riseT_root.png'%plotDir)
        hist_data_spline       = plot_histo_root(data_splines                 , 100,   0,   1,                   'data_spline', '(data-spline)/data', 'Amplitude alg comparison','%s/data_spline.png'%plotDir)

        range_mean, range_meanE, range_std, range_stdE = fit_histo_gaus(hist_range, 0, vertical_range*2, 'fit_signal_range', 'Voltage [V]', 'Pulse range fit', '%s/signal_range_fit.png'%plotDir)
        amplitude_mean, amplitude_meanE, amplitude_std, amplitude_stdE = fit_histo_gaus(hist_amplitudes, 0, vertical_range*2, 'fit_signal_amplitude', 'Voltage [V]', 'Pulse amplitude fit', '%s/signal_amplitude_fit.png'%plotDir)
        arrivalT_mean, arrivalT_meanE, arrivalT_std, arrivalT_stdE = fit_histo_gaus(hist_arrivalTs, arrival_beg, arrival_end, 'fit_signal_arrivalT', 'Time [index]', 'Pulse arrival time fit', '%s/signal_arrivalT_fit.png'%plotDir)

        if (len(chSig_ranges)>0):
            SDE = len(pulseRanges) / len(chSig_ranges)
            SDE_error = math.sqrt((SDE*(1-SDE))/len(chSig_ranges))
        else:
            SDE, SDE_error = 0, 0

        with open('%s/%s/%s.txt'%(args.outputDir,baseDir,args.txtfilename),'a') as f:
            f.write('%s ; Total Events : %d ; SDE=%.3f pm %.3f\n'%(basename, len(chSig_ranges), SDE, SDE_error))
            f.write('Range Mean fit=%.5f pm %.5f ; Range std fit=%.5f pm %.5f ; fit Time jitter=%.3f pm %.3f\n'%(range_mean,range_meanE,range_std,range_stdE,arrivalT_std,arrivalT_stdE))
            f.write('Range Mean=%.5f pm %.5f ; Range std=%.5f\n'%(np.mean(pulseRanges),np.std(pulseRanges)/math.sqrt(len(pulseRanges)),np.std(pulseRanges)))
            f.write('Amplitude Mean fit=%.5f pm %.5f ; Amplitude std fit=%.5f pm %.5f\n'%(amplitude_mean,amplitude_meanE,amplitude_std,amplitude_stdE))

        print('%s ; Total Events : %d ; SDE=%.3f pm %.3f\n'%(basename, len(chSig_ranges), SDE, SDE_error))
        print('Range Mean fit=%.5f pm %.5f ; Range std fit=%.5f pm %.5f ; fit Time jitter=%.3f pm %.3f\n'%(range_mean,range_meanE,range_std,range_stdE,arrivalT_std,arrivalT_stdE))
        print('Range Mean=%.5f pm %.5f ; Range std=%.5f\n'%(np.mean(pulseRanges),np.std(pulseRanges)/math.sqrt(len(pulseRanges)),np.std(pulseRanges)))
        print('Amplitude Mean fit=%.5f pm %.5f ; Amplitude std fit=%.5f pm %.5f\n'%(amplitude_mean,amplitude_meanE,amplitude_std,amplitude_stdE))

        print('\n\n')
        hfile.Write()
        hfile.Close()
        ROOT.gROOT.GetListOfFiles().Remove(hfile)

if __name__ == "__main__":

    createDir(args.outputDir)

    for in_filename in args.in_filenames:
        # baseDir = in_filename.split('/')[-2]
        # Temperature = in_filename.split('uW_')[1].split('K_')[0]
        # Voltage = in_filename.split('/')[-1].split('mV')[0]
        # print(f'Temperature : {Temperature}, Bias Voltage : {Voltage}')
        # SDE, range_average, range_std = SingleTDMS_analysis(in_filename)
        SingleTDMS_analysis(in_filename)
