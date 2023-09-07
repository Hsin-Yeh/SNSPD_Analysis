#!/usr/bin/env python3

from nptdms import TdmsFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all='ignore')
from enum import Enum
import json
import math

# User defined functions
from ..utils.Timing_Analyzer import *
from ..utils.tdmsUtils import *
from ..utils.plotUtils import *
from ..config import config

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-d',default="./plots/test",type=str,help='output directory')
parser.add_argument('--avgCount','-a',default=20,type=int,help='average pulse counts')
parser.add_argument('--report','-r',default=100,type=int,help='report every x events')
parser.add_argument('--debug_report','-b',action="store_true",help='report every x events')
parser.add_argument('--display_report','-p',action="store_true",help='report every x events')
parser.add_argument('--doSingle',action="store_true",help='do single pulse analysis')
parser.add_argument('--subset','-s',default=-1,type=int,help='Process a subset of data. -1 = all')
parser.add_argument('--txtfilename','-t',default="test",type=str,help='results txt file')
args = parser.parse_args()


def debugPrint(string):
    if (config.DEBUG): print(string)

def Advanced_pulse_analysis(data, trigT, event, histo_pulse):
    data_xIndex = np.arange(len(data))
    # Cubic Spline Fit
    data_spline = CubicSpline(data_xIndex, data)
    # Find turning point
    data_turning_pedestals, data_turning_peaks, data_turning_ranges = Get_turning_times(data_spline, 0.02, 0, len(data), 'Rise', config.DEBUG)
    if (len(data_turning_peaks)<1):
        print(f'Abnormal Event{event}. Pass Event Selection, but can\'t find turning points')
        return
    # Get pulse amplitude --> Defined as range between pulse rising turning points
    data_amplitude = max(data_turning_ranges)
    histo_pulse["amplitude"].Fill(data_amplitude)
    imax = data_turning_ranges.index(data_amplitude)

    # Get 50% pulse amplitude level
    data_10 = data_turning_peaks[imax].y*0.1 + data_turning_pedestals[imax].y*0.9
    data_50 = data_turning_peaks[imax].y*0.5 + data_turning_pedestals[imax].y*0.5
    data_90 = data_turning_peaks[imax].y*0.9 + data_turning_pedestals[imax].y*0.1
    # Get Arrival time
    data_arrivalT = Get_Function_Arrival(data_spline, data_50, data_turning_pedestals[imax].x, data_turning_peaks[imax].x) + config.Pulse_startT + trigT  #int(chTrig_arrivalT) + 205
    histo_pulse["arrivalT"].Fill(data_arrivalT)
    # Get Rise time
    data_riseT = Get_Function_RiseFall_Range(data_spline, data_10, data_90, data_turning_pedestals[imax].x, data_turning_peaks[imax].x)
    histo_pulse["riseT"].Fill(data_riseT)

    debugPrint(f'Pulse amplitude = {data_amplitude:.4f}, arrival Time = {data_arrivalT:.4f}, rise Time = {data_riseT:.4f}')
    display_spline_fit(data_spline, data_xIndex)

def Simple_pulse_analysis(data, event, histo_sb, histo_pulse):
    event_display(data, f'Waveform#{event}')

    histo_sb["pre_std"].Fill(np.std(data[config.prePulse_startT:config.prePulse_endT]))
    histo_sb["pos_mean"].Fill(np.mean(data[config.prePulse_startT:config.prePulse_endT]))
    histo_sb["pre_range"].Fill(np.ptp(data[config.prePulse_startT:config.prePulse_endT]))
    histo_sb["pos_std"].Fill(np.std(data[config.postPulse_startT:config.postPulse_endT]))
    histo_sb["pre_mean"].Fill(np.mean(data[config.postPulse_startT:config.postPulse_endT]))
    histo_sb["pos_range"].Fill(np.ptp(data[config.postPulse_startT:config.postPulse_endT]))

    # Pulse region
    data_pulse = data[config.Pulse_startT:config.Pulse_endT]
    data_range = np.ptp(data_pulse)
    histo_pulse["range"].Fill(data_range)
    debugPrint(f'Range = {data_range:.5f}')
    event_display(data_pulse, f'Waveform#{event}')

def SingleTDMS_analysis(in_filename):
    with TdmsFile.open(in_filename) as tdms_file:
        # Read Meta Data (Basic information)
        metadata = tdms_file.properties
        metadata_df = pd.DataFrame(metadata.items(), columns=['metaKey', 'metaValue'])
        print(metadata_df)
        totalEvents = int(metadata_df.loc[metadata_df['metaKey'] == 'Total Events', 'metaValue'].iloc[0])
        recordlength = int(metadata_df.loc[metadata_df['metaKey'] == 'record length', 'metaValue'].iloc[0])
        vertical_range = float(metadata_df.loc[metadata_df['metaKey'] == 'vertical range Sig', 'metaValue'].iloc[0])
        # Read Groups and Channels
        Read_Groups_and_Channels(tdms_file)
        chSig_total = tdms_file['ADC Readout Channels']['chSig']
        chTrig_total = tdms_file['ADC Readout Channels']['chTrig']

        # initialize parameter
        chSig_average = np.zeros(1000)
        chTrig_arrivalT_average = 0
        pulseCount = 0
        plotDisplay = True

        for event in range(totalEvents):
            # Choose a subset of the whole data to do the analysis. -1 = run All
            if (event == args.subset ): break
            # Loop progress
            if ((event+1)%args.report==0): print (f"==========Processing {event}/{totalEvents} event==========")
            # Read chSig into np array
            chSig = chSig_total[event * recordlength:(event+1) * recordlength]
            chTrig = chTrig_total[event * recordlength:(event+1) * recordlength]
            event_display_2ch(chSig,chTrig,f'Waveform', 0.02)
            # Get chTrig (trigger) arrival times
            x_index = np.arange(len(chTrig))
            chTrig_spline = CubicSpline(x_index, chTrig)
            chTrig_turning_pedestals, chTrig_turning_peaks, chTrig_turning_ranges = Get_turning_times(chTrig_spline, 0.1, 0, len(chTrig), 'Fall', config.DEBUG)
            # Loop over laser pulse
            for ipulse, (chTrig_turning_pedestal, chTrig_turning_peak) in enumerate(zip(chTrig_turning_pedestals, chTrig_turning_peaks)):
                debugPrint(f'==========Event{event}_Pulse{ipulse}==========')
                # Skip last pulse due to distortion of the oscilloscop at the boundary
                if (ipulse >= config.NpulsePerTrigger-1): continue
                # Skip unreasonable turning points
                if ( chTrig_turning_peak.x < 0 or chTrig_turning_pedestal.x < 0 ): continue
                # Define time of arrival at the 50% level of the falling slope
                chTrig_arrivalT = Get_Function_Arrival(chTrig_spline, chTrig_turning_pedestal.y-0.1, chTrig_turning_pedestal.x, chTrig_turning_peak.x)
                if (chTrig_arrivalT<0) : continue
                chSig_average = np.add(chSig_average,chSig[int(chTrig_arrivalT):int(chTrig_arrivalT) + 1000])
                chTrig_arrivalT_average = chTrig_arrivalT_average + ( chTrig_arrivalT - int(chTrig_arrivalT) )
                pulseCount = pulseCount + 1
                if (args.doSingle):
                    Simple_pulse_analysis(chSig[int(chTrig_arrivalT):int(chTrig_arrivalT) + 1000], event, sb, Pulse)
                    Advanced_pulse_analysis(chSig[int(chTrig_arrivalT):int(chTrig_arrivalT) + 1000], chTrig_arrivalT - int(chTrig_arrivalT), event, Pulse)
                    if (event==1 and ipulse == 3):
                        for i in range(1000): Pulse_display.SetPoint(i,i,chSig[int(chTrig_arrivalT) + i])

            # Analysis after averaging pulses
            if (pulseCount>args.avgCount):
                chSig_average = chSig_average/pulseCount
                chTrig_arrivalT_average = chTrig_arrivalT_average/pulseCount
                Simple_pulse_analysis(chSig_average, event, sb_avg, Pulse_avg)
                Advanced_pulse_analysis(chSig_average, chTrig_arrivalT_average, event, Pulse_avg)
                if (plotDisplay):
                    for i in range(1000): Pulse_avg_display.SetPoint(i,i,chSig_average[i])
                    plotDisplay = False
                chSig_average = np.zeros(1000)
                chTrig_arrivalT_average = 0
                pulseCount = 0
            else:
                continue

if __name__ == "__main__":

    in_filename = args.in_filenames[0]
    if (args.debug_report==True): config.DEBUG = True
    if (args.display_report==True): config.DISPLAY = True

    # Make Directories
    if(in_filename.find('.txt')!=-1):
        basename = in_filename.rsplit('/',1)[1].split('.txt')[0]
    else:
        basename = in_filename.rsplit('/',1)[1].split('.tdms')[0]
    baseDir = in_filename.split('/')[-2]
    plotDir = args.outputDir + '/' + baseDir + '/' + basename
    avg_plotDir = args.outputDir + '/' + baseDir + '/' + basename + '/Avg_' + str(args.avgCount)
    createDir(args.outputDir)
    createDir(baseDir)
    createDir(plotDir)
    createDir(avg_plotDir)
    # Create root filen
    hfile = ROOT.TFile(f'{avg_plotDir}/{basename}.root', 'RECREATE', 'analysis histograms of {basename} measurements' )

    # Histogram collection
    sb, sb_avg, Pulse, Pulse_avg = {}, {}, {}, {}
    sb["pre_std"] = ROOT.TH1F("prePulse_std", "prePulse_std", 50, 0, 0.1)
    sb["pos_std"] = ROOT.TH1F("posPulse_std", "posPulse_std", 50, 0, 0.1)
    sb["pre_mean"] = ROOT.TH1F("prePulse_mean", "prePulse_mean", 50, -0.005, 0.005)
    sb["pos_mean"] = ROOT.TH1F("posPulse_mean", "posPulse_mean", 50, -0.005, 0.005)
    sb["pre_range"] = ROOT.TH1F("prePulse_range", "prePulse_range", 1000, 0, 0.2)
    sb["pos_range"] = ROOT.TH1F("posPulse_range", "posPulse_range", 1000, 0, 0.2)

    sb_avg["pre_std"] = ROOT.TH1F(f"prePulse_avg_{args.avgCount}_std", f"prePulse_avg_{args.avgCount}_std", 50, 0, 0.01)
    sb_avg["pos_std"] = ROOT.TH1F(f"posPulse_avg_{args.avgCount}_std", f"posPulse_avg_{args.avgCount}_std", 50, 0, 0.01)
    sb_avg["pre_mean"] = ROOT.TH1F(f"prePulse_avg_{args.avgCount}_mean", f"prePulse_avg_{args.avgCount}_mean", 50, -0.005, 0.005)
    sb_avg["pos_mean"] = ROOT.TH1F(f"posPulse_avg_{args.avgCount}_mean", f"posPulse_avg_{args.avgCount}_mean", 50, -0.005, 0.005)
    sb_avg["pre_range"] = ROOT.TH1F(f"prePulse_avg_{args.avgCount}_range", f"prePulse_avg_{args.avgCount}_range", 1000, 0, 0.2)
    sb_avg["pos_range"] = ROOT.TH1F(f"posPulse_avg_{args.avgCount}_range", f"posPulse_avg_{args.avgCount}_range", 1000, 0, 0.2)

    Pulse["range"] = ROOT.TH1F("Pulse_range", "Pulse Range", 1024, 0, 0.5)
    Pulse["amplitude"] = ROOT.TH1F("Pulse_amplitude", "Pulse Amplitude", 1024, 0, 0.5)
    Pulse["arrivalT"] = ROOT.TH1F("Pulse_arrivalT", "Pulse arrival time", 100, 441, 445)
    Pulse["riseT"] = ROOT.TH1F("Pulse_riseT", "Pulse rising time", 100, 0, 6)

    Pulse_avg["range"] = ROOT.TH1F(f"Pulse_avg_{args.avgCount}_range", f"Average {args.avgCount} Pulse Range", 1024, 0, 0.5)
    Pulse_avg["amplitude"] = ROOT.TH1F(f"Pulse_avg_{args.avgCount}_amplitude", f"Average {args.avgCount} Pulse Amplitude", 1024, 0, 0.5)
    Pulse_avg["arrivalT"] = ROOT.TH1F(f"Pulse_avg_{args.avgCount}_arrivalT", f"Average {args.avgCount} Pulse arrival time", 100, 441, 445)
    Pulse_avg["riseT"] = ROOT.TH1F(f"Pulse_avg_{args.avgCount}_riseT", f"Average {args.avgCount} Pulse rising time", 100, 0, 6)

    Pulse_display = ROOT.TGraph()
    Pulse_avg_display = ROOT.TGraph()
    Pulse_display.SetName("Pulse_display")
    Pulse_avg_display.SetName(f"Pulse_avg_{args.avgCount}_display")

    #################### Start Analysis ####################
    SingleTDMS_analysis(in_filename)
    #################### End Analysis ####################

    ###############################################
    #################### Plots ####################
    ###############################################
    c1 = ROOT.TCanvas()
    Pulse_avg_display.SetMarkerStyle(4)
    Pulse_avg_display.SetMarkerSize(0.5)
    Pulse_avg_display.Draw("ALP")
    c1.SaveAs(f"{avg_plotDir}/{Pulse_avg_display.GetName()}.png")
    for key, hist in sb_avg.items():
        hist.Draw("HIST")
        c1.SaveAs(f"{avg_plotDir}/{hist.GetName()}.png")
    for key, hist in Pulse_avg.items():
        hist.Draw("HIST")
        c1.SaveAs(f"{avg_plotDir}/{hist.GetName()}.png")
    if (args.doSingle):
        Pulse_display.Draw("ALP")
        c1.SaveAs(f"{plotDir}/{Pulse_display.GetName()}.png")
        for key, hist in sb.items():
            hist.Draw("HIST")
            c1.SaveAs(f"{plotDir}/{hist.GetName()}.png")
        for key, hist in Pulse.items():
            hist.Draw("HIST")
            c1.SaveAs(f"{plotDir}/{hist.GetName()}.png")

    c2 = ROOT.TCanvas("c2","c2",2560,1280)
    c2.Divide(3,2)
    c2.cd(1)
    Pulse_avg_display.Draw("ALP")
    c2.cd(2)
    Pulse_avg["amplitude"].Draw("HIST")
    c2.cd(3)
    Pulse_avg["arrivalT"].Draw("HIST")
    c2.cd(4)
    sb_avg["pre_range"].Draw("HIST")
    c2.cd(5)
    sb_avg["pos_range"].Draw("HIST")
    c2.cd(6)
    sb_avg["pos_std"].Draw("HIST")
    c2.SaveAs(f"{avg_plotDir}/test.png")

    # End Info
    SNR = Pulse_avg["amplitude"].GetMean() / sb_avg["pos_range"].GetMean()
    print(f"Average Pulse amplitude = {Pulse_avg['amplitude'].GetMean():.3f}, Sideband range = {sb_avg['pos_range'].GetMean():.3f}, SNR = {SNR:.2f}")

    # Write and Close Root file
    Pulse_display.Write()
    Pulse_avg_display.Write()
    hfile.Write()
    hfile.Close()
    ROOT.gROOT.GetListOfFiles().Remove(hfile)
