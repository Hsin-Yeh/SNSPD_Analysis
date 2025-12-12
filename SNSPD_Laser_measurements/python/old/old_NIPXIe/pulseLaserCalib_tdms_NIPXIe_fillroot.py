#!/usr/bin/env python3

 #!/usr/bin/env python3

# printf '%s\n' /Volumes/T7/SNSPD/20230821/4.7K/*uW/*-1250mV.tdms | parallel --progress --jobs 3 python3 -m python.NIPXIE.pulseLaserCalib_tdms_NIPXIe_average {} --doSingle -d plots/20230821/20231011_1/.
# python3 -m python.NIPXIE.pulseLaserCalib_tdms_NIPXIe_fillroot -d plots/20240221/20231011_1/.

from nptdms import TdmsFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from array import array
np.seterr(all='ignore')
from enum import Enum
import json
import math
import datetime

# User defined functions
from ..utils.Timing_Analyzer import *
from ..utils.tdmsUtils import *
from ..utils.plotUtils import *
from ..utils.osUtils import *
from ..utils.fitfunctions import *
from ..config import SNSPD_5_config as cf

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-d',default="./plots",type=str,help='output directory')
parser.add_argument('--report','-r',default=1000,type=int,help='report every x events')
parser.add_argument('--checkSingleEvent','-c',default=-1,type=int,help='Check Single Event')
parser.add_argument('--debug_report','-b',action="store_true",help='report every x events')
parser.add_argument('--display_report','-p',action="store_true",help='report every x events')
parser.add_argument('--doAdvanced',action="store_true",help='do single pulse analysis')
parser.add_argument('--doAverage',action="store_true",help='do average pulse analysis')
parser.add_argument('--dryRun',action="store_true",help='dry run')
parser.add_argument('--subset','-s',default=-1,type=int,help='Process a subset of data. -1 = all')
args = parser.parse_args()


def debugPrint(string):
    if (cf.DEBUG): print(string)

def init_loop_variables():
    pre_std[0], pos_std[0], pre_mean[0], pos_mean[0], pre_range[0], pos_range[0], pre_max[0], pos_max[0] = -1,-1,-1,-1,-1,-1,-1,-1
    pulse_rise_range[0], pulse_fall_range[0], pulse_fall_range_ptp[0] = -1,-1,-1
    pulse_amplitude[0], pulse_arrivalT[0], pulse_riseT[0], pulse_pre_range[0] = -1,-1,-1,-1
    pulse_fall_tau[0], pulse_fall_A[0], pulse_fall_t0[0], pulse_fall_C[0], pulse_fall_status[0] = -1,-1,-1,-1,-1
    pulse_rise_tau[0] = -1
    pulse_max[0], pulse_min[0], pulse_max_T[0], pulse_min_T[0] = -1,-1,-1,-1

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

def Advanced_pulse_analysis(data, trigT, event):
    data_xIndex = np.arange(len(data))
    # Cubic Spline Fit
    data_spline = CubicSpline(data_xIndex, data)
    # Find turning point
    data_turning_pedestals, data_turning_peaks, data_turning_ranges = Get_turning_times(data_spline, cf.threshold, 0, len(data), 'Rise', cf.DEBUG)
    if (len(data_turning_peaks)<1):
        # print(f'Abnormal Event{event}. Pass Event Selection, but can\'t find turning points')
        return
    # Get pulse amplitude --> Defined as range between pulse rising turning points
    pulse_amplitude[0] = max(data_turning_ranges)
    imax = data_turning_ranges.index(pulse_amplitude)

    # Get 50% pulse amplitude level
    data_10 = data_turning_peaks[imax].y*0.1 + data_turning_pedestals[imax].y*0.9
    data_50 = data_turning_peaks[imax].y*0.5 + data_turning_pedestals[imax].y*0.5
    data_90 = data_turning_peaks[imax].y*0.9 + data_turning_pedestals[imax].y*0.1
    # Get Arrival time
    pulse_arrivalT[0] = Get_Function_Arrival(data_spline, data_50, data_turning_pedestals[imax].x, data_turning_peaks[imax].x) + cf.Pulse_startT + trigT  #int(chTrig_arrivalT) + 205
    # Get Rise time
    pulse_riseT[0] = Get_Function_RiseFall_Range(data_spline, data_10, data_90, data_turning_pedestals[imax].x, data_turning_peaks[imax].x)
    # display_spline_fit(data_spline, data_xIndex)
    debugPrint(f'Pulse amplitude = {pulse_amplitude[0]:.4f}, arrival Time = {pulse_arrivalT[0]:.4f}, rise Time = {pulse_riseT[0]:.4f}')

def Simple_pulse_analysis(data, event, ipulse):
    # event_display(data, f'Waveform#{event}_{ipulse}')
    pre_std[0] = (np.std(data[cf.prePulse_startT:cf.prePulse_endT]))
    pos_mean[0] = (np.mean(data[cf.prePulse_startT:cf.prePulse_endT]))
    pre_range[0] = (np.ptp(data[cf.prePulse_startT:cf.prePulse_endT]))
    pos_std[0] = (np.std(data[cf.postPulse_startT:cf.postPulse_endT]))
    pre_mean[0] = (np.mean(data[cf.postPulse_startT:cf.postPulse_endT]))
    pos_range[0] = (np.ptp(data[cf.postPulse_startT:cf.postPulse_endT]))
    pre_max[0] = np.max(data[cf.prePulse_startT:cf.prePulse_endT])
    pos_max[0] = np.max(data[cf.prePulse_startT:cf.prePulse_endT])
    # Pulse region
    pulse_max[0] = np.max(data[cf.Pulse_startT:cf.Pulse_endT])
    pulse_min[0] = np.min(data[cf.Pulse_startT:cf.Pulse_endT])
    pulse_max_T[0] = cf.Pulse_startT + np.argmax(data[cf.Pulse_startT:cf.Pulse_endT])
    pulse_min_T[0] = cf.Pulse_rise_endT + np.argmin(data[cf.Pulse_rise_endT:cf.Pulse_endT])
    pulse_rise_range[0] = data[cf.Pulse_rise_endT] - data[cf.Pulse_startT]
    pulse_fall_range[0] = data[cf.Pulse_rise_endT] - data[cf.Pulse_fall_endT]
    pulse_rise_range_ptb[0] = pulse_max[0] - pre_mean[0]
    pulse_fall_range_ptp[0] = np.ptp(data[cf.Pulse_startT:cf.Pulse_endT])
    pulse_pre_range = (pulse_fall_range[0]-pre_range[0])/pre_range[0]
    debugPrint(f'Rise Range = {pulse_rise_range[0]:.5f}, Fall Range = {pulse_fall_range[0]:.5f}, Pre-range = {pre_range[0]:.5f}, (pulse-pre)/pre = {pulse_pre_range:.5f}')
    debugPrint(f'maxT = {pulse_max_T[0]:.0f}, max = {pulse_max[0]:.2f}')

def Common_mode_analysis(chSig_average, data, event):
    chSig_average = np.add(chSig_average,data)
    if (cf.DISPLAY): event_display(chSig_average,f'Waveform{event}')
    return chSig_average

def Find_Trigger_time_predefined():
    # chTrig_arrivalTs = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    # chTrig_arrivalTs = [98]
    chTrig_arrivalTs = [58]
    return chTrig_arrivalTs

def Find_Trigger_time_splineFit(chTrig):
    chTrig_arrivalTs = []
    # Get chTrig (trigger) arrival times
    x_index = np.arange(len(chTrig))
    chTrig_spline = CubicSpline(x_index, chTrig)
    chTrig_turning_pedestals, chTrig_turning_peaks, chTrig_turning_ranges = Get_turning_times(chTrig_spline, 0.1, 0, len(chTrig), 'Fall', cf.DEBUG)
    # Loop over laser pulse
    for ipulse, (chTrig_turning_pedestal, chTrig_turning_peak) in enumerate(zip(chTrig_turning_pedestals, chTrig_turning_peaks)):
        # Skip unreasonable turning points
        if ( chTrig_turning_peak.x < 0 or chTrig_turning_pedestal.x < 0 ): continue
        # Define time of arrival at the 50% level of the falling slope
        chTrig_arrivalT = Get_Function_Arrival(chTrig_spline, chTrig_turning_pedestal.y-0.1, chTrig_turning_pedestal.x, chTrig_turning_peak.x)
        if (chTrig_arrivalT<0) : continue
        chTrig_arrivalTs.append(chTrig_arrivalT)
    return chTrig_arrivalTs

def single_pulse_spectrum(chSig,Pulse_spectrums,event):
    graph = ROOT.TGraph()
    graph.SetName(f"Pulse_spectrum_Evt{event}")
    graph.SetTitle(f"Pulse_spectrum_Evt{event}")
    graph.GetXaxis().SetTitle(f"index(0.4ns)")
    graph.GetYaxis().SetTitle(f"Voltage(V)")
    graph.SetMarkerStyle(4)
    graph.SetMarkerSize(0.5)
    for i in range(len(chSig)): graph.SetPoint(i,i,chSig[i])
    if(event<cf.avgMaxCount): Pulse_spectrums.append(graph)
    if(pulse_max[0] > pre_max[0]):
        graph_clone = graph.Clone()
        Fit_pulse_fall(graph_clone)

def FWHM(data,value):
    data_xIndex = np.arange(len(data))
    # Cubic Spline Fit
    data_spline=CubicSpline(data_xIndex,data)
    rise_50, fall_50 = Get_Function_FWHM(data_spline,value,cf.Pulse_startT,cf.Pulse_endT)
    pulse_FWHM[0] = fall_50 - rise_50
    debugPrint(f"{value},{rise_50}, {fall_50}, {fall_50-rise_50}")
    if (cf.DISPLAY): display_spline_fit(data_spline,data_xIndex)

def Fit_pulse_fall(graph):
    fitFunc_fall, fitResult_fall = Fit_time_constant_fall(graph,pulse_max_T[0],pulse_max_T[0]+30,"SQR","sames",ROOT.kRed)
    pulse_fall_tau[0] = fitFunc_fall.GetParameter(1) if fitResult_fall.Status()<2 else -1
    pulse_fall_A[0] = fitFunc_fall.GetParameter(0) if fitResult_fall.Status()<2 else -1
    pulse_fall_t0[0] = fitFunc_fall.GetParameter(2) if fitResult_fall.Status()<2 else -1
    pulse_fall_C[0] = fitFunc_fall.GetParameter(3) if fitResult_fall.Status()<2 else -1
    pulse_fall_status[0] = fitResult_fall.Status()
    debugPrint(f"fall:{pulse_fall_tau[0]:.2f}, fall_fit_status: {fitResult_fall.Status()}")
    if (cf.DISPLAY==True):
        c_pulse = ROOT.TCanvas()
        graph.Draw()
        graph.GetXaxis().SetRangeUser(280,400)
        c_pulse.Update()
        stat = graph.FindObject("stats")
        stat.SetTextColor(ROOT.kRed);
        statText=ROOT.TLatex();
        statText.SetTextColor(ROOT.kRed)
        statText.SetTextSize(0.04)
        statText.DrawLatexNDC(stat.GetX1NDC(),stat.GetY1NDC()-0.05,"Fall: Ae^{-(t-t0)/#tau}+C")
        c_pulse.Update()
        ROOT.gPad.WaitPrimitive()

def Fit_pulse_rise_fall(graph,event):
    fitFunc_rise, fitResult_rise = Fit_time_constant_rise(graph,pulse_max_T[0]-100,pulse_max_T[0]+1,"SQR")
    graph1 = graph.Clone()
    fitFunc_fall, fitResult_fall = Fit_time_constant_fall(graph1,pulse_max_T[0],pulse_max_T[0]+30,"SQR","sames",ROOT.kAzure)
    pulse_rise_tau[0] = fitFunc_rise.GetParameter(1) if fitFunc_rise.GetParameter(1)>0 else -1
    pulse_fall_tau[0] = fitFunc_fall.GetParameter(1) if fitFunc_fall.GetParameter(1)>0 else -1
    debugPrint(f"fall:{pulse_fall_tau[0]:.2f}, fall_fit_status: {fitResult_fall.Status()}, rise:{pulse_rise_tau[0]:.2f}, rise_fit_status: {fitResult_rise.Status()}")
    if (cf.DISPLAY==True):
        c_pulse = ROOT.TCanvas()
        graph.Draw()
        graph1.Draw("sames")
        graph.GetXaxis().SetRangeUser(280,400)
        c_pulse.Update()
        stat = graph.FindObject("stats")
        stat1 = graph1.FindObject("stats")
        stat.SetTextColor(ROOT.kRed);
        stat1.SetTextColor(ROOT.kAzure);
        height = stat1.GetY2NDC() - stat1.GetY1NDC();
        stat1.SetY1NDC(stat.GetY1NDC() - height - 0.1);
        stat1.SetY2NDC(stat.GetY1NDC() - 0.1);
        stat1.Draw("sames");
        statText=ROOT.TLatex();
        statText.SetTextColor(ROOT.kRed)
        statText.SetTextSize(0.04)
        statText.DrawLatexNDC(stat.GetX1NDC(),stat.GetY1NDC()-0.05,"Rise: Ae^{(t-t0)/#tau}+C")
        stat1Text=ROOT.TLatex();
        stat1Text.SetTextColor(ROOT.kAzure)
        stat1Text.SetTextSize(0.04)
        stat1Text.DrawLatexNDC(stat1.GetX1NDC(),stat1.GetY1NDC()-0.05,"Fall: Ae^{-(t-t0)/#tau}+C")
        c_pulse.Update()
        ROOT.gPad.WaitPrimitive()

def FFT(data,dt):
    # Perform FFT
    mag = np.fft.fft(data, cf.freq_steps)
    freqs = np.fft.fftfreq(cf.freq_steps, dt)
    positive_indices = np.where(freqs > 0)
    positive_freqs = freqs[positive_indices]
    positive_mags = mag[positive_indices]
    # positive_mags_abs =  np.abs(positive_mags)
    # positive_mags_abs /= np.sum(positive_mags_abs)
    if (cf.DISPLAY): event_display_fft(data,positive_freqs,positive_mags,0,500e6)
    return positive_freqs, positive_mags

def Stack_spectrums(Pulse_spectrums):
    c_stack_spectrum = ROOT.TCanvas("c_stack_spectrum","Stack_spectrums",1500,900)
    for i, graph in enumerate(Pulse_spectrums):
        if (i==0):
            graph.GetYaxis().SetRangeUser(-0.5,1.5);
            graph.GetXaxis().SetRangeUser(300,400);
            graph.Draw("ALP PLC PMC")
        else:
            graph.Draw("LPSame PLC PMC")
    c_stack_spectrum.Write()

def average_plots(chSig_average,title):
    c1 = ROOT.TCanvas()
    Pulse_avg_display = ROOT.TGraph()
    Pulse_avg_display.SetName(f"Pulse_avg_display")
    Pulse_avg_display.SetTitle(title)
    Pulse_avg_display.GetXaxis().SetTitle(f"index(0.4ns)")
    Pulse_avg_display.GetYaxis().SetTitle(f"Voltage(V)")
    for i, sig in enumerate(chSig_average): Pulse_avg_display.SetPoint(i,i,sig)
    Pulse_avg_display.SetMarkerStyle(4)
    Pulse_avg_display.SetMarkerSize(0.5)
    Pulse_avg_display.Draw("ALP")
    leg = ROOT.TLegend(0.6,0.6,0.8,0.8)
    leg.AddEntry(Pulse_avg_display,f"{cf.avgMaxCount}-event-average signal spectrum","lp")
    # c1.SaveAs(f"{outDir}/{Pulse_avg_display.GetName()}.png")
    Pulse_avg_display.Write()

def FFT_plot(freqs, mags, title=""):
    c_fft = ROOT.TCanvas()
    fft_display = ROOT.TGraph()
    fft_display.SetName(title)
    fft_display.SetTitle(title)
    fft_display.GetXaxis().SetTitle(f"frequency(Hz)")
    fft_display.GetYaxis().SetTitle(f"Normalized Magnitude")
    for i, (freq,mag) in enumerate(zip(freqs,np.abs(mags))): fft_display.SetPoint(i,freq,mag)
    fft_display.SetMarkerStyle(4)
    fft_display.SetMarkerSize(0.5)
    fft_display.Draw("ALP")
    fft_display.Write()

###############################################
#################### Main  ####################
###############################################
def SingleTDMS_analysis():
    with TdmsFile.open(in_filename) as tdms_file:
        # Read Meta Data (Basic information)
        metadata = tdms_file.properties
        metadata_df = pd.DataFrame(metadata.items(), columns=['metaKey', 'metaValue'])
        print(metadata_df)
        totalEvents = int(metadata_df.loc[metadata_df['metaKey'] == 'Total Events', 'metaValue'].iloc[0])
        recordlength = int(metadata_df.loc[metadata_df['metaKey'] == 'record length', 'metaValue'].iloc[0])
        vertical_range = float(metadata_df.loc[metadata_df['metaKey'] == 'vertical range Sig', 'metaValue'].iloc[0])
        SampleRate = float(metadata_df.loc[metadata_df['metaKey'] == 'actual sample rate', 'metaValue'].iloc[0])
        dt = 1/SampleRate
        metadata_df.to_json(metaFileName,orient="records",lines=True) # Write metadata to json file
        # Read Groups and Channels
        Read_Groups_and_Channels(tdms_file)
        chSig_total = tdms_file['ADC Readout Channels']['chSig']
        # chTrig_total = tdms_file['ADC Readout Channels']['chTrig']
        # Initialize histos
        h_fft_2d = ROOT.TH2D("h_fft_2d","h_fft_2d",int(cf.freq_steps/2),0,SampleRate/2,1000,0,0.01)

        # Initialize variables
        chSig_average = np.zeros(recordlength)
        pulseCount, avgCount = 0,0
        Pulse_spectrums = []
        # Start Loop
        print (f"==========Start Looping at {datetime.datetime.now()}==========")
        for event in range(totalEvents):
            # if (args.dryRun): break
            if (event == args.subset): break # Choose a subset of the whole data to do the analysis. -1 = run All
            if (outtree.GetEntries() == cf.totalTreeEvents): break
            if (args.checkSingleEvent!=-1 and event!=args.checkSingleEvent): continue
            if ((event)%args.report==0): print (f"==========Processing {event}/{totalEvents} event==========") # Loop progress
            chSig = chSig_total[event * recordlength:(event+1) * recordlength]  # Read chSig into np array
            # chTrig = chTrig_total[event * recordlength:(event+1) * recordlength] # Read chTrig into np array
            # event_display_2ch(chSig,chTrig,f'Waveform', 0.02)
            # chSig_diff = np.diff(chSig)
            # event_display_2ch(chSig,chSig_diff,f'Waveform', 0.04)
            chTrig_arrivalTs = Find_Trigger_time_splineFit(chTrig) if args.doAdvanced else Find_Trigger_time_predefined() # Find Trigger timeing
            for ipulse, chTrig_arrivalT in enumerate(chTrig_arrivalTs):
                debugPrint(f'==========Event{event}_Pulse{ipulse}==========')
                init_loop_variables()
                pulseCount = pulseCount + 1
                Simple_pulse_analysis(chSig, event, ipulse) # Record simple signal and sideband ranges into histogram
                if Sideband_selection(): # Record extra information if Sideband selection passed
                    debugPrint("pass sideband selection")
                    single_pulse_spectrum(chSig, Pulse_spectrums, event) # Draw pulse spectrum to a graph and Fit falling time constant
                    FWHM(chSig,pulse_rise_range[0]/2)
                    if (args.doAdvanced): Advanced_pulse_analysis(chSig, chTrig_arrivalT, event) # Do advanced analysis (Rising time, timing jitter, sophisticated amplitude)
                    # if (cf.DISPLAY): event_display_2ch(chSig,chTrig,f'Waveform{event}', 0.02)
                    if (event<cf.avgMaxCount):
                        avgCount+=1
                        chSig_average = Common_mode_analysis(chSig_average, chSig, event) # Create average signal spectrum
                        freqs, mags = FFT(chSig,dt)
                        for freq, mag in zip(freqs,np.abs(mags)): h_fft_2d.Fill(freq,mag)
                    outtree.Fill() # Fill tree
                else:
                    debugPrint(f"fail sideband selection: {pre_std[0]}, {pos_std[0]}, {pre_range[0]}, {pos_range[0]}")
        print (f"==========End Looping at {datetime.datetime.now()}==========")
    # Output some numbers
    print(f"TotalEvents:{totalEvents}, TriggerPulse_Count:{pulseCount}, PassSideband_Count: {outtree.GetEntries()}")
    # Plots
    chSig_average = chSig_average/avgCount
    freqs, mags = FFT(chSig_average,dt)
    FFT_plot(freqs,mags,"fft_average")
    freqs, mags = FFT(chSig_average[cf.Pulse_startT:cf.Pulse_endT],dt)
    FFT_plot(freqs,mags,"fft_average_pulse_peak")
    freqs, mags = FFT(chSig_average[cf.Pulse_startT:500],dt)
    FFT_plot(freqs,mags,"fft_average_pulse_full")
    freqs, mags = FFT(chSig_average[0:250],dt)
    FFT_plot(freqs,mags,"fft_average_prepulse")
    average_plots(chSig_average,basename)
    Stack_spectrums(Pulse_spectrums)
    h_fft_2d.Write()

if __name__ == "__main__":

    if (args.debug_report==True): cf.DEBUG = True
    if (args.display_report==True): cf.DISPLAY = True
    ROOT.gStyle.SetPalette(ROOT.kBird)

    ########## Init ##########
    for in_filename in args.in_filenames:
        print("\n##############################")
        print(f"input file: {in_filename}")
        basename = in_filename.rsplit('/',1)[1].split('.tdms')[0]
        baseDir = in_filename.split('SNSPD_rawdata/')[1].rsplit('/',1)[0]
        outDir = args.outputDir + '/' + baseDir + '/' + basename
        metaFileName = outDir + '/' + in_filename.rsplit('/',1)[1].split('.tdms')[0] + ".json"
        createDir(outDir)
        # Create root filen
        outfile = ROOT.TFile(f'{outDir}/{basename}.root', 'RECREATE', f'analysis histograms of {basename} measurements' )
        outtree = ROOT.TTree("Result_tree","Pulse laser analysis results")
        out_inputname = ROOT.TNamed("inputDataName",in_filename)
        out_metaname = ROOT.TNamed("metaDataName",metaFileName)
        # Define variables for branch
        pre_std, pos_std, pre_mean, pos_mean, pre_range, pos_range, pre_max, pos_max = array('f',[0]),array('f',[0]),array('f',[0]),array('f',[0]),array('f',[0]),array('f',[0]),array('f',[0]),array('f',[0])
        pulse_rise_range, pulse_fall_range, pulse_rise_range_ptb, pulse_fall_range_ptp = array('f',[0]),array('f',[0]),array('f',[0]),array('f',[0])
        pulse_amplitude, pulse_arrivalT, pulse_riseT, pulse_pre_range = array('f',[0]),array('f',[0]),array('f',[0]),array('f',[0])
        pulse_FWHM = array('f',[0])
        pulse_fall_tau, pulse_fall_A, pulse_fall_t0, pulse_fall_C, pulse_fall_status = array('f',[0]),array('f',[0]),array('f',[0]),array('f',[0]),array('f',[0])
        pulse_rise_tau = array('f',[0])
        pulse_max, pulse_min, pulse_max_T, pulse_min_T = array('f',[0]),array('f',[0]),array('f',[0]),array('f',[0])
        # init branches
        outtree.Branch('pre_std', pre_std, 'pre_std/F')
        outtree.Branch('pre_mean', pre_mean, 'pre_mean/F')
        outtree.Branch('pre_range', pre_range, 'pre_range/F')
        outtree.Branch('pre_max', pre_max, 'pre_max/F')
        outtree.Branch('pos_std', pos_std, 'pos_std/F')
        outtree.Branch('pos_mean', pos_mean, 'pos_mean/F')
        outtree.Branch('pos_range', pos_range, 'pos_range/F')
        outtree.Branch('pos_max', pos_max, 'pos_max/F')
        outtree.Branch('pulse_rise_range', pulse_rise_range, 'pulse_rise_range/F')
        outtree.Branch('pulse_rise_range_ptb', pulse_rise_range_ptb, 'pulse_rise_range_ptb/F')
        outtree.Branch('pulse_fall_range', pulse_fall_range, 'pulse_fall_range/F')
        outtree.Branch('pulse_fall_range_ptp', pulse_fall_range_ptp, 'pulse_fall_range_ptp/F')
        outtree.Branch('pulse_max', pulse_max, 'pulse_max/F')
        outtree.Branch('pulse_min', pulse_min, 'pulse_min/F')
        outtree.Branch('pulse_max_T', pulse_max_T, 'pulse_max_T/F')
        outtree.Branch('pulse_min_T', pulse_min_T, 'pulse_min_T/F')
        outtree.Branch('pulse_rise_tau', pulse_rise_tau, 'pulse_rise_tau/F')
        outtree.Branch('pulse_fall_tau', pulse_fall_tau, 'pulse_fall_tau/F')
        outtree.Branch('pulse_fall_A', pulse_fall_A, 'pulse_fall_A/F')
        outtree.Branch('pulse_fall_t0', pulse_fall_t0, 'pulse_fall_t0/F')
        outtree.Branch('pulse_fall_C', pulse_fall_C, 'pulse_fall_C/F')
        outtree.Branch('pulse_fall_status', pulse_fall_status, 'pulse_fall_status/F')
        outtree.Branch('pulse_FWHM', pulse_FWHM, 'pulse_FWHM/F')
        # outtree.Branch('pulse_pre_range', pulse_pre_range, 'pulse_pre_range/F')
        if (args.doAdvanced):
            outtree.Branch('pulse_amplitude', pulse_amplitude, 'pulse_amplitude/F')
            outtree.Branch('pulse_arrivalT', pulse_arrivalT, 'pulse_arrivalT/F')
            outtree.Branch('pulse_riseT', pulse_riseT, 'pulse_riseT/F')
        ########## End Init ##########

        # Start Analysis
        SingleTDMS_analysis()
        # plots()
        # End Analysis
        print (f'Output: {outDir}/{basename}.root')
        out_inputname.Write()
        out_metaname.Write()
        outtree.Write()
        outfile.Close()
