#!/usr/bin/env python3

from nptdms import TdmsFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Timing_Analyzer import *

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-d',default="./",type=str,help='output directory')
parser.add_argument('--report','-r',default=100,type=int,help='report every x events')
parser.add_argument('--debug_report','-b',default=-1,type=int,help='report every x events')
parser.add_argument('--display_report','-p',default=-1,type=int,help='report every x events')
args = parser.parse_args()

DEBUG = False

def debugPrint(string):
    if (DEBUG): print(string)

def Read_Groups_and_Channels(tdms_file):
    # Loop through each group and print the channel names
    for group in tdms_file.groups():
        print(f"Group '{group.name}':")
        for channel in group.channels():
            print(f"- Channel '{channel.name}':")

def event_display(np,title='Waveform'):
    # Create a line plot of the data
    plt.plot(range(len(np)), np)
    # Add labels to the plot
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')
    # Display the plot
    plt.show()

def event_display_2ch(np1, np2, title):
    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the three arrays
    ax.plot(range(len(np2)), np2-0.15, label='ch2', marker='o',fillstyle='none')
    ax.plot(range(len(np1)), np1, label='ch1', marker='o',fillstyle='none')
    # Add labels to the plot
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    # ax.set_ylim(-0.25,0.15)
    # ax.set_xlim(760,780)
    plt.grid()
    # Display the plot
    plt.show()

def display_spline_fit(spline_func, x_index):
    x_spline_range = np.linspace(x_index.min(), x_index.max(), num=10000)
    y_spline = spline_func(x_spline_range)
    plt.plot(x_spline_range, y_spline, '-', label='Spline Fit')
    plt.show()


def plot_histo(np1, nbin, rangemin, rangemax, xTitle, title, saveTitle):
    fig, ax = plt.subplots()
    ax.hist(np1, bins=nbin, range=(rangemin, rangemax), alpha=0.5, color='blue', edgecolor='black')
    mean, std = np.mean(np1), np.std(np1)
    textstr = f'$\mu={mean:.4f}$\n$\sigma={std:.4f}$'
    ax.text(0.73, 0.93, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top')
    ax.set_xlabel(xTitle)
    ax.set_ylabel('Events')
    ax.set_title(title)
    plt.savefig(saveTitle)
    plt.show()

def plot_2histo(np1, np2, nbin, rangemin, rangemax, label1, label2, title, saveTitle='test.png'):
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(8,5))
    n1, bins1, patches1 = ax1.hist(np1, bins=nbin, range=(rangemin, rangemax), alpha=0.6, color='red', label=label1)
    n2, bins2, patches2 = ax2.hist(np2, bins=nbin, range=(rangemin, rangemax), alpha=0.6, color='blue', label=label2)
    ax1.legend()
    ax2.legend()
    # Add mean and standard deviation to the plot
    mean1, std1 = np.mean(np1), np.std(np1)
    mean2, std2 = np.mean(np2), np.std(np2)
    textstr1 = f'$\mu_1={mean1:.4f}$\n$\sigma_1={std1:.4f}$'
    textstr2 = f'$\mu_2={mean2:.4f}$\n$\sigma_2={std2:.4f}$'
    ax1.text(0.8, 0.75, textstr1, transform=ax1.transAxes, fontsize=14, verticalalignment='top')
    ax2.text(0.8, 0.75, textstr2, transform=ax2.transAxes, fontsize=14, verticalalignment='top')
    fig.suptitle(title)
    plt.savefig(saveTitle)
    plt.show()


def event_selection(*arrays):
    threshold = 0.03
    diff_threshold = -0.1
    # if (len(chdata)>0 and (chdata > threshold).any()):
    if (len(arrays[0])>0 and (arrays[0]>threshold).any() and (arrays[1]>diff_threshold).any()):
        return True
    else:
        return False

# def find_trigger_sample(np):
#     np

if __name__ == "__main__":

    NpulsePerTrigger=10

    for in_filename in args.in_filenames:
        with TdmsFile.open(in_filename) as tdms_file:
            # Read Meta Data (Basic information)
            metadata = tdms_file.properties
            metadata_df = pd.DataFrame(metadata.items(), columns=['metaKey', 'metaValue'])
            print(metadata_df)
            totalEvents = metadata_df.loc[metadata_df['metaKey'] == 'Total Events', 'metaValue'].iloc[0]
            # Read Groups and Channels
            Read_Groups_and_Channels(tdms_file)

            channel_sum = 0.0
            channel_length = 0
            nPass=0
            ranges=[]
            # Sideband region numpy
            prePulse_mean, postPulse_mean, prePulse_stdev, postPulse_stdev, prePulse_range, postPulse_range, prePulse_integral, postPulse_integral = [], [], [], [], [], [], [], []
            # Signal region numpy
            ch1_pulse_spline_ranges, ch1_pulse_diff_ranges, ch1_pulse_amplitudes, ch1_pulse_arrivalTs, ch1_pulse_riseTs, ch1_pulse_spline_integrals = [], [], [], [], [], []
            # Start Looping through events
            for event, chunk in enumerate(tdms_file.data_chunks()):
                # Loop progress
                if ((event+1)%args.report==0): print (f"==========Processing {event}/{totalEvents} event==========")
                DEBUG = True if (event+1)%args.debug_report==0 else False
                DISPLAY = True if (event+1)%args.display_report==0 else False
                # Skip chunk larger than totalEvents
                if (event > int(totalEvents)-1): continue
                # if (event > 100): continue
                # Read ch1 into np array
                ch1 = chunk['ADC Readout Channels']['ch1']._data()
                ch2= chunk['ADC Readout Channels']['ch2']._data()
                if (DISPLAY): event_display_2ch(ch1,ch2,f'Waveform')

                ch1_diff = np.diff(ch1)
                ch2_diff = np.diff(ch2)
                # Create a spline interpolation function for the data
                x_index = np.arange(len(ch1))
                ch1_spline = CubicSpline(x_index, ch1)
                ch2_spline = CubicSpline(x_index, ch2)
                # Get ch2 (trigger) arrival times
                ch2_turning_pedestals, ch2_turning_peaks = Get_turning_times(ch2_spline, 0.4, 0, len(ch2), 'Fall', DEBUG)
                for ipulse, (ch2_turning_pedestal, ch2_turning_peak) in enumerate(zip(ch2_turning_pedestals, ch2_turning_peaks)):
                    # Skip last pulse due to distortion of the oscilloscop at the boundary
                    if (ipulse >= NpulsePerTrigger-1): continue
                    # Skip unreasonable turning points
                    if ( ch2_turning_peak.x < 0 or ch2_turning_pedestal.x < 0 ): continue
                    # Define time of arrival at the 50% level of the falling slope
                    ch2_arrivalT = Get_Function_Arrival(ch2_spline, (ch2_turning_pedestal.y+ch2_turning_peak.y)/2, ch2_turning_pedestal.x, ch2_turning_peak.x)
                    # Define signal pulse region
                    Pulse_startT =  int(ch2_arrivalT) + 210
                    Pulse_endT =  int(ch2_arrivalT) + 250
                    # Define pre-pulse (sideband) region
                    prePulse_startT =  int(ch2_arrivalT) + 10
                    prePulse_endT =  int(ch2_arrivalT) + 180
                    # Define post-pulse (sideband) region
                    postPulse_startT =  int(ch2_arrivalT) + 300
                    postPulse_endT =  int(ch2_arrivalT) + 800
                    # event_display_2ch(ch1[prePulse_startT:postPulse_endT], ch1_diff[prePulse_startT:postPulse_endT], f'Waveform#{event}_pulse{ipulse}')
                    # Sideband characteristic
                    prePulse_mean.append(np.mean(ch1[prePulse_startT:prePulse_endT])) # mean
                    postPulse_mean.append(np.mean(ch1[postPulse_startT:postPulse_endT]))
                    prePulse_stdev.append(np.std(ch1[prePulse_startT:prePulse_endT])) # stdev
                    postPulse_stdev.append(np.std(ch1[postPulse_startT:postPulse_endT]))
                    prePulse_range.append(np.ptp(ch1[prePulse_startT:prePulse_endT])) # max - min
                    postPulse_range.append(np.ptp(ch1[postPulse_startT:postPulse_endT]))
                    prePulse_integral.append(np.sum(ch1[prePulse_startT:prePulse_endT])) # max - min
                    postPulse_integral.append(np.sum(ch1[postPulse_startT:postPulse_endT]))

                    # Pulse pre-selection using sideband region
                    if (prePulse_range[-1] < 0.057 or prePulse_stdev[-1] < 0.013 or postPulse_range[-1] < 0.075 or postPulse_stdev[-1] < 0.014):
                        debugPrint(f'Event{event}_Pulse{ipulse} pass preselection')
                        # Pulse region
                        ch1_pulse = ch1[Pulse_startT:Pulse_endT]
                        ch1_pulse_xIndex = np.arange(len(ch1_pulse))
                        # Cubic Spline Fit
                        ch1_pulse_spline = CubicSpline(ch1_pulse_xIndex, ch1_pulse)
                        # Pulse spline range
                        ch1_pulse_spline_range = Get_FunctionMax(ch1_pulse_spline, 7, 25).y - Get_FunctionMin(ch1_pulse_spline, 7, 25).y
                        ch1_pulse_spline_ranges.append(ch1_pulse_spline_range)
                        # Pulse spline integral
                        ch1_pulse_spline_integral, error = Get_function_integral(ch1_pulse_spline, 7, 25)
                        ch1_pulse_spline_integrals.append(ch1_pulse_spline_integral)
                        # Derivative of pulse region
                        ch1_pulse_diff = ch1_diff[Pulse_startT:Pulse_endT]
                        ch1_pulse_diff_xIndex = np.arange(len(ch1_pulse_diff))
                        ch1_pulse_diff_spline = ch1_pulse_spline.derivative()
                        # Diff spline range
                        ch1_pulse_diff_range = Get_FunctionMax(ch1_pulse_diff_spline, 8, 19).y - Get_FunctionMin(ch1_pulse_diff_spline, 8, 19).y
                        ch1_pulse_diff_ranges.append(ch1_pulse_diff_range)

                        debugPrint(f'Pulse range = {ch1_pulse_spline_range}, Diff range = {ch1_pulse_diff_range}')
                        # Event Selection
                        if (ch1_pulse_spline_range > 0.025 and ch1_pulse_diff_range > 0.01):
                            debugPrint('Pass event selection')
                            # Find turning point
                            ch1_pulse_turning_pedestals, ch1_pulse_turning_peaks = Get_turning_times(ch1_pulse_spline, 0.04, 6, 25, 'Rise', DEBUG)
                            if (len(ch1_pulse_turning_peaks)>0):
                                # Get pulse amplitude --> Defined as range between pulse rising turning points
                                ch1_pulse_amplitude = ch1_pulse_turning_peaks[0].y - ch1_pulse_turning_pedestals[0].y
                                ch1_pulse_amplitudes.append(ch1_pulse_amplitude)
                                # Get 50% pulse amplitude level
                                ch1_pulse_10 = ch1_pulse_turning_peaks[0].y*0.1 + ch1_pulse_turning_pedestals[0].y*0.9
                                ch1_pulse_50 = ch1_pulse_turning_peaks[0].y*0.5 + ch1_pulse_turning_pedestals[0].y*0.5
                                ch1_pulse_90 = ch1_pulse_turning_peaks[0].y*0.9 + ch1_pulse_turning_pedestals[0].y*0.1
                                # Get Arrival time
                                ch1_pulse_arrivalT = Get_Function_Arrival(ch1_pulse_spline, ch1_pulse_50, ch1_pulse_turning_pedestals[0].x, ch1_pulse_turning_peaks[0].x) + Pulse_startT - ch2_arrivalT
                                ch1_pulse_arrivalTs.append(ch1_pulse_arrivalT)
                                # Get Rise time
                                ch1_pulse_riseT = Get_Function_RiseFall_Range(ch1_pulse_spline, ch1_pulse_10, ch1_pulse_90, ch1_pulse_turning_pedestals[0].x, ch1_pulse_turning_peaks[0].x)
                                ch1_pulse_riseTs.append(ch1_pulse_riseT)

                                debugPrint(f'Pulse amplitude = {ch1_pulse_amplitude}, arrival Time = {ch1_pulse_arrivalT}, rise Time = {ch1_pulse_riseT}')
                            else:
                                print(f'Abnormal Event{event}_Pulse{ipulse}. Pass Event Selection, but can\'t find turning points')
                                # event_display_2ch(ch1_pulse_diff, ch1_pulse, f'Wavform#{event}_pulse{ipulse}')
                            # Create a check point for amplitude
                            if (ch1_pulse_amplitude < 0):
                                print('Abnormal Event{event}_Pulse{ipulse}. Pulse amplitude is negative')
                                exit()
                        else:
                            debugPrint('Fail event selection')
                            # event_display_2ch(ch1_pulse_diff, ch1_pulse, f'Wavform#{event}_pulse{ipulse}')

                        # ch1_pulse_diff_turning_pedestals, ch1_pulse_diff_turning_peaks = Get_turning_times(ch1_pulse_diff_spline, 0.02, 0, 'Rise', DEBUG)
                        # display_spline_fit(ch1_pulse_spline, ch1_pulse_xIndex)
                        if (DISPLAY): event_display_2ch(ch1_pulse_diff, ch1_pulse, f'Wavform#{event}_pulse{ipulse}')
                    else:
                        debugPrint (f'Event{event}_Pulse{ipulse} fail preselection ')

                    #     # event_display(event, ch1_pulse)


                # Selection
                # Get_Xs(spline_func)
                # if (event_selection(ch1[4200:4240], diffs[4200:4240])):
                    # Count events passing the selection
                    # nPass+=1
            #         # Count total samples
            #         channel_length += len(ch1)
            #         # Sum over all samples, useful to calculate pedestal average
            #         channel_sum += ch1[:].sum()
            #         # Calculate pulse amplitude(range) --> Maximum - pedestal
            #         pedestal_average = np.mean(ch1[0:25])
            #         maximum = np.max(ch1)
            #         range = maximum - pedestal_average
            #         ranges.append(range)

            #         indices = np.where(ch1 > 0.3)[0]
            #         # if (len(indices) > 0): print(indices[0])
                # event_display_2ch(event, ch1,ch2)

            # channel_mean = channel_sum / channel_length
            # print(channel_length, channel_mean)

            # Sideband Region
            # Plot two histograms side-by-side
            plot_2histo(prePulse_mean, postPulse_mean, 50, -0.01, 0.01, 'prePulse', 'postPulse', 'Sideband mean', f'{args.outputDir}/sideband_mean.png')
            plot_2histo(prePulse_stdev, postPulse_stdev, 50, 0, 0.05, 'prePulse', 'postPulse', 'Sideband stdev', f'{args.outputDir}/sideband_stdev.png')
            plot_2histo(prePulse_range, postPulse_range, 50, 0, 0.2, 'prePulse', 'postPulse', 'Sideband range', f'{args.outputDir}/sideband_range.png')
            plot_2histo(prePulse_integral, postPulse_integral, 50, -1, 1, 'prePulse', 'postPulse', 'Sideband integral', f'{args.outputDir}/sideband_integral.png')
            # Signal Region
            # plot_2histo(ch1_pulse_amplitudes, ch1_pulse_ranges, 50, 0, 0.2, 'Spline amplitude', 'Range', 'Pulse amplitude')
            plot_histo(ch1_pulse_amplitudes, 256, 0, 0.5, 'Voltage [V]', 'Pulse amplitude',f'{args.outputDir}/signal_amplitude.png')
            plot_histo(ch1_pulse_diff_ranges, 50, 0, 0.1, 'Voltage [V]', 'Signal Region differentiate range',f'{args.outputDir}/signal_diff.png')
            plot_histo(ch1_pulse_arrivalTs, 50, 220, 230, 'Time [index]', 'Pulse arrival time',f'{args.outputDir}/signal_arrivalT.png')
            plot_histo(ch1_pulse_riseTs, 50, 0, 7, 'Time [index]', 'Pulse rise time',f'{args.outputDir}/signal_riseT.png')
            plot_histo(ch1_pulse_spline_integrals, 50, -1, 1, 'Voltage [V]', 'Signal Region Integral',f'{args.outputDir}/signal_integral.png')