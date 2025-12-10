#!/usr/bin/env python3

import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

def colors(i):
    # Colors for different data sets using tab10 palette
    colors = plt.cm.tab10.colors[:10]
    return colors[i]

def markers(i):
    markers = ['o', 's', 'x', '^', 'v', '+', '.', 'd', 'p', 'h',
               '1', '2', '3', '4', '8', '|', '_', ',', '*', 'H']
    return markers[int(i%20)]

def lines(i):
    lines = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, ()), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1))]
    return lines[i]

def event_display(np,title='Waveform'):
    # Create a line plot of the data
    plt.plot(range(len(np)), np)
    # Add labels to the plot
    plt.title(title,fontsize=15)
    plt.xlabel('Index [0.4ns]',fontsize=15)
    plt.ylabel('ADC [Volts]',fontsize=15)
    plt.tight_layout()
    # Display the plot
    plt.show()

def event_display_spline(array, spline_func, title='Waveform'):
    # Create a line plot of the data
    plt.plot(range(len(array)), array, marker='o', linestyle='none', fillstyle='none', label='Signal data')
    x_spline_range = np.linspace(0, len(array), num=100000)
    y_spline = spline_func(x_spline_range)
    plt.plot(x_spline_range, y_spline, '-', label='Spline Fit')
    # Add labels to the plot
    # plt.title(title,fontsize=15)
    plt.xlabel('Index [0.4ns]',fontsize=15)
    plt.ylabel('ADC [Volts]',fontsize=15)
    plt.legend(fontsize='large')
    plt.tight_layout()
    # Display the plot
    # if (config.DISPLAY): plt.show()
    plt.close('all')

def event_display_2ch(np1, np2, title='Waveform', offset=0.15):
    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the three arrays
    ax.plot(range(len(np2)), np2-offset, label='trigger', marker='o',fillstyle='none')
    ax.plot(range(len(np1)), np1, label='signal', marker='o',fillstyle='none')
    # Add labels to the plot
    ax.legend(fontsize='large')
    ax.set_title(title, fontsize=15)
    ax.set_xlabel('Time sample (0.4 ns per sample)',fontsize=15)
    ax.set_ylabel('ADC (Volts)',fontsize=15)
    plt.tight_layout()
    # ax.set_ylim(-0.25,0.15)
    # ax.set_xlim(760,780)
    plt.grid()
    # Display the plot
    plt.show()
    plt.close('all')

def event_display_fft(np1, freqs, mags, xmin, xmax):
    # Create a new figure
    plt.figure(figsize=(10, 6))
    # Plot the time domain signal
    plt.subplot(2, 1, 1)
    plt.plot(range(len(np1)), np1, label='signal',marker='o',fillstyle='none')
    plt.title('Time Domain Signal')
    plt.xlabel('Index (0.4ns)',fontsize=15)
    plt.ylabel('ADC (Volts)',fontsize=15)
    # Plot the frequency domain signal
    plt.subplot(2, 1, 2)
    plt.plot(freqs, np.abs(mags))
    plt.title('Frequency Domain Signal')
    plt.xlabel('Frequency (Hz)',fontsize=15)
    plt.ylabel('Magnitude',fontsize=15)
    plt.xlim(xmin, xmax)  # Limit x-axis to frequencies below 500 MHz
    plt.tight_layout()
    plt.show()
    plt.close('all')

# def event_display_TGraph2matpoltlib(graphs,):
#     graphs
#     plt.plot(range(len(np)), np)
#     # Add labels to the plot
#     plt.title(title,fontsize=15)
#     plt.xlabel('Index [0.4ns]',fontsize=15)
#     plt.ylabel('ADC [Volts]',fontsize=15)
#     plt.tight_layout()
#     # Display the plot
#     plt.show()

def display_spline_fit(spline_func, x_index):
    x_spline_range = np.linspace(x_index.min(), x_index.max(), num=10000)
    y_spline = spline_func(x_spline_range)
    plt.plot(x_spline_range, y_spline, '-', label='Spline Fit')
    plt.show()
    plt.close('all')

def plot_histo(np1, nbin, rangemin, rangemax, xTitle, title, saveTitle):
    fig, ax = plt.subplots()
    ax.hist(np1, bins=nbin, range=(rangemin, rangemax), alpha=0.6, color='blue')
    Events, mean, std = len(np1), np.mean(np1), np.std(np1)
    textstr = f'Events={Events}\nmean={mean:.4f}\nsigma={std:.4f}'
    ax.text(0.73, 0.93, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top')
    ax.set_xlabel(xTitle)
    ax.set_ylabel('Events')
    ax.set_title(title)
    plt.savefig(saveTitle)
    # if (config.DISPLAY): plt.show()
    plt.close('all')
    return mean, std

def plot_2histo(np1, np2, nbin, rangemin, rangemax, label1, label2, title, saveTitle='test.png'):
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(8,5))
    n1, bins1, patches1 = ax1.hist(np1, bins=nbin, range=(rangemin, rangemax), alpha=0.6, color='red', label=label1)
    n2, bins2, patches2 = ax2.hist(np2, bins=nbin, range=(rangemin, rangemax), alpha=0.6, color='blue', label=label2)
    ax1.legend()
    ax2.legend()
    # Add mean and standard deviation to the plot
    Events1, mean1, std1 = len(np1), np.mean(np1), np.std(np1)
    Events2, mean2, std2 = len(np2), np.mean(np2), np.std(np2)
    textstr1 = f'Events={Events1}\nmean1={mean1:.4f}\nsigma1={std1:.4f}'
    textstr2 = f'Events={Events1}\nmean2={mean2:.4f}\nsigma2={std2:.4f}'
    ax1.text(0.7, 0.75, textstr1, transform=ax1.transAxes, fontsize=14, verticalalignment='top')
    ax2.text(0.7, 0.75, textstr2, transform=ax2.transAxes, fontsize=14, verticalalignment='top')
    fig.suptitle(title)
    plt.savefig(saveTitle)
    # if (config.DISPLAY): plt.show()
    array = [mean1, std1, mean2, std2]
    plt.close('all')
    return array

def savefig(fig,name):
    fig.savefig(f'{name}.png')
    fig.savefig(f'{name}.pdf')
    print(f'{name}.png')
    print(f'{name}.pdf')
