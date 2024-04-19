#!/usr/bin/env python3

import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ROOT
# import root_numpy

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
    ax.set_xlabel('Index [0.4ns]',fontsize=15)
    ax.set_ylabel('ADC [Volts]',fontsize=15)
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
    if (config.DISPLAY): plt.show()
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

def plot_histo_root(np1, nbin, rangemin, rangemax, name, xTitle, title, saveTitle):
    c1 = ROOT.TCanvas()
    hist = ROOT.TH1F(name, name, nbin, rangemin, rangemax)
    array = np.array(np1, dtype=float)
    # root_numpy.fill_hist(hist, array, weights=None, return_indices=False)
    # Draw hist
    hist.SetTitle(title)
    hist.GetXaxis().SetTitle(xTitle)
    hist.Draw()
    c1.SaveAs(saveTitle)
    return hist

def fit_histo_gaus(hist, rangemin, rangemax, name, xTitle, title, saveTitle):
    c1 = ROOT.TCanvas()
    fit = ROOT.TF1("fit","gausn",rangemin, rangemax)
    fit.SetLineWidth(2)
    hist.Fit("fit",'IRQ')
    mean = fit.GetParameter(1)
    mean_error = fit.GetParError(1)
    std = fit.GetParameter(2)
    std_error = fit.GetParError(2)
    integral = fit.Integral(rangemin, rangemax)
    # Draw hist
    hist.SetTitle(title)
    hist.GetXaxis().SetTitle(xTitle)
    hist.Draw()
    ROOT.gPad.Update()
    st = hist.GetListOfFunctions().FindObject("stats")
    st.AddText(f"Integral={integral}")
    fit.Draw("same")
    c1.SaveAs(saveTitle)
    return mean, mean_error, std, std_error, integral

def fit_histo_gaus_limit(hist, rangemin, rangemax, mean_min, mean_max, sigma_min, sigma_max, c_min, c_max, name, xTitle, title, saveTitle):
    c1 = ROOT.TCanvas()
    fit = ROOT.TF1("fit","gausn",rangemin, rangemax)
    fit.SetLineWidth(2)
    fit.SetParLimits(0,c_min,c_max)
    fit.SetParLimits(1,mean_min,mean_max)
    fit.SetParLimits(2,sigma_min,sigma_max)
    fit.SetParameter(1,mean_min)
    fit.SetParameter(2,sigma_min)
    hist.Fit("fit",'IQBR')
    mean = fit.GetParameter(1)
    mean_error = fit.GetParError(1)
    std = fit.GetParameter(2)
    std_error = fit.GetParError(2)
    const = fit.GetParameter(0)
    const_error = fit.GetParError(0)
    integral = fit.Integral(rangemin, rangemax)
    # Draw hist
    hist.SetTitle(title)
    hist.GetXaxis().SetTitle(xTitle)
    hist.Draw()
    ROOT.gPad.Update()
    st = hist.GetListOfFunctions().FindObject("stats")
    st.AddText(f"Integral={integral}")
    fit.Draw("same")
    c1.SaveAs(saveTitle)
    return mean, mean_error, std, std_error, const, const_error, integral

def color(i):
    colorwheel = [416, 600, 800, 632, 880, 432, 616, 860, 820, 900, 420, 620, 820, 652, 1000, 452, 636, 842, 863, 823]
    # colorindex = int(i/11) + int(i%11)
    return colorwheel[i]
