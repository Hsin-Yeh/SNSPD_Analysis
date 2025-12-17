#!/usr/bin/env python3
# NI-5162 specs: https://www.ni.com/docs/en-US/bundle/pxie-5162-specs/page/specs.html

# python3 -m python.NIPXIE.plot_multiHistograms --res plots/SNSPD_5/Laser/Ch3/20240504/11p47K/Pulse/515/10000kHz/*nW/240degrees/*/*/*.root
# python3 -m python.NIPXIE.plot_multiHistograms plots/SNSPD_5/Laser/Ch3/20240503/4p68/Pulse/515/10000kHz/*nW/240degrees/*/*/*.root
# python3 -m python.NIPXIE.plot_multiHistograms plots/SNSPD_5/Laser/Ch3/20240503/4p68/Pulse/515/10000kHz/*nW/240degrees/*/*/*.root plots/SNSPD_5/Laser/Ch3/20240504/11p47K/Pulse/515/10000kHz/*nW/240degrees/*/*/*.root

# python3 -m python.NIPXIE.plot_multiHistograms --res plots/SNSPD_5/Laser/Ch3/20240504/11p47K/Pulse/515/10000kHz/*nW/240degrees/*/*/*.root ; python3 -m python.NIPXIE.plot_multiHistograms plots/SNSPD_5/Laser/Ch3/20240503/4p68/Pulse/515/10000kHz/*nW/240degrees/*/*/*.root ; ./scripts/cpPaperPlots.sh

import numpy as np
np.seterr(all='ignore')
import ROOT
from array import array
import argparse
import math
import json
import pandas as pd

from ..utils.plotUtils import colors, markers, lines, savefig
from ..utils.osUtils import createDir
from ..utils.fitfunctions import alt_expo
from ..utils.Results import *
from ..config import SNSPD_5_2_config as cf

import warnings

parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--debug','-d',action="store_true",help='debug mode')
parser.add_argument('--initPlot','-i',action="store_true",help='save histograms')
parser.add_argument('--sweepBias',action="store_true",help='sweep bias mode')
parser.add_argument('--sweepAll',action="store_true",help='Run all plots for each sweep variable')
parser.add_argument('--sweepPolarization',action="store_true",help='Run plots for polarization')
parser.add_argument('--fit','-f',action="store_true",help='do fit')
parser.add_argument('--res',action="store_true",help='do resistivie mode analysis')
args = parser.parse_args()

def getkey(photon_number,bias_voltage,polarization,sample_temperature):
    key = str(photon_number) + '_' + str(bias_voltage) + 'mV_' + str(polarization) + 'degrees_' + str(sample_temperature) + 'K'
    return key

def Normalized_Graph_sweep(Vars,statname, ytit="", plotDir="./"):
    createDir(f"{plotDir}/Norm")
    fig, ax = plt.subplots()
    yvals,xvals=[],[]
    photon_subset = [699, 777, 855, 933, 1011, 1088, 1166, 1244, 1322, 1399, 1477, 1555, 1788, 2333] if not args.res else [1555, 3111, 4666, 6222, 7777, 9333, 11666, 15555, 23333]
    bias_subset = [800,850,900,950,1000,1030] if not args.res else [84, 94, 104, 124, 134, 144, 154, 164, 174]
    index=0
    for bv,n_bv in zip(Vars["bv"],Vars["n_bv"]):
        if (bv not in bias_subset): continue
        for (pho,n_pho,eV) in zip(Vars["pho"],Vars["n_pho"],Vars["eV"]):
            key = getkey(pho,bv,Vars["pol"][0],Temps[0])
            try:
                value = Stats[key][statname]
                yvals.append(value)
                xvals.append(eV/1000)
            except KeyError:
                print(key)
                continue
        ax.plot(xvals, yvals, marker=markers(index), markersize=8, fillstyle='full', linestyle='-', alpha=1, label=f'{n_bv:.2f}')
        index+=1
        xvals.clear()
        yvals.clear()
    # ax.grid(True)
    # if(args.res): ax.set_xlabel(r'Laser Intensity ($P_{Laser}$ / $P_0^{cal})$',fontsize=15)
    # else: ax.set_xlabel(r'Laser Intensity ($P_{Laser}$ / $P_0^{sc})$',fontsize=15)
    ax.set_xlabel('Energy (keV)',fontsize=15)
    ax.set_ylabel(ytit,fontsize=15)
    # ax.set_title(f'{title}_sweep_intensity',fontsize=15)
    handles, labels = ax.get_legend_handles_labels() # Get the current handles and labels from the axes
    handles = handles[::-1] # Reverse the order of handles and labels
    labels = labels[::-1]
    Ibias=r'$I_{bias}$'
    Isw=r'$I_{sw}$'
    if (Temps[0]=="11p47K"): legend = ax.legend(handles, labels, title=f'{Ibias} / {Isw}({SampleT_norm:.2f}Tc)')
    else: legend = ax.legend(handles, labels, title=f'{Ibias} / {Isw}({SampleT_norm:.2f}Tc)',loc='upper right', bbox_to_anchor=(0.93, 0.98))
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    savefig(fig,f"{plotDir}/Norm/{statname}_sweep_intensity")
    ax.set_yscale('log')
    savefig(fig,f"{plotDir}/Norm/{statname}_sweep_intensity_log")
    plt.close("all")

    fig1, ax1 = plt.subplots()
    photon_subset = [777, 1011, 1322, 1555, 2333] if not args.res else [4666, 6222, 7777, 9333, 11666, 15555, 23333]
    bias_subset = [800, 850, 900, 950, 1000, 1010, 1020] if not args.res else [84, 94, 104, 124, 134, 144, 154, 164, 174]
    for i, (pho,n_pho,eV) in enumerate(zip(Vars["pho"],Vars["n_pho"],Vars["eV"])):
        if (pho not in photon_subset): continue
        for ibv, (bv,n_bv) in enumerate(zip(Vars["bv"],Vars["n_bv"])):
            if (bv not in bias_subset): continue
            key = getkey(pho,bv,Vars["pol"][0],Temps[0])
            try:
                value = Stats[key][statname]
                yvals.append(value)
                xvals.append(n_bv)
            except KeyError:
                print(key)
                continue
        ax1.plot(xvals, yvals, marker=markers(i), markersize=8, fillstyle='full', alpha=1, label=f'{eV/1000:.1f}')
        xvals.clear()
        yvals.clear()
    # ax1.grid(True)
    ax1.set_xlabel(f'Bias Current ({Ibias} / {Isw}({SampleT_norm:.2f}Tc))',fontsize=15)
    ax1.set_ylabel(ytit,fontsize=15)
    # if(args.res): legend = ax1.legend(title=r'$P_{Laser}$ / $P_0^{cal}$')
    # else: legend = ax1.legend(title=r'$P_{Laser}$ / $P_0^{sc}$')
    ax1.legend(title='Energy (keV)')
    fig1.tight_layout()
    savefig(fig1,f"{plotDir}/Norm/{statname}_sweep_bias")
    plt.close()

def Normalized_Graph_sweep_singleIntensity(pho, statname, ytit="", plotDir="./"):
    createDir(f"{plotDir}/Norm")
    fig, ax = plt.subplots()
    yvals,xvals=[],[]
    index=0
    for bv,n_bv in zip(Vars["bv"],Vars["n_bv"]):
        key = getkey(pho,bv,Vars["pol"][0],Temps[0])
        try:
            value = Stats[key][statname]
            yvals.append(value)
            xvals.append(n_bv)
        except KeyError:
            print(key)
            continue
    ax.plot(xvals, yvals, marker="o", markersize=8, fillstyle='full', linestyle='-', alpha=1)
    ax.set_xlabel(r'Normalized Bias Current ($I_{bias}$ / $I_{sw}$)',fontsize=15)
    ax.set_ylabel(ytit,fontsize=15)
    # ax.set_title(f'{title}_sweep_intensity',fontsize=15)
    plt.tight_layout()
    savefig(fig,f"{plotDir}/Norm/{statname}_sweep_bias_{pho}")
    plt.close('all')

def model_function(E, A, B, C):
    return np.sqrt((A / E)**2 + (B / np.sqrt(E))**2 + C**2)

def Resolution(Vars,plotDir):
    createDir(f"{plotDir}/Reso/")
    fig, ax = plt.subplots()
    index=0
    fit_results = []
    bias_subset = [84, 94, 104, 124, 134, 144, 154, 164, 174]
    for i, (bv,n_bv) in enumerate(zip(Vars["bv"],Vars["n_bv"])):
        if (bv not in bias_subset): continue
        yvals,xvals=[],[]
        for (pho,n_pho,eV) in zip(Vars["pho"],Vars["n_pho"],Vars["eV"]):
            if (pho < 2500): continue
            key = getkey(pho,bv,Vars["pol"][0],Temps[0])
            try:
                mean = Stats[key]["avg"]
                sigma = Stats[key]["pulse_range_ptp_err"] / Stats[key]["avg"]
                xvals.append(eV/1000)
                yvals.append(sigma)
            except KeyError:
                print(key)
                continue
        initial_guess = [1, 1, 1]  # Initial guess for A, B, C
        params, covariance = curve_fit(model_function, xvals, yvals, p0=initial_guess, bounds=(0, np.inf)) # Perform the curve fit
        A_fit, B_fit, C_fit = params
        E_fit = np.linspace(min(xvals)-2, max(xvals)+2, 100)
        y_fit = model_function(E_fit, A_fit, B_fit, C_fit)
        sqrt_keV = r'$\sqrt{keV}$'
        exp = model_function(1, A_fit, B_fit, C_fit)
        print(f'{n_bv}: A = {A_fit} keV, B = {B_fit} {sqrt_keV}, C = {C_fit}, 1MeV = {exp}')
        ax.plot(xvals, yvals, marker=markers(i), markersize=8, fillstyle='none', linestyle='none', alpha=1, label=f'{n_bv}: A = {A_fit:.2f} keV, B = {B_fit:.2f} {sqrt_keV}, C = {C_fit:.2f}')
        ax.plot(E_fit, y_fit, color=colors(i), linestyle=lines(i))
    ax.set_xlabel(r'Energy (keV)',fontsize=15)
    ax.set_ylabel(r'Relative Resolution $\sigma / \overline{A}$',fontsize=15)
    # Customize legend to include line styles for markers
    handles, labels = ax.get_legend_handles_labels()
    for i in range(len(handles)):
        handles[i] = plt.Line2D([], [], color=colors(i), marker=markers(i), fillstyle='none', linestyle=lines(i), label=labels[i])
    Ibias=r'$I_{bias}$'
    Isw=r'$I_{sw}$'
    legend = ax.legend(handles, labels, title=f'{Ibias} / {Isw}({SampleT_norm:.2f}Tc): Fit results')
    plt.tight_layout()
    savefig(fig,f"{plotDir}/Reso/resolution")
    plt.close('all')

def multi_histo_matplotlib(Vars,histname,xtit,plotDir):
    # photon_subset = [1.0, 3.0, 5.0, 7.5, 10.0, 15.0] if args.res else [1.0, 1.3, 1.5, 2.0]
    photon_subset = [776, 1182, 1447, 1754, 2211, 2603]
    photon_subset = [2603, 3278, 4793, 8630, 18238, 30255]
    photon_subset = [936, 1162, 1427, 1572, 1757, 1987, 2154, 2368, 2598, 2995, 3278]
    for (bv,n_bv) in zip(Vars["bv"],Vars["n_bv"]):
        fig, ax = plt.subplots()
        for (ipho, (pho,n_pho,eV)) in enumerate(zip(Vars["pho"],Vars["n_pho"],Vars["eV"])):
            # if (ipho%2==1): continue
            if (pho not in photon_subset): continue
            key = getkey(pho,bv,Vars["pol"][0],Temps[0])
            try:
                [lst1, lst2] = Stats[key][histname]
                bin_contents = np.array(lst1)
                bin_edges = np.array(lst2)
                non_zero_bins = np.where(bin_contents > 0)[0] # Find the rightmost bin with non-zero content
                if len(non_zero_bins) > 0:
                    min_non_zero_bin_edge = bin_edges[non_zero_bins[0] + 1]
                    max_non_zero_bin_edge = bin_edges[non_zero_bins[-1] + 1]
                else:
                    max_non_zero_bin_edge = bin_edges[-1]
                ax.hist(bin_edges[:-1], bins=bin_edges, weights=bin_contents, histtype='step', linewidth=2, label=f'{eV/1000:.1f}')
            except KeyError:
                print(key)
                continue
        # Customize the plot
        ax.set_yscale('log')
        if (args.res):
            ax.set_xlim(0, max_non_zero_bin_edge*1.2)
            ax.set_ylim(top=0.4)
        else: ax.set_xlim(-0.1, max_non_zero_bin_edge*1.2)
        ax.set_xlim(-0.1,0.5)
        ax.set_xlabel(xtit, fontsize=15)
        ax.set_ylabel('Probability', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=10)
        # if(args.res): legend = ax.legend(title=r'$P_{Laser}$ / $P_0^{cal}$')
        # else: legend = ax.legend(title=r'$P_{Laser}$ / $P_0^{sc}$')
        legend = ax.legend(title='Energy (keV)')
        legend.get_title().set_fontsize('large')  # Adjust the font size of the legend title
        Ibias=r'$I_{bias}$'
        Isw=r'$I_{sw}$'
        ax.text(0.3, 0.95, f'{Ibias} = {n_bv}{Isw}({SampleT_norm:.2f}Tc)', transform=ax.transAxes, verticalalignment='top', fontsize=15)
        fig.tight_layout()
        savefig(fig,f"{plotDir}/multi/histos_sweep_photon_{bv}mV")
        fig.clf()  # Clear the figure
        plt.close(fig)

def spectrum_sweep_bias(plotDir="./"):
    photon_subset = [699, 777, 855, 933, 1011, 1088, 1166, 1244, 1322, 1399, 1477, 1555, 1788, 2333] if not args.res else [1555, 3111, 4666, 6222, 7777, 9333, 11666, 15555, 23333]
    bias_subset = [800,850,900,950,1000,1030] if not args.res else [84, 94, 104, 124, 134, 144, 154, 164, 174]
    createDir(f"{plotDir}/spectrums")
    xvals=[]
    for i in range(1000):
        xvals.append(i*0.4-122)
    xmin,xmax = 0,18
    for (pho,n_pho) in zip(Vars["pho"],Vars["n_pho"]):
        fig, ax = plt.subplots()
        pad_xnum = 5
        # fig_multi, ax_multi = plt.subplots(int(len(Vars["bv"])/pad_xnum+1), pad_xnum, figsize=(36, 6*(int(len(Vars["bv"])/pad_xnum)+1)), sharex=True, sharey=True)
        index=0
        for (bv,n_bv) in zip(Vars["bv"],Vars["n_bv"]):
            if (bv not in bias_subset): continue
            key = getkey(pho,bv,Vars["pol"][0],Temps[0])
            try:
                graph = Stats[key]["spectrum"]
            except KeyError:
                print(key)
                continue
            ax.plot(xvals, graph, marker=markers(index), markersize=6, fillstyle='none', alpha=0.75, label=f'{n_bv}')
            # ax multi
            # ax1 = ax_multi.flatten()[index]
            # ax1.plot(xvals, graph, marker='o', markersize=6, fillstyle='none', alpha=0.75, label=f'{bv}mV')
            # ax1.tick_params(axis='both', which='major', labelsize=20) # Set larger ticks and labels
            # ax1.legend(fontsize=30)
            # ax1.grid()
            # if index // pad_xnum == int(len(Vars["bv"])/pad_xnum):  ax1.set_xlabel('Time (ns)', fontsize=30) # Add x and y titles only to the leftmost and bottom-most subplot
            # if index % pad_xnum == 0:  ax1.set_ylabel('Voltage (V)', fontsize=30)
            # ax1.set_xlim(xmin,xmax)
            # ax1.legend()
            index+=1
        # fig_multi.tight_layout()
        # fig_multi.savefig(f"{plotDir}/spectrums/multi_{int(n_pho)}.png", format='png')
        # print(f"{plotDir}/spectrums/multi_{int(n_pho)}.png")
        # # ax.grid(True)
        ax.set_xlabel('Time (ns)',fontsize=15)
        ax.set_ylabel('Voltage (V)',fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlim(xmin,xmax)
        handles, labels = ax.get_legend_handles_labels() # Get the current handles and labels from the axes
        handles = handles[::-1] # Reverse the order of handles and labels
        labels = labels[::-1]
        Ibias=r'$I_{bias}$'
        Isw=r'$I_{sw}$'
        legend = ax.legend(handles,labels,title=f'{Ibias} / {Isw}({SampleT_norm:.2f}Tc)')
        legend.get_title().set_fontsize('large')  # Adjust the font size of the legend title
        P0=r'$P_{0}^{cal}$' if (args.res) else r'$P_{0}^{sc}$'
        # ax.text(0.25, 0.95, f'P = {n_pho}{P0}', transform=ax.transAxes, verticalalignment='top', fontsize=15)
        fig.tight_layout()
        savefig(fig,f"{plotDir}/spectrums/comb_{pho}")
        plt.close("all")

def Compare_Temp(statname, ytit="", plotDir="./"):
    createDir(f"{plotDir}/Compare")
    yvals,xvals=[],[]
    fig, ax = plt.subplots()
    for i,temp in enumerate(Temps):
        print(temp)
        for bv,n_bv in zip(VarsTemp[temp]["bv"],VarsTemp[temp]["n_bv"]):
            key = getkey(15555 if temp=="11p47K" else 1555,bv,VarsTemp[temp]["pol"][0],temp)
            try:
                value = Stats[key][statname]
                yvals.append(value)
                xvals.append(n_bv)
            except KeyError:
                print(key)
                continue
        if 'p' in temp: temp = temp.replace('p', '.')  # Replace 'p' with '.'
        if 'K' in temp: temp = temp.replace('K', '')  # Remove 'K' if 'k' is present
        ax.plot(xvals, yvals, marker=markers(i), markersize=10, fillstyle='full', linestyle='none', alpha=1, label=f"{temp}K")
        yvals,xvals=[],[]
    ax.set_xlabel(r'Bias Current ($I_{bias}$ / $I_{sw}$)',fontsize=15)
    ax.set_ylabel(ytit,fontsize=15)
    ax.legend()
    fig.tight_layout()
    savefig(fig,f"{plotDir}/Compare/{statname}")

def plots():
    print("\n==================== Start plotting ====================")
    for temp in Temps:
        sweepAll_plotDir = args.in_filenames[0].rsplit('/',5)[0]
        createDir(sweepAll_plotDir)
        createDir(f"{sweepAll_plotDir}/multi")
        Normalized_Graph_sweep(VarsTemp[temp],"pulse_range_ptp",ytit=r"Signal Amplitude Mean $\overline{A}_{signal}$ (V)",plotDir=sweepAll_plotDir)
        # Normalized_Graph_sweep(VarsTemp[temp],"pulse_fall_tau",ytit=r"Fall Time Constant \tau_{fall} (ns)",plotDir=sweepAll_plotDir)
        # Normalized_Graph_sweep(VarsTemp[temp],"avg",ytit="Signal Amplitude (V)",plotDir=sweepAll_plotDir)
        # Normalized_Graph_sweep(VarsTemp[temp],"resist",ytit=r"Resistance ($Omega$)",plotDir=sweepAll_plotDir)
        # Normalized_Graph_sweep_singleIntensity(15555 if args.res else 1555, "pulse_fall_tau", ytit="Fall Time Constant (ns)", plotDir=sweepAll_plotDir)
        # Resolution(VarsTemp[temp],sweepAll_plotDir)
        multi_histo_matplotlib(VarsTemp[temp],"h_pulse_fall_range_ptp",r"Signal Amplitude (V)",sweepAll_plotDir)
        # spectrum_sweep_bias(sweepAll_plotDir)
    # Compare_Temp("pulse_fall_tau",ytit=r"Fall Time Constant $\tau_{fall}$ (ns)",plotDir=sweepAll_plotDir)

if __name__ == "__main__":
    Temps = []
    VarsTemp, Stats, subsets = {},{},{}
    SampleTc = 12
    SampleT=11.5 if args.res else 4.7
    SampleT_norm = SampleT/SampleTc

    # SNSPD_5-2
    # # Critical voltage
    # Vcs = {}
    # Vcs["4.68"] = 1080
    # Vcs["4p72K"] = 1080
    # Vcs["9p25K"] = 520
    # Vcs["11p47K"] = 135
    # # Critical current
    # Ics = {}
    # Ics["4.68"] = 108e-6
    # Ics["4p72K"] = 108e-6
    # Ics["9p25K"] = 52e-6
    # Ics["11p47K"] = 13e-6

    # SNSPD_6-1
    # Critical voltage
    Vcs = {}
    Vcs["4p77K"] = 380
    Vcs["7p37K"] = 260
    Vcs["8p57K"] = 240
    Vcs["9p27K"] = 170
    # Critical current
    Ics = {}
    Ics["4p77K"] = 38e-6
    Ics["7p37K"] = 26e-6
    Ics["8p57K"] = 24e-6
    Ics["9p27K"] = 17e-6

    # All bias: [74, 84, 94, 104, 114, 124, 134, 144, 154, 164, 174], [500, 600, 700, 800, 850, 900, 950, 1000, 1010, 1020, 1030, 1040, 1050]
    subsets["photon"] = [699,1011,1399,2333]
    # subsets["bv"] = [84, 94, 104, 124, 134, 144, 154, 164, 174] if args.res else [800, 850, 900, 950, 1000, 1010, 1020, 1030, 1050]
    subsets["bv"] = [84, 94, 104, 124, 134, 144, 154, 164, 174, 800, 850, 900, 950, 1000, 1010, 1020, 1030]

    for i, in_filename in enumerate(args.in_filenames):
        print (f"{i}/{len(args.in_filenames)}: {in_filename}")
        runFlag, info = getvars(in_filename,Temps,VarsTemp,Vcs,Ics,subsets)
        if (runFlag==False): continue
        key = getkey(info['photon_number'],info['bias_voltage'],info['polarization'],info['sample_temperature'])
        print(key)
        stat= getstat(in_filename,key) if args.initPlot else readstat(in_filename,key)
        Stats[key]=stat
    SortVars(Temps,VarsTemp,Vcs,Ics)
    plots() # Plot them together
