#!/usr/bin/env python3

# python3 python/NIPXIE/plot_examples.py Stats/4p68K/*_1000mV_240degrees_4p68K_stats.json
# python3 python/NIPXIE/plot_examples.py Stats/11p47K/*_104mV_240degrees_11p47KK_stats.json


import numpy as np
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import errno

parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
args = parser.parse_args()

##### Utilities #####
def savefig(fig,name):
    fig.savefig(f'{name}.png')
    print(f'{name}.png')

def createDir(Dir):
    try:
        os.makedirs(Dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

##### Graphs, histos and waveforms #####
def graphs(statname, ytit=""):
    fig, ax = plt.subplots()
    yvals,xvals=[],[]
    for key, stat in Stats.items():
        photon_number = float(key.split('_')[0])
        value = stat[statname]
        xvals.append(photon_number)
        yvals.append(value)
    ax.plot(xvals, yvals, marker='o', markersize=8, fillstyle='full', linestyle='none')
    ax.set_xlabel('Photon Number',fontsize=15)
    ax.set_ylabel(ytit,fontsize=15)
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    savefig(fig,f'{plotDir}/Graph_{statname}_sweepPhoton')
    plt.close(fig)

def graphs_sweepbias(statname, ytit=""):
    fig, ax = plt.subplots()
    yvals,xvals=[],[]
    for key, stat in Stats.items():
        photon_number = float(key.split('_')[0])
        bias = float(key.split('_')[1].split('mV')[0])
        value = stat[statname]
        xvals.append(bias/10)
        yvals.append(value)
    ax.plot(xvals, yvals, marker='o', markersize=8, fillstyle='full', linestyle='none')
    ax.set_xlabel(r'Bias Current ($\mu$A)',fontsize=15)
    ax.set_ylabel(ytit,fontsize=15)
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    savefig(fig,f'{plotDir}/Graph_{statname}_sweepBias')
    plt.close(fig)

def histos(histname,xtit):
    fig, ax = plt.subplots()
    for key, stat in Stats.items():
        photon_number = float(key.split('_')[0])
        [lst1, lst2] = stat[histname]
        bin_contents = np.array(lst1)
        bin_edges = np.array(lst2)
        ax.hist(bin_edges[:-1], bins=bin_edges, weights=bin_contents, histtype='step', linewidth=2, label=photon_number)
    # Customize the plot
    ax.set_yscale('log')
    ax.set_xlabel(xtit, fontsize=15)
    ax.set_ylabel('Probability', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(title='photon_number')
    fig.tight_layout()
    savefig(fig,f'{plotDir}/multi_histo_{histname}')
    plt.close(fig)

def spectrums():
    xvals=[]
    fig, ax = plt.subplots()
    for i in range(1000):
        xvals.append(i*0.4)
    for key, stat in Stats.items():
        graph = stat["spectrum"]
        ax.plot(xvals, graph, marker='o', markersize=6, fillstyle='none', alpha=0.75, label=key)
    ax.set_xlabel('Time (ns)',fontsize=15)
    ax.set_ylabel('Voltage (V)',fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(loc='upper right')
    ax.set_xlim(110,230)
    fig.tight_layout()
    savefig(fig,f'{plotDir}/spectrums')
    plt.close(fig)

##### Choose what to plot #####
def plots():
    print("\n==================== Start plotting ====================")
    histos("h_pulse_fall_range_ptp","Signal Amplitude (V)")
    graphs("pulse_range_ptp","Signal Amplitude (V)")
    graphs_sweepbias("pulse_range_ptp","Signal Amplitude (V)")
    graphs("eff","Efficiency")
    spectrums()

##### Read the input json files #####
def readjson():
    print("\n==================== Read json ====================")
    for i, in_filename in enumerate(args.in_filenames):
        key = in_filename.split('/')[-1].split('_stats.json')[0]
        with open(in_filename, 'r') as file:
            stat = json.load(file)
        if (i == 0): print(f"Keys stored in the stat dict: {stat.keys()}\n")
        pho,bv,pol,temp = key.split('_')
        print(pho,bv,pol,temp)
        Stats[key]=stat

if __name__ == "__main__":
    Stats = {}
    plotDir = "plots/" + args.in_filenames[0].split('_')[3].split('degrees')[0]
    createDir(plotDir)
    readjson()
    plots()
