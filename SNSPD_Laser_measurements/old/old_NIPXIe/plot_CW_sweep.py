#!/usr/bin/env python3

import ROOT
from array import array
import argparse
import matplotlib.pyplot as plt
import numpy as np
import math

from ..utils.plotUtils import *
from ..utils.tdmsUtils import *
from ..utils.osUtils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-o',default="./output",type=str,help='output directory')
parser.add_argument('--report','-r',default=10000,type=int,help='report every x events')
parser.add_argument('--debug','-d',action="store_true",help='debug mode')
parser.add_argument('--initPlot','-i',action="store_true",help='save histograms')
args = parser.parse_args()

def project(tree, hist, var, cut, title="", xtit="", ytit="", outDir="plots/test/", saveTitle="", xmin=-1, xmax=-1):
    # print (f'projecting var: {var}, cut: {cut} from tree: {tree.GetName()} into hist: {h.GetName()}')
    tree.Project(hist.GetName(),var,cut)
    # move overflow to last bin
    nbins = hist.GetNbinsX();
    lastBin = hist.GetBinContent(nbins);
    overflow = hist.GetBinContent(nbins+1);
    firstBin = hist.GetBinContent(1);
    underflow = hist.GetBinContent(0);
    hist.SetBinContent(nbins,lastBin+overflow);
    hist.SetBinContent(nbins+1, 0.0);
    hist.SetBinContent(1,firstBin+underflow);
    hist.SetBinContent(0, 0.0);
    if (args.initPlot):
        c_hist = ROOT.TCanvas()
        hist.GetXaxis().SetTitle(xtit)
        hist.GetYaxis().SetTitle(ytit)
        hist.SetTitle(title)
        hist.Draw()
        if (xmax!=-1): hist.GetXaxis().SetRangeUser(xmin,xmax)
        c_hist.SaveAs(f"{outDir}/{saveTitle}.png")

def calculate_CW_photon(laser_power,recordlength,sampleRate,attenuation,wavelength):
    c = 299792458
    h = 6.62607015e-34
    SNSPD_power = laser_power*attenuation
    frequency = c / wavelength
    single_photon_energy = h*frequency
    gate=recordlength/sampleRate
    print(gate)
    photon_number = (SNSPD_power*gate)/single_photon_energy
    return int(photon_number)

def getkey(photon_number,bias_voltage,polarization,sample_temperature):
    key = str(photon_number) + '_' + str(bias_voltage) + 'mV_' + str(polarization) + 'degrees_' + str(sample_temperature) + 'K'
    return key

def get_info(in_filename):
    # Read the JSON file into a DataFrame
    jsonfile = in_filename.replace('.root', '.json')
    df = pd.read_json(jsonfile, lines=True)
    laser_power        = float(df[df['metaKey'] == 'Laser Power (uW)']['metaValue'].iloc[0])
    bias_voltage       = int(float(df[df['metaKey'] == 'Bias Voltage (mV)']['metaValue'].iloc[0]))
    bias_current       = int(float(df[df['metaKey'] == 'Bias Current (uA)']['metaValue'].iloc[0]))
    vertical_range     = float(df[df['metaKey'] == 'vertical range Sig']['metaValue'].iloc[0])
    vertical_offset    = float(df[df['metaKey'] == 'vertical offset Sig']['metaValue'].iloc[0])
    wavelength         = float(df[df['metaKey'] == 'Laser Wavelength (nm)']['metaValue'].iloc[0]) * 1e-9
    polarization       = int(df[df['metaKey'] == 'Polarization']['metaValue'].iloc[0])
    sample_temperature = df[df['metaKey'] == 'Sample Temperature (K)']['metaValue'].iloc[0]
    recordlength       = float(df[df['metaKey'] == 'record length']['metaValue'].iloc[0])
    sampleRate         = float(df[df['metaKey'] == 'actual sample rate']['metaValue'].iloc[0])
    attenuation = 0.03
    photon_number = calculate_CW_photon(laser_power*1e-6,recordlength,sampleRate,attenuation,wavelength)
    if photon_number not in Photons:
        Photons.append(photon_number)
    if bias_voltage not in BVs:
        BVs.append(bias_voltage)
        BCs.append(bias_current)
    if polarization not in Polars:
        Polars.append(polarization)
    # if sample_temperature not in Temps:
    Temps.append(sample_temperature)
    return laser_power, bias_voltage, bias_current, photon_number, polarization, sample_temperature, vertical_range, vertical_offset

def sort_bias(bias, var):
    bias_array = np.array(bias)
    var_array = np.array(var)
    sorted_bias_array = np.sort(bias_array)
    sorted_var_array= var_array[bias_array.argsort()]
    return sorted_bias_array, sorted_var_array

def Graph_sweep(polar, temp, photons, bvs, bcs, stat, title="graph", ytit="", plotDir="./"):
    fig, ax = plt.subplots()
    yvals,xvals=[],[]
    for i, bv in enumerate(bvs):
        for photon in photons:
            key = getkey(photon,bv,polar,temp)
            try:
                value = stat[key]
                yvals.append(value)
                xvals.append(photon)
            except KeyError:
                continue
        ax.plot(xvals, yvals, marker=markers(i), markersize=6, fillstyle='none', alpha=0.75, label=f'{bv}mV')
        xvals.clear()
        yvals.clear()
    ax.grid(True)
    ax.set_xlabel('Photon number',fontsize=15)
    ax.set_ylabel(ytit,fontsize=15)
    ax.set_title(f'{title}_sweep_photon',fontsize=15)
    ax.legend(title='Bias Voltage')
    plt.tight_layout()
    plt.savefig(f"{plotDir}/{title}_sweep_photon.png", format='png')
    print(f"{plotDir}/{title}_sweep_photon.png")
    ax.set_yscale('log')
    ax.set_title(f'{title}_sweep_photon_log',fontsize=15)
    plt.savefig(f"{plotDir}/{title}_sweep_photon_log.png", format='png')
    print(f"{plotDir}/{title}_sweep_photon_log.png")
    plt.close()

    fig1, ax1 = plt.subplots()
    for i, photon in enumerate(photons):
        for ibv, bv in enumerate(bvs):
            key = getkey(photon,bv,polar,temp)
            try:
                value = stat[key]
                yvals.append(value)
                xvals.append(bv)
            except KeyError:
                continue
        ax1.plot(xvals, yvals, marker=markers(i), markersize=6, fillstyle='none', alpha=0.75, label=f'{photon}')
        xvals.clear()
        yvals.clear()
    ax1.grid(True)
    ax1.set_xlabel(r'Bias Current ($\mu$A)',fontsize=15)
    ax1.set_ylabel(ytit,fontsize=15)
    ax1.set_title(f'{title}_sweep_bias',fontsize=15)
    ax1.legend(title='Photon number')
    plt.tight_layout()
    plt.savefig(f"{plotDir}/{title}_sweep_bias.png", format='png')
    print(f"{plotDir}/{title}_sweep_bias.png")
    ax1.set_yscale('log')
    ax1.set_title(f'{title}_sweep_bias_log',fontsize=15)
    plt.savefig(f"{plotDir}/{title}_sweep_bias_log.png", format='png')
    print(f"{plotDir}/{title}_sweep_bias_log.png")
    plt.close()

def gettree():
    for i, in_filename in enumerate(args.in_filenames):
        print (f"{i}/{len(args.in_filenames)}: {in_filename}")
        laser_power, bias_voltage, bias_current, photon_number, polarization, sample_temperature, vertical_range, vertical_offset = get_info(in_filename)
        # if (bias_voltage > 530): continue
        basename = getkey(photon_number,bias_voltage,polarization,sample_temperature)
        plotDir= in_filename.rsplit("/",1)[0]
        infile = ROOT.TFile.Open(in_filename)
        # Get Tree
        intree = infile.Get('SNSPD_data')
        # Initialize variables
        if (vertical_range==0.05): nbin=pow(2,6)
        elif (vertical_range==0.1): nbin=pow(2,6.6)
        elif (vertical_range>0.1 and vertical_range<=5): nbin=pow(2,7)
        else:
            print("Invalid vertical range!!!")
            break
        range_min, range_max= 0, vertical_range
        binsize = float((range_max-range_min)/nbin)
        maxCount = int(intree.GetMaximum("pulseCount"))
        # initialize histo
        h_count = ROOT.TH1F("h_count","h_count",maxCount,0,maxCount)
        h_pulseRange = ROOT.TH1F("h_pulseRange","h_pulseRange",nbin,range_min,range_max)
        # Project variables to histos
        project(intree,h_count,"pulseCount","",f"{laser_power}uW_{bias_voltage}mV_{bias_current}uA","Pulse Counts","Event",plotDir,"h_count")
        project(intree,h_pulseRange,"pulseRange","",f"{laser_power}uW_{bias_voltage}mV_{bias_current}uA","Pulse Range (V)","Event",plotDir,"h_range")
        # Calculate
        count = h_count.GetMean()
        count_err = h_count.GetRMS() / math.sqrt(h_count.Integral())
        pulse_range = h_pulseRange.GetMean()
        pulse_range_error = h_pulseRange.GetRMS()
        eff = count/photon_number
        # Fill stats dict
        effs[basename] = eff
        counts[basename] = count
        count_errs[basename] = count_err
        pulse_ranges[basename] = pulse_range
        print(f"{basename}: count:{count}, {eff*100:.6f}%, {pulse_range*1000:.1f}mV+-{pulse_range_error*1000:.2f}mV")

def plots():
    print("\n==================== Start plotting ====================")
    # Sort sweep variables
    Photons.sort()
    print(Photons)
    BVs.sort()
    BCs.sort()
    Polars.sort()
    # Plots
    sweepAll_plotDir = args.in_filenames[0].rsplit('/',5)[0]
    Graph_sweep(Polars[0],Temps[0],Photons,BVs,BCs,counts,title="g_count",ytit="Pulse Counts",plotDir=sweepAll_plotDir)

if __name__ == "__main__":
    Photons,BVs,BCs,Polars,Temps = [],[],[],[],[] # List for sweep variables
    effs,counts,count_errs,pulse_ranges = {},{},{},{}
    gettree() # loop over the input files
    plots() # Plot them together
