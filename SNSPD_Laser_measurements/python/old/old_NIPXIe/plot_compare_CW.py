#!/usr/bin/env python3

import ROOT
from array import array
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-o',default="./output",type=str,help='output directory')
parser.add_argument('--report','-r',default=10000,type=int,help='report every x events')
parser.add_argument('--debug','-d',action="store_true",help='debug mode')
args = parser.parse_args()

def color(i):
    colorwheel = [416, 600, 800, 632, 880, 432, 616, 860, 820, 900, 420, 620, 820, 652, 1000, 452, 636, 842, 863, 823]
    # colorindex = int(i/11) + int(i%11)
    return colorwheel[i]

def getgraph(in_filename):
    plotDir= in_filename.rsplit("/",1)[0]
    infile = ROOT.TFile.Open(in_filename)
    g_count = infile.Get('g_count')
    g_pulse_range = infile.Get('g_pulse_range')
    return g_count, g_pulse_range

def plot_compare():
    g_counts, g_pulse_ranges, g_pres, powers=[],[],[],[]
    for in_filename in args.in_filenames:
        power = in_filename.split('uW/')[0].split('/')[-1]
        print(power)
        g_count, g_pulse_range = getgraph(in_filename)
        g_counts.append(g_count)
        g_pulse_ranges.append(g_pulse_range)
        powers.append(power)

    c1 = ROOT.TCanvas()
    plot_Dir = args.in_filenames[0].rsplit("/",2)[0]
    leg_count = ROOT.TLegend(0.2,0.8,0.8,0.9)
    leg_count.SetNColumns(3)
    for i, (g_count,power) in enumerate(zip(g_counts,powers)):
        if(i==0):
            g_count.Draw("AP PMC")
            g_count.GetYaxis().SetRangeUser(0,20)
            g_count.GetYaxis().SetTitle("Pulse Count / 40#mus")
            g_count.GetXaxis().SetTitle("Bias Current (nA)")
        else: g_count.Draw("PSame PMC")
        leg_count.AddEntry(g_count,f'{power}nW','p')
        # g_count.SetMarkerColor(color(i))
        g_count.SetMarkerStyle(i+20)
    leg_count.Draw()
    c1.SaveAs("CW_count_compare.png")

    for i, g_pulse_range in enumerate(g_pulse_ranges):
        if(i==0):
            g_pulse_range.Draw("AP PMC")
            g_pulse_range.GetYaxis().SetRangeUser(0,3.2)
            g_pulse_range.GetXaxis().SetTitle("Bias Current (nA)")
        else: g_pulse_range.Draw("PSame PMC")
        # g_pulse_range.SetMarkerColor(color(i))
        g_pulse_range.SetMarkerStyle(i+20)
    leg_count.Draw()
    c1.SaveAs("CW_amp_compare.png")

if __name__ == "__main__":
    ROOT.gStyle.SetPalette(ROOT.kBird)
    plot_compare()
