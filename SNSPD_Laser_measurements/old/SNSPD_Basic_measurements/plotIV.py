#!/usr/bin/env python

import os
import errno
import ROOT
from array import array

import argparse
parser = argparse.ArgumentParser(description='draw IV results')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
args = parser.parse_args()

def color(i):
    if i == 0: return 1
    elif i == 1: return 2
    elif i == 2: return 3
    elif i == 3: return 4
    elif i == 4: return 6
    elif i == 5: return 7
    elif i == 6: return 8
    elif i == 7: return 9
    elif i == 8: return 40
    elif i == 9: return 41
    elif i == 10: return 42
    elif i == 11: return 43
    elif i == 12: return 44
    elif i == 13: return 45
    elif i == 14: return 46
    elif i == 15: return 47
    elif i == 16: return 48
    elif i == 17: return 49
    else: return 0

def main():
    plotDir= "plots/" + args.in_filenames[0].split("data/")[1].rsplit("/",1)[0]
    try:
        os.makedirs(plotDir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Plot directory exists.')
        else:
            raise

    c1 = ROOT.TCanvas()
    g_IV={}
    for infile in args.in_filenames:
        with open (infile) as f:
            Lines = f.readlines()

        baseName= infile.split("/")[-1].split(".txt")[0]


        Currents, Volts, Resists = array('d'), array('d'), array('d')

        for i, Line in enumerate(Lines):
            V, I, R = Line.split()
            Volts.append(float(V))
            Currents.append(float(I))
            Resists.append(float(R))

        l = ROOT.TLegend(0.6, 0.55, 0.8, 0.7)
        g_IV[baseName] = ROOT.TGraph(len(Lines), Currents, Volts)

        c1.SetLogy(False)
        c1.SetLogx(False)
        g_IV[baseName].SetTitle("%s" % baseName)
        g_IV[baseName].GetXaxis().SetTitle("Current (A)")
        g_IV[baseName].GetYaxis().SetTitle("Voltage (V)")
        g_IV[baseName].SetMarkerColor(1)
        g_IV[baseName].SetMarkerSize(0.5)
        g_IV[baseName].Draw("APE")
        g_IV[baseName].GetXaxis().SetRangeUser(0., 0.00018)
        c1.Print("plots/IV_%s.png" % infile.split("/")[-1].split(".txt")[0])
        c1.SetLogy()
        c1.SetLogx()
        c1.Print("plots/IV_%s_log.png" % infile.split("/")[-1].split(".txt")[0])

    leg_multi = ROOT.TLegend(0.2, 0.6, 0.6, 0.8)
    leg_multi.SetTextSize(0.03)
    for i, gID in enumerate(g_IV):
        g_IV[gID].SetTitle("")
        g_IV[gID].GetXaxis().SetTitle("Current (A)")
        g_IV[gID].GetYaxis().SetTitle("Voltage (V)")
        g_IV[gID].SetLineColor(color(i))
        g_IV[gID].SetMarkerColor(color(i))
        g_IV[gID].SetMarkerStyle(i+20)
        # g_IV[gID].GetYaxis().SetRangeUser(0.01,0.4)
        g_IV[gID].GetXaxis().SetRangeUser(0,0.0011)
        leg_multi.AddEntry(g_IV[gID], gID, "P")
        if i == 0:
            g_IV[gID].Draw("APE")
        else:
            g_IV[gID].Draw("Psame")
    leg_multi.Draw("same")
    c1.SetLogx(0)
    c1.SetLogy(0)
    c1.Print("%s/IV_multi.png" % plotDir)
    c1.SetLogx(1)
    c1.SetLogy(1)
    c1.Print("%s/IV_multi_log.png" % plotDir)

if __name__ == '__main__':
    main()
