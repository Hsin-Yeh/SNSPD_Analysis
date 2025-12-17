#!/usr/bin/env python

import ROOT
import csv
import argparse
from array import array

parser = argparse.ArgumentParser(description='make basic plots of laser calibration and convert csv to root')
parser.add_argument('--simfile','-s',default="",type=str,help='input simulation file')
parser.add_argument('--datafile','-d',default="",type=str,help='input data file')
args = parser.parse_args()

sim_reflecs, sim_lamdas = array( 'd' ), array( 'd' )
data_reflecs, data_lamdas = array( 'd' ), array( 'd' )
with open(args.simfile, newline='') as csvfile:
    rows = csv.reader(csvfile, delimiter=',', quotechar="|")
    for (i, row) in enumerate(rows):
        if (i>2 and len(row)>1):
            sim_lamdas.append(float(row[0])*1e6)
            sim_reflecs.append(float(row[1][1:]))

with open(args.datafile, newline='') as csvfile:
    rows = csv.reader(csvfile, delimiter='\t', quotechar="|")
    for (i, row) in enumerate(rows):
        if (i>-1 and len(row)>1):
            data_lamdas.append(float(row[0]))
            data_reflecs.append(float(row[1]))


c1 = ROOT.TCanvas("c1","",1280,720)
ROOT.gStyle.SetPadLeftMargin(0.05)
ROOT.gStyle.SetPadRightMargin(0.05)
g_sim = ROOT.TGraph(len(sim_lamdas),sim_lamdas,sim_reflecs)
g_sim.SetTitle("")
g_sim.GetYaxis().SetTitle("Reflectivity")
g_sim.GetXaxis().SetTitle("Incident laser wavelength (#mum)")
g_sim.SetLineColor(2)
g_sim.SetLineWidth(2)
g_sim.GetXaxis().CenterTitle(True)
g_sim.GetYaxis().CenterTitle(True)
g_sim.GetYaxis().SetTitleOffset(0.6)
g_sim.Draw("AL")
g_data = ROOT.TGraph(len(data_lamdas),data_lamdas,data_reflecs)
g_data.SetLineColor(1)
g_data.SetLineWidth(2)
g_data.SetMarkerColor(1)
g_data.SetMarkerSize(0.3)
g_data.Draw("Psame")

leg = ROOT.TLegend(0.7,0.65,0.85,0.8)
leg.AddEntry(g_data,"FTIR data","l")
leg.AddEntry(g_sim,"FDTD simulation","l")
leg.Draw("same")

c1.SaveAs("test.root")
c1.SaveAs("test.png")
