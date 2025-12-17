#!/usr/bin/env python

import ROOT
from array import array

import argparse
parser = argparse.ArgumentParser(description='draw SQUID measurement results')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
args = parser.parse_args()

for infile in args.in_filenames:
    with open (infile) as f:
        Lines = f.readlines()

    Temps, Resists = array('d'), array('d')
    nPoints=0

    for Line in Lines:
        temp, resist = Line.split()
        Temps.append(float(temp))
        Resists.append(float(resist))
        nPoints = nPoints+1

    c1 = ROOT.TCanvas()
    l = ROOT.TLegend(0.6, 0.55, 0.8, 0.7)
    g = ROOT.TGraph(nPoints, Temps, Resists)

    g.SetTitle("infile")
    g.GetYaxis().SetTitle("Resistance (#Omega)")
    g.GetXaxis().SetTitle("T(K)")
    g.SetLineColor(2)
    g.SetLineWidth(2)
    # g.SetLineStyle(10)
    g.Draw("AC");
    # g_FC.SetMarkerStyle(24)

    # l.AddEntry(g_FC, "FC", "P")
    # l.AddEntry(g_ZFC, "ZFC", "P")
    # l.Draw("same")

    c1.SaveAs(Form("plots/RT_%s.png",infile))
    c1.SaveAs(Form("plots/RT_%s.pdf",infile))
