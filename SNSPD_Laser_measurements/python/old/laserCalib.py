#!/usr/bin/env python

import ROOT
import csv
import argparse
import datetime
import os
import errno
from array import array

# sample width = 0.4ns
Nheaderlines = 20 #Header lines
eventSampleLength = 1000 # Record length
sampleMin = 190 #sample window minimum
sampleMax = 210 #sample window maximum

parser = argparse.ArgumentParser(description='make basic plots of laser calibration and convert csv to root')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outName','-o',default="default",type=str,help='out Name')
args = parser.parse_args()

if (args.outName=="default"):
    outName = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
else:
    outName = args.outName

# Create output directory
try:
    os.mkdir(f'plots/{outName}')
except OSError as e:
    if e.errno == errno.EEXIST:
        print('Output directory exists.')
    else:
        raise


outfile = ROOT.TFile.Open(f"output_root/{outName}.root","RECREATE")
outTree = ROOT.TTree("ScopeData","ScopeData")
data = array( 'f', [ 0 ] * eventSampleLength)
outTree.Branch('data',data,"data[%d]/F" %(eventSampleLength))

h_sum = ROOT.TH1F("h_sum","Sum of samples",100,-0.05,0.1)
h_range = ROOT.TH1F("h_range","Sample range",100,0,0.02)
h_max = ROOT.TH1F("h_max","Sample max",25,0,0.005)
h_maxTS = ROOT.TH1F("h_maxTS","Sample max timesample",1000,0,1000)
h_value = ROOT.TH1F("h_pedestal","Pedestal",100,-0.005,0.005)

c1 = ROOT.TCanvas()
for ifile,infile in enumerate(args.in_filenames):
    if (infile.find(".csv")==0): continue
    if (ifile%1000==0): print (f"Processed {ifile}/{len(args.in_filenames)} file")

    with open(infile, newline='') as csvfile:
        rows = csv.reader(csvfile)

        max = min = sum = 0
        samplings, timesamples = array( 'd' ), array( 'd' )
        for i, row in enumerate(rows):
            if (i<=Nheaderlines): continue
            if (len(row)<1): continue

            TS = i-Nheaderlines-1
            value = float(row[1])
            data[TS] = value

            # sum range selection
            if (sampleMin!=-1 and TS<sampleMin): continue
            if (sampleMax!=-1 and TS>sampleMax): continue

            samplings.append(value)
            timesamples.append(TS)
            sum += value
            if   value > max:
                max = value
                maxTS = TS
            elif value < min: min = value
            h_value.Fill(value)

        if (ifile == 0):
            c1.SetLeftMargin(0.17)
            g = ROOT.TGraph(len(samplings),timesamples,samplings)
            g.Draw("AL")
            g.SetTitle("Event Display")
            g.GetXaxis().SetTitle("Timesample / 0.4ns")
            g.GetYaxis().SetTitle("Scope value / V")
            g.GetYaxis().SetTitleOffset(1.3)
            c1.SaveAs(f"plots/{outName}/{outName}_evdis.png")
            g.Write()

        h_sum.Fill(sum)
        h_range.Fill(max-min)
        h_max.Fill(max)
        h_maxTS.Fill(maxTS)
        outTree.Fill()

# fit = ROOT.TF1("fit","[0]*exp(-0.5*((x-[1])/[2])**2)+[3]*exp(-0.5*((x-([1]+[5]))/[2])**2)+[4]*exp(-0.5*((x-([1]+2*[5]))/[2])**2)")
# fit.SetParameters(200,-0.1,0.01,100,50,0.03)

# h_sum.Fit("fit")
# fit.Draw("same")
h_sum.Draw("HIST")
c1.SaveAs(f'plots/{outName}/{outName}_sum.png')

h_range.Draw("HIST")
c1.SaveAs(f'plots/{outName}/{outName}_range.png')

h_max.Draw("HIST")
h_max.GetXaxis().SetRangeUser(0,0.08)
h_max.GetXaxis().SetTitle("Scope Value (V)")
c1.SaveAs(f'plots/{outName}/{outName}_max.png')

fit = ROOT.TF1("fit","gaus(0)",-0.01,0.01)
h_value.Fit("fit")
h_value.Draw("HIST")
fit.Draw("same")
c1.SaveAs(f'plots/{outName}/{outName}_allvalue.png')

ROOT.gStyle.SetOptStat(1);
h_maxTS.SetStats(1)
h_maxTS.Draw("HIST")
c1.SaveAs(f'plots/{outName}/{outName}_maxTS.png')


outTree.Write()
h_sum.Write()
h_max.Write()
h_range.Write()
outfile.Close()
