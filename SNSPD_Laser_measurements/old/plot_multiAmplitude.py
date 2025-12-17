#!/usr/bin/env python

import ROOT
import argparse

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

if __name__ == "__main__":
    info = r'$T=4.6K\quad V_{Bias}=2V$'

    c1 = ROOT.TCanvas("c1","c1",1500,600)
    leg = ROOT.TLegend(0.6,0.45,0.85,0.85)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0);

    # Sort file
    params, files = [], []
    for in_filename in args.in_filenames:
        # param = float(in_filename.split('/')[-1].split('uW')[0])
        param = float(in_filename.split('/')[-1].split('uW')[0])
        if (param<0.4):continue
        params.append(param)
        files.append(in_filename)
    sorted_data = sorted(zip(params, files))
    sorted_params, sorted_files = zip(*sorted_data)

    h_amplitude = {}
    for ifile, (param, in_filename) in enumerate(zip(sorted_params, sorted_files)):
        print (param)
        infile = ROOT.TFile.Open(in_filename)
        h_amplitude[param] = infile.Get('signal_amplitude')
        h_amplitude[param].GetYaxis().SetTitleOffset(0.7)
        h_amplitude[param].GetYaxis().SetTitle("Normalized Entries / 0.195mV")
        h_amplitude[param].GetXaxis().SetTitle("Signal Amplitude [V]")
        h_amplitude[param].SetTitle("")
        h_amplitude[param].SetDirectory(0)
        h_amplitude[param].SetLineColor(color(ifile))
        h_amplitude[param].GetXaxis().SetRangeUser(0,0.06)
        factor = 1
        h_amplitude[param].Scale(factor/h_amplitude[param].GetEntries());
        leg.AddEntry(h_amplitude[param], f"{param}uW", "l")
        # Draw
        if (ifile==0): h_amplitude[param].Draw("HIST")
        else: h_amplitude[param].Draw("HISTsame")
        infile.Close()

    # Plot
    leg.Draw()

    infoText=ROOT.TLatex(0.35,0.92, "T=4.6K, V_{Bias}=2V, 532nm pulsed laser, 2.5MHz Repetition rate");
    # infoText=ROOT.TLatex(0.3,0.92, "T=4.6K, I_{Laser}=3uW, 532nm pulsed laser, 2.5MHz Repetition rate");
    infoText.SetNDC(1);
    # infoText.SetTextFont(60);
    # infoText.SetLineColor(0);
    infoText.SetLineStyle(2);
    # infoText.SetLineWidth(1);
    # infoText.SetTextSize(0.04);
    infoText.Draw("same");

    outputDir = args.in_filenames[0].rsplit('/',2)[0]
    c1.SaveAs(f"{outputDir}/multiAmplitude.png")
