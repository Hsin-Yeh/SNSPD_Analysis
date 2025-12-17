#!/usr/bin/env python

import ROOT
import numpy as np
from array import array
import math

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-d',default="./plots",type=str,help='output directory')
parser.add_argument('--report','-r',default=100,type=int,help='report every x events')
parser.add_argument('--debug_report','-b',default=10000,type=int,help='report every x events')
parser.add_argument('--display_report','-p',default=10000,type=int,help='report every x events')
parser.add_argument('--subset','-s',default=-1,type=int,help='Process a subset of data. -1 = all')
args = parser.parse_args()

npeaks = 1

def color(i):
    colorwheel = [416, 600, 632, 400, 616, 432, 800, 820, 840, 860, 880, 900]
    colorindex = int(i/12) + int(i%12)
    return colorwheel[colorindex]

def fpeaks(x, par):
    # return par[0]*ROOT.TMath.Gaus(x[0],par[1],par[2])
    result = 0
    for p in range(npeaks):
        norm  = par[2*p+2]
        mean  = par[0] * (p) + par[1]
        sigma = par[2*p+3]
        result += norm*ROOT.TMath.Gaus(x[0],mean,sigma)
    return result;

if __name__ == "__main__":
    c1 = ROOT.TCanvas()
    for in_filename in args.in_filenames:
        infile = ROOT.TFile.Open(in_filename)
        h_amplitude = infile.Get('signal_amplitude')
        h_amplitude.Draw()
        h_amplitude.GetXaxis().SetRangeUser(0,0.02)
        c1.SaveAs(f"{args.outputDir}/signal_amplitude_zoomin.png")
        h_amplitude_2 = h_amplitude.Clone()

        # ########################################
        # # Equidistance Gaussian fit
        # ########################################
        npeaks = 5
        f1 = ROOT.TF1("f1", fpeaks, 0.0035, 0.009,12)
        f1.SetLineWidth(3)
        f1.SetNpx(10000)
        # f1.SetParameters(100,0.1,0.01,100,0.15,0.01)
        # f1.SetParameters(0.01,0.08,0.003,100,100,100,100,100,100,100)

        # f1.SetParameter(0, 0.00117) # separation
        # f1.SetParameter(1, 0.0053) # first peak mean

        f1.SetParameter(0, 0.0008) # separation
        f1.SetParameter(1, 0.004) # first peak mean

        f1.SetParameter(2, 38) # norm
        f1.SetParameter(3, 0.00025) # width
        f1.SetParameter(4, 38) # norm
        f1.SetParameter(5, 0.00025) # width
        f1.SetParameter(6, 20) # norm
        f1.SetParameter(7, 0.00025) # width
        f1.SetParameter(8, 20) # norm
        f1.SetParameter(9, 0.00025) # width
        f1.SetParameter(10, 20) # norm
        f1.SetParameter(11, 0.00025) # width
        f1.SetParameter(12, 100) # norm
        f1.SetParameter(13, 0.00025) # width
        # f1.SetParameter(14, 100) # norm
        # f1.SetParameter(15, 0.00025) # width
        # f1.SetParameter(16, 100) # norm
        # f1.SetParameter(17, 0.003) # width
        # f1.SetParameter(18, 100) # norm
        # f1.SetParameter(19, 0.003) # width
        # f1.SetParameter(20, 100) # norm
        # f1.SetParameter(21, 0.003) # width
        # f1.SetParameter(22, 100) # norm
        # f1.SetParameter(23, 0.003) # width
        # f1.SetParameter(24, 100) # norm
        # f1.SetParameter(25, 0.003) # width
        # f1.SetParameter(26, 100) # norm
        # f1.SetParameter(27, 0.003) # width
        # f1.SetParameter(28, 100) # norm
        # f1.SetParameter(29, 0.003) # width
        # f1.SetParameter(30, 100) # norm
        # f1.SetParameter(31, 0.003) # width
        # f1.SetParameter(32, 100) # norm
        # f1.SetParameter(33, 0.003) # width
        # f1.SetParameter(34, 100) # norm
        # f1.SetParameter(35, 0.003) # width
        # f1.SetParameter(36, 100) # norm
        # f1.SetParameter(37, 0.003) # width
        # f1.SetParameter(38, 100) # norm
        # f1.SetParameter(39, 0.003) # width
        # f1.SetParameter(40, 100) # norm
        # f1.SetParameter(41, 0.003) # width


        f1.SetParLimits(0,  0.0007, 0.0011) # separation
        f1.SetParLimits(1,  0.0037,0.005) # first peak mean
        f1.SetParLimits(2,  0, 1000) # norm
        f1.SetParLimits(3,  0.0002, 0.0003) # width
        f1.SetParLimits(4,  0, 1000) # norm
        f1.SetParLimits(5,  0.0002, 0.0003) # width
        f1.SetParLimits(6,  0, 1000) # norm
        f1.SetParLimits(7,  0.0002, 0.0003) # width
        f1.SetParLimits(8,  0, 1000) # norm
        f1.SetParLimits(9,  0.0002, 0.0003) # width
        f1.SetParLimits(10, 0, 1000) # norm
        f1.SetParLimits(11, 0.0002, 0.0003) # width
        # f1.SetParLimits(12, 0, 1000) # norm
        # f1.SetParLimits(13, 0.0002, 0.0003) # width
        # f1.SetParLimits(14, 0, 1000) # norm
        # f1.SetParLimits(15, 0.0002, 0.0003) # width
        # f1.SetParLimits(16, 0, 1000) # norm
        # f1.SetParLimits(17, 0.001, 0.005) # width
        # f1.SetParLimits(18, 0, 1000) # norm
        # f1.SetParLimits(19, 0.001, 0.005) # width
        # f1.SetParLimits(20, 0, 1000) # norm
        # f1.SetParLimits(21, 0.001, 0.005) # width
        # f1.SetParLimits(22, 0, 1000) # norm
        # f1.SetParLimits(23, 0.001, 0.005) # width
        # f1.SetParLimits(24, 0, 1000) # norm
        # f1.SetParLimits(25, 0.001, 0.005) # width
        # f1.SetParLimits(26, 0, 1000) # norm
        # f1.SetParLimits(27, 0.001, 0.005) # width
        # f1.SetParLimits(28, 0, 1000) # norm
        # f1.SetParLimits(29, 0.001, 0.005) # width
        # f1.SetParLimits(30, 0, 1000) # norm
        # f1.SetParLimits(31, 0.001, 0.005) # width
        # f1.SetParLimits(32, 0, 1000) # norm
        # f1.SetParLimits(33, 0.001, 0.005) # width
        # f1.SetParLimits(34, 0, 1000) # norm
        # f1.SetParLimits(35, 0.001, 0.005) # width
        # f1.SetParLimits(36, 0, 1000) # norm
        # f1.SetParLimits(37, 0.001, 0.005) # width
        # f1.SetParLimits(38, 0, 1000) # norm
        # f1.SetParLimits(39, 0.001, 0.005) # width
        # f1.SetParLimits(40, 0, 1000) # norm
        # f1.SetParLimits(41, 0.001, 0.005) # width


        # ROOT.gStyle.SetOptFit(0000)
        f = {}
        peakIndex, xerrors, norms, means, sigmas, normErrors = array('d'), array('d'), array('d'), array('d'), array('d'), array('d')
        leg = ROOT.TLegend(0.65,0.25,0.85,0.65)
        leg.AddEntry(f1,"Total Fit","l")
        h_amplitude_2.Fit(f1,'R')
        h_amplitude_2.SetStats(0)
        h_amplitude_2.SetTitle("")
        h_amplitude_2.Draw()
        h_amplitude_2.GetXaxis().SetRangeUser(0,0.02)
        h_amplitude_2.GetYaxis().SetTitle("Entries / 0.195mV")
        for ipeak in range(npeaks):
            fitname = 'f_' + str(ipeak)
            f[ipeak] = ROOT.TF1(fitname, "gaus(0)",0.002,0.02,4)
            norm = float(f1.GetParameter(ipeak*2+2))
            mean = float(f1.GetParameter(0)*ipeak+f1.GetParameter(1))
            sigma = float(f1.GetParameter(ipeak*2+3))
            f[ipeak].SetParameters(norm, mean, sigma)
            f[ipeak].SetLineColor(color(ipeak))
            f[ipeak].SetNpx(10000)
            f[ipeak].Draw("same")
            leg.AddEntry(f[ipeak],"%d photons"%(ipeak+4),"l")
            # Append to array
            norms.append(norm)
            means.append(mean)
            sigmas.append(sigma)
            peakIndex.append(ipeak+4)
            normErrors.append(f1.GetParError(ipeak*2+2))
            # normErrors.append(math.sqrt(norm))
            # normErrors.append(0)
            xerrors.append(0.5)

        leg.SetFillStyle(0)
        leg.SetBorderSize(0);
        leg.Draw("same")

        chilat=ROOT.TLatex(0.55,0.7, "#chi^{2} / ndf = %.1f / %.1f"%(f1.GetChisquare(),f1.GetNDF()));
        chilat.SetNDC(1);
        chilat.SetTextSize(0.04);
        chilat.Draw()

        lat=ROOT.TLatex(0.55,0.78, "#Delta_{peak} = %.5f #pm %.5f V"%(f1.GetParameter(0),f1.GetParError(0)));
        lat.SetNDC(1);
        lat.SetTextSize(0.04);
        lat.Draw();

        info=ROOT.TLatex(0.15,0.9, "4.6K,   0.5#muW,   532nm pulsed laser,   Binning = 0.5mV / 2^{8} (range/8bits)");
        info.SetNDC(1);
        info.SetTextSize(0.035);
        info.Draw();
        c1.SaveAs(f"{args.outputDir}/multiGaussianFit.png")

        ROOT.gStyle.SetOptFit(1)
        g_means = ROOT.TGraphErrors(npeaks, peakIndex, means, xerrors, sigmas)
        linearfit = ROOT.TF1("linearfit","[0]+[1]*x", 0, 20)
        g_means.Fit(linearfit)
        g_means.Draw("APE")
        linearfit.Draw("same")
        c1.SaveAs(f"{args.outputDir}/means.png")

        # ########################################
        # # Poisson fit
        # ########################################

        g_norms = ROOT.TGraphErrors(npeaks, peakIndex, norms, xerrors, normErrors)
        poissonfit =  ROOT.TF1("poissonfit","[0]*TMath::Poisson(x,[1])",0,9);
        poissonfit.SetParameter(0, 100)
        poissonfit.SetParameter(1, 4)
        poissonfit.SetParName(0,"Norm")
        poissonfit.SetParName(1,"mean")

        g_norms.Fit(poissonfit,'R')
        g_norms.GetXaxis().SetTitle("nPhotons")
        g_norms.GetYaxis().SetTitle("Poisson(x)")
        g_norms.GetXaxis().SetRangeUser(4,19)
        g_norms.SetTitle("")
        poissonfit.Draw()
        g_norms.Draw("PE")
        norm_2d = np.column_stack((peakIndex, norms))
        std_dev_x = np.var(norm_2d, axis=0)[0]
        # lat=ROOT.TLatex(0.6,0.8, "#sigma_{calc}^{2} = %.3f"%std_dev_x);
        # lat.SetNDC(1)
        # lat.Draw()
        # lat=ROOT.TLatex(0.6,0.7, "Fano Factor = #frac{#sigma_{calc}^{2}}{#mu_{fit}} = %.3f"%(float(std_dev_x)/float(poissonfit.GetParameter(1))));
        # lat.SetNDC(1)
        # lat.Draw()

        c1.SaveAs(f"{args.outputDir}/norms.png")

        g_altnorms = ROOT.TGraphErrors(npeaks, peakIndex, norms, xerrors, normErrors)
        altpoissonfit = ROOT.TF1("f1","[0]*TMath::Power(([1]/[2]),(x/[2]))*(TMath::Exp(-([1]/[2])))/TMath::Gamma((x/[2])+1.)", 0, 9) # Alternative poisson from here https://root-forum.cern.ch/t/fitting-a-poisson-distribution-to-a-histogram/12078
        altpoissonfit.SetParameter(0, 100)
        altpoissonfit.SetParameter(1, 4)
        altpoissonfit.SetParameter(2, 0.9)

        altpoissonfit.SetParLimits(1, 0, 12)
        altpoissonfit.SetParName(0, "Norm")
        altpoissonfit.SetParName(1, "mean")
        altpoissonfit.SetParName(2, "factor")

        g_altnorms.Fit(altpoissonfit,'R')
        g_altnorms.SetTitle("")
        altpoissonfit.SetTitle("")
        g_altnorms.GetXaxis().SetTitle("nPhotons * factor")
        g_altnorms.GetYaxis().SetTitle("Poisson(x/factor)")
        altpoissonfit.Draw()
        g_altnorms.Draw("PE")
        info.SetNDC(1);
        info.SetTextSize(0.035);
        info.Draw();
        ROOT.gStyle.SetStatY(0.88);
        ROOT.gStyle.SetStatX(0.88);
        c1.SaveAs(f"{args.outputDir}/altnorms.png")
