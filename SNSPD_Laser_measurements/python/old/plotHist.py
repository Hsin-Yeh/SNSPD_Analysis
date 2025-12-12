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

########################################
# Poisson fit
########################################
def fit_poisson(peakIndex, norms, xerrors, normErrors):
    print(peakIndex, norms, xerrors, normErrors)

    g_norms = ROOT.TGraphErrors(npeaks, peakIndex, norms, xerrors, normErrors)
    poissonfit =  ROOT.TF1("poissonfit","[0]*TMath::Poisson(x,[1])",4,19);
    poissonfit.SetParameter(0, 100)
    poissonfit.SetParameter(1, 10)
    poissonfit.SetParName(0,"Norm")
    poissonfit.SetParName(1,"mean")

    g_norms.Fit(poissonfit,'R')
    g_norms.GetXaxis().SetTitle("nPhotons")
    g_norms.GetYaxis().SetTitle("Poisson(x)")
    g_norms.SetTitle("")
    g_norms.Draw("APE")
    poissonfit.Draw("same")

    norm_2d = np.column_stack((peakIndex, norms))
    var_x = np.std(norm_2d, axis=0, ddof=1)[0]
    lat=ROOT.TLatex(0.6,0.8, "#sigma_{calc}^{2} = %.3f"%var_x);
    lat.SetNDC(1);
    lat.SetTextSize(0.04);
    lat.Draw()
    # lat=ROOT.TLatex(0.6,0.7, "Fano Factor #frac{#sigma_{calc}^{2}}{#mu_{fit}} = %.3f"%(float(var_x)/float(poissonfit.GetParameter(1))));
    # lat.SetNDC(1);
    # lat.SetTextSize(0.04);
    # lat.Draw()

    c1.SaveAs(f"{args.outputDir}/norms.png")

    g_altnorms = ROOT.TGraphErrors(npeaks, peakIndex, norms, xerrors, normErrors)
    altpoissonfit = ROOT.TF1("f1","[0]*TMath::Power(([1]/[2]),(x/[2]))*(TMath::Exp(-([1]/[2])))/TMath::Gamma((x/[2])+1.)", 4, 19) # Alternative poisson from here https://root-forum.cern.ch/t/fitting-a-poisson-distribution-to-a-histogram/12078
    altpoissonfit.SetParameter(0, 100)
    altpoissonfit.SetParameter(1, 10)
    altpoissonfit.SetParameter(2, 0.9)

    altpoissonfit.SetParLimits(1, 9, 12)
    altpoissonfit.SetParName(0, "Norm")
    altpoissonfit.SetParName(1, "mean")
    altpoissonfit.SetParName(2, "factor")

    g_altnorms.Fit(altpoissonfit,'R')
    g_altnorms.SetTitle("")
    g_altnorms.GetXaxis().SetTitle("nPhotons * factor")
    g_altnorms.GetYaxis().SetTitle("Poisson(x/factor)")
    g_altnorms.Draw("APE")
    altpoissonfit.Draw("same")
    c1.SaveAs(f"{args.outputDir}/altnorms.png")

########################################
# Equidistance Gaussian fit
########################################
def fit_multiGaussians_equidistance(h):
    npeaks = 16
    f1 = ROOT.TF1("f1", fpeaks, 0.02, 0.23, 34)
    f1.SetLineWidth(4)
    f1.SetNpx(10000)
    # f1.SetParameters(100,0.1,0.01,100,0.15,0.01)
    # f1.SetParameters(0.01,0.08,0.003,100,100,100,100,100,100,100)
    f1.SetParameter(0, 0.01) # separation
    f1.SetParameter(1, 0.04) # first peak mean
    f1.SetParameter(2, 10) # norm
    f1.SetParameter(3, 0.003) # width
    f1.SetParameter(4, 100) # norm
    f1.SetParameter(5, 0.003) # width
    f1.SetParameter(6, 100) # norm
    f1.SetParameter(7, 0.003) # width
    f1.SetParameter(8, 100) # norm
    f1.SetParameter(9, 0.003) # width
    f1.SetParameter(10, 100) # norm
    f1.SetParameter(11, 0.003) # width
    f1.SetParameter(12, 100) # norm
    f1.SetParameter(13, 0.003) # width
    f1.SetParameter(14, 100) # norm
    f1.SetParameter(15, 0.003) # width
    f1.SetParameter(16, 100) # norm
    f1.SetParameter(17, 0.003) # width
    f1.SetParameter(18, 100) # norm
    f1.SetParameter(19, 0.003) # width
    f1.SetParameter(20, 100) # norm
    f1.SetParameter(21, 0.003) # width
    f1.SetParameter(22, 100) # norm
    f1.SetParameter(23, 0.003) # width
    f1.SetParameter(24, 100) # norm
    f1.SetParameter(25, 0.003) # width
    f1.SetParameter(26, 100) # norm
    f1.SetParameter(27, 0.003) # width
    f1.SetParameter(28, 100) # norm
    f1.SetParameter(29, 0.003) # width
    f1.SetParameter(30, 100) # norm
    f1.SetParameter(31, 0.003) # width
    f1.SetParameter(32, 100) # norm
    f1.SetParameter(33, 0.003) # width
    f1.SetParameter(34, 100) # norm
    f1.SetParameter(35, 0.003) # width
    f1.SetParameter(36, 100) # norm
    f1.SetParameter(37, 0.003) # width
    f1.SetParameter(38, 100) # norm
    f1.SetParameter(39, 0.003) # width
    f1.SetParameter(40, 100) # norm
    f1.SetParameter(41, 0.003) # width


    f1.SetParLimits(0,  0.001, 0.015) # separation
    f1.SetParLimits(1,  0.04,0.06) # first peak mean
    f1.SetParLimits(2,  0, 1000) # norm
    f1.SetParLimits(3,  0.001, 0.005) # width
    f1.SetParLimits(4,  0, 1000) # norm
    f1.SetParLimits(5,  0.001, 0.005) # width
    f1.SetParLimits(6,  0, 1000) # norm
    f1.SetParLimits(7,  0.001, 0.005) # width
    f1.SetParLimits(8,  0, 1000) # norm
    f1.SetParLimits(9,  0.001, 0.005) # width
    f1.SetParLimits(10, 0, 1000) # norm
    f1.SetParLimits(11, 0.001, 0.005) # width
    f1.SetParLimits(12, 0, 1000) # norm
    f1.SetParLimits(13, 0.001, 0.005) # width
    f1.SetParLimits(14, 0, 1000) # norm
    f1.SetParLimits(15, 0.001, 0.005) # width
    f1.SetParLimits(16, 0, 1000) # norm
    f1.SetParLimits(17, 0.001, 0.005) # width
    f1.SetParLimits(18, 0, 1000) # norm
    f1.SetParLimits(19, 0.001, 0.005) # width
    f1.SetParLimits(20, 0, 1000) # norm
    f1.SetParLimits(21, 0.001, 0.005) # width
    f1.SetParLimits(22, 0, 1000) # norm
    f1.SetParLimits(23, 0.001, 0.005) # width
    f1.SetParLimits(24, 0, 1000) # norm
    f1.SetParLimits(25, 0.001, 0.005) # width
    f1.SetParLimits(26, 0, 1000) # norm
    f1.SetParLimits(27, 0.001, 0.005) # width
    f1.SetParLimits(28, 0, 1000) # norm
    f1.SetParLimits(29, 0.001, 0.005) # width
    f1.SetParLimits(30, 0, 1000) # norm
    f1.SetParLimits(31, 0.001, 0.005) # width
    f1.SetParLimits(32, 0, 1000) # norm
    f1.SetParLimits(33, 0.001, 0.005) # width
    f1.SetParLimits(34, 0, 1000) # norm
    f1.SetParLimits(35, 0.001, 0.005) # width
    f1.SetParLimits(36, 0, 1000) # norm
    f1.SetParLimits(37, 0.001, 0.005) # width
    f1.SetParLimits(38, 0, 1000) # norm
    f1.SetParLimits(39, 0.001, 0.005) # width
    f1.SetParLimits(40, 0, 1000) # norm
    f1.SetParLimits(41, 0.001, 0.005) # width


    ROOT.gStyle.SetOptFit(0000)
    f = {}
    peakIndex, xerrors, norms, means, sigmas, normErrors = array('d'), array('d'), array('d'), array('d'), array('d'), array('d')
    leg = ROOT.TLegend(0.15,0.25,0.3,0.85)
    leg.AddEntry(f1,"Total Fit","l")
    h.Fit(f1,'R')
    h.SetStats(0)
    h.Draw()
    h.GetYaxis().SetTitle("Entries / 3mV")
    for ipeak in range(npeaks):
        fitname = 'f_' + str(ipeak)
        f[ipeak] = ROOT.TF1(fitname, "gaus(0)",0.02,0.23,4)
        norm = float(f1.GetParameter(ipeak*2+2))
        mean = float(f1.GetParameter(0)*ipeak+f1.GetParameter(1))
        sigma = float(f1.GetParameter(ipeak*2+3))
        f[ipeak].SetParameters(norm, mean, sigma)
        f[ipeak].SetLineColor(color(ipeak))
        f[ipeak].SetNpx(10000)
        f[ipeak].Draw("same")
        leg.AddEntry(f[ipeak],"%d photons"%(ipeak+3),"l")
        # Append to array
        norms.append(norm)
        means.append(mean)
        sigmas.append(sigma)
        peakIndex.append(ipeak+3)
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

    c1.SaveAs(f"{args.outputDir}/multiGaussianFit.png")

    ROOT.gStyle.SetOptFit(1)
    g_means = ROOT.TGraphErrors(npeaks, peakIndex, means, xerrors, sigmas)
    g_means.Draw("APE")
    c1.SaveAs(f"{args.outputDir}/means.png")

    return peakIndex, norms, xerrors, normErrors, means, sigmas


def fit_multiGaussians(h):
    # Define the parameter array for the total function.
    par = np.zeros(36)

    g1 = ROOT.TF1("g1", "gaus", 0.05, 0.2)
    g1 = ROOT.TF1("g1", "gaus", 0.05, 0.2)
    g2 = ROOT.TF1("g2", "gaus", 0.05, 0.2)
    g3 = ROOT.TF1("g3", "gaus", 0.05, 0.2)
    g4 = ROOT.TF1("g4", "gaus", 0.05, 0.2)
    g5 = ROOT.TF1("g5", "gaus", 0.05, 0.2)
    g6 = ROOT.TF1("g6", "gaus", 0.05, 0.2)
    g7 = ROOT.TF1("g7", "gaus", 0.05, 0.2)
    g8 = ROOT.TF1("g8", "gaus", 0.05, 0.2)
    g9 = ROOT.TF1("g9", "gaus", 0.05, 0.2)
    g10 = ROOT.TF1("g10", "gaus", 0.05, 0.2)
    g11 = ROOT.TF1("g11", "gaus", 0.05, 0.2)
    g12 = ROOT.TF1("g12", "gaus", 0.05, 0.2)

    # The total is the sum of the three, each has three parameters.
    total = ROOT.TF1("total", "gaus(0)+gaus(3)+gaus(6)+gaus(9)+gaus(12)+gaus(15)+gaus(18)+gaus(21)+gaus(24)+gaus(27)+gaus(30)+gaus(33)", 0.05, 0.2)
    total.SetLineColor(4)
    total.SetLineWidth(3)

    # Fit each function and add it to the list of functions. By default, TH1::Fit()
    # fits the function on the defined histogram range. You can specify the "R"
    # option in the second parameter of TH1::Fit() to restrict the fit to the range
    # specified in the TF1 constructor. Alternatively, you can also specify the
    # range in the call to TH1::Fit(), which we demonstrate here with the 3rd
    # Gaussian. The "+" option needs to be added to the later fits to not replace
    # existing fitted functions in the histogram.
    h.Fit(g1, "", "", 0.058, 0.065)
    h.Fit(g2, "+", "", 0.065, 0.082)
    h.Fit(g3, "+", "", 0.08, 0.09);
    h.Fit(g4, "+", "", 0.092, 0.105);
    h.Fit(g5, "+", "", 0.105, 0.12);
    h.Fit(g6, "+", "", 0.116, 0.132);
    h.Fit(g7, "+", "", 0.128, 0.145);
    h.Fit(g8, "+", "", 0.142, 0.155);
    h.Fit(g9, "+", "", 0.153, 0.163);
    h.Fit(g10, "+", "", 0.165, 0.178);
    h.Fit(g11, "+", "", 0.175, 0.185);
    h.Fit(g12, "+", "", 0.189, 0.2);

    # Get the parameters from the fit.
    g1.GetParameters(par[:3])
    g2.GetParameters(par[3:6])
    g3.GetParameters(par[6:9])
    g4.GetParameters(par[9:12])
    g5.GetParameters(par[12:15])
    g6.GetParameters(par[15:18])
    g7.GetParameters(par[18:21])
    g8.GetParameters(par[21:24])
    g9.GetParameters(par[24:27])
    g10.GetParameters(par[27:30])
    g11.GetParameters(par[30:33])
    g12.GetParameters(par[33:])

    # Use the parameters on the sum.
    total.SetParameters(par)
    h.Fit(total, "+", "", 0.058,0.2)
    h.Draw()
    c1.SaveAs(f"{args.outputDir}/test.png")



if __name__ == "__main__":
    c1 = ROOT.TCanvas()
    for in_filename in args.in_filenames:
        infile = ROOT.TFile.Open(in_filename)
        h_amplitude = infile.Get('signal_range')

        h_amplitude_clone = {}
        for i in range(1,6):
            h_amplitude_clone[i] = h_amplitude.Clone()
            h_amplitude_clone[i].Rebin(i)
            h_amplitude_clone[i].GetYaxis().SetTitle(f"Entries / {i}mV")
            h_amplitude_clone[i].Draw()
            c1.SaveAs(f"{args.outputDir}/signal_range_{i}mVbinning.png")

        # fit_multiGaussians(h_amplitude_clone[3])
        # peakIndex, norms, xerrors, normErrors, means, sigmas = fit_multiGaussians_equidistance(h_amplitude_clone[3])
        # fit_poisson(peakIndex, norms, xerrors, normErrors)
