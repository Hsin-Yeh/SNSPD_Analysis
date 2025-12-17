#!/usr/bin/env python

import ROOT

if __name__ == "__main__":
    rnd = ROOT.TRandom3()
    h = ROOT.TH1D("h","Poisson data",20,0,20)
    for i in range (100000):
        n = rnd.Poisson(10.5)
        h.Fill(n)
    c1 = ROOT.TCanvas()

    h1 = ROOT.TH1D("h1","h1",300,0,0.3)
    for nPhoton in range (1, 20):
        print(h.GetBinContent(nPhoton))
        for i in range (int(h.GetBinContent(nPhoton))):
            n = rnd.Gaus(0.005+nPhoton*0.01182 , 0.005)
            h1.Fill(n)

    c2 = ROOT.TCanvas()
    h_clone = {}
    for i in range(1,7):
        h_clone[i] = h1.Clone()
        h_clone[i].Rebin(i)
        h_clone[i].GetYaxis().SetTitle(f"Entries / {i}mV")
        h_clone[i].Draw()
        c2.SaveAs(f"./Generated_{i}mVbinning.png")
