import os
import errno
import ROOT
from array import array

import argparse
parser = argparse.ArgumentParser(description='draw SQUID measurement results')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--FC','-f',action="store_true",help='include FC in plot')
args = parser.parse_args()

def color(i):
    if i == 0: return 1
    elif i == 1: return 2
    elif i == 2: return 3
    elif i == 3: return 4
    elif i == 4: return 5
    elif i == 5: return 6
    elif i == 6: return 7
    elif i == 7: return 8
    elif i == 8: return 9
    elif i == 9: return 40
    elif i == 10: return 41
    elif i == 11: return 42
    elif i == 12: return 43
    elif i == 13: return 44
    elif i == 14: return 45
    elif i == 15: return 46
    elif i == 16: return 47
    elif i == 17: return 48
    elif i == 18: return 49
    else: return 0


plotDir= "plots/" + args.in_filenames[0].split("/")[-2]
try:
    os.makedirs(plotDir)
except OSError as e:
    if e.errno == errno.EEXIST:
        print('Plot directory exists.')
    else:
        raise

g_ZFC_list={}
for infile in args.in_filenames:
    with open (infile) as f:
        Lines = f.readlines()

    baseName= infile.rsplit("/",1)[1]

    if args.FC == True:

        Temp_ZFC, Temperr_ZFC, Mt_ZFC, Mterr_ZFC = array('d'), array('d'), array('d'), array('d')
        Temp_FC, Temperr_FC, Mt_FC, Mterr_FC = array('d'), array('d'), array('d'), array('d')

        steps = int((len(Lines)-1)/2)

        for i, Line in enumerate(Lines):
            if (i>0 and i<=steps):
                time, Temp, Oe, Mt, Mterr = Line.split(",")
                Temp_ZFC.append(float(Temp))
                Mt_ZFC.append(float(Mt)*1000)
                Mterr_ZFC.append(float(Mterr))
                Temperr_ZFC.append(0)
            elif (i > steps):
                time, Temp, Oe, Mt, Mterr = Line.split(",")
                Temp_FC.append(float(Temp))
                Mt_FC.append(float(Mt)*1000)
                Mterr_FC.append(float(Mterr))
                Temperr_FC.append(0)

        c1 = ROOT.TCanvas()
        l = ROOT.TLegend(0.65, 0.3, 0.85, 0.5)
        g_FC = ROOT.TGraphErrors(steps, Temp_FC, Mt_FC, Temperr_FC, Mterr_FC)
        g_ZFC_list[baseName] = ROOT.TGraphErrors(steps, Temp_ZFC, Mt_ZFC, Temperr_ZFC, Mterr_ZFC)

        g_ZFC_list[baseName].SetTitle("%s" % (baseName))
        g_ZFC_list[baseName].GetYaxis().SetTitle("M(10^{-3} emu)")
        g_ZFC_list[baseName].GetXaxis().SetTitle("T(K)")
        g_ZFC_list[baseName].SetMarkerColor(4)
        # g_FC.SetMarkerStyle(24)
        g_FC.SetMarkerColor(2)
        # g_FC.SetMarkerStyle(24)
        g_ZFC_list[baseName].Draw("APE")
        g_FC.Draw("PEsame")
        # g_FC.GetYaxis().SetRangeUser(-0.005, 0.005)
        # g_FC.GetXaxis().SetRangeUser(6, 9)

        l.AddEntry(g_FC, "FC", "P")
        l.AddEntry(g_ZFC_list[baseName], "ZFC", "P")
        l.Draw("same")

        c1.Print("%s/MT_%s.png" % (plotDir,baseName))
        c1.Print("%s/MT_%s.pdf" % (plotDir,baseName))

    else:

        Temp_ZFC, Temperr_ZFC, Mt_ZFC, Mterr_ZFC = array('d'), array('d'), array('d'), array('d')

        for i, Line in enumerate(Lines[1:]):
            time, Temp, Oe, Mt, Mterr = Line.split(",")
            Temp_ZFC.append(float(Temp))
            Mt_ZFC.append(float(Mt)*1000)
            Mterr_ZFC.append(float(Mterr))
            Temperr_ZFC.append(0)

        c1 = ROOT.TCanvas()
        l = ROOT.TLegend(0.6, 0.55, 0.8, 0.7)
        g_ZFC_list[baseName] = ROOT.TGraphErrors(len(Temp_ZFC), Temp_ZFC, Mt_ZFC, Temperr_ZFC, Mterr_ZFC)

        g_ZFC_list[baseName].SetTitle("%s" % infile.rsplit("/",1)[1])
        g_ZFC_list[baseName].GetYaxis().SetTitle("M(10^{-3} emu)")
        g_ZFC_list[baseName].GetXaxis().SetTitle("T(K)")
        g_ZFC_list[baseName].SetMarkerColor(4)
        g_ZFC_list[baseName].Draw("APE")

        l.AddEntry(g_ZFC_list[baseName], "ZFC", "P")
        l.Draw("same")

        c1.Print("%s/MT_%s.png" % (plotDir,baseName) )
        c1.Print("%s/MT_%s.pdf" % (plotDir,baseName) )

# c1.SetLogy();
leg_multi = ROOT.TLegend(0.15, 0.7, 0.35, 0.85)
leg_multi.SetTextSize(0.03)
for i, gID in enumerate(g_ZFC_list):
    # g_ZFC_list[gID].SetTitle("Purity 99.5% NbN Target")
    g_ZFC_list[gID].SetTitle("")
    g_ZFC_list[gID].GetYaxis().SetTitle("M(10^{-3} emu)")
    g_ZFC_list[gID].GetXaxis().SetTitle("T(K)")
    g_ZFC_list[gID].SetLineColor(color(i))
    g_ZFC_list[gID].SetMarkerColor(color(i))
    g_ZFC_list[gID].SetMarkerStyle(i+20)
    leg_multi.AddEntry(g_ZFC_list[gID], gID, "P")
    if i == 0:
        g_ZFC_list[gID].Draw("APE")
        g_ZFC_list[gID].GetYaxis().SetRangeUser(-0.2,0.1)
        g_ZFC_list[gID].GetXaxis().SetRangeUser(11,12.2)
    else:
        g_ZFC_list[gID].Draw("Psame")

leg_multi.Draw("same")
c1.Print("%s/MT_multi.png" % plotDir)
