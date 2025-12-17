#!/usr/bin/env python3

import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd

from .plot_multiHistograms import *
from ..utils.osUtils import createDir
from ..utils.plotUtils import savefig

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
args = parser.parse_args()


def plot(statname, ytit="", plotDir="./"):
    createDir(f"{plotDir}/Compare")
    yvals,xvals=[],[]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    for temp in Vars["temp"]:
        for bv,n_bv in zip(Vars["bv"],Vars["n_bv"]):
            key = getkey(15555 if args.res else 1555,bv,Vars["pol"][0],temp)
            try:
                value = Stats[key][statname]
                yvals.append(value)
                xvals.append(bc)
            except KeyError:
                print(key)
                continue
    savefig(fig,f"{plotDir}/Compare/{statname}")

if __name__ == '__main__':
    Vars, Vcs, Ics, Stats = {},{},{},{}
    Vars["pho"], Vars["bv"], Vars["bc"], Vars["n_pho"], Vars["n_bv"], Vars["n_bc"], Vars["pol"], Vars["temp"] = [],[],[],[],[],[],[],[] # List for sweep variables
    SampleTc = 12
    SampleT=11.5 if args.res else 4.7
    SampleT_norm = SampleT/SampleTc
    Vcs["4p68"] = 1060
    Vcs["11p47K"] = 130
    Ics["4p68"] = 106e-6
    Ics["11p47K"] = 13e-6

    for i, in_filename in enumerate(args.in_filenames):
        print (f"{i}/{len(args.in_filenames)}: {in_filename}")
        runFlag, info = getvars(in_filename,Vars,Vcs,Ics,subsets)
        if (runFlag==False): continue
        key = getkey(info['photon_number'],info['bias_voltage'],info['polarization'],info['sample_temperature'])
        print(key)
        stat=readstat(in_filename,key)
        Stats[key]=stat
    SortVars(Vars)
    plots() # Plot them together
