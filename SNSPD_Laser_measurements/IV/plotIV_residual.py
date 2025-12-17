#!/usr/bin/env python3

import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np

from ..utils.osUtils import createDir
from ..utils.plotUtils import savefig,markers

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
args = parser.parse_args()

def read_file(in_filename,subset=10000000):
    print(f"Processing {in_filename}")
    Vars={}
    # Info
    temp_str = in_filename.split('/')[-1].split('K')[0].split('_')[-1]
    Vars["temp"] = float(temp_str.replace('p', '.'))
    Vars["Sample"] = in_filename.split('SNSPD_rawdata/')[1].split('/')[0]
    Vars["Ch"] = in_filename.split('SNSPD_rawdata/')[1].split('/')[2]
    Vars["Source"] = in_filename.split('IV_')[1].split('_Source/')[0]
    Vars["time"] = in_filename.rsplit('/')[-1].split('K_',)[1].split('.txt')[0]
    Vars["basename"] = in_filename.rsplit('/')[-1].split('.txt')[0]
    # Read the data from the file
    with open(in_filename, 'r') as f:
        # hysteresis = int(f.readline().split('=')[1].strip())
        hysteresis = 2
        df = pd.read_csv(f, delim_whitespace=True, names=['Volts', 'Currents', 'Resists'])
    split_index = len(df) // hysteresis
    df_1 = df.iloc[:split_index].copy()
    df_1["Currents"] *= 1e6
    df_1['Volts'] *= 1e3
    df_1['Resists'] /= 1e3
    subset_df = df_1[df_1['Currents'] < subset]
    return subset_df, Vars

def bare():
    Bare_file="/wk_cms3/wuhsinyeh/SNSPD/SNSPD_rawdata/Bare/IV_Current_Source/Ch0/Bare_Ch0_300K_20240524_1310.txt"
    # Bare_file="/Volumes/HV620S/SNSPD/SNSPD_rawdata/Bare/IV_Current_Source/Ch0/Bare_Ch0_300K_20240524_1310.txt"
    df_bare, Vars_bare = read_file(Bare_file,100000)
    cubic_spline = CubicSpline(df_bare['Currents'], df_bare['Resists'])
    return cubic_spline

def residual(df,cubic_spline):
    fitted_y_data = cubic_spline(df['Currents'])
    residuals = df['Resists'] - fitted_y_data
    return [df['Currents'], residuals]

def find_ic(df):
    isw_1 = df[df['Resists'] > 0.1].iloc[0]['Currents']
    isw_2 = df[df['Resists'] > 352].iloc[0]['Currents']
    width = isw_2 - isw_1
    print(isw_1,isw_2,width)
    return isw_1, isw_2, width

def plot(temps,graphs,xtitle,ytitle,plotDir,savetitle,xmin=-1,xmax=-1,ymin=-1,ymax=-1):
    fig,ax = plt.subplots()
    for i,temp in enumerate(temps):
        # label=f'{temp/SampleTc:.2f}'
        lab=f'{temp}'
        ax.plot(graphs[temp][0], graphs[temp][1], marker=markers(i), fillstyle='none', linestyle='none', label=lab)
    ax.set_ylabel(ytitle,fontsize=15)
    ax.set_xlabel(xtitle,fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.minorticks_on()
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    # ax.legend(title=r'$T$ / $T_{C}$')
    ax.legend(title=r'$T$(K)')
    fig.tight_layout()
    # ax.set_xlim(left=10)
    # ax.set_yscale('log')
    savefig(fig,f"{plotDir}/{savetitle}")
    ax.set_xlim(-10,40)
    ax.set_ylim(9,60)
    savefig(fig,f"{plotDir}/{savetitle}_zoomin")

if __name__ == '__main__':
    SampleTc = 12
    resists, temps, times={}, [], []
    # bare_spline = bare()
    for i, in_filename in enumerate(args.in_filenames):
        df, Vars = read_file(in_filename)
        # resists[Vars["temp"]] = (residual(df,bare_spline))
        resists[Vars["temp"]] = ([df['Currents'],df['Resists']])
        resists[Vars["time"]] = ([df['Currents'],df['Resists']])
        temps.append(Vars["temp"])
        times.append(Vars["time"])
        # find_ic(df)

    temps.sort()
    plotDir = 'plots/' + Vars["Sample"] + '/IV/' + Vars["Ch"]
    createDir(plotDir)
    plot(temps, resists, r"Bias Current ($\mu$A)", r"Resistance (k$\Omega$)", plotDir, "IR_residual")
    # plot(times, resists, r"Bias Current ($\mu$A)", r"Resistance (k$\Omega$)", plotDir, "IR_residual")


# 0.38    127     127.28   0.3
# 0.83:   32.1    37.5    5.4
# 0.85:   29.2    34    4.8
# 0.92:   13.6    21.33    7.73
# 0.96:   6.6     17.36    10.76
