#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='draw IV results')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-o',default="./plots",type=str,help='output directory')
parser.add_argument('--hysteresis',action="store_true",help='It is a hysteresis data taking')
args = parser.parse_args()

from ..utils.osUtils import createDir
from ..utils.plotUtils import savefig

def single_plot(df,x,y,temp,title,xlabel,ylabel,plotDir,savetitle,xmin=-1,xmax=-1,ymin=-1,ymax=-1):
    fig, ax = plt.subplots()
    ax.plot(df[x], df[y], label=f"{temp}K", marker='.', linestyle='None')
    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    if (xmin!=-1): ax.set_xlim(xmin,xmax)
    if (ymin!=-1): ax.set_ylim(ymin,ymax)
    ax.legend(fontsize=13)
    ax.grid(True)
    ax.grid(which='minor', linestyle=':', linewidth='0.5')
    fig.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    savefig(fig,f"{plotDir}/{savetitle}.png")
    plt.close()

def single_plot_with_fit(spline,df,x,y,temp,title,xlabel,ylabel,plotDir,savetitle,xmin=-1,xmax=-1,ymin=-1,ymax=-1):
    fig, ax = plt.subplots()
    ax.plot(df[x], df[y], label=f"{temp}K", marker='.', linestyle='None')
    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    if (xmin!=-1): ax.set_xlim(xmin,xmax)
    if (ymin!=-1): ax.set_ylim(ymin,ymax)
    ax.legend(fontsize=13)
    ax.grid(True)
    ax.grid(which='minor', linestyle=':', linewidth='0.5')
    fig.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()

    # Plot the fitted curve
    x_fit = np.linspace(min(df["Currents"]), max(df["Currents"]), 1000)
    # y_fit = decaying_exponential(x_fit, *params)
    y_fit = spline(x_fit)
    ax.plot(x_fit, y_fit, label='Fitted curve', color='red')

    savefig(fig,f"{plotDir}/{savetitle}.png")
    plt.close()

def read_file(in_filename,subset):
    print(f"Processing {in_filename}")
    Vars={}
    # Info
    temp_str = in_filename.split('/')[-1].split('K')[0].split('_')[-1]
    Vars["temp"] = float(temp_str.replace('p', '.'))
    Vars["Sample"] = in_filename.split('SNSPD_rawdata/')[1].split('/')[0]
    Vars["Ch"] = in_filename.split('SNSPD_rawdata/')[1].split('/')[2]
    Vars["Source"] = in_filename.split('IV_')[1].split('_Source/')[0]
    Vars["basename"] = in_filename.rsplit('/')[-1].split('.txt')[0]
    Vars["plotDir"] = args.outputDir + '/' + Vars['Sample'] + '/IV/' + Vars['Ch'] + '/' + Vars['basename']
    createDir(Vars['plotDir'])
    # Read the data from the file
    df = pd.read_csv(in_filename, header=None, delim_whitespace=True, names=['Volts', 'Currents', 'Resists'])
    split_index = len(df) // 4
    df_1 = df.iloc[:split_index].copy()
    df_1["Currents"] *= 1e6
    df_1['Resists'] *= 1e-3
    df_1['Volts'] *= 1e3
    subset_df = df_1[df_1['Currents'] < subset]
    return subset_df, Vars

# Define a decaying exponential function
def decaying_exponential(x, a, b, c):
    return a * np.exp(-b * x) + c


def fit_baseline(df,Vars):
    initial_guess = [1, 1, 1] # Initial guess for the parameters
    params, params_covariance = curve_fit(decaying_exponential, df['Currents'], df['Resists'], p0=initial_guess) # Perform the curve fitting
    spline = UnivariateSpline(df['Currents'], df['Resists'], s=1)  # s is the smoothing factor
    cubic_spline = CubicSpline(df['Currents'], df['Resists'])
    single_plot_with_fit(cubic_spline, df, 'Currents', 'Resists', Vars['temp'], f"{Vars['Sample']}-{Vars['Ch']}", r'Current ($\mu$A)', r'Resistance (k$\Omega$)', Vars['plotDir'], f"IR_{Vars['basename']}")

if __name__ == '__main__':
    for in_filename in args.in_filenames:
        df, Vars = read_file(in_filename)
        fit_baseline(df,Vars)
