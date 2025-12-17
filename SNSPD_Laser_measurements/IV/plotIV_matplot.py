#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
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

def sort_temp(temp,var):
    temp_array = np.array(temp)
    var_array = np.array(var)
    sorted_temp_array = np.sort(temp_array)
    sorted_var_array= var_array[temp_array.argsort()]
    return sorted_temp_array, sorted_var_array

def ic_vs_t(T, Ic0, T0):
    return Ic0 * np.exp(-T / T0)

def find_ic(array, threshold, start_index=0):
    # Find the index of the first element exceeding the threshold starting from the given index
    ic_index = np.argmax(array[start_index:] > threshold)
    # # Adjust the deviation_index to the absolute index
    # ic_index += start_index
    # # Check later indices
    # for i in range(ic_index + 1, len(array)):
    #     if array[i] < threshold:
    #         # Recursively call itself with the next index
    #         return find_ic(array, threshold, i + 1)
    # # If no other elements below the threshold are found, return the first deviating element
    return ic_index

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

def multi_plot(temps,Chs,Samples,dfs,x,y,title,xtitle,ytitle,plotDir,savetitle,xmin=-1,xmax=-1,ymin=-1,ymax=-1):
    print(f"processing multiplot {y} vs {x}")
    sort_temps, sort_dfs = sort_temp(temps,dfs)
    pad_xnum = 5
    # Create a multi-pad canvas
    fig, axs = plt.subplots(int(len(dfs)/pad_xnum+1), pad_xnum, figsize=(36, 6*(int(len(dfs)/pad_xnum)+1)), sharex=True, sharey=True)
    fig_combine, ax_combine = plt.subplots(1,2,figsize=(24, 12))
    for i, (ax,temp,Ch,Sample,df) in enumerate(zip(axs.flatten(),sort_temps, Chs, Samples, sort_dfs)):
        # Combine plot
        if (args.hysteresis):
            split_index = len(df) // 2
            df_1 = df.iloc[:split_index]
            df_2 = df.iloc[split_index:split_index*2]
        else:
            df_1 = df
            df_2 = df
        # Multi pad plot
        ax.plot(df_1[x], df_1[y], marker='.', linestyle='-', label=f'{temp}K_{x}_ascending')
        ax.plot(df_2[x], df_2[y], marker='.', linestyle='-', label=f'{temp}K_{x}_descending')
        ax.grid(True)
        ax.set_ylim(0, None)
        ax.set_xlim(0, None)
        ax.legend(fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20) # Set larger ticks and labels
        if i // pad_xnum == int(len(dfs)/pad_xnum):  ax.set_xlabel(xtitle, fontsize=20) # Add x and y titles only to the leftmost and bottom-most subplot
        if i % pad_xnum == 0:  ax.set_ylabel(ytitle, fontsize=20)
        ax_combine[0].plot(df_1[x], df_1[y], marker='.', linestyle='-', label=f'{temp}K')
        ax_combine[1].plot(df_1[x], df_1[y], marker='o', linestyle='-', label=f'{temp}K')
    fig.tight_layout()
    savefig(fig,f"{plotDir}/{savetitle}_multi")

    ax_combine[0].set_ylabel(ytitle,fontsize=25)
    ax_combine[0].set_xlabel(xtitle,fontsize=25)
    ax_combine[0].tick_params(axis='both', which='major', labelsize=20)
    ax_combine[0].set_title(f'{title} Sweep {x} ascending',fontsize=25)
    if (ymin!=-1): ax_combine[1].set_ylim(ymin,ymax)
    if (xmin!=-1): ax_combine[1].set_xlim(xmin,xmax)
    ax_combine[1].set_title(f'{title}_zoomin',fontsize=25)
    ax_combine[1].tick_params(axis='both', which='major', labelsize=20)
    ax_combine[1].set_xlabel(xtitle,fontsize=25)
    ax_combine[1].set_ylabel(ytitle,fontsize=25)
    legend = ax_combine[0].legend(title=f'Sample Temperature',fontsize=20)
    legend.get_title().set_fontsize('20')
    fig_combine.tight_layout()
    savefig(fig_combine,f"{plotDir}/{savetitle}")

def plot(temps,graphs,xtitle,ytitle,plotDir,savetitle,xmin=-1,xmax=-1,ymin=-1,ymax=-1):
    fig,ax = plt.subplots()
    for i,temp in enumerate(temps):
        ax.plot(graphs[temp][0], graphs[temp][1], marker=markers(i), fillstyle='none', linestyle='none', label=f'{temp/SampleTc:.2f}')
    ax.set_ylabel(ytitle,fontsize=15)
    ax.set_xlabel(xtitle,fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(title=r'$T$ / $T_{C}$')
    fig.tight_layout()
    ax.set_xlim(left=5)
    # ax.set_yscale('log')
    savefig(fig,f"{plotDir}/{savetitle}")
    plt.show()
    ax.set_xlim(13,14.5)
    ax.set_ylim(-0.15,0.3)
    savefig(fig,f"{plotDir}/{savetitle}_zoomin")

def ic_plot(temps,dfs):
    ics = []
    for (temp,df) in zip(temps,dfs):
        index = find_ic(df['Resists'], 11000)
        ic = df['Currents'][index]
        ics.append(ic)
    sort_temps, sort_ics = sort_temp(temps,ics)

    # Fit the data
    popt, pcov = curve_fit(ic_vs_t, sort_temps, sort_ics, p0=(1.0, 1.0))

    # Extract the fitted parameters
    y0_fit, x0_fit = popt
    # Generate y values using the fitted parameters
    y_fit = ic_vs_t(sort_temps, y0_fit, x0_fit)

    plt.plot(sort_temps,sort_ics, marker='o', linestyle='None')
    plt.plot(sort_temps, y_fit, 'r-', label=f'Fit: y = {y0_fit:.2f} * exp(-x / {x0_fit:.2f})')
    plt.title('')
    plt.xlabel('Temp(K)', fontsize=15)
    plt.ylabel(r'Critical Current ($\mu$A)', fontsize=15)
    plt.grid(True)
    plt.grid(which='minor', linestyle=':', linewidth='0.5')
    plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.tight_layout()
    plt.savefig("ic")
    plt.close()

def read_file():
    temps, Chs, Samples, Sources, dfs = [],[],[],[],[]
    for in_filename in args.in_filenames:
        print(f"Processing {in_filename}")
        # Info
        temp_str = in_filename.split('/')[-1].split('K')[0].split('_')[-1]
        temp = float(temp_str.replace('p', '.'))
        Sample = in_filename.split('SNSPD_rawdata/')[1].split('/')[0]
        Ch = in_filename.split('SNSPD_rawdata/')[1].split('/')[2]
        Source = in_filename.split('IV_')[1].split('_Source/')[0]
        basename = in_filename.rsplit('/')[-1].split('.txt')[0]
        plotDir = args.outputDir + '/' + Sample + '/IV/' + Ch + '/' + basename
        createDir(plotDir)
        # Read the data from the file
        df = pd.read_csv(in_filename, header=None, delim_whitespace=True, names=['Volts', 'Currents', 'Resists'])
        df['Currents'] *= 1e6
        df['Resists'] *= 1e-3
        df['Volts'] *= 1e3
        # single plots
        # single_plot(df, 'Currents',   'Volts', temp, f"{Sample}-{Ch}", r'Current ($\mu$A)',  'Voltage (mV)',           plotDir, f"IV_{basename}")
        # single_plot(df, 'Volts',   'Currents', temp, f"{Sample}-{Ch}", 'Voltage (mV)',      r'Current ($\mu$A)',       plotDir, f"VI_{basename}")
        # single_plot(df, 'Volts'   , 'Resists', temp, f"{Sample}-{Ch}", r'Voltage (mV)',     r'Resistance (k$\Omega$)', plotDir, f"VR_{basename}")
        # single_plot(df, 'Currents', 'Resists', temp, f"{Sample}-{Ch}", r'Current ($\mu$A)', r'Resistance (k$\Omega$)', plotDir, f"IR_{basename}")
        # single_plot(df, 'Currents', 'Resists', temp, f"{Sample}-{Ch}", r'Current ($\mu$A)', r'Resistance (k$\Omega$)', plotDir, f"IR_zoomin_{basename}",-1,-1,5,25)
        # Append to array
        temps.append(temp)
        Chs.append(Ch)
        Samples.append(Sample)
        Sources.append(Source)
        dfs.append(df)
    return temps, Chs, Samples, Sources, dfs

def main():
    temps, Chs, Samples, Sources, dfs = read_file()
    chDir = args.outputDir + '/' + Samples[0] + '/IV/' + Chs[0]
    if (Sources[0]=="Voltage"):
        plot(temps, Chs, Samples, dfs, 'Volts',  'Currents',  f"{Samples[0]}-{Chs[0]}", 'Voltage (mV)',  r'Current ($\mu$A)',       chDir, f"VI_combine",0,700,-1,-1)
        plot(temps, Chs, Samples, dfs, 'Volts',  'Resists',   f"{Samples[0]}-{Chs[0]}", r'Voltage (mV)', r'Resistance (k$\Omega$)', chDir, f"VR_combine",0,700,5,25)
    elif (Sources[0]=="Current"):
        plot(temps, Chs, Samples, dfs, 'Currents', 'Volts',   f"{Samples[0]}-{Chs[0]}", r'Current ($\mu$A)', 'Voltage (mV)',            chDir, f"IV_combine",0,70,0,700)
        plot(temps, Chs, Samples, dfs, 'Currents', 'Resists', f"{Samples[0]}-{Chs[0]}", r'Current ($\mu$A)', r'Resistance (k$\Omega$)', chDir, f"IR_combine",1,30,9.8,10.5)
    # ic_plot(temps,dfs)

if __name__ == '__main__':
    main()

