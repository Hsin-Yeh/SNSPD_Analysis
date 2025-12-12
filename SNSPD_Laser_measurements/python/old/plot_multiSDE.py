#!/usr/bin/env python
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
from scipy.special import erf

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-o',default="./output",type=str,help='output directory')
args = parser.parse_args()

def color(i):
    colorwheel = [416, 600, 800, 632, 880, 432, 616, 840, 860, 820, 900]
    # colorindex = int(i/11) + int(i%11)
    return colorwheel[i]

def markerstyle(i):
    style = ["o","v","s"]
    return style[i]

if __name__ == "__main__":
    info = r'$T=4.6K\quad V_{Bias}=2V$'

    temps, files = [], []
    for in_filename in args.in_filenames:
        Temperature = float(in_filename.split('/')[-2].split('uW_')[1].split('K_')[0])
        temps.append(Temperature)
        files.append(in_filename)
    sorted_data = sorted(zip(temps, files), reverse=True)
    sorted_temps, sorted_files = zip(*sorted_data)

    fig1, ax1 = plt.subplots()
    # fig2, ax2 = plt.subplots()

    for ifile, (temp, in_filename) in enumerate(zip(sorted_temps, sorted_files)):
        # Open the file and read its contents
        with open(in_filename, 'r') as file:
            file_content = file.read()

        pattern = r'([\d.]+)mV ; Total Events.+?SDE=([\d.]+) pm ([\d.]+)\nRange Mean fit=([\d.]+) pm ([\d.]+) ; Range std fit=([\d.]+) pm ([\d.]+) ; fit Time jitter=([\d.]+) pm ([\d.]+)\nRange Mean=([\d.]+) pm ([\d.]+) ; Range std=([\d.]+)'
        matches = re.findall(pattern, file_content)

        # Extract the names and corresponding data from the matches
        names                  = np.array([float(match[0]) for match in matches])
        sdes                   = np.array([float(match[1]) for match in matches])
        sde_errors             = np.array([float(match[2]) for match in matches])
        range_mean_fits        = np.array([float(match[3]) for match in matches])
        range_mean_fit_errors  = np.array([float(match[4]) for match in matches])
        range_std_fits         = np.array([float(match[5]) for match in matches])
        range_std_fit_errors   = np.array([float(match[6]) for match in matches])
        fit_time_jitters       = np.array([float(match[7]) for match in matches])
        fit_time_jitter_errors = np.array([float(match[8]) for match in matches])
        range_means            = np.array([float(match[9]) for match in matches])
        range_mean_errors      = np.array([float(match[10]) for match in matches])
        range_stds             = np.array([float(match[11]) for match in matches])
        # Sort the data based on the names
        sorted_data = sorted(zip(names, sdes, sde_errors, range_mean_fits, range_mean_fit_errors, range_std_fits, range_std_fit_errors, fit_time_jitters, fit_time_jitter_errors, range_means, range_mean_errors, range_stds))
        # Unzip the sorted data
        sorted_names, sorted_sdes, sorted_sde_errors, sorted_range_mean_fits, sorted_range_mean_fit_errors, sorted_range_std_fits, sorted_range_std_fit_errors, sorted_fit_time_jitters, sorted_fit_time_jitter_errors, sorted_range_means, sorted_range_mean_errors, sorted_range_stds = zip(*sorted_data)

        # Plot the sde
        ax1.plot(sorted_names, sorted_sdes, 'o-',label=f"{temp}K")
        # ax2.plot(sorted_names, sorted_range_means, 'o',label=f"{temp}K")

    ax1.set_xlabel('Bias Voltage [V]', fontsize=15)
    ax1.set_ylabel('Efficiency', fontsize=15)
    ax1.legend(fontsize=13)
    ax1.grid(True)
    ax1.grid(which='minor', linestyle=':', linewidth='0.5')
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.tight_layout()
    plt.savefig(f"./SDE_temp.png")

    # ax2.set_xlabel('Bias Voltage [V]', fontsize=15)
    # ax2.set_ylabel('Signal Amplitude Mean', fontsize=15)
    # ax2.legend(fontsize=13)
    # ax2.grid(True)
    # ax2.grid(which='minor', linestyle=':', linewidth='0.5')
    # ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    # plt.tight_layout()
    # plt.savefig(f"./Amplitude_temp.png")

    plt.show()
