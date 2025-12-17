#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-o',default="./output",type=str,help='output directory')
parser.add_argument('--report','-r',default=10000,type=int,help='report every x events')
parser.add_argument('--debug','-d',action="store_true",help='debug mode')
args = parser.parse_args()

for ifile, in_filename in enumerate(args.in_filenames):
    # Read the data from the provided text
    df = pd.read_csv(in_filename, delimiter=',')

    # Extract the columns of interest
    temperature = df['Temperature (K)']
    dc_moment = df['DC Moment Fixed Ctr (emu)']
    dc_moment_err = df['DC Moment Err Fixed Ctr (emu)']

    power = in_filename.split('/')[-1].split('W-')[0].split('-')[-1]
    # N2 = in_filename.split('/')[-1].split('-')[0].split('_')[-1]
    # Ar = in_filename.split('/')[-1].split('-')[0].split('_')[0].split('NbN')[1]
    # Plot DC moment Fixed Ctr with error vs temperature
    # plt.errorbar(temperature, dc_moment, yerr=dc_moment_err, fmt='o', capsize=3, label=f"{Ar}:{N2}")
    plt.errorbar(temperature, dc_moment, yerr=dc_moment_err, fmt='o', capsize=3, label=f"{power}")

# Set the plot labels and title
plt.xlabel('Temperature (K)',fontsize=15)
plt.ylabel('DC Moment Fixed Ctr (emu)',fontsize=15)
# plt.title('RF Power = 120W',fontsize=20)
plt.title(r'Ar:$N_{2}$ = 12:0.2',fontsize=20)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
# plt.legend(title=r'      Ar:$N_{2}$')
plt.legend(title=r'RF Power (W)')
plt.grid(True)
plt.grid(True, which='minor', linestyle=':', linewidth='0.5')  # Minor grid
plt.minorticks_on()  # Enable minor ticks
plt.tight_layout()
# plt.savefig(f"{args.outputDir}/MT_flowrate.png")
plt.savefig(f"{args.outputDir}/MT_power.png")
# plt.title('RF Power = 120W',fontsize=20)
# plt.xlim(6,14)
plt.xlim(10,13)
plt.ylim(-0.00002, 0.000005)
plt.tight_layout()
# plt.savefig(f"{args.outputDir}/MT_flowrate_zoomin.png")
plt.savefig(f"{args.outputDir}/MT_power_zoomin.png")
# Show the plot
plt.show()
