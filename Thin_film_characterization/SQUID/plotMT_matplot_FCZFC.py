#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-o',default="./",type=str,help='output directory')
parser.add_argument('--report','-r',default=10000,type=int,help='report every x events')
parser.add_argument('--debug','-d',action="store_true",help='debug mode')
args = parser.parse_args()

for ifile, in_filename in enumerate(args.in_filenames):
    # Read the data from the file
    data = pd.read_csv(in_filename, skiprows=0, delimiter=',')

    # Extract the columns
    temperature = data['Temperature (K)']
    moment = data['DC Moment Fixed Ctr (emu)']
    dc_moment_err = data['DC Moment Err Fixed Ctr (emu)']


    # Split the data into ZFC and FC
    half_length = len(temperature) // 2
    zfc_temperature = temperature[:half_length]
    zfc_moment = moment[:half_length]
    zfc_moment_err = dc_moment_err[:half_length]
    fc_temperature = temperature[half_length:]
    fc_moment = moment[half_length:]
    fc_moment_err = dc_moment_err[half_length:]

    if (in_filename.find('inplane')!=-1): label = 'Magnetic field in plane'
    else: label = 'Magnetic field out of plane'
    # Plot ZFC data
    plt.errorbar(zfc_temperature, zfc_moment, yerr=zfc_moment_err, fmt='o', capsize=3, label='ZFC')
    # plt.errorbar(zfc_temperature, zfc_moment, yerr=zfc_moment_err, fmt='o', capsize=3, label=f'{label}')

    # Plot FC data
    plt.errorbar(fc_temperature, fc_moment, yerr=fc_moment_err, fmt='o', capsize=3, label='FC')

plt.xlabel('Temperature (K)',fontsize=15)
plt.ylabel('DC Moment Fixed Ctr (emu)',fontsize=15)
# plt.title(r'7nm NbN on MgO, RF power=130W, Ar:$N_{2}$=36:0.1(sccm)',fontsize=11)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize='x-large')
plt.grid(True)
plt.grid(True, which='minor', linestyle=':', linewidth='0.5')  # Minor grid
plt.minorticks_on()  # Enable minor ticks
plt.tight_layout()
plt.savefig(f"{args.outputDir}/MT_7nm.png")
plt.xlim(5,10)
plt.ylim(-0.5e-5, 2e-5)
plt.tight_layout()
# plt.savefig(f"{args.outputDir}/MT_flowrate_zoomin.png")
plt.savefig(f"{args.outputDir}/MT_7nm_zoomin.png")
plt.show()
