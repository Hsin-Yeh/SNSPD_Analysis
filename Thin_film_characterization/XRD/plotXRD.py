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

# Read the data from the provided text
df = pd.read_csv(args.in_filenames[0], delimiter=',', skiprows=32)

# Extract the columns of interest
angle = df['Angle']
intensity = df['Intensity']

# Plot angle versus value
plt.plot(angle, intensity)

# Set the plot labels and title
plt.semilogy(angle, intensity)
plt.xlabel(r'2$\theta$ ($\degree$)',fontsize=15)
plt.ylabel('Intensity (a.u.)',fontsize=15)
# plt.title('Angle vs Intensity (log scale)')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True)
plt.grid(True, which='minor', linestyle=':', linewidth='0.5')  # Minor grid
plt.minorticks_on()  # Enable minor ticks
plt.tight_layout()
plt.savefig(f"{args.outputDir}/XRD_7nm.png")
plt.xlim(30,50)
info="MgO(200)"
plt.text(43.5, 1e7, info, fontsize=13, horizontalalignment='left')
info=r"$\delta$-NbN(200)"
plt.text(41.5, 3e5, info, fontsize=13, horizontalalignment='right')
# plt.ylim(-0.5e-5, 2e-5)
plt.tight_layout()
# plt.savefig(f"{args.outputDir}/MT_flowrate_zoomin.png")
plt.savefig(f"{args.outputDir}/XRD_7nm_zoomin.png")
# Show the plot
plt.show()
