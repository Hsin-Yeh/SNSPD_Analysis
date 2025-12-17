#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import argparse
parser = argparse.ArgumentParser(description='draw IV results')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
args = parser.parse_args()

# Plot initialize
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

temps, files = [], []
for in_filename in args.in_filenames:
    Temperature = float(in_filename.split('/')[-1].split('K')[0])
    temps.append(Temperature)
    files.append(in_filename)
    sorted_data = sorted(zip(temps, files), reverse=True)
    sorted_temps, sorted_files = zip(*sorted_data)

for ifile, (temp, in_filename) in enumerate(zip(sorted_temps, sorted_files)):
    with open (in_filename) as f:
        Lines = f.readlines()

    Currents, Volts, Resists = [], [], []
    for i, Line in enumerate(Lines):
        V, I, R = Line.split()
        Volts.append(float(V))
        Currents.append(float(I)*1e+6)
        Resists.append(float(R))

    ax1.plot(Currents, Volts, '',label=f"{temp}K")
    ax2.plot(Volts, Resists, '',label=f"{temp}K")
    ax3.plot(Currents, Resists, '',label=f"{temp}K")
    ax4.plot(Currents, Volts, '',label=f"{temp}K")

ax1.set_xlabel(r'Current ($\mu$A)', fontsize=15)
ax1.set_ylabel('Voltage (V)', fontsize=15)
ax1.legend(fontsize=13)
ax1.grid(True)
ax1.grid(which='minor', linestyle=':', linewidth='0.5')
ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.tight_layout()
ax1.get_figure()
plt.savefig(f"./IV.png")

ax2.set_xlabel('Voltage (V)', fontsize=15)
ax2.set_ylabel(r'Resistance ($\Omega$)', fontsize=15)
ax2.legend(fontsize=13)
ax2.grid(True)
ax2.grid(which='minor', linestyle=':', linewidth='0.5')
ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.tight_layout()
ax2.get_figure()
plt.savefig(f"./VR.png")

ax3.set_xlabel(r'Current ($\mu$A)', fontsize=15)
ax3.set_ylabel(r'Resistance ($\Omega$)', fontsize=15)
ax3.legend(fontsize=13)
ax3.grid(True)
ax3.grid(which='minor', linestyle=':', linewidth='0.5')
ax3.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax3.yaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.tight_layout()
ax3.get_figure()
plt.savefig(f"./IR.png")

ax4.set_xlabel(r'Current ($\mu$A)', fontsize=15)
ax4.set_ylabel('Voltage (V)', fontsize=15)
ax4.legend(fontsize=13)
ax4.grid(True)
ax4.grid(which='minor', linestyle=':', linewidth='0.5')
ax4.set_xlim(0, 220)
ax4.set_ylim(0, 0.05)
ax4.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax4.yaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.tight_layout()
ax4.get_figure()
plt.savefig(f"./IV_zoomin.png")

plt.show()
