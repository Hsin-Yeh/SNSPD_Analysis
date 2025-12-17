#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.interpolate import interp1d
import matplotlib.ticker as ticker

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-o',default="./",type=str,help='output directory')
args = parser.parse_args()

# Open the file and read its contents
with open(args.in_filenames[0], 'r') as file:
    file_content = file.read()

# info = r'$T=4.6K,\quad V_{Bias}=2V,\quad 532nm\, pulsed\, laser$'
Temp="4.7"
info = f'{Temp}K' + r'$\quad I_{Laser}=3\mu W,\quad 532nm\, pulsed\, laser$'
outputDir = args.in_filenames[0].rsplit('/',1)[0]


# Define regular expressions to extract data
# pattern = r'([\d.]+)mV ; Total Events.+?SDE=([\d.]+) pm ([\d.]+)\nRange Mean fit=([\d.]+) pm ([\d.]+) ; Range std fit=([\d.]+) pm ([\d.]+) ; fit Time jitter=([\d.]+) pm ([\d.]+)\nRange Mean=([\d.]+) pm ([\d.]+) ; Range std=([\d.]+)\nAmplitude Mean fit=([\d.]+) pm ([\d.]+) ; Amplitude std fit=([\d.]+) pm ([\d.]+)'
pattern = r"(-?\d+)mV ; Total Events : (\d+) ; SDE=([\d.]+) pm ([\d.]+)\nRange Mean fit=([-.\d]+) pm ([\d.]+) ; Range std fit=([-.\d]+) pm ([\d.]+) ; fit Time jitter=([-.\d]+) pm ([\d.]+)\nRange Mean=([-.\d]+) pm ([\d.]+) ; Range std=([-.\d]+)\nAmplitude Mean fit=([-.\d]+) pm ([\d.]+) ; Amplitude std fit=([-.\d]+) pm ([\d.]+)"
matches = re.findall(pattern, file_content)
# Extract the names and corresponding data from the matches
names                     = np.array([-float(match[0]) for match in matches])
sdes                      = np.array([float(match[1]) for match in matches])
sde_errors                = np.array([float(match[2]) for match in matches])
range_mean_fits           = np.array([float(match[3]) for match in matches])
range_mean_fit_errors     = np.array([float(match[4]) for match in matches])
range_std_fits            = np.array([float(match[5]) for match in matches])
range_std_fit_errors      = np.array([float(match[6]) for match in matches])
fit_time_jitters          = np.array([float(match[7]) for match in matches])
fit_time_jitter_errors    = np.array([float(match[8]) for match in matches])
range_means               = np.array([float(match[9]) for match in matches])
range_mean_errors         = np.array([float(match[10]) for match in matches])
range_stds                = np.array([float(match[11]) for match in matches])
amplitude_mean_fit_values = np.array([float(match[12]) for match in matches])
amplitude_mean_fit_errors = np.array([float(match[13]) for match in matches])
amplitude_std_fit_values  = np.array([float(match[14]) for match in matches])
amplitude_std_fit_errors  = np.array([float(match[15]) for match in matches])

# names = names / 1000
# Sort the data based on the names
sorted_data = sorted(zip(names, sdes, sde_errors, range_mean_fits, range_mean_fit_errors, range_std_fits, range_std_fit_errors, fit_time_jitters, fit_time_jitter_errors, range_means, range_mean_errors, range_stds, amplitude_mean_fit_values, amplitude_mean_fit_errors, amplitude_std_fit_values, amplitude_std_fit_errors))
# Unzip the sorted data
sorted_names, sorted_sdes, sorted_sde_errors, sorted_range_mean_fits, sorted_range_mean_fit_errors, sorted_range_std_fits, sorted_range_std_fit_errors, sorted_fit_time_jitters, sorted_fit_time_jitter_errors, sorted_range_means, sorted_range_mean_errors, sorted_range_stds, sorted_amplitude_mean_fits, sorted_amplitude_mean_fit_errors, sorted_amplitude_std_fits, sorted_amplitude_std_fit_errors = zip(*sorted_data)
print(sorted_names, sorted_amplitude_mean_fits, sorted_sdes)

##############################
# Plot the Range Mean
##############################
plt.errorbar(sorted_names, sorted_range_means, yerr=sorted_range_mean_errors, fmt='o')
plt.xlabel('Bias Voltage [V]', fontsize=15)
plt.ylabel(r'Pulse amplitude mean ($\overline{A}_{signal}$)[mV]', fontsize=15)
plt.title(info, loc='right', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(1)
plt.tight_layout()
plt.savefig(f"{outputDir}/Mean.png")
plt.show()

# Plot the Range std
plt.errorbar(sorted_names, sorted_range_stds, yerr=sorted_range_std_fit_errors, fmt='o')
plt.xlabel('Bias Voltage [V]', fontsize=15)
plt.ylabel('Range Std [mV]', fontsize=15)
plt.title(info, loc='right', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(1)
plt.tight_layout()
plt.savefig(f"{outputDir}/Std.png")
plt.show()

# Plot the Range std Fit
plt.errorbar(sorted_names, sorted_range_std_fits, yerr=sorted_range_std_fit_errors, fmt='o')
plt.xlabel('Bias Voltage [V]', fontsize=15)
plt.ylabel('Range Std Fit [mV]', fontsize=15)
plt.title(info, loc='right', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(1)
plt.tight_layout()
plt.savefig(f"{outputDir}/Std_fits.png")
plt.show()

# Plot the Time Jitter Fit
plt.errorbar(sorted_names, sorted_fit_time_jitters, yerr=sorted_fit_time_jitter_errors, fmt='o', label='Data')
# plt.plot(x_fit, y_fit, 'r:', label='Linear Fit')
plt.xlabel('Bias Voltage [V]', fontsize=15)
plt.ylabel('Time Jitter Fit [0.4ns]', fontsize=15)
plt.title(info, loc='right', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0, 0.8)  # Set x-axis limits
plt.grid(1)
plt.savefig(f"{outputDir}/Time_jitter.png")
plt.show()

# Plot the sde
plt.errorbar(sorted_names, sorted_sdes, yerr=sorted_sde_errors, fmt='o', label='Data')
# Fit with linear spline
f = interp1d(sorted_names, sorted_sdes, kind='linear')
# Generate finer x-values for evaluating the fit
x_fine = np.linspace(min(sorted_names), max(sorted_names), 1000)
# Evaluate the fit at finer resolution
y_fine = f(x_fine)
# Find the first index where the fit reaches 0.9
idx = np.where(y_fine >= 0.9)[0][0]
x_90 = x_fine[idx]
y_90 = y_fine[idx]

# Plot the point where fit reaches 0.9
plt.plot(x_90, y_90, 'go', label=f'90% @ {x_90:.2f}V')
plt.axvline(x=x_90, color='g', linestyle='--')

plt.xlabel('Bias Voltage [V]', fontsize=15)
plt.ylabel('Efficiency', fontsize=15)
plt.title(info, loc='right', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{outputDir}/SDE.png")
plt.show()


with open (f'../SNSPD_Basic_measurements/data/20230614_IV/{Temp}K_10kohm_dark.txt') as f:
    Lines = f.readlines()

Currents, Volts, Resists = [], [], []
for i, Line in enumerate(Lines):
    V, I, R = Line.split()
    Volts.append(float(V))
    Currents.append(float(I)*1e+6)
    Resists.append(float(R))

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 8))

ax1.errorbar(sorted_names, sorted_range_means, yerr=sorted_range_mean_errors, fmt='o')
ax2.errorbar(sorted_names, sorted_sdes, yerr=sorted_sde_errors, fmt='o-')
ax2.axvline(x=x_90, color='r', linestyle='--',label=f'90% @ {x_90:.2f}V')
ax3.errorbar(Volts, Resists, fmt='-')
ax1.axvline(x=0.9, color='g', linestyle='--')
ax2.axvline(x=0.9, color='g', linestyle='--')
ax3.axvline(x=0.9, color='g', linestyle='--',label="Critical Point @ 0.9V")
ax3.set_xlim(0.,2.4)
ax3.set_ylim(9800,10200)
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax1.set_ylabel(r'$\overline{A}_{signal}$ [mV]',fontsize=15)
ax2.set_ylabel('Efficiency',fontsize=15)
ax3.set_ylabel('Resistance',fontsize=15)
ax3.set_xlabel('Bias Voltage [V]',fontsize=15)
ax1.set_title(info, loc='right', fontsize=13)
ax1.grid(which='minor', linestyle=':', linewidth='0.5')
ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.grid(which='minor', linestyle=':', linewidth='0.5')
ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax3.grid(which='minor', linestyle=':', linewidth='0.5')
ax3.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax3.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.legend(fontsize='large')
ax3.legend(fontsize='large')
plt.tight_layout()
plt.savefig(f"{outputDir}/Amplitude_SDE_VR_{Temp}K.png")
plt.show()
