#!/usr/bin/env python

import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-o',default="./output",type=str,help='output directory')
args = parser.parse_args()

# Open the file and read its contents
with open(args.in_filenames[0], 'r') as file:
    file_content = file.read()

# Use regular expressions to extract the required data
pattern = r'([\d.]+)degrees ; Total Events.+?SDE=([\d.]+) pm ([\d.]+)\nRange Mean fit=([\d.]+) pm ([\d.]+) ; Range std fit=([\d.]+) pm ([\d.]+) ; fit Time jitter=([\d.]+) pm ([\d.]+)\nRange Mean=([\d.]+) pm ([\d.]+) ; Range std=([\d.]+)'
matches = re.findall(pattern, file_content)
print(matches)
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

##############################
# Plot the Range Mean
##############################
plt.errorbar(sorted_names, sorted_range_means, yerr=sorted_range_mean_errors, fmt='o')
plt.xlabel('Bias Voltage [V]', fontsize=15)
plt.ylabel('Range Mean [mV]', fontsize=15)
plt.title('', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(1)
plt.tight_layout()
plt.savefig(f"{args.outputDir}/Mean.png")
plt.show()

# Plot the Range std
plt.errorbar(sorted_names, sorted_range_stds, yerr=sorted_range_std_fit_errors, fmt='o')
plt.xlabel('Bias Voltage [V]', fontsize=15)
plt.ylabel('Range Std [mV]', fontsize=15)
plt.title('', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(1)
plt.tight_layout()
plt.savefig(f"{args.outputDir}/Std.png")
plt.show()

# Plot the Range std Fit
plt.errorbar(sorted_names, sorted_range_std_fits, yerr=sorted_range_std_fit_errors, fmt='o')
plt.xlabel('Bias Voltage [V]', fontsize=15)
plt.ylabel('Range Std Fit [mV]', fontsize=15)
plt.title('', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(1)
plt.tight_layout()
plt.savefig(f"{args.outputDir}/Std_fits.png")
plt.show()

# Plot the Time Jitter Fit
# sorted_fit_time_jitters = np.array(sorted_fit_time_jitters)
# # Fit the data with a linear fit for x < 20
# mask = sorted_names > 2
# x_fit = sorted_names[mask]
# y_fit = sorted_fit_time_jitters[mask]
# fit_coeffs = np.polyfit(x_fit, y_fit, 0)
# fit_line = np.polyval(fit_coeffs, sorted_names)
# x_fit = np.linspace(min(x_fit), max(x_fit), 100)
# y_fit = np.polyval(fit_coeffs, x_fit)

plt.errorbar(sorted_names, sorted_fit_time_jitters, yerr=sorted_fit_time_jitter_errors, fmt='o', label='Data')
# plt.plot(x_fit, y_fit, 'r:', label='Linear Fit')
plt.xlabel('Bias Voltage [V]', fontsize=15)
plt.ylabel('Time Jitter Fit [0.4ns]', fontsize=15)
plt.title('', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(1)
# fit_results = f'Average Time Jitter = {fit_coeffs[0]:.3f} (index)\n                                   {fit_coeffs[0]*0.4:.3f} (ns)'
# plt.text(35, 0.25, fit_results, fontsize=15, horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
# plt.tight_layout()
# # Specify the coordinates and dimensions of the zoomed-in plot within the main plot
# zoom_coords = [0.6, 0.6, 0.3, 0.3]  # [left, bottom, width, height]
# # Create the zoomed-in plot in the specified region
# ax_zoom = plt.axes(zoom_coords)
# plt.errorbar(sorted_names, sorted_fit_time_jitters, yerr=sorted_fit_time_jitter_errors, fmt='o', label='Data')
# plt.plot(x_fit, y_fit, 'r:', label='Linear Fit')
# ax_zoom.set_ylim(min(y_fit)*0.9, max(y_fit)*1.1)  # Adjust the x-axis limits as desired
# ax_zoom.set_title('Zoomed-in Plot for x > 2 ')
# ax_zoom.grid(True)
# plt.savefig(f"{args.outputDir}/Time_jitter.png")
plt.show()

# Plot the sde
plt.errorbar(sorted_names, sorted_sdes, yerr=sorted_sde_errors, fmt='o')
plt.xlabel('Bias Voltage [V]', fontsize=15)
plt.ylabel('Efficiency', fontsize=15)
plt.title('', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{args.outputDir}/SDE.png")
plt.show()
