#!/usr/bin/env python

import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-o',default="./output",type=str,help='output directory')
args = parser.parse_args()

# Open the file and read its contents
with open(args.in_filenames[0], 'r') as file:
    file_content = file.read()

info = r'$T=4.6K,\quad V_{Bias}=2V,\quad 532nm\, pulsed\, laser$'
outputDir = args.in_filenames[0].rsplit('/',1)[0]

# Use regular expressions to extract the required data
pattern = r'([\d.]+)uW ; Total Events.+?SDE=([\d.]+) pm ([\d.]+)\nRange Mean fit=([\d.]+) pm ([\d.]+) ; Range std fit=([\d.]+) pm ([\d.]+) ; fit Time jitter=([\d.]+) pm ([\d.]+)\nRange Mean=([\d.]+) pm ([\d.]+) ; Range std=([\d.]+)'
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
print(sorted_names, sorted_sdes)

# Resolution Plot
range_std_ratio = [std / mean for std, mean in zip(sorted_range_std_fits, sorted_range_mean_fits)]
# Define the relative resolution fit function
def func(x, stochastic, noise, constant):
    return np.sqrt((stochastic/np.sqrt(x))**2 + (noise/x)**2 + constant**2)
# Perform the curve fit
popt, pcov = curve_fit(func, sorted_range_mean_fits, range_std_ratio, bounds=(0, np.inf))
# Extract the optimized parameters
stochastic_opt, noise_opt, constant_opt = popt
# Generate x values for the curve
x_curve = np.linspace(min(sorted_range_mean_fits), max(sorted_range_mean_fits), 100)
# Calculate the y values for the curve fit
y_curve = func(x_curve, stochastic_opt, noise_opt, constant_opt)
# plot
fig, ax = plt.subplots()
plt.errorbar(sorted_range_mean_fits, range_std_ratio, xerr=sorted_range_mean_fit_errors, yerr=sorted_range_std_fit_errors, fmt='o')
plt.plot(x_curve, y_curve, 'r-', label='Curve Fit')
plt.xlabel(r'Signal Amplitude Mean ($\overline{A}_{signal}$) [V]', fontsize=15)
plt.ylabel(r'Relative resolution ($\sigma_{A_{signal}}\, /\, \overline{A}_{signal}$)', fontsize=15)
plt.title('', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(1)
# Add fit results as text annotations
fit_function = r'$(\frac{\sigma}{\mu})^2 = (\frac{Stochastic}{\sqrt{\mu}})^2 + (\frac{Noise}{\mu})^2 + (c)^2$'
fit_results = f'\nStochastic = {stochastic_opt:.4f}\nNoise = {noise_opt:.4f}\nc = {constant_opt:.4f}'
plt.text(0.05, 0.21, fit_function + fit_results, fontsize=15, horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
plt.text(max(sorted_range_mean_fits), max(range_std_ratio)*1.05, info, fontsize=13, horizontalalignment='right')
plt.tight_layout()
plt.savefig(f"{outputDir}/Resolution.png")
plt.show()

##############################
# Plot the Range Mean Fit
##############################
# Define the x and y data
sorted_names = np.array(sorted_names)
sorted_range_mean_fits = np.array(sorted_range_mean_fits)
# Fit the data with a linear fit for x < 20
mask = sorted_names < 33
x_fit = sorted_names[mask]
y_fit = sorted_range_mean_fits[mask]
fit_coeffs = np.polyfit(x_fit, y_fit, 2)
fit_line = np.polyval(fit_coeffs, sorted_names)
x_fit = np.linspace(0, 33, 500)
y_fit = np.polyval(fit_coeffs, x_fit)
# Create a plot with the curve and data
plt.errorbar(sorted_names, sorted_range_mean_fits, yerr=sorted_range_mean_errors, fmt='o', label='Data')
plt.plot(x_fit, y_fit, 'r:', label='Quadratic Fit')
plt.xlabel(r'Power Meter Intensity ($I_{Laser}$) [$\mu$W]', fontsize=15)
plt.ylabel(r'Signal amplitude mean ($\overline{A}_{signal}$) [V]', fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
# plt.legend()
fit_function = r'$a + bx + cx^2$'
fit_results = f'\na = {fit_coeffs[2]:.1E}\nb = {fit_coeffs[1]:.1E}\nc = {fit_coeffs[0]:.1E}'
plt.text(20, 0.01, fit_function + fit_results, fontsize=15, horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
plt.text(max(sorted_names), max(sorted_range_mean_fits)*1.07, info, fontsize=13, horizontalalignment='right')
plt.grid(True)
plt.tight_layout()
# Specify the coordinates and dimensions of the zoomed-in plot within the main plot
zoom_coords = [0.6, 0.2, 0.3, 0.3]  # [left, bottom, width, height]
# Create the zoomed-in plot in the specified region
ax_zoom = plt.axes(zoom_coords)
plt.errorbar(sorted_names, sorted_range_mean_fits, yerr=sorted_range_mean_fit_errors, fmt='o', label='Data')
plt.plot(x_fit, y_fit, 'r:', label='Quadratic Fit')
ax_zoom.set_xlim(0, 33)  # Adjust the x-axis limits as desired
ax_zoom.set_ylim(0, max(y_fit) * 1.1)  # Adjust the y-axis limits as desired
ax_zoom.set_title('Zoomed-in Plot for x < 32')
ax_zoom.grid(True)
ax_zoom.legend()
plt.savefig(f"{outputDir}/Mean_Fits.png")
plt.show()

##############################
# Plot the Range Mean
##############################
plt.errorbar(sorted_names, sorted_range_means, yerr=sorted_range_mean_errors, fmt='o')
print(sorted_names, sorted_range_means)
plt.xlabel(r'Power Meter Intensity ($I_{Laser}$) [$\mu$W]', fontsize=15)
plt.ylabel('Range Mean [V]', fontsize=15)
plt.title('', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(1)
plt.text(max(sorted_names), max(sorted_range_means)*1.07, info, fontsize=13, horizontalalignment='right')
plt.tight_layout()
plt.savefig(f"{outputDir}/Mean.png")
plt.show()

# Plot the Range std
plt.errorbar(sorted_names, sorted_range_stds, yerr=sorted_range_std_fit_errors, fmt='o')
print(sorted_names, sorted_range_stds)
plt.xlabel(r'Power Meter Intensity ($I_{Laser}$) [$\mu$W]', fontsize=15)
plt.ylabel('Range Std [V]', fontsize=15)
plt.title('', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(1)
plt.text(max(sorted_names), max(sorted_range_stds)*1.06, info, fontsize=13, horizontalalignment='right')
plt.tight_layout()
plt.savefig(f"{outputDir}/Std.png")
plt.show()

# Plot the Range std Fit
plt.errorbar(sorted_names, sorted_range_std_fits, yerr=sorted_range_std_fit_errors, fmt='o')
plt.xlabel(r'Power Meter Intensity ($I_{Laser}$) [$\mu$W]', fontsize=15)
plt.ylabel('Range Std Fit [V]', fontsize=15)
plt.title('', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(1)
plt.text(max(sorted_names), max(sorted_range_std_fits)*1.12, info, fontsize=13, horizontalalignment='right')
plt.tight_layout()
plt.savefig(f"{outputDir}/Std_fits.png")
plt.show()

# Plot the Time Jitter Fit
sorted_fit_time_jitters = np.array(sorted_fit_time_jitters)
# Fit the data with a linear fit for x < 20
mask = sorted_names > 2
x_fit = sorted_names[mask]
y_fit = sorted_fit_time_jitters[mask]
fit_coeffs = np.polyfit(x_fit, y_fit, 0)
fit_line = np.polyval(fit_coeffs, sorted_names)
x_fit = np.linspace(min(x_fit), max(x_fit), 100)
y_fit = np.polyval(fit_coeffs, x_fit)

plt.errorbar(sorted_names, sorted_fit_time_jitters, yerr=sorted_fit_time_jitter_errors, fmt='o', label='Data')
plt.plot(x_fit, y_fit, 'r:', label='Linear Fit')
plt.xlabel(r'Power Meter Intensity ($I_{Laser}$) [$\mu$W]', fontsize=15)
plt.ylabel('Time Jitter Fit [0.4ns]', fontsize=15)
plt.title('', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(1)
fit_results = f'Time Jitter > 2uW = {fit_coeffs[0]:.3f} (index)\n                                 {fit_coeffs[0]*0.4:.3f} (ns)'
plt.text(35, 0.23, fit_results, fontsize=15, horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
plt.text(max(sorted_names), 0.525, info, fontsize=13, horizontalalignment='right')
plt.tight_layout()
# Specify the coordinates and dimensions of the zoomed-in plot within the main plot
zoom_coords = [0.6, 0.55, 0.3, 0.3]  # [left, bottom, width, height]
# Create the zoomed-in plot in the specified region
ax_zoom = plt.axes(zoom_coords)
plt.errorbar(sorted_names, sorted_fit_time_jitters, yerr=sorted_fit_time_jitter_errors, fmt='o', label='Data')
plt.plot(x_fit, y_fit, 'r:', label='Linear Fit')
ax_zoom.set_ylim(min(y_fit)*0.9, max(y_fit)*1.1)  # Adjust the x-axis limits as desired
ax_zoom.set_title('Zoomed-in Plot for x > 2 ')
ax_zoom.grid(True)
plt.savefig(f"{outputDir}/Time_jitter.png")
plt.show()

# Plot the sde
plt.errorbar(sorted_names, sorted_sdes, yerr=sorted_sde_errors, fmt='o', label='Data')
# Define the erf function
def erf_func(x, a, b):
    return erf(a * x) + b
# Perform the fit
fit_params, _ = curve_fit(erf_func, sorted_names, sorted_sdes)
# Generate a finer x grid for plotting the fit curve
x_fit = np.linspace(-0.1, max(sorted_names), 1000)
# Calculate the fit curve
y_fit = erf_func(x_fit, *fit_params)
# Find the x position where the function has a value of 0.9
x_0_9 = x_fit[np.abs(y_fit - 0.9).argmin()]

plt.plot(x_fit, y_fit, 'r:', label='Erf Function Fit')
plt.xlabel(r'Power Meter Intensity ($I_{Laser}$) [$\mu$W]', fontsize=15)
plt.ylabel('Efficiency', fontsize=15)
plt.title('', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.text(max(sorted_names), max(sorted_sdes)*1.07, info, fontsize=13, horizontalalignment='right')
fit_results = 'Turn on (90%) at ' f'{x_0_9:.3f}' + r'$\mu$W'
plt.text(68, 0.1, fit_results, fontsize=15, horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
plt.legend()
plt.tight_layout()
# Specify the coordinates and dimensions of the zoomed-in plot within the main plot
zoom_coords = [0.55, 0.43, 0.35, 0.35]  # [left, bottom, width, height]
# Create the zoomed-in plot in the specified region
ax_zoom = plt.axes(zoom_coords)
plt.errorbar(sorted_names, sorted_sdes, yerr=sorted_sde_errors, fmt='o', label='Data')
plt.plot(x_fit, y_fit, 'r:', label='Linear Fit')
plt.axhline(y=0.9, color='green', linestyle='--', label='90% Horizontal Line')
plt.axvline(x=x_0_9, color='green', linestyle='--', label='90% Vertical Line')
ax_zoom.set_xlim(-2, 5)  # Adjust the x-axis limits as desired
ax_zoom.grid(True)
plt.savefig(f"{outputDir}/SDE.png")
plt.show()
