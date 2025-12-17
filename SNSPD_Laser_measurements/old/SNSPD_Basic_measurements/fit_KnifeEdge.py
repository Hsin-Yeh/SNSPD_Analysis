#!/usr/bin/env python
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
import matplotlib.pyplot as plt
from scipy.stats import norm


# Define the cumulative Gaussian function
def cumulative_gaussian(x, constant, Norm, mean, sigma):
    return constant + (Norm-constant) * (1 + erf((x - mean) / (sigma * np.sqrt(2)))) / 2

# Define the data points
x_data = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 6.5, 8])
y_data = np.array([2.05, 3.1, 5.52, 9.4, 14.8, 24.5, 32.5, 40.9, 49.6, 56.2, 61.5, 64.7, 66.5, 66.8, 66.9, 67.2])

# Perform curve fitting
popt, _ = curve_fit(cumulative_gaussian, x_data, y_data)

# Extract the fitted parameters
constant, Norm, mean, sigma = popt
print("Fitted Parameters:")
print(f"a = {constant}")
print(f"b = {Norm}")
print(f"c = {mean}")
print(f"d = {sigma}")

# Generate x values for the fitted curve
x_fit = np.linspace(x_data.min(), x_data.max(), 100)

# Calculate y values for the fitted curve
y_fit = cumulative_gaussian(x_fit, constant, Norm, mean, sigma)

# Plot the data points and the fitted curve
plt.plot(x_data, y_data, 'bo', label='Data')
plt.plot(x_fit, y_fit, 'r-', label='Fitted Curve')
plt.xlabel('Knife Position [mm]')
plt.ylabel('Laser Intensity [uW]')
plt.legend(loc='lower right')

# Add text annotations for the fit results
text = f'Cumulative Gaussian Fit Results:\nconstant = {constant:.2f}\nNorm = {Norm:.2f}\nmean = {mean:.2f}\nsigma = {sigma:.2f}'
plt.text(0.5, 50, text, fontsize=12)
plt.savefig("knife_edge.png")
plt.show()

# Define the width in terms of sigma
apperature = 4.7

# Calculate the area within the specified width
lower_bound = mean - apperature/2
upper_bound = mean + apperature/2
area = norm.cdf(upper_bound, loc=mean, scale=sigma) - norm.cdf(lower_bound, loc=mean, scale=sigma)

# Calculate the percentage of the area
percentage = area * 100

print(f"Percentage of the area within apperature {apperature}mm ({(apperature/2)/sigma:.2f}sigma) : {percentage:.2f}%")
