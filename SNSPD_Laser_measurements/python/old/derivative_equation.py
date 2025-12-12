#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def r_function(x):
    # Define the trapzoid function r(x)
    if 2 <= x <= 6:
        return 3.0
    else:
        return 0.0

def differential_equation(y, x):
    y_prime = y[1]
    r = r_function(x)
    return [y_prime, y[0] - y[1] * r]

# Define the initial condition
y0 = 0.0

# Define the x values
x = np.linspace(0, 10, 1000)

# Solve the differential equation using odeint
solution = odeint(differential_equation, [y0, y0], x)
y = solution[:, 0]  # Extract the y values from the solution

# Plot the solution
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Solution of y'(x) + (y(x) * r(x))' = y(0) - y(x)")
plt.grid(True)
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# def trapezoidal_function(x, a, b, c, d):
#     y = np.zeros_like(x)
#     mask = np.logical_and(x >= a, x <= b)
#     y[mask] = (x[mask] - a) / (b - a)
#     mask = np.logical_and(x > b, x <= c)
#     y[mask] = 1
#     mask = np.logical_and(x > c, x <= d)
#     y[mask] = (d - x[mask]) / (d - c)
#     return y

# # Define the x values
# x = np.linspace(0, 10, 1000)

# # Define the parameters for the trapezoidal function
# a = 2
# b = 4
# c = 6
# d = 8

# # Compute the y values of the trapezoidal function
# y = trapezoidal_function(x, a, b, c, d)

# # Compute the derivative using numpy.gradient
# dy_dx = np.gradient(y, x)

# # Plot the trapezoidal function
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Trapezoidal Function')

# # Plot the derivative
# plt.subplot(2, 1, 2)
# plt.plot(x, dy_dx, 'r')
# plt.xlabel('x')
# plt.ylabel("dy/dx")
# plt.title('Derivative of Trapezoidal Function')

# plt.tight_layout()
# plt.show()
