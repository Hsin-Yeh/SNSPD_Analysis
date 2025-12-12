#!/usr/bin/env python

from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.optimize import fmin
from scipy.optimize import brentq
from scipy.integrate import quad
import numpy as np

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def Get_Xs(inputfunc, value, rangeMin, rangeMax):
    # Define the function to find roots for
    def f(x):
        return inputfunc(x) - value
    # Find the roots using the Brent method
    roots = []
    for i in range(int(rangeMin), int(rangeMax)+1):
        # f(i) and f(i+1) should be opposite sign
        if ( f(i) * f(i+1) < 0):
            # brentq root finding
            root = brentq(f, i, i+1)
            roots.append(root)
    return roots

def Get_Function_Arrival(inputfunc, value, rangeMin, rangeMax):
    time = Get_Xs(inputfunc, value, rangeMin, rangeMax)
    # print(f"Time of Arrival: {time[0]}")
    if (len(time)>0): return time[0]
    else: return -999

def Get_Function_FWHM(inputfunc, value, rangeMin, rangeMax):
    time = Get_Xs(inputfunc, value, rangeMin, rangeMax)
    # print(f"Time of Arrival: {time[0]}")
    if (len(time)>1): return time[0],time[1]
    else: return -999,0


def Get_df_Arrival(df, chName, value):
    idx = (np.abs(df[chName] - value)).idxmin()
    print(f"Time of Arrival: {idx} {df.loc[idx,'Time']} ")


def Get_Function_RiseFall_Range(inputfunc, value10, value90, rangeMin, rangeMax):
    # Get the time of 10% amplitude
    time10 = Get_Xs(inputfunc, value10, rangeMin, rangeMax)
    # Get the time of 90% amplitude
    time90 = Get_Xs(inputfunc, value90, rangeMin, rangeMax)
    if (len(time10)>0 and len(time90)>0):
        start_time = time10[0]
        end_time = time90[0]
        RiseFall_time = end_time - start_time
    else:
        RiseFall_time = -999
    return RiseFall_time
    # if ( value > 0 ): print(f"Edge Start: {start_time}, Edge End: {end_time}, Rise Time: {RiseFall_time}")
    # if ( value < 0 ): print(f"Edge Start: {start_time}, Edge End: {end_time}, Fall Time: {RiseFall_time}")

def Get_df_RiseFall_Range(df, chName, value):
    # Find the pulse start index
    start_index = (df[chName] < 0.1 * value).idxmin()
    # Find the index of the first value in the CH2 column that is less than 0.9 times the minimum
    end_index = (df[chName] < 0.9 * value).idxmin()
    # Find the time difference between the minimum and the falling point
    RiseFall_time = df.loc[end_index, 'Time'] - df.loc[start_index, 'Time']
    # Print the Rise/Fall time
    if ( value > 0 ): print(f"Edge Start: {start_index}, Edge End: {end_index}, Rise Time: {RiseFall_time}")
    if ( value < 0 ): print(f"Edge Start: {start_index}, Edge End: {end_index}, Fall Time: {RiseFall_time}")

def Get_turning_times(inputfunc, range_threshold, rangeMin, rangeMax, Rise_Fall, debug=False):
    # Obtain the first derivative of the spline function
    roots = inputfunc.derivative().roots()
    p_pedestals=[]
    p_peaks=[]
    p_ranges=[]
    # Find the turning value
    for iroot, root in enumerate(roots):
        if ( roots[iroot-1] > rangeMin and root < rangeMax and root > roots[iroot-1] ):
            # print (iroot, root, inputfunc(root), inputfunc(roots[iroot-1]))
            if (Rise_Fall == 'Rise' and inputfunc(root) - inputfunc(roots[iroot-1]) > range_threshold):
                p_pedestals.append(Point(roots[iroot-1], inputfunc(roots[iroot-1])))
                p_peaks.append(Point(root, inputfunc(root)))
                p_ranges.append(inputfunc(root)-inputfunc(roots[iroot-1]))
                if (debug): print(f"Rise Turning Point: ({p_pedestals[-1].x:.4f},{p_pedestals[-1].y:.4f}) ({p_peaks[-1].x:.4f},{p_peaks[-1].y:.4f}) ({p_ranges[-1]:.4f})")
            elif (Rise_Fall == 'Fall' and inputfunc(roots[iroot-1]) - inputfunc(root)  > range_threshold):
                p_pedestals.append(Point(roots[iroot-1], inputfunc(roots[iroot-1])))
                p_peaks.append(Point(root, inputfunc(root)))
                p_ranges.append(inputfunc(root)-inputfunc(roots[iroot-1]))
                if (debug): print(f"Fall Turning Point: ({p_pedestals[-1].x:.4f},{p_pedestals[-1].y:.4f}) ({p_peaks[-1].x:.4f},{p_peaks[-1].y:.4f}) ({p_ranges[-1]:.4f})")
    return p_pedestals, p_peaks, p_ranges

def Get_FunctionMax(inputfunc, start, end, debug=False):
    # Define the function
    def f(x):
        return inputfunc(x)
    # Find the maximum value within the range [-5, 5]
    result = minimize_scalar(lambda x: -f(x), bounds=(start, end), method='bounded')
    # Print the maximum value
    if (debug):
        print("Maximum value:", -result.fun)
        # Print the x-value at which the maximum occurs
        print("Location of maximum:", result.x)
    p = Point(result.x, -result.fun)
    # p = Point(0,0)
    return p

def Get_FunctionMin(inputfunc, start, end, debug=False):
    # Define the function
    def f(x):
        return inputfunc(x)
    # Find the minimum value within the range [-5, 5]
    result = minimize_scalar(lambda x: -f(x), bounds=(start, end), method='bounded')
    # Print the minimum value
    if (debug):
        print("Minimum value:", result.fun)
        # Print the x-value at which the minimum occurs
        print("Location of minimum:", result.x)
    p = Point(result.x, result.fun)
    return p

def Get_dfMax(df):
    # Find the index of the maximum value in the CH2 column
    max_index = df_slice['CH2'].idxmax()
    # Find the value of the maximum in the CH2 column
    max_value = df.loc[max_index, 'CH2']
    p = Point(max_index, max_value)
    return p

def Get_dfMin(df):
    # Find the index of the minimum value in the CH2 column
    min_index = df_slice['CH2'].idxmin()
    # Find the value of the minimum in the CH2 column
    min_value = df.loc[min_index, 'CH2']
    p = Point(min_index, min_value)
    return p

def Get_npMax(array):
    # Find the index of the maximum value in the NumPy array
    max_index = np.argmax(array)
    # Find the value of the maximum in the NumPy array
    max_value = np.max(array)
    p = Point(max_index, max_value)
    return p

def Get_npMin(array):
    # Find the index of the minimum value in the NumPy array
    min_index = np.argmin(array)
    # Find the value of the minimum in the NumPy array
    min_value = np.min(array)
    p = Point(min_index, min_value)
    return p

def Get_Average_Plateau(df,chName,value):
    # Filter out sample points above 95% of the minimum
    filtered = df[abs(df[chName]) >= abs(value) * 0.95]
    # Calculate the average of the remaining sample points
    average = filtered[chName].mean()
    print("The average of sample points with 95% of the minimum is:", average)

def Get_indices_overThreshold(df, chName, threshold):
    # Find the indices where the CH2 values are less than the threshold
    indices = np.where(abs(df[chName]) > abs(threshold))[0]
    print (f"Sample points with value above threshold: {indices}")

def Get_function_integral(f, rangemin, rangemax):
    # Calculate the integral of f(x) from 0 to 1
    result, error = quad(f, rangemin, rangemax)
    # Print the result
    # print(f'Integral of f(x) from {rangemin} to {rangemax}:', result, error)
    return result, error
