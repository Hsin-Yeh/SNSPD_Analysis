#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline, CubicSpline
from Timing_Analyzer import *
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
HEADERROWS=0 #20
SEPERATOR='\t'

parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-o',default="./output",type=str,help='output directory')
parser.add_argument('--report','-r',default=10000,type=int,help='report every x events')
parser.add_argument('--debug','-d',action="store_true",help='debug mode')
args = parser.parse_args()

if __name__ == "__main__":

    #Metadata
    metadata_df = pd.read_csv(args.in_filenames[0], header=None, nrows=HEADERROWS, sep=SEPERATOR)
    # Convert the DataFrame to a dictionary
    metadata = dict(zip(metadata_df[0], metadata_df[1]))
    # Print the metadata dictionary
    print(metadata_df)

    for ifile, in_filename in enumerate(args.in_filenames):
        if (in_filename.find(".csv")==0): continue
        if (ifile%args.report==0): print (f"Processed {ifile}/{len(args.in_filenames)} file")

        # Read the csv file into a pandas DataFrame
        df = pd.read_csv(in_filename, skiprows=HEADERROWS, sep=SEPERATOR)

        # Rename the columns
        df.columns = ['Time', 'CH1']

        # Convert the data types
        df['Time'] = pd.to_numeric(df['Time'])
        df['CH1'] = pd.to_numeric(df['CH1'])

        # Check if there are any values less than -0.2 in the CH2 column
        if (df['CH1'] > 0.).any():
            print("There are values less than -0.2 in the CH1 column.")
            data_start = 0
            data_end = 10000
            # Slice the DataFrame to only include data between index 400 and 600
            df_slice = df.iloc[data_start:data_end]
            # Generate a new x-axis range for the spline function
            x_spline = df_slice['Time'].values
            x_spline_range = np.linspace(x_spline.min(), x_spline.max(), num=1000)
            # Create a spline interpolation function for the data
            spline_func = CubicSpline(df_slice['Time'], df_slice['CH1'])
            # Evaluate the spline function at the new x-axis range
            y_spline = spline_func(x_spline_range)

            # Get the minimum point of this function
            minPoint = Get_FunctionMin(spline_func,data_start,data_end)
            # Get the average of the trigger plateau
            Get_Average_Plateau(df_slice, 'CH1', minPoint.y)
            # Get Inidices over threshold
            Get_indices_overThreshold(df, 'CH1', minPoint.y*0.9)
            # Get Turning Point
            # turning_point_pedestal, turning_point_peak = Get_turning_time(spline_func, minPoint.y*0.9)
            # Get the Fall time using interpolate function
            # Get_Function_RiseFall_Range(spline_func, minPoint.y, data_start, data_end)
            # Get pulse arrival time with interpolate function
            # Get_Function_Arrival(spline_func, (turning_point_peak.y + turning_point_pedestal.y)/2, data_start, data_end)

            # Create the plot
            plt.plot(df_slice['Time'], df_slice['CH1'])
            # plt.plot(df_slice['CH1'],'o',fillstyle='none')
            plt.xlabel('Time (s)')
            # plt.xlabel('Index')
            plt.ylabel('CH1')
            plt.title('Waveform')
            # Plot the spline function with a solid line
            # plt.plot(x_spline_range, y_spline, '-', label='Spline Fit')
            plt.grid()
            # Add a legend
            plt.legend()
            plt.show()
        else:
            print("There are no values less than -0.2 in the CH1 column.")
