#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-o',default="./",type=str,help='output directory')
parser.add_argument('--outfilename','-n',default="_test",type=str,help='output file name')
parser.add_argument('--report','-r',default=10000,type=int,help='report every x events')
parser.add_argument('--debug','-d',action="store_true",help='debug mode')
args = parser.parse_args()

def plot_real():
    plt.figure(figsize=(10,6))
    for in_filename in args.in_filenames:
        # Read the data from the provided text
        df = pd.read_csv(in_filename, delimiter='\t')
        x = df.iloc[:, 0]
        plt.plot(x, df['e1'], label=in_filename.split('/')[-1].split('.txt')[0])

    # Read the data from the provided text
    # df_0 = pd.read_csv("/Volumes/HV620S/SNSPD/Ellipsometry/20230606/NbN_0606_real_flowrate.txt", delimiter='\t')
    # x_0 = df_0.iloc[:, 0]
    # plt.plot(x_0, df_0['12:0.2'], label="12:0.2 (20230322)")

    # df_2 = pd.read_csv("/Volumes/HV620S/SNSPD/Ellipsometry/20230921/0921NbN_800C_12_0.1.txt", delimiter='\t')
    # x_2 = df_2.iloc[:, 0]
    # plt.plot(x_2, df_2['e1'], label="0921NbN_800C_12_0.1")

    # df_3 = pd.read_csv("/Volumes/HV620S/SNSPD/Ellipsometry/20230921/0922NbN_800C_12_0.1.txt", delimiter='\t')
    # x_3 = df_3.iloc[:, 0]
    # plt.plot(x_3, df_3['e1'], label="0922NbN_800C_12_0.1")

    # df_4 = pd.read_csv("/Volumes/HV620S/SNSPD/Ellipsometry/20230921/0924NbN_800C_12_0.3.txt", delimiter='\t')
    # x_4 = df_4.iloc[:, 0]
    # plt.plot(x_4, df_4['e1'], label="0924NbN_800C_12_0.3")

    df_1 = pd.read_csv("/Volumes/HV620S/SNSPD/Ellipsometry/20230606/NbN_0606_7nm.txt", delimiter='\t')
    x_1 = df_1.iloc[:, 0]
    plt.plot(x_1, df_1['n']*df_1['n']-df_1['k']*df_1['k'], label="NbN_0606_7nm")

    # Set the plot labels and title
    plt.xlabel(r'Wavelength $\lambda$ [nm]',fontsize=15)
    plt.ylabel(r'$Re(\varepsilon)$',fontsize=15)
    # if (args.in_filenames[0].find("real")!=-1): plt.ylabel(r'$Re(\varepsilon)$',fontsize=15)
    # else: plt.ylabel(r'$Im(\varepsilon)$',fontsize=15)
    # plt.title("RF Power = 120W",fontsize=20)
    # plt.title(r"Ar:$N_{2}$ = 12:0.2",fontsize=20)
    # plt.title('Refractive index',fontsize=20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # plt.legend(title=r'      Ar:$N_{2}$')
    plt.legend(title='')
    # plt.legend(title='',fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.grid(True, which='minor', linestyle=':', linewidth='0.5')  # Minor grid
    plt.minorticks_on()  # Enable minor ticks
    plt.tight_layout()
    plt.savefig(f"{args.outputDir}/real{args.outfilename}.png")
    plt.show()



def plot_imag():

    plt.figure(figsize=(10,6))
    for in_filename in args.in_filenames:
        # Read the data from the provided text
        df = pd.read_csv(in_filename, delimiter='\t')
        x = df.iloc[:, 0]
        plt.plot(x, df['e2'], label=in_filename.split('/')[-1].split('.txt')[0])

    # Read the data from the provided text
    # df_0 = pd.read_csv("/Volumes/HV620S/SNSPD/Ellipsometry/20230606/NbN_0606_imaginary_flowrate.txt", delimiter='\t')
    # x_0 = df_0.iloc[:, 0]
    # plt.plot(x_0, df_0['12:0.2'], label="12:0.2 (20230322)")

    # # Read the data from the provided text
    # Read the data from the provided text
    # df_2 = pd.read_csv("/Volumes/HV620S/SNSPD/Ellipsometry/20230921/0921NbN_800C_12_0.1.txt", delimiter='\t')
    # x_2 = df_2.iloc[:, 0]
    # plt.plot(x_2, df_2['e2'], label="0921NbN_800C_12_0.1")

    # Read the data from the provided text
    # df_3 = pd.read_csv("/Volumes/HV620S/SNSPD/Ellipsometry/20230921/0922NbN_800C_12_0.1.txt", delimiter='\t')
    # x_3 = df_3.iloc[:, 0]
    # plt.plot(x_3, df_3['e2'], label="0922NbN_800C_12_0.1")

    # Read the data from the provided text
    # df_4 = pd.read_csv("/Volumes/HV620S/SNSPD/Ellipsometry/20230921/0924NbN_800C_12_0.3.txt", delimiter='\t')
    # x_4 = df_4.iloc[:, 0]
    # plt.plot(x_4, df_4['e2'], label="0924NbN_800C_12_0.3")

    df_1 = pd.read_csv("/Volumes/HV620S/SNSPD/Ellipsometry/20230606/NbN_0606_7nm.txt", delimiter='\t')
    x_1 = df_1.iloc[:, 0]
    # plt.plot(x_1, df_1['n']*df_1['n']-df_1['k']*df_1['k'], label="36:0.1 (20230606)")
    plt.plot(x_1, 2*df_1['n']*df_1['k'], label="NbN_0606_7nm")

    # Set the plot labels and title
    plt.xlabel(r'Wavelength $\lambda$ [nm]',fontsize=15)
    plt.ylabel(r'$Im(\varepsilon)$',fontsize=15)
    # if (args.in_filenames[0].find("real")!=-1): plt.ylabel(r'$Re(\varepsilon)$',fontsize=15)
    # else: plt.ylabel(r'$Im(\varepsilon)$',fontsize=15)
    # plt.title("RF Power = 120W",fontsize=20)
    # plt.title(r"Ar:$N_{2}$ = 12:0.2",fontsize=20)
    # plt.title('Refractive index',fontsize=20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # plt.legend(title=r'      Ar:$N_{2}$')
    plt.legend(title='')
    plt.grid(True)
    plt.grid(True, which='minor', linestyle=':', linewidth='0.5')  # Minor grid
    plt.minorticks_on()  # Enable minor ticks
    plt.tight_layout()
    plt.savefig(f"{args.outputDir}/imag{args.outfilename}.png")
    plt.show()

    # # Extract the first column and get the column names
    # x = df.iloc[:, 0]
    # column_names = df.columns[1:]

    # # Plot the value of each column against the first column
    # for column in column_names:
    #     values = df[column]
    # plt.plot(x, values, label=column)

def plot_imag_real():
    for in_filename in args.in_filenames:
        # Read the data from the provided text
        df = pd.read_csv(in_filename, delimiter='\t')
        x = df.iloc[:, 0]

        # Create figure and axis objects
        fig, ax1 = plt.subplots()

        # Plot the first line on the left y-axis
        ax1.plot(x, df['e1'], 'tab:blue', label=in_filename.split('/')[-1].split('.txt')[0])
        ax1.set_xlabel(r'Wavelength $\lambda$ [nm]',fontsize=15)
        ax1.set_ylabel(r'$Re(\varepsilon)$',fontsize=15, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        # Create a twin axis sharing the x-axis
        ax2 = ax1.twinx()

        # Plot the second line on the right y-axis
        ax2.plot(x, df['e2'], 'tab:red', label=in_filename.split('/')[-1].split('.txt')[0])
        ax2.set_ylabel(r'$Im(\varepsilon)$',fontsize=15, color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Add title
        plt.title(in_filename.split('/')[-1].split('.txt')[0])
        plt.savefig(f"{args.outputDir}/{in_filename.split('/')[-1].split('.txt')[0]}.png")
        plt.show()


if __name__ == "__main__":
    # plot_real()
    # plot_imag()
    plot_imag_real()
