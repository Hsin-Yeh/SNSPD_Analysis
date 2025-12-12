#!/usr/bin/env python

from nptdms import TdmsFile
import pandas as pd
import numpy as np
from array import array
import ROOT
import errno
import os

# User defined functions
from utils.tdmsUtils import *

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outputDir','-d',default="./output_root",type=str,help='output directory')
parser.add_argument('--report','-r',default=100,type=int,help='report every x events')
args = parser.parse_args()

def Convert_Single_TDMS(in_filename):
    with TdmsFile.open(in_filename) as tdms_file:
        baseName = in_filename.rsplit('/',1)[1].split('.tdms')[0]
        baseDir= in_filename.split('Laser/')[1].rsplit('/',1)[0]
        pulseFlag = True if in_filename.find('Pulse') != -1 else False
        # make outputDir
        try:
            os.makedirs(f'{args.outputDir}/{baseDir}')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('output directory exists.')
            else:
                raise
        # Read Meta Data (Basic information)
        metadata = tdms_file.properties
        metadata_df = pd.DataFrame(metadata.items(), columns=['metaKey', 'metaValue'])
        print(metadata_df)
        totalEvents = int(metadata_df.loc[metadata_df['metaKey'] == 'Total Events', 'metaValue'].iloc[0])
        recordlength = int(metadata_df.loc[metadata_df['metaKey'] == 'record length', 'metaValue'].iloc[0])
        vertical_range = float(metadata_df.loc[metadata_df['metaKey'] == 'vertical range Sig', 'metaValue'].iloc[0])
        # Write metadata to json file
        metadata_df.to_json(f'{args.outputDir}/{baseDir}/{baseName}.json',orient="records",lines=True)
        # Write metadata to txt file
        metadata_df = metadata_df.reset_index()  # make sure indexes pair with number of rows
        with open(f'{args.outputDir}/{baseDir}/{baseName}.txt', 'w') as txtFile:
            for index, row in metadata_df.iterrows():
                txtFile.write(f"{row['metaKey']} : {row['metaValue']}\n")
        # Create output tree
        outtree = ROOT.TTree("SNSPD_data", "SNSPD_data")
        chSig = array( 'f', [ 0 ] * recordlength)
        if (pulseFlag): chTrig = array( 'f', [ 0 ] * recordlength)
        outtree.Branch('chSig',chSig,"chSig[%d]/F" %(recordlength))
        if (pulseFlag): outtree.Branch('chTrig',chTrig,"chTrig[%d]/F" %(recordlength))
        # Read Groups and Channels
        Read_Groups_and_Channels(tdms_file)
        # Start Loop
        print (f"==========Start Looping==========")
        for event, chunk in enumerate(tdms_file.data_chunks()):
            # Loop progress
            if ((event+1)%args.report==0): print (f"==========Processing {event}/{totalEvents} event==========")
            # Skip chunk larger than totalEvents
            if (event > totalEvents-1): break
            # Read data into np array
            chSig_chunk = chunk['ADC Readout Channels']['chSig']._data()
            if (pulseFlag): chTrig_chunk = chunk['ADC Readout Channels']['chTrig']._data()
            # copy to tree branch
            for i in range(0, recordlength):
                chSig[i] = chSig_chunk[i];
                if (pulseFlag): chTrig[i] = chTrig_chunk[i];
            # Fill Tree
            outtree.Fill()
    # Output
    rootFileName = f"{args.outputDir}/{baseDir}/{baseName}.root"
    print(f"output file: {rootFileName}")
    rootFile = ROOT.TFile.Open(rootFileName,"RECREATE")
    rootFile.cd()
    outtree.Write()
    rootFile.Close()


if __name__ == "__main__":

    # Loop the input tdms files
    for in_filename in args.in_filenames:
        Convert_Single_TDMS(in_filename)
