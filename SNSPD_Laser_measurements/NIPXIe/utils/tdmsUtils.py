#!/usr/bin/env python3

from nptdms import TdmsFile

def Read_Groups_and_Channels(tdms_file):
    # Loop through each group and print the channel names
    for group in tdms_file.groups():
        print(f"Group '{group.name}':")
        for channel in group.channels():
            print(f"- Channel '{channel.name}':")
