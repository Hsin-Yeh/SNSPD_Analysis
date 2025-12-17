#!/usr/bin/env python

from nptdms import TdmsFile

def Read_Groups_and_Channels(tdms_file):
    # Loop through each group and print the channel names
    for group in tdms_file.groups():
        print("Group '{}':".format(group.name))
        for channel in group.channels():
            print("- Channel '{}':".format(channel.name))
