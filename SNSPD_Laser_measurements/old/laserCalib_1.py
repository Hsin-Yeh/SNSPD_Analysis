#!/usr/bin/env python

import ROOT
import argparse
from array import array

parser = argparse.ArgumentParser(description='make basic plots of laser calibration and convert csv to root')
parser.add_argument('in_filenames',nargs="+",help='input filenames')
parser.add_argument('--outName','-p',default="default",type=str,help='out Name')

outfile = ROOT.TFile.Open(f"output_root/{outName}.root","RECREATE")
