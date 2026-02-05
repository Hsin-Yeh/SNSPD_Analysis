#!/bin/bash
cd /Users/ya/Documents/Projects/SNSPD/SNSPD_Analysis/SNSPD_Laser_measurements/TCSPC
python3 extract_and_compare_power_sweeps.py
echo "Power sweep comparison complete"
ls -lh /Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/combined/
