#!/usr/bin/env python3
"""
Main plotting script for SNSPD analysis - unified interface
Can generate all plots with a single command
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
import plot_utils
import plot_rates_vs_bias_v2
import plot_rates_vs_power

from plot_utils import read_analysis_files, group_by_power, group_by_bias, print_data_summary
from plot_rates_vs_bias_v2 import plot_single_power_vs_bias, plot_multi_power_vs_bias
from plot_rates_vs_power import plot_single_bias_vs_power, plot_multi_bias_vs_power

# Import pulse characteristic plotting functions
try:
    import plot_pulse_characteristics
    from plot_pulse_characteristics import plot_pulse_ptp_vs_bias, plot_pulse_ptp_vs_power
    HAS_PULSE_PLOTS = True
except ImportError:
    HAS_PULSE_PLOTS = False

parser = argparse.ArgumentParser(description='Generate all SNSPD rate plots')
parser.add_argument('--input_dir', '-i', nargs='+', default=['./plots/test'], 
                   help='Directory or directories containing analysis.json files (supports multiple)')
parser.add_argument('--output_dir', '-d', default='.', help='Output directory for plots')
parser.add_argument('--pattern', '-p', default='*_analysis.json', help='File pattern to match')
parser.add_argument('--mode', '-m', choices=['all', 'vs_bias', 'vs_power', 'pulse'], default='all',
                   help='Plot mode: all, vs_bias, vs_power, or pulse')
parser.add_argument('--log_scale', action='store_true', help='Use log scale for power plots')
parser.add_argument('--recursive', '-r', action='store_true', default=True, help='Search recursively in subdirectories')

def main():
    args = parser.parse_args()
    
    print("="*60)
    print("SNSPD Analysis Plotting Tool")
    print("="*60)
    input_dirs = args.input_dir if isinstance(args.input_dir, list) else [args.input_dir]
    print(f"\nReading analysis files from: {', '.join(input_dirs)}")
    print(f"File pattern: {args.pattern}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Recursive search: {args.recursive}")
    
    # Read data
    data = read_analysis_files(args.input_dir, args.pattern, args.recursive)
    
    if not data:
        print("\nNo valid data found!")
        return
    
    print_data_summary(data)
    
    # Group data
    power_groups = group_by_power(data)
    bias_groups = group_by_bias(data)
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots based on mode
    if args.mode in ['all', 'vs_bias']:
        print("\n" + "="*60)
        print("GENERATING RATES VS BIAS VOLTAGE PLOTS")
        print("="*60)
        
        # Individual power plots
        print("\n--- Individual Power Levels ---")
        for power, power_data in sorted(power_groups.items()):
            plot_single_power_vs_bias(power, power_data, args.output_dir)
        
        # Combined multi-power plot
        if len(power_groups) > 1:
            print("\n--- Combined Multi-Power Comparison ---")
            plot_multi_power_vs_bias(power_groups, args.output_dir)
    
    if args.mode in ['all', 'vs_power']:
        print("\n" + "="*60)
        print("GENERATING RATES VS POWER PLOTS")
        print("="*60)
        
        # Individual bias plots
        print("\n--- Individual Bias Voltages ---")
        for bias, bias_data in sorted(bias_groups.items()):
            plot_single_bias_vs_power(bias, bias_data, args.output_dir, args.log_scale)
        
        # Combined multi-bias plot
        if len(bias_groups) > 1:
            print("\n--- Combined Multi-Bias Comparison ---")
            plot_multi_bias_vs_power(bias_groups, args.output_dir, args.log_scale)
    
    if args.mode in ['all', 'pulse'] and HAS_PULSE_PLOTS:
        print("\n" + "="*60)
        print("GENERATING PULSE CHARACTERISTIC PLOTS")
        print("="*60)
        
        print("\n--- Pulse PTP vs Bias Voltage ---")
        plot_pulse_ptp_vs_bias(power_groups, args.output_dir)
        
        print("\n--- Pulse PTP vs Power ---")
        plot_pulse_ptp_vs_power(bias_groups, args.output_dir)
    
    print("\n" + "="*60)
    print("PLOTTING COMPLETE")
    print("="*60)
    print(f"\nAll plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
