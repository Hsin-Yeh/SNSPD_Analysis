#!/usr/bin/env python3
"""
Stage 3: Comparison plotting across multiple measurements

Reads statistical analysis results from analyze_events.py (Stage 2) and generates
comparison plots for rates vs bias, rates vs power, and pulse characteristics.

Workflow:
  Stage 1: SelfTrigger.py     (TDMS → event JSON)
  Stage 2: analyze_events.py  (event JSON → statistics JSON)
  Stage 3: plot_all.py        (statistics JSON → comparison plots)  ← THIS SCRIPT

The script expects either:
  - *_analysis.json files from Stage 1 (contains summary_statistics)
  - statistics_*.json files from Stage 2 (contains fitted statistics)

For best results, use statistics_*.json files which include error estimates.
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

parser = argparse.ArgumentParser(
    description='Stage 3: Generate comparison plots from statistical analysis',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Use Stage 1 analysis files (original)
  python plot_all.py -i plots/test/ -p '*_analysis.json'
  
  # Use Stage 2 statistics files (recommended - includes errors)
  python plot_all.py -i output/ -p 'statistics_*.json'
  
  # Multiple directories with custom output location
  python plot_all.py -i dir1/ dir2/ dir3/ -d output/comparison_plots/
  
  # Only generate vs_bias plots
  python plot_all.py -i output/ -m vs_bias
"""
)
parser.add_argument('--input_dir', '-i', nargs='+', default=['./plots/test'], 
                   help='Directory or directories containing analysis.json or statistics_*.json files')
parser.add_argument('--output_dir', '-d', default='.', help='Output directory for plots')
parser.add_argument('--pattern', '-p', default='*_analysis.json', 
                   help='File pattern to match (e.g., *_analysis.json or statistics_*.json)')
parser.add_argument('--mode', '-m', choices=['all', 'vs_bias', 'vs_power', 'pulse'], default='all',
                   help='Plot mode: all, vs_bias, vs_power, or pulse')
parser.add_argument('--log_scale', action='store_true', help='Use log scale for power plots')
parser.add_argument('--recursive', '-r', action='store_true', default=True, help='Search recursively in subdirectories')

def main():
    args = parser.parse_args()
    
    print("="*70)
    print("Stage 3: SNSPD Comparison Plotting")
    print("="*70)
    input_dirs = args.input_dir if isinstance(args.input_dir, list) else [args.input_dir]
    print(f"\nReading analysis files from: {', '.join(input_dirs)}")
    print(f"File pattern: {args.pattern}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Recursive search: {args.recursive}")
    
    # Read data (works with both *_analysis.json and statistics_*.json)
    data = read_analysis_files(args.input_dir, args.pattern, args.recursive)
    
    if not data:
        print("\nNo valid data found!")
        return
    
    print_data_summary(data)
    
    # Group data
    power_groups = group_by_power(data)
    bias_groups = group_by_bias(data)
    
    # Extract sample name from input directory path
    # Example: ../../SNSPD_analyzed_json/SMSPD_3/ -> SMSPD_3
    first_input_dir = input_dirs[0]
    path_parts = Path(first_input_dir).parts
    
    # Find the component after 'SNSPD_analyzed_json'
    sample_name = None
    for i, part in enumerate(path_parts):
        if 'SNSPD_analyzed_json' in part and i + 1 < len(path_parts):
            sample_name = path_parts[i + 1]
            break
    
    # Fallback to last component if SNSPD_analyzed_json not found
    if sample_name is None:
        sample_name = os.path.basename(os.path.normpath(first_input_dir))
    
    # Create output directory structure: output/{sample_name}/
    output_dir = os.path.join(args.output_dir, 'output', sample_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Sample name: {sample_name}")
    print(f"Plots will be saved to: {output_dir}")
    
    # Generate plots based on mode
    if args.mode in ['all', 'vs_bias']:
        print("\n" + "="*60)
        print("GENERATING RATES VS BIAS VOLTAGE PLOTS")
        print("="*60)
        
        # Combined multi-power plot
        if len(power_groups) > 1:
            print("\n--- Combined Multi-Power Comparison ---")
            plot_multi_power_vs_bias(power_groups, output_dir)
    
    if args.mode in ['all', 'vs_power']:
        print("\n" + "="*60)
        print("GENERATING RATES VS POWER PLOTS")
        print("="*60)
        
        # Combined multi-bias plot
        if len(bias_groups) > 1:
            print("\n--- Combined Multi-Bias Comparison ---")
            plot_multi_bias_vs_power(bias_groups, output_dir, args.log_scale)
    
    if args.mode in ['all', 'pulse'] and HAS_PULSE_PLOTS:
        print("\n" + "="*60)
        print("GENERATING PULSE CHARACTERISTIC PLOTS")
        print("="*60)
        
        print("\n--- Pulse PTP vs Bias Voltage ---")
        plot_pulse_ptp_vs_bias(power_groups, output_dir)
        
        print("\n--- Pulse PTP vs Power ---")
        plot_pulse_ptp_vs_power(bias_groups, output_dir)
    
    print("\n" + "="*60)
    print("PLOTTING COMPLETE")
    print("="*60)
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
