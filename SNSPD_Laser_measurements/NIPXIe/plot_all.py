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
import utils.plot_utils as plot_utils
import plot_statistics_vs_power_bias

from utils.plot_utils import read_analysis_files, group_by_power, group_by_bias, print_data_summary

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
parser.add_argument('--loglog_fit_range', type=str, default=None, help='Fit range for log-log power plots, format: min,max (nW)')

def find_common_path(directories):
    """
    Find the longest common path component from a list of directories.
    
    Example:
      Input: ['/path/to/SMSPD_3/Laser/10000kHz/93nW/data',
              '/path/to/SMSPD_3/Laser/10000kHz/207nW/data']
      Output: 'SMSPD_3/Laser/10000kHz' (common parent directory)
    
    Returns the deepest common path component, or 'output' if none found.
    """
    if not directories or len(directories) == 1:
        # Single directory: use last meaningful component
        if directories:
            path_parts = Path(directories[0]).parts
            # Skip generic names like 'data', 'plots', '20251210_071435'
            skip_names = {'data', 'plots', 'stage1_events', 'stage2_statistics', 'stage3_plots'}
            for part in reversed(path_parts):
                if part not in skip_names and not part.startswith('202'):
                    return part
        return 'output'
    
    # Multiple directories: find common path
    paths = [Path(d).parts for d in directories]
    
    # Find longest common prefix
    common_parts = []
    for parts in zip(*paths):
        if len(set(parts)) == 1:  # All paths have same component
            common_parts.append(parts[0])
        else:
            break
    
    # Find the deepest meaningful common component
    skip_names = {'data', 'plots', 'stage1_events', 'stage2_statistics', 'stage3_plots', 
                  'SNSPD_analyzed_output', 'SNSPD_rawdata'}
    
    for part in reversed(common_parts):
        if part not in skip_names and not part.startswith('202'):
            return part
    
    return 'output'

def find_common_parent_path(directories):
    """
    Find the common directory name starting after 'stage2_statistics' among all input directories.
    Example:
      Input: ['/path/to/SMSPD_3/Laser/10000kHz/93nW/data',
              '/path/to/SMSPD_3/Laser/10000kHz/207nW/data']
        Output: 'SMSPD_3/Laser/10000kHz'
    """
    if not directories:
        return ''
    split_paths = [list(Path(d).resolve().parts) for d in directories]
    # Find longest common prefix
    common = []
    for parts in zip(*split_paths):
        if len(set(parts)) == 1:
            common.append(parts[0])
        else:
            break
    # Find index of 'stage2_statistics' in common path
    if 'stage2_statistics' in common:
        idx = common.index('stage2_statistics') + 1
        # Take everything after 'stage2_statistics'
        result = common[idx:]
    else:
        # If not found, skip generic names from the end
        skip_names = {'data', 'plots', 'stage1_events', 'stage2_statistics', 'stage3_plots', 'SNSPD_analyzed_output', 'SNSPD_rawdata'}
        result = common[:]
        while result and (result[-1] in skip_names or result[-1].startswith('20')):
            result.pop()
    return os.path.join(*result) if result else ''

def determine_stage3_output_dir(input_dirs):
    """
    Output directory is always under ~/SNSPD_analyzed_output/stage3_plots/<common_parent_path>
    """
    home = str(Path.home())
    analyzed_output = os.path.join(home, 'SNSPD_analyzed_output')
    print(f"\nUsing analyzed output base directory: {analyzed_output}")
    stage3_dir = os.path.join(analyzed_output, 'stage3_plots')
    print(f"Stage 3 plots will be saved under: {stage3_dir}")
    common_parent = find_common_parent_path(input_dirs)
    print(f"Common parent path for input directories: {common_parent}")
    output_dir = os.path.join(stage3_dir, common_parent)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

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
    
    # Parse loglog_fit_range argument
    loglog_fit_range = None
    if args.loglog_fit_range:
        try:
            fit_min, fit_max = map(float, args.loglog_fit_range.split(','))
            loglog_fit_range = (fit_min, fit_max)
        except Exception as e:
            print(f"Warning: Could not parse --loglog_fit_range '{args.loglog_fit_range}': {e}")
    
    # Read data (works with both *_analysis.json and statistics_*.json)
    data = read_analysis_files(args.input_dir, args.pattern, args.recursive)
    
    if not data:
        print("\nNo valid data found!")
        return
    
    # print_data_summary(data)
    
    # Group data
    power_groups = group_by_power(data)
    bias_groups = group_by_bias(data)
    
    # Use new output directory strategy
    output_dir = determine_stage3_output_dir(input_dirs)
    print(f"Plots will be saved to: {output_dir}")
    
    # Generate plots based on mode
    # Generate statistics vs bias/power plots
    if args.mode in ['all', 'vs_bias']:
        print("\n" + "="*60)
        print("GENERATING RATES VS BIAS VOLTAGE PLOTS")
        print("="*60)
        
        print("\n--- Statistics vs Bias (all powers) ---")
        for power, pdata in power_groups.items():
            plot_statistics_vs_power_bias.plot_statistics_vs_bias(power, pdata, output_dir) 
    if args.mode in ['all', 'vs_power']:
        print("\n" + "="*60)
        print("GENERATING RATES VS POWER PLOTS")
        print("="*60)
        
        print("\n--- Statistics vs Power (all biases) ---")
        for bias, bdata in bias_groups.items():
            plot_statistics_vs_power_bias.plot_statistics_vs_power(bias, bdata, output_dir, loglog_fit_range=loglog_fit_range)
        
        print("\n--- Stacked laser sync histograms vs Power (all biases) ---")
        for bias, bdata in bias_groups.items():
            plot_statistics_vs_power_bias.plot_laser_sync_histogram_stack_vs_power(bdata, output_dir)
          
    # Generate multi-variable overlay plots
    if args.mode in ['all', 'vs_bias']:
        print("\n--- Multi-variable overlay: all powers vs bias ---")
        plot_statistics_vs_power_bias.plot_multi_statistics_vs_bias(power_groups, output_dir)
    if args.mode in ['all', 'vs_power']:
        print("\n--- Multi-variable overlay: all biases vs power ---")
        plot_statistics_vs_power_bias.plot_multi_statistics_vs_power(bias_groups, output_dir, loglog_fit_range=loglog_fit_range)
          
    print("\n" + "="*60)
    print("PLOTTING COMPLETE")
    print("="*60)
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
