#!/usr/bin/env python3
"""
Plot event-by-event variables vs event number for different bias voltages
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
import plot_utils

parser = argparse.ArgumentParser(description='Plot event-by-event data from analysis.json files')
parser.add_argument('--input_dir', '-i', nargs='+', default=['./plots/test'], 
                   help='Directory or directories containing analysis.json files (supports multiple)')
parser.add_argument('--output_dir', '-d', default='.', help='Output directory for plots')
parser.add_argument('--pattern', '-p', default='*_analysis.json', help='File pattern to match')
parser.add_argument('--recursive', '-r', action='store_true', default=True, help='Search recursively in subdirectories')
parser.add_argument('--power', type=int, help='Select specific power level (nW) to plot. If not specified, uses first available power.')
parser.add_argument('--max_events', type=int, default=None, help='Maximum number of events to plot')
parser.add_argument('--variables', nargs='+', 
                   default=['pulse_fall_range_ptp', 'pulse_time_interval', 'pulse_time'],
                   help='Variables to plot (default: pulse_fall_range_ptp pulse_time_interval pulse_time)')

def read_event_data(input_dir, pattern, recursive=True, power_filter=None):
    """Read event-by-event data from analysis files"""
    
    # Handle single directory or list of directories
    if isinstance(input_dir, str):
        input_dirs = [input_dir]
    else:
        input_dirs = input_dir
    
    files = []
    for directory in input_dirs:
        if recursive:
            search_pattern = os.path.join(directory, '**', pattern)
            files.extend(plot_utils.glob.glob(search_pattern, recursive=True))
        else:
            search_pattern = os.path.join(directory, pattern)
            files.extend(plot_utils.glob.glob(search_pattern))
    
    data_by_bias = {}
    
    for filepath in files:
        filename = os.path.basename(filepath)
        bias_voltage = plot_utils.extract_bias_voltage(filename)
        power = plot_utils.extract_power(filename)
        
        if bias_voltage is None:
            continue
        
        # Filter by power if specified
        if power_filter is not None and power != power_filter:
            continue
        
        try:
            with open(filepath, 'r') as f:
                analysis = json.load(f)
            
            event_data = analysis.get('event_by_event_data', [])
            
            if event_data:
                if bias_voltage not in data_by_bias:
                    data_by_bias[bias_voltage] = {
                        'power': power,
                        'events': event_data,
                        'filename': filename
                    }
                else:
                    # If multiple files for same bias, keep the one with more events
                    if len(event_data) > len(data_by_bias[bias_voltage]['events']):
                        data_by_bias[bias_voltage] = {
                            'power': power,
                            'events': event_data,
                            'filename': filename
                        }
        
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
    
    return data_by_bias

def plot_variables_vs_event(data_by_bias, variables, output_dir, max_events=None):
    """Plot multiple variables vs event number for different bias voltages"""
    
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 5*n_vars))
    
    if n_vars == 1:
        axes = [axes]
    
    # Define colors
    unique_biases = sorted(data_by_bias.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_biases)))
    color_map = {bias: colors[i] for i, bias in enumerate(unique_biases)}
    
    # Variable display names
    var_labels = {
        'pulse_fall_range_ptp': 'Pulse Fall Range PTP (V)',
        'pulse_time_interval': 'Pulse Time Interval (s)',
        'pulse_time': 'Pulse Time (s)',
        'trigger_check': 'Trigger Check',
        'pulse_rise_range_ptb': 'Pulse Rise Range PTB (V)',
        'pulse_max': 'Pulse Max (V)',
        'pulse_min': 'Pulse Min (V)'
    }
    
    print("\n=== Plotting event-by-event data ===")
    
    for var_idx, variable in enumerate(variables):
        ax = axes[var_idx]
        
        for bias in sorted(data_by_bias.keys()):
            data_info = data_by_bias[bias]
            events = data_info['events']
            power = data_info['power']
            
            # Extract variable data
            event_numbers = [e['event_number'] for e in events if variable in e]
            var_values = [e[variable] for e in events if variable in e]
            
            if not event_numbers:
                print(f"  Warning: No data for {variable} at {bias}mV")
                continue
            
            # Apply max_events limit
            if max_events is not None:
                event_numbers = event_numbers[:max_events]
                var_values = var_values[:max_events]
            
            color = color_map[bias]
            label = f'{bias} mV' if len(unique_biases) <= 10 else None
            
            # Use scatter for better visibility with many points
            ax.scatter(event_numbers, var_values, c=[color], alpha=0.6, s=1, 
                      label=label, rasterized=True)
            
            print(f"  {bias}mV ({power}nW): {len(event_numbers)} events, "
                  f"{variable} range: [{np.min(var_values):.3e}, {np.max(var_values):.3e}]")
        
        # Configure subplot
        ax.set_xlabel('Event Number', fontsize=11)
        ylabel = var_labels.get(variable, variable)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{ylabel} vs Event Number', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Only show legend if not too crowded
        if len(unique_biases) <= 10:
            ax.legend(fontsize=8, markerscale=5, ncol=min(5, len(unique_biases)))
    
    plt.tight_layout()
    
    # Save plot
    var_str = '_'.join(variables[:3])  # Use first 3 variables in filename
    output_path = os.path.join(output_dir, f'event_data_{var_str}_vs_event.png')
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved: {output_path}")
    plt.close()

def main():
    args = parser.parse_args()
    
    print("="*60)
    print("SNSPD Event-by-Event Data Plotting")
    print("="*60)
    input_dirs = args.input_dir if isinstance(args.input_dir, list) else [args.input_dir]
    print(f"\nReading analysis files from: {', '.join(input_dirs)}")
    print(f"File pattern: {args.pattern}")
    print(f"Recursive search: {args.recursive}")
    if args.power:
        print(f"Power filter: {args.power} nW")
    print(f"Variables to plot: {', '.join(args.variables)}")
    if args.max_events:
        print(f"Max events: {args.max_events}")
    
    # Read event data
    data_by_bias = read_event_data(args.input_dir, args.pattern, args.recursive, args.power)
    
    if not data_by_bias:
        print("\nNo event data found!")
        return
    
    print(f"\nFound event data for {len(data_by_bias)} bias voltages:")
    for bias, info in sorted(data_by_bias.items()):
        print(f"  {bias} mV ({info['power']} nW): {len(info['events'])} events")
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    plot_variables_vs_event(data_by_bias, args.variables, args.output_dir, args.max_events)
    
    print("\n" + "="*60)
    print("PLOTTING COMPLETE")
    print("="*60)
    print(f"\nAll plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
