#!/usr/bin/env python3
"""
Generic counter data plotter - works with different measurement folders
Usage: python plot_counter_generic.py <data_folder> [--bias 68,70,72] [--powers all/369,446,534]
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plot_style import setup_atlas_style

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit
import re
import argparse

def parse_filename(filename):
    """Extract power and timestamp from filename"""
    match = re.search(r'(\d+)nW_(\d{8}_\d{4})\.txt', filename)
    if match:
        power_nw = int(match.group(1))
        timestamp_str = match.group(2)
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M')
        return power_nw, timestamp
    return None, None

def read_counter_file(filepath):
    """Read counter data file and return bias voltages and median count rates"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header
    data_lines = lines[1:]
    
    bias_voltages = []
    count_rates = []
    
    for line in data_lines:
        parts = line.strip().split('\t')
        if len(parts) < 7:
            continue
            
        bias_voltage = float(parts[0])
        # measurements start from index 6 (7th column)
        measurements = [float(x) for x in parts[6:]]
        
        # Calculate median count rate (not mean)
        median_count_rate = np.median(measurements)
        
        bias_voltages.append(bias_voltage)
        count_rates.append(median_count_rate)
    
    return np.array(bias_voltages), np.array(count_rates)

def find_latest_files(data_dir):
    """Find the latest file for each power level in the directory"""
    # Ensure data_dir is a Path object
    data_dir = Path(data_dir)
    
    # Find all power directories
    power_dirs = [d for d in data_dir.iterdir() if d.is_dir() and 'nW' in d.name]
    
    power_files = {}
    dark_files = []  # Store all dark count files with timestamps
    
    for power_dir in power_dirs:
        power_match = re.search(r'(\d+)nW', power_dir.name)
        if not power_match:
            continue
        
        power_nw = int(power_match.group(1))
        
        # Find all files in this power directory
        txt_files = list(power_dir.glob('*.txt'))
        if not txt_files:
            continue
        
        # Sort by timestamp and take latest
        files_with_time = []
        for f in txt_files:
            _, timestamp = parse_filename(f.name)
            if timestamp:
                files_with_time.append((f, timestamp))
        
        if files_with_time:
            latest_file = max(files_with_time, key=lambda x: x[1])[0]
            if power_nw == 0:
                # Store all dark count files for matching
                dark_files.extend(files_with_time)
            else:
                power_files[power_nw] = latest_file
    
    return power_files, dark_files

def find_closest_dark_file(signal_timestamp, dark_files):
    """Find the dark count file with timestamp earlier and closest to signal file"""
    best_dark = None
    min_time_diff = None
    
    for dark_file, dark_time in dark_files:
        if dark_time <= signal_timestamp:
            time_diff = (signal_timestamp - dark_time).total_seconds()
            if min_time_diff is None or time_diff < min_time_diff:
                min_time_diff = time_diff
                best_dark = dark_file
    
    return best_dark

def get_available_bias_voltages(filepath):
    """Extract all unique bias voltages from a data file"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    bias_voltages = set()
    for line in lines[1:]:  # Skip header
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            try:
                bias_mv = int(round(float(parts[0]) * 1000))
                if bias_mv > 0:
                    bias_voltages.add(bias_mv)
            except ValueError:
                continue
    
    return sorted(list(bias_voltages))


def select_bias_voltages(bias_spec, available_biases):
    """
    Select bias voltages based on specification.
    
    Args:
        bias_spec: Can be:
            - "all" or -1: Use all available bias voltages
            - Percentage string like "20%" or "50%": Select percentage of bias voltages evenly spaced
            - Comma-separated values like "66,68,70,72,74": Specific bias voltages
        available_biases: List of available bias voltages
    
    Returns:
        List of selected bias voltages
    """
    if isinstance(bias_spec, str):
        if bias_spec.lower() == "all" or bias_spec == "-1":
            return available_biases
        elif bias_spec.endswith('%'):
            # Percentage-based selection
            try:
                percentage = float(bias_spec.rstrip('%'))
                n_total = len(available_biases)
                n_select = max(1, int(n_total * percentage / 100))
                # Select evenly spaced indices
                if n_select >= n_total:
                    return available_biases
                indices = np.linspace(0, n_total - 1, n_select, dtype=int)
                return [available_biases[i] for i in indices]
            except ValueError:
                print(f"Warning: Invalid percentage '{bias_spec}', using default")
                return [66, 68, 70, 72, 74]
        else:
            # Comma-separated values
            return [float(b.strip()) for b in bias_spec.split(',')]
    elif bias_spec == -1:
        return available_biases
    else:
        return bias_spec


def main():
    # Setup ATLAS plotting style
    setup_atlas_style()
    
    parser = argparse.ArgumentParser(description='Plot counter data from a measurement folder')
    parser.add_argument('data_folder', type=str, help='Path to data folder (e.g., /path/to/SMSPD_data/SMSPD_3/test/2-7/6K)')
    parser.add_argument('--bias', type=str, default='66,68,70,72,74', 
                        help='Bias voltages: "all"/-1 for all, "20%%"/"50%%" for percentage, or comma-separated values in mV (default: 66,68,70,72,74)')
    parser.add_argument('--powers', type=str, default='all', 
                        help='Power levels: "all" for all available, or comma-separated values in nW (default: all)')
    parser.add_argument('--dark-subtract-mode', type=str, default='closest', 
                        help='Dark count subtraction method: "closest" (closest in time) or "latest" (latest file) (default: closest)')
    parser.add_argument('--remove-lowest', type=int, default=0, help='Number of lowest power points to remove (default: 0)')
    parser.add_argument('--tolerance', type=float, default=1.5, help='Bias voltage tolerance in mV (default: 1.5)')
    parser.add_argument('--linear-fit', type=str, default='false', help='Enable linear fit: "true" or "false" (default: false)')
    parser.add_argument('--fit-range', type=str, default='all', help='Fit range: "all" or "min_power,max_power" in nW (default: all)')
    parser.add_argument('--fit-line-range', type=str, default='all', help='Fit line display range: "all" or "min_power,max_power" in nW (default: all)')
    parser.add_argument('--loglog', type=str, default='false', help='Use log-log scale for power plot: "true" or "false" (default: false)')
    parser.add_argument('--yaxis-scale', type=str, default='auto', help='Y-axis scale: "auto" or "min,max" (default: auto)')
    parser.add_argument('--measurement-name', type=str, default=None, help='Measurement name for output (default: derived from folder path)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_folder)
    if not data_dir.exists():
        print(f"Error: Data folder {data_dir} does not exist!")
        return
    
    # Find latest files for each power
    power_files, dark_files = find_latest_files(data_dir)
    
    if not power_files:
        print(f"No signal data files found in {data_dir}")
        return
    
    # Get available bias voltages from the first data file
    first_file = next(iter(power_files.values()))
    available_biases = get_available_bias_voltages(first_file)
    print(f"Available bias voltages: {available_biases} mV")
    
    # Parse target bias voltages based on specification
    target_biases_mv = select_bias_voltages(args.bias, available_biases)
    print(f"Selected bias voltages: {target_biases_mv} mV")
    
    # Filter power levels based on --powers argument
    if args.powers.lower() != 'all':
        try:
            selected_powers = [float(p.strip()) for p in args.powers.split(',')]
            # Filter power_files to only include selected powers
            power_files = {p: f for p, f in power_files.items() if p in selected_powers}
            print(f"Selected powers: {selected_powers} nW")
        except ValueError:
            print(f"Warning: Invalid power specification '{args.powers}', using all powers")
    
    if not power_files:
        print(f"No signal data files found in {data_dir}")
        return
    
    print(f"Processing data from: {data_dir}")
    print(f"Found {len(power_files)} power levels with data")
    for power, filepath in sorted(power_files.items()):
        print(f"  {power} nW: {filepath.name}")
    
    if dark_files:
        print(f"Found {len(dark_files)} dark count files")
    
    # Get measurement name from argument or derive from path
    if args.measurement_name:
        measurement_name = args.measurement_name
    else:
        # Extract measurement name from path hierarchy
        # Structure: /SMSPD_3/{measurement}/2-7/6K/ or /SMSPD_3/{measurement}/6K/
        if data_dir.name == '6K':
            if data_dir.parent.name == '2-7':
                # Path: .../measurement/2-7/6K
                measurement_name = data_dir.parent.parent.name
            else:
                # Path: .../measurement/6K
                measurement_name = data_dir.parent.name
        else:
            measurement_name = data_dir.name
    
    # Create output directory based on measurement name
    output_dir = Path('output') / measurement_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create two subplots with ATLAS style
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Colors for different powers - use distinguishable colors
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(power_files))))
    
    # Data structure for second plot
    row_data = {}
    
    # Use the latest (last) dark count file for plotting
    latest_dark_file = None
    if dark_files:
        latest_dark_file = max(dark_files, key=lambda x: x[1])[0]
    
    for idx, (power, filepath) in enumerate(sorted(power_files.items())):
        print(f"\nProcessing {power} nW from {filepath.name}")
        
        # Select dark count file based on mode
        _, signal_timestamp = parse_filename(filepath.name)
        dark_file_for_subtraction = None
        
        if args.dark_subtract_mode.lower() == 'latest':
            # Use latest dark count file for all signals
            dark_file_for_subtraction = latest_dark_file
            if dark_file_for_subtraction:
                print(f"  Using latest dark count file: {dark_file_for_subtraction.name}")
        else:
            # Use closest dark count file (default)
            if dark_files and signal_timestamp:
                dark_file_for_subtraction = find_closest_dark_file(signal_timestamp, dark_files)
                if dark_file_for_subtraction:
                    print(f"  Using closest dark count file: {dark_file_for_subtraction.name}")
        
        # Read signal data
        bias_voltages, count_rates = read_counter_file(filepath)
        
        # Read dark counts from the selected dark file
        signal_dark_rates = {}
        if dark_file_for_subtraction:
            dark_bias, dark_rates = read_counter_file(dark_file_for_subtraction)
            for bias, rate in zip(dark_bias, dark_rates):
                signal_dark_rates[bias] = rate
        
        # Subtract dark counts for plot 1 only
        if signal_dark_rates:
            dark_bias_array = np.array(list(signal_dark_rates.keys()))
            dark_rates_array = np.array(list(signal_dark_rates.values()))
            dark_interp = np.interp(bias_voltages, dark_bias_array, dark_rates_array)
            count_rates_subtracted = count_rates - dark_interp
        else:
            count_rates_subtracted = count_rates
        
        # Plot 1: Count rate vs bias voltage (dark subtracted)
        ax1.plot(bias_voltages * 1000, count_rates_subtracted, 'o',
                label=f'{power} nW', color=colors[idx], alpha=0.9)
        
        # Collect data for second plot - key by actual bias voltage in mV
        for bias, rate in zip(bias_voltages, count_rates):
            bias_mv = bias * 1000  # Convert to mV for consistent keying
            if bias_mv not in row_data:
                row_data[bias_mv] = {}
            row_data[bias_mv][power] = rate
    
    # Configure first plot - ATLAS style
    ax1.set_xlabel('Bias Voltage (mV)')
    ax1.set_ylabel('Count Rate (counts/s)')
    
    title_suffix = ' (Dark Subtracted)' if latest_dark_file else ''
    ax1.text(0.05, 0.95, f'{measurement_name}{title_suffix}', 
             transform=ax1.transAxes, fontsize=18, fontweight='bold',
             verticalalignment='top')
    ax1.legend(loc='best', frameon=True)
    ax1.set_ylim(bottom=0)
    
    # Enable minor ticks
    ax1.minorticks_on()
    
    # Plot dark count curve from latest dark file on same y-axis
    if latest_dark_file:
        dark_bias, dark_rates = read_counter_file(latest_dark_file)
        ax1.plot(dark_bias * 1000, dark_rates, 's:', color='dimgray', 
                alpha=0.7, label='Dark Count')
        ax1.legend(loc='best', frameon=True)
    
    # Plot 2: Count rate vs power for selected bias voltages
    colors_biases = plt.cm.rainbow(np.linspace(0, 1, len(target_biases_mv)))
    color_idx = 0
    
    # Sort bias voltages and filter to those with multiple power points
    all_bias_mv = sorted(row_data.keys())
    valid_bias_mv = [bias_mv for bias_mv in all_bias_mv if len(row_data[bias_mv]) > 1]
    
    if valid_bias_mv:
        for bias_mv in valid_bias_mv:
            # Check if this bias is close to any target
            matched = False
            for target in target_biases_mv:
                if abs(bias_mv - target) <= args.tolerance:
                    matched = True
                    break
            
            if not matched:
                continue
            
            power_rate_pairs = sorted(row_data[bias_mv].items())
            powers = np.array([p for p, rate in power_rate_pairs])
            rates = np.array([rate for p, rate in power_rate_pairs])
            
            # Filter out non-positive rates and very small values for log-log plots
            if args.loglog.lower() == 'true':
                # For log-log plots, remove zeros and very small values
                min_threshold = 1e-3  # Minimum threshold for log scale
                valid_mask = (rates > min_threshold) & (powers > min_threshold)
            else:
                # For linear plots, just remove non-positive rates
                valid_mask = rates > 0
            
            powers_valid = powers[valid_mask]
            rates_valid = rates[valid_mask]
            
            if len(powers_valid) < 2:
                continue
            
            plot_color = colors_biases[color_idx % len(colors_biases)]
            
            # Perform linear fit if enabled
            if args.linear_fit.lower() == 'true' and len(powers_valid) >= 2:
                # Determine fit range
                if args.fit_range.lower() == 'all':
                    fit_mask = np.ones(len(powers_valid), dtype=bool)
                else:
                    try:
                        fit_min, fit_max = map(float, args.fit_range.split(','))
                        fit_mask = (powers_valid >= fit_min) & (powers_valid <= fit_max)
                    except:
                        print(f"Warning: Invalid fit range '{args.fit_range}', using all data")
                        fit_mask = np.ones(len(powers_valid), dtype=bool)
                
                powers_fit = powers_valid[fit_mask]
                rates_fit = rates_valid[fit_mask]
                
                if len(powers_fit) >= 2:
                    try:
                        # Check if log-log scale is enabled for power-law fit
                        if args.loglog.lower() == 'true':
                            # Power-law fit: Rate = A * Power^n
                            # In log space: log(Rate) = log(A) + n*log(Power)
                            log_powers_fit = np.log(powers_fit)
                            log_rates_fit = np.log(rates_fit)
                            
                            coeffs, cov = np.polyfit(log_powers_fit, log_rates_fit, 1, cov=True)
                            n = coeffs[0]  # power exponent
                            log_A = coeffs[1]
                            A = np.exp(log_A)
                            n_err = np.sqrt(cov[0, 0])
                            log_A_err = np.sqrt(cov[1, 1])
                            A_err = A * log_A_err  # Error propagation for A
                            
                            # Calculate chi-squared in log space
                            log_fit_rates = n * log_powers_fit + log_A
                            log_residuals = log_rates_fit - log_fit_rates
                            
                            # Uncertainty in log space
                            log_uncertainties = 1.0 / np.sqrt(np.abs(rates_fit))  # Approximate
                            log_uncertainties[log_uncertainties == 0] = 1
                            
                            chi_squared = np.sum((log_residuals / log_uncertainties)**2)
                            ndf = len(powers_fit) - 2
                            chi2_ndf = chi_squared / ndf if ndf > 0 else 0
                            
                            # Plot data points with power-law fit label
                            label_combined = f'{bias_mv:.1f}mV: n={n:.2f}±{n_err:.2f}, χ²/ndf={chi2_ndf:.2f}'
                            ax2.plot(powers_valid, rates_valid, 'o', 
                                    label=label_combined, color=plot_color, alpha=0.9)
                            
                            print(f"  Bias {bias_mv:.1f} mV power-law fit:")
                            print(f"    Rate = ({A:.2e} ± {A_err:.2e}) * Power^({n:.3f} ± {n_err:.3f})")
                            print(f"    χ²/ndf = {chi_squared:.2f}/{ndf} = {chi2_ndf:.3f}")
                            
                        else:
                            # Linear fit: Rate = slope * Power + intercept using polyfit with covariance
                            coeffs, cov = np.polyfit(powers_fit, rates_fit, 1, cov=True)
                            slope = coeffs[0]
                            intercept = coeffs[1]
                            slope_err = np.sqrt(cov[0, 0])
                            intercept_err = np.sqrt(cov[1, 1])
                            
                            # Calculate chi-squared and ndf
                            fit_rates = slope * powers_fit + intercept
                            residuals = rates_fit - fit_rates
                            
                            # Use Poisson uncertainty: sigma = sqrt(counts)
                            # For count rates, we approximate uncertainty as sqrt(rate)
                            uncertainties = np.sqrt(np.abs(rates_fit))
                            uncertainties[uncertainties == 0] = 1  # Avoid division by zero
                            
                            chi_squared = np.sum((residuals / uncertainties)**2)
                            ndf = len(powers_fit) - 2  # number of data points - number of parameters
                            chi2_ndf = chi_squared / ndf if ndf > 0 else 0
                            
                            # Plot data points with combined label (compact format)
                            label_combined = f'{bias_mv:.1f}mV: {slope:.2f}±{slope_err:.2f}, χ²/ndf={chi2_ndf:.2f}'
                            ax2.plot(powers_valid, rates_valid, 'o', 
                                    label=label_combined, color=plot_color, alpha=0.9)
                            
                            print(f"  Bias {bias_mv:.1f} mV linear fit:")
                            print(f"    Rate = ({slope:.3f} ± {slope_err:.3f}) * Power + ({intercept:.1f} ± {intercept_err:.1f})")
                            print(f"    χ²/ndf = {chi_squared:.2f}/{ndf} = {chi2_ndf:.3f}")
                        
                        # Plot fit line with specified range
                        if args.fit_line_range.lower() == 'all':
                            power_fit_min = powers_valid.min()
                            power_fit_max = powers_valid.max()
                        else:
                            try:
                                line_min, line_max = map(float, args.fit_line_range.split(','))
                                power_fit_min = max(line_min, powers_valid.min())
                                power_fit_max = min(line_max, powers_valid.max())
                            except:
                                print(f"Warning: Invalid fit line range '{args.fit_line_range}', using all data")
                                power_fit_min = powers_valid.min()
                                power_fit_max = powers_valid.max()
                        
                        power_fit_line = np.linspace(power_fit_min, power_fit_max, 100)
                        
                        # Calculate fit line based on fit type
                        if args.loglog.lower() == 'true':
                            # Power-law: Rate = A * Power^n
                            rate_fit_line = A * power_fit_line**n
                        else:
                            # Linear: Rate = slope * Power + intercept
                            rate_fit_line = slope * power_fit_line + intercept
                        
                        ax2.plot(power_fit_line, rate_fit_line, '--', color=plot_color, 
                                alpha=0.7)
                        
                    except Exception as e:
                        print(f"  Fit failed for {bias_mv:.1f} mV: {e}")
                        # Plot data only if fit fails
                        ax2.plot(powers_valid, rates_valid, 'o', 
                                label=f'{bias_mv:.1f} mV', color=plot_color, alpha=0.9)
            else:
                # Plot data points without fit
                ax2.plot(powers_valid, rates_valid, 'o', 
                        label=f'{bias_mv:.1f} mV', color=plot_color, alpha=0.9)
            
            color_idx += 1
    
    # Add dark count reference lines if available (use latest dark file)
    # Filter out very small dark count values (< 1) to avoid extending y-axis range
    if latest_dark_file:
        dark_bias, dark_rates = read_counter_file(latest_dark_file)
        dark_dict = {bias * 1000: rate for bias, rate in zip(dark_bias, dark_rates) if rate >= 1.0}
        
        # Show dark count line for each selected bias voltage (no legend labels)
        for target in target_biases_mv:
            for bias_mv, dark_rate in dark_dict.items():
                if abs(bias_mv - target) <= args.tolerance:
                    ax2.axhline(y=dark_rate, color='dimgray', linestyle=':', 
                               alpha=0.5)
                    break
    
    # Configure second plot - ATLAS style
    ax2.set_xlabel('Optical Power (nW)')
    ax2.set_ylabel('Count Rate (counts/s)')
    ax2.text(0.05, 0.95, f'{measurement_name} (Raw data)', 
             transform=ax2.transAxes, fontsize=18, fontweight='bold',
             verticalalignment='top')
    
    # Dynamic legend: more columns for more entries, smaller font, tighter spacing
    num_entries = len([h for h in ax2.get_legend_handles_labels()[0]])
    if num_entries > 10:
        legend_ncol = 4
        legend_fontsize = 11
    elif num_entries > 6:
        legend_ncol = 3
        legend_fontsize = 12
    else:
        legend_ncol = 2
        legend_fontsize = 13
    
    # Position legend based on plot type
    if args.loglog.lower() == 'true':
        legend_loc = 'lower right'
        legend_bbox = (1, 0)
    else:
        legend_loc = 'upper left'
        legend_bbox = (0, 1)
    
    ax2.legend(fontsize=legend_fontsize, loc=legend_loc, ncol=legend_ncol, 
              frameon=True, bbox_to_anchor=legend_bbox,
              columnspacing=0.8, handlelength=1.5, handletextpad=0.5)
    
    # Enable minor ticks
    ax2.minorticks_on()
    
    # Apply log-log scale if requested
    if args.loglog.lower() == 'true':
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    else:
        ax2.set_ylim(bottom=0)
    
    # Apply custom y-axis scale if specified
    if args.yaxis_scale.lower() != 'auto':
        try:
            ymin, ymax = map(float, args.yaxis_scale.split(','))
            ax2.set_ylim(ymin, ymax)
        except:
            print(f"Warning: Invalid y-axis scale '{args.yaxis_scale}', using auto")
    
    plt.tight_layout()
    
    # Save plot with descriptive name using measurement name
    output_file = output_dir / f'{measurement_name}_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    main()
