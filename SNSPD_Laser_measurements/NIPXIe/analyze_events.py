#!/usr/bin/env python3
"""
Stage 2 Analysis: Statistical analysis of event-by-event JSON data

Reads JSON files from SelfTrigger (Stage 1), performs statistical analysis including:
- Histogram fitting (Gaussian fits for timing variables)
- Mean, median, standard deviation calculations
- Error propagation and uncertainties
- Outputs analyzed statistics to JSON for Stage 3 (plot_all)

Workflow:
  Stage 1: SelfTrigger.py     (TDMS → event JSON)
  Stage 2: analyze_events.py  (event JSON → statistics JSON)  ← THIS SCRIPT
  Stage 3: plot_all.py        (statistics JSON → comparison plots)
"""

import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import norm
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Stage 2: Statistical analysis of event-by-event JSON data',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Full analysis with plots
  python analyze_events.py event0_analysis.json -d output/
  
  # Statistics only (no plots), for batch processing
  python analyze_events.py *.json --no-plots
  
  # Custom binning for better fits
  python analyze_events.py event0_analysis.json -b 100
"""
)
parser.add_argument('in_filenames', nargs="+", help='Input JSON analysis files from SelfTrigger')
parser.add_argument('--output_dir', '-d', default='.', help='Output directory for plots and statistics')
parser.add_argument('--bins', '-b', type=int, default=50, help='Number of bins for histograms')
parser.add_argument('--no-plots', action='store_true', help='Skip plot generation, only compute statistics')
args = parser.parse_args()

def read_file(filename):
    """Read JSON file and extract event-by-event data"""
    with open(filename) as f:
        data = json.load(f)
    
    # Extract event data from current JSON structure
    if isinstance(data, dict):
        if 'event_by_event_data' in data:
            events = data['event_by_event_data']
            metadata = data.get('metadata', {})
            summary = data.get('summary_statistics', {})
            return events, metadata, summary
        elif 'events' in data:
            return data['events'], {}, {}
        elif 'data' in data:
            return data['data'], {}, {}
    
    # If data is already a list
    if isinstance(data, list):
        return data, {}, {}
    
    return [], {}, {}

def plot_variable_vs_event(data, variable_name, filename, metadata):
    """Plot a variable vs event number"""
    event_numbers = [event['event_number'] for event in data if variable_name in event]
    values = [event[variable_name] for event in data if variable_name in event]
    
    if not values:
        return None
    
    plt.figure(figsize=(12, 6))
    plt.plot(event_numbers, values, marker='o', linestyle='', markersize=2, alpha=0.5)
    plt.xlabel('Event Number', fontsize=12)
    plt.ylabel(variable_name, fontsize=12)
    
    # Add metadata to title
    bias_mV = metadata.get('Bias Voltage (mV)', 'N/A')
    power_uW = metadata.get('Laser Power (uW)', 'N/A')
    title = f'{variable_name} vs Event Number\n'
    title += f'Bias: {bias_mV} mV, Power: {power_uW} µW'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def gaussian(x, amplitude, mean, sigma):
    """Gaussian function for curve fitting"""
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

def compute_variable_statistics(data, variable_name, bins=50):
    """
    Compute comprehensive statistics for a variable
    
    Returns dict with:
        - mean, std, median, sem (standard error of mean)
        - For trigger_check: Gaussian fit parameters (mu, sigma, amplitude) and errors
    """
    values = [event[variable_name] for event in data if variable_name in event]
    
    if not values:
        return None
    
    values_array = np.array(values)
    stats = {
        'variable': variable_name,
        'count': len(values),
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array, ddof=1)),  # Sample std
        'sem': float(np.std(values_array, ddof=1) / np.sqrt(len(values_array))),  # Standard error
        'median': float(np.median(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'q25': float(np.percentile(values_array, 25)),
        'q75': float(np.percentile(values_array, 75))
    }
    
    # Add Gaussian fit for trigger_check
    if variable_name == 'trigger_check':
        try:
            # Use more bins for trigger_check
            bins_to_use = max(bins, 100)
            counts, bin_edges = np.histogram(values_array, bins=bins_to_use)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Initial guess for Gaussian parameters
            amplitude_guess = np.max(counts)
            mean_guess = stats['mean']
            sigma_guess = stats['std']
            
            # Perform Gaussian fit
            popt, pcov = curve_fit(
                gaussian, bin_centers, counts, 
                p0=[amplitude_guess, mean_guess, sigma_guess],
                maxfev=10000
            )
            
            # Extract fit parameters and errors
            perr = np.sqrt(np.diag(pcov))
            
            stats['gaussian_fit'] = {
                'amplitude': float(popt[0]),
                'amplitude_err': float(perr[0]),
                'mu': float(popt[1]),
                'mu_err': float(perr[1]),
                'sigma': float(popt[2]),
                'sigma_err': float(perr[2]),
                'fit_success': True
            }
            
        except Exception as e:
            stats['gaussian_fit'] = {
                'fit_success': False,
                'error_message': str(e)
            }
    
    return stats

def plot_variable_histogram(data, variable_name, filename, metadata, bins=50, stats_dict=None):
    """Plot histogram of a variable with optional pre-computed statistics"""
    values = [event[variable_name] for event in data if variable_name in event]
    
    if not values:
        return None, None
    
    # Compute statistics if not provided
    if stats_dict is None:
        stats_dict = compute_variable_statistics(data, variable_name, bins)
    
    # Use more bins for trigger_check to get better fit
    if variable_name == 'trigger_check':
        bins = 100
    
    plt.figure(figsize=(12, 6))
    counts, bin_edges, patches = plt.hist(values, bins=bins, alpha=0.7, edgecolor='black')
    plt.xlabel(variable_name, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.yscale('log')
    
    # Add statistics from computed values
    mean_val = stats_dict['mean']
    std_val = stats_dict['std']
    median_val = stats_dict['median']
    sem_val = stats_dict['sem']
    
    plt.axvline(mean_val, color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_val:.4f} ± {sem_val:.4f}')
    plt.axvline(median_val, color='g', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=1, label=f'±1σ: {std_val:.4f}')
    plt.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=1)
    
    # Add Gaussian fit for trigger_check
    if variable_name == 'trigger_check' and 'gaussian_fit' in stats_dict:
        fit = stats_dict['gaussian_fit']
        if fit['fit_success']:
            # Generate smooth curve for plotting
            x_fit = np.linspace(min(values), max(values), 1000)
            y_fit = gaussian(x_fit, fit['amplitude'], fit['mu'], fit['sigma'])
            
            # Plot the fit
            plt.plot(x_fit, y_fit, 'b-', linewidth=2.5, 
                    label=f'Gaussian Fit: μ={fit["mu"]:.2f}±{fit["mu_err"]:.2f}, σ={fit["sigma"]:.2f}±{fit["sigma_err"]:.2f}')
            
            # Print fit results
            print(f"  {variable_name} Gaussian fit: μ={fit['mu']:.4f}±{fit['mu_err']:.4f}, "
                  f"σ={fit['sigma']:.4f}±{fit['sigma_err']:.4f}, A={fit['amplitude']:.2f}±{fit['amplitude_err']:.2f}")
    
    # Add metadata to title
    bias_mV = metadata.get('Bias Voltage (mV)', 'N/A')
    power_uW = metadata.get('Laser Power (uW)', 'N/A')
    title = f'{variable_name} Histogram\n'
    title += f'Bias: {bias_mV} mV, Power: {power_uW} µW'
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print(f"  {variable_name}: mean={mean_val:.4f}±{sem_val:.4f}, std={std_val:.4f}, median={median_val:.4f}")
    
    return plt.gcf(), stats_dict

def plot_2d_correlation(data, var_x, var_y, filename, metadata, bins=50):
    """Plot 2D correlation between two variables"""
    # Filter events that have both variables
    values_x = []
    values_y = []
    for event in data:
        if var_x in event and var_y in event:
            values_x.append(event[var_x])
            values_y.append(event[var_y])
    
    if not values_x or not values_y:
        return None
    
    plt.figure(figsize=(10, 8))
    plt.hist2d(values_x, values_y, bins=bins, cmap='viridis', cmin=1)
    plt.colorbar(label='Counts')
    
    # Calculate and display correlation
    correlation = np.corrcoef(values_x, values_y)[0, 1]
    
    plt.xlabel(var_x, fontsize=12)
    plt.ylabel(var_y, fontsize=12)
    
    # Add metadata to title
    bias_mV = metadata.get('Bias Voltage (mV)', 'N/A')
    power_uW = metadata.get('Laser Power (uW)', 'N/A')
    title = f'2D Correlation: {var_x} vs {var_y}\n'
    title += f'Bias: {bias_mV} mV, Power: {power_uW} µW\n'
    title += f'Correlation: {correlation:.3f}'
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print(f"  Correlation ({var_x} vs {var_y}): {correlation:.4f}")
    
    return plt.gcf()

def plot_summary_comparison(summary, metadata, filename):
    """Plot comparison of signal vs dark event characteristics"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract signal and all event statistics
    signal_ptp_mean = summary.get('signal_pulse_fall_range_ptp_mean', 0)
    signal_ptp_std = summary.get('signal_pulse_fall_range_ptp_std', 0)
    all_ptp_mean = summary.get('pulse_fall_range_ptp_mean', 0)
    all_ptp_std = summary.get('pulse_fall_range_ptp_std', 0)
    
    signal_rise_mean = summary.get('signal_rise_amplitude_mean', 0)
    signal_rise_std = summary.get('signal_rise_amplitude_std', 0)
    all_rise_mean = summary.get('rise_amplitude_mean', 0)
    all_rise_std = summary.get('rise_amplitude_std', 0)
    
    # Plot 1: Pulse Fall Range PTP
    ax = axes[0, 0]
    categories = ['All Events', 'Signal Events']
    means = [all_ptp_mean, signal_ptp_mean]
    stds = [all_ptp_std, signal_ptp_std]
    ax.bar(categories, means, yerr=stds, capsize=5, alpha=0.7, color=['blue', 'green'])
    ax.set_ylabel('Pulse Fall Range PTP (V)', fontsize=11)
    ax.set_title('Pulse Fall Range PTP Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Rise Amplitude
    ax = axes[0, 1]
    means = [all_rise_mean, signal_rise_mean]
    stds = [all_rise_std, signal_rise_std]
    ax.bar(categories, means, yerr=stds, capsize=5, alpha=0.7, color=['blue', 'green'])
    ax.set_ylabel('Rise Amplitude (V)', fontsize=11)
    ax.set_title('Rise Amplitude Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Rate summary
    ax = axes[1, 0]
    count_rate = summary.get('count_rate', 0)
    signal_rate = summary.get('signal_rate', 0)
    dark_rate = summary.get('dark_count_rate', 0)
    
    categories = ['Count Rate', 'Signal Rate', 'Dark Rate']
    rates = [count_rate, signal_rate, dark_rate]
    colors = ['purple', 'green', 'red']
    ax.bar(categories, rates, alpha=0.7, color=colors)
    ax.set_ylabel('Rate (Hz)', fontsize=11)
    ax.set_title('Event Rates', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Efficiency
    ax = axes[1, 1]
    efficiency = summary.get('efficiency', 0)
    efficiency_error = summary.get('efficiency_error', 0)
    ax.bar(['Efficiency'], [efficiency * 100], yerr=[efficiency_error * 100], 
           capsize=5, alpha=0.7, color='orange')
    ax.set_ylabel('Efficiency (%)', fontsize=11)
    ax.set_title('Detection Efficiency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(efficiency * 100 * 1.2, 0.1))
    
    # Overall title
    bias_mV = metadata.get('Bias Voltage (mV)', 'N/A')
    power_uW = metadata.get('Laser Power (uW)', 'N/A')
    fig.suptitle(f'Summary Statistics\nBias: {bias_mV} mV, Power: {power_uW} µW', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Variables to analyze
    variables_to_plot = [
        'pre_mean', 'pulse_max', 'pulse_min', 
        'pulse_time', 'pulse_time_interval',
        'pulse_rise_range_ptb', 'pulse_fall_range_ptp',
        'rise_amplitude', 'rise_time_10_90', 'fall_time_90_10',
        'trigger_check'
    ]
    
    # 2D correlations to plot
    correlations_to_plot = [
        ('pulse_time_interval', 'pulse_fall_range_ptp'),
        ('pulse_time_interval', 'rise_amplitude'),
        ('pulse_fall_range_ptp', 'rise_amplitude'),
        ('rise_time_10_90', 'fall_time_90_10'),
        ('trigger_check', 'pulse_fall_range_ptp'),
        ('trigger_check', 'rise_amplitude'),
        ('trigger_check', 'pulse_time_interval')
    ]
    
    for filename in args.in_filenames:
        print(f"\n{'='*70}")
        print(f"Processing: {filename}")
        print(f"{'='*70}")
        
        events, metadata, summary = read_file(filename)
        
        if not events:
            print(f"  Warning: No event data found in {filename}")
            continue
        
        print(f"  Total events: {len(events)}")
        
        # Print summary statistics if available
        if summary:
            print(f"\n  Summary Statistics from Stage 1:")
            print(f"    Count Rate: {summary.get('count_rate', 0):.2f} Hz")
            print(f"    Signal Rate: {summary.get('signal_rate', 0):.2f} Hz")
            print(f"    Dark Count Rate: {summary.get('dark_count_rate', 0):.2f} Hz")
            print(f"    Efficiency: {summary.get('efficiency', 0):.6f}")
        
        # Create base output filename
        base_name = Path(filename).stem
        
        # =====================================================================
        # STAGE 2: COMPUTE STATISTICS FOR ALL VARIABLES
        # =====================================================================
        print(f"\n  Computing statistics for all variables:")
        variable_statistics = {}
        
        for var in variables_to_plot:
            stats = compute_variable_statistics(events, var, bins=args.bins)
            if stats:
                variable_statistics[var] = stats
        
        # Prepare output statistics JSON
        output_statistics = {
            'input_file': filename,
            'metadata': metadata,
            'summary_from_stage1': summary,
            'variable_statistics': variable_statistics,
            'analysis_timestamp': datetime.now().isoformat(),
            'bins_used': args.bins
        }
        
        # Save statistics to JSON
        stats_output_path = os.path.join(args.output_dir, f"statistics_{base_name}.json")
        with open(stats_output_path, 'w') as f:
            json.dump(output_statistics, f, indent=2)
        print(f"\n  ✓ Statistics saved to: statistics_{base_name}.json")
        
        # =====================================================================
        # PLOTTING (optional, skip if --no-plots)
        # =====================================================================
        if args.no_plots:
            print(f"  Skipping plots (--no-plots enabled)")
            continue
        
        # Plot summary comparison
        if summary:
            try:
                fig = plot_summary_comparison(summary, metadata, filename)
                output_path = os.path.join(args.output_dir, f"summary_{base_name}.png")
                fig.savefig(output_path, dpi=150)
                plt.close(fig)
                print(f"\n  Saved: summary_{base_name}.png")
            except Exception as e:
                print(f"  Warning: Could not create summary plot - {e}")
        
        # Plot individual variables
        print(f"\n  Plotting individual variables:")
        for var in variables_to_plot:
            try:
                # Get pre-computed statistics
                stats = variable_statistics.get(var)
                
                # Plot histogram
                fig, _ = plot_variable_histogram(events, var, filename, metadata, 
                                                  bins=args.bins, stats_dict=stats)
                if fig:
                    output_path = os.path.join(args.output_dir, f"{var}_histogram_{base_name}.png")
                    fig.savefig(output_path, dpi=150)
                    plt.close(fig)
                
                # Plot vs event number
                fig = plot_variable_vs_event(events, var, filename, metadata)
                if fig:
                    output_path = os.path.join(args.output_dir, f"{var}_vs_event_{base_name}.png")
                    fig.savefig(output_path, dpi=150)
                    plt.close(fig)
                    
            except Exception as e:
                print(f"  Warning: Could not plot '{var}' - {e}")
        
        # Plot 2D correlations
        print(f"\n  Plotting 2D correlations:")
        for var_x, var_y in correlations_to_plot:
            try:
                fig = plot_2d_correlation(events, var_x, var_y, filename, metadata, bins=args.bins)
                if fig:
                    output_path = os.path.join(args.output_dir, f"correlation_{var_x}_vs_{var_y}_{base_name}.png")
                    fig.savefig(output_path, dpi=150)
                    plt.close(fig)
            except Exception as e:
                print(f"  Warning: Could not create correlation plot ({var_x} vs {var_y}) - {e}")
        
        print(f"\n  Completed processing: {filename}")
    
    print(f"\n{'='*70}")
    print(f"Stage 2 Analysis Complete")
    print(f"Statistics saved to: {args.output_dir}")
    if not args.no_plots:
        print(f"Plots saved to: {args.output_dir}")
    print(f"{'='*70}\n")
            print(f"    Signal Rate: {summary.get('signal_rate', 0):.2f} Hz")
            print(f"    Dark Count Rate: {summary.get('dark_count_rate', 0):.2f} Hz")
            print(f"    Efficiency: {summary.get('efficiency', 0):.6f}")
        
        # Create base output filename
        base_name = Path(filename).stem
        
        # Plot summary comparison
        if summary:
            try:
                fig = plot_summary_comparison(summary, metadata, filename)
                output_path = os.path.join(args.output_dir, f"summary_{base_name}.png")
                fig.savefig(output_path, dpi=150)
                plt.close(fig)
                print(f"\n  Saved: summary_{base_name}.png")
            except Exception as e:
                print(f"  Warning: Could not create summary plot - {e}")
        
        # Plot individual variables
        print(f"\n  Plotting individual variables:")
        for var in variables_to_plot:
            try:
                # Plot histogram
                fig = plot_variable_histogram(events, var, filename, metadata, bins=args.bins)
                if fig:
                    output_path = os.path.join(args.output_dir, f"{var}_histogram_{base_name}.png")
                    fig.savefig(output_path, dpi=150)
                    plt.close(fig)
                
                # Plot vs event number
                fig = plot_variable_vs_event(events, var, filename, metadata)
                if fig:
                    output_path = os.path.join(args.output_dir, f"{var}_vs_event_{base_name}.png")
                    fig.savefig(output_path, dpi=150)
                    plt.close(fig)
                    
            except Exception as e:
                print(f"  Warning: Could not plot '{var}' - {e}")
        
        # Plot 2D correlations
        print(f"\n  Plotting 2D correlations:")
        for var_x, var_y in correlations_to_plot:
            try:
                fig = plot_2d_correlation(events, var_x, var_y, filename, metadata, bins=args.bins)
                if fig:
                    output_path = os.path.join(args.output_dir, f"correlation_{var_x}_vs_{var_y}_{base_name}.png")
                    fig.savefig(output_path, dpi=150)
                    plt.close(fig)
            except Exception as e:
                print(f"  Warning: Could not create correlation plot ({var_x} vs {var_y}) - {e}")
        
        print(f"\n  Completed processing: {filename}")
    
    print(f"\n{'='*70}")
    print(f"All plots saved to: {args.output_dir}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
