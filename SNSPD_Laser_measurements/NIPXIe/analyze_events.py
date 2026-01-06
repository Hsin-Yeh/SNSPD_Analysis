# --- Custom selection functions for analysis variables ---
def is_true_laser_sync(event):
    # Check both possible field names for laser sync timing
    sync_val = event.get('laser_sync_arrival', event.get('laser_sync_time', 0))
    return 194 < sync_val < 203

def is_dark_laser_sync(event):
    # Check both possible field names for laser sync timing
    sync_val = event.get('laser_sync_arrival', event.get('laser_sync_time', 0))
    return sync_val <= 194 or sync_val >= 203
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
import shutil

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
  python analyze_events.py event0_analysis.json -b 200
"""
)
parser.add_argument('in_filenames', nargs="+", help='Input JSON analysis files from SelfTrigger')
parser.add_argument('--output_dir', '-d', default='.', help='Output directory for plots and statistics')
# Removed --bins argument, now using bins_dict per variable
parser.add_argument('--no-plots', action='store_true', help='Skip plot generation, only compute statistics')
parser.add_argument('--reset', action='store_true', help='Remove all output files and folders')
parser.add_argument('--scan', action='store_true', help='Scan for missing output files')
parser.add_argument('--update', action='store_true', help='Analyze only missing outputs')
parser.add_argument('--restart', action='store_true', help='Reset and analyze all outputs')

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

def compute_variable_statistics(data, variable_name, bins=50, selection=None):
    """
    Compute comprehensive statistics for a variable
    
    Returns dict with:
        - mean, std, median, sem (standard error of mean)
        - For laser_sync_arrival: Gaussian fit parameters (mu, sigma, amplitude) and errors
    """
    if selection:
        if callable(selection):
            values = [event[variable_name] for event in data if variable_name in event and selection(event)]
        else:
            # Evaluate string with event dict in scope
            values = [event[variable_name] for event in data if variable_name in event and eval(selection, {}, event)]
    else:
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
    }    
    return stats

def fit_gaussian_to_histogram(values_array, stats, bins):
    """Fit a Gaussian to the histogram of values_array and return fit results as a dict."""
    from scipy.optimize import curve_fit
    import numpy as np
    try:
        bins_to_use = max(bins, 200)
        counts, bin_edges = np.histogram(values_array, bins=bins_to_use)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        amplitude_guess = np.max(counts)
        mean_guess = stats['mean']
        sigma_guess = abs(stats['std'])

        popt, pcov = curve_fit(
            gaussian, bin_centers, counts,
            p0=[amplitude_guess, mean_guess, sigma_guess],
            bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]),
            maxfev=10000
        )
        perr = np.sqrt(np.diag(pcov))
        return {
            'amplitude': float(popt[0]),
            'amplitude_err': float(perr[0]),
            'mu': float(popt[1]),
            'mu_err': float(perr[1]),
            'sigma': float(popt[2]),
            'sigma_err': float(perr[2]),
            'fit_success': True
        }
    except Exception as e:
        return {
            'fit_success': False,
            'error_message': str(e)
        }
    
    return stats

def plot_variable_histogram(data, variable_name, filename, metadata, bins=200, stats_dict=None, xlim=None, selection=None, title_name=None):
    """Plot histogram of a variable with optional pre-computed statistics"""
    if selection:
        if callable(selection):
            values = [event[variable_name] for event in data if variable_name in event and selection(event)]
        else:
            values = [event[variable_name] for event in data if variable_name in event and eval(selection, {}, event)]
    else:
        values = [event[variable_name] for event in data if variable_name in event]
    
    if not values:
        return None, None

    # Handle xlim and overflow bins
    if xlim is not None:
        lower, upper = xlim
        # Assign values below lower to lower, above upper to upper
        clipped_values = []
        for v in values:
            if v < lower:
                clipped_values.append(lower)
            elif v > upper:
                clipped_values.append(upper)
            else:
                clipped_values.append(v)
        values = clipped_values
        # Create bins with first and last bin as overflow bins
        bin_edges = np.linspace(lower, upper, bins - 1)
        # Add one bin below and one above for under/overflow
        bin_edges = np.concatenate(([lower - (bin_edges[1] - bin_edges[0])], bin_edges, [upper + (bin_edges[1] - bin_edges[0])]))
    else:
        bin_edges = bins

    plt.figure(figsize=(12, 6))
    counts, bin_edges, patches = plt.hist(values, bins=bin_edges, alpha=0.7, edgecolor='black')
    plt.xlabel(variable_name, fontsize=12)
    bin_size = bin_edges[1] - bin_edges[0]
    plt.ylabel(f'Events / {bin_size:.3g}', fontsize=12)
    
    # Add statistics from computed values
    mean_val = stats_dict['mean']
    std_val = stats_dict['std']
    sem_val = stats_dict['sem']

    # Add Gaussian fit 
    if 'gaussian_fit' in stats_dict:
        fit = stats_dict['gaussian_fit']
        if fit['fit_success']:
            # Generate smooth curve for plotting
            x_fit = np.linspace(min(values), max(values), 1000)
            y_fit = gaussian(x_fit, fit['amplitude'], fit['mu'], fit['sigma'])
            plt.plot(x_fit, y_fit, 'b-', linewidth=2.5)
            print(f"  {variable_name} Gaussian fit: μ={fit['mu']:.4f}±{fit['mu_err']:.4f}, "
                  f"σ={fit['sigma']:.4f}±{fit['sigma_err']:.4f}, A={fit['amplitude']:.2f}±{fit['amplitude_err']:.2f}")

    # Add metadata to title
    bias_mV = metadata.get('Bias Voltage (mV)', 'N/A')
    power_uW = metadata.get('Laser Power (uW)', 'N/A')
    title = f'{title_name} Histogram\n'
    title += f'Bias: {bias_mV} mV, Power: {power_uW} µW'
    plt.title(title, fontsize=14, fontweight='bold')

    # Build statistical box text
    stats_text = (
        f"Total Events: {stats_dict['count']}\n"
        f"Mean = {mean_val:.4f} ± {sem_val:.4f}\n"
        f"Std = {std_val:.4f}"
    )
    # Add fit statistics if available
    if 'gaussian_fit' in stats_dict:
        fit = stats_dict['gaussian_fit']
        if fit['fit_success']:
            stats_text += (
                f"\nGaussian Fit:\n"
                f"μ = {fit['mu']:.4f} ± {fit['mu_err']:.4f}\n"
                f"σ = {fit['sigma']:.4f} ± {fit['sigma_err']:.4f}\n"
                f"A = {fit['amplitude']:.2f} ± {fit['amplitude_err']:.2f}"
            )
        else:
            stats_text += f"\nGaussian Fit Failed:\n{fit['error_message']}"
    plt.gca().text(0.98, 0.98, stats_text, fontsize=11, va='top', ha='right', transform=plt.gca().transAxes,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))

    # Remove legend from histogram
    # plt.legend(fontsize=10)  # Removed
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    print(f"  {variable_name}: mean={mean_val:.4f}±{sem_val:.4f}, std={std_val:.4f}")

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
    # title += f'Correlation: {correlation:.3f}'
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print(f"  Correlation ({var_x} vs {var_y}): {correlation:.4f}")
    
    return plt.gcf()

def determine_output_directory(input_filename: str, default_dir: str, stage: str = 'stage2_statistics') -> str:
    """
    Determine output directory based on input file location, matching SelfTrigger.py logic.
    By default, uses 'stage2_statistics' for Stage 2 outputs.
    """
    abs_path = os.path.abspath(input_filename)
    path_parts = abs_path.split(os.sep)
    for i, part in enumerate(path_parts):
        if part in ['SNSPD_rawdata', 'SNSPD_data', 'SNSPD_analyzed_output']:
            base_path = os.sep.join(path_parts[:i])
            output_base_dir = os.path.join(base_path, 'SNSPD_analyzed_output', stage)
            subdirs = path_parts[i+2:-1]  # Get subdirectory structure
            if subdirs:
                return os.path.join(output_base_dir, *subdirs) + '/'
            return output_base_dir + '/'
    return default_dir + '/'

def calculate_key_variables(events, signal_selection, dark_selection, time_window_s, source_rate):
    """
    Calculate signal count rate, dark count rate, efficiency, and errors.
    - events: list of event dicts
    - signal_selection: function(event) -> bool
    - dark_selection: function(event) -> bool
    - time_window_s: total acquisition time in seconds
    Returns: dict with keys: signal_count_rate, dark_count_rate, efficiency, error_signal, error_dark, error_efficiency
    """
    n_signal = sum(1 for event in events if signal_selection(event))
    n_dark = sum(1 for event in events if dark_selection(event))
    n_total = len(events)
    # Count rates
    count_rate = n_total / time_window_s if time_window_s > 0 else 0
    signal_count_rate = n_signal / time_window_s if time_window_s > 0 else 0
    dark_count_rate = n_dark / time_window_s if time_window_s > 0 else 0
    efficiency = signal_count_rate / source_rate if source_rate > 0 else 0
    # Poisson errors
    signal_count_rate_error = (n_signal ** 0.5) / time_window_s if time_window_s > 0 else 0
    dark_count_rate_error = (n_dark ** 0.5) / time_window_s if time_window_s > 0 else 0
    count_rate_error = (n_total ** 0.5) / time_window_s if time_window_s > 0 else 0
    # Binomial error for efficiency
    efficiency_error = ((efficiency * (1 - efficiency) / n_total) ** 0.5) if n_total > 0 else 0
    return {
        'count_rate': count_rate,
        'signal_count_rate': signal_count_rate,
        'dark_count_rate': dark_count_rate,
        'efficiency': efficiency,
        'signal_count_rate_error': signal_count_rate_error,
        'dark_count_rate_error': dark_count_rate_error,
        'count_rate_error': count_rate_error,
        'efficiency_error': efficiency_error,
        'n_signal': n_signal,
        'n_dark': n_dark,
        'n_total': n_total,
        'time_window_s': time_window_s,
    }

def get_analysis_config():
    analysis_variables = [
        {
            'name': 'True_laser_sync_time',
            'variable': 'laser_sync_time',
            'bins': 200,
            'xlim': [197, 200],
            'selection': is_true_laser_sync,
            'gaussian_fit': False,
            'histogram_plot': True,
            'event_plot': False,
        },
        {
            'name': 'Dark_laser_sync_time',
            'variable': 'laser_sync_time',
            'bins': 200,
            'xlim': None,
            'selection': is_dark_laser_sync,
            'gaussian_fit': False,
            'histogram_plot': False,
            'event_plot': False,
        },
        {
            'name': 'signal_fall_amplitude',
            'variable': 'signal_fall_amplitude',
            'bins': 80,
            'xlim': None,
            'selection': None,
            'gaussian_fit': False,
            'histogram_plot': False,
            'event_plot': False,
        },
        {
            'name': 'True_signal_fall_amplitude',
            'variable': 'signal_fall_amplitude',
            'bins': 128,
            'xlim': [0, 2],
            'selection': is_true_laser_sync,
            'gaussian_fit': False,
            'histogram_plot': True,
            'event_plot': False,
        },
        {
            'name': 'Dark_signal_fall_amplitude',
            'variable': 'signal_fall_amplitude',
            'bins': 128,
            'xlim': [0, 2],
            'selection': is_dark_laser_sync,
            'gaussian_fit': False,
            'histogram_plot': True,
            'event_plot': False,
        },
        {
            'name': 'True_arrival_time_20',
            'variable': 'arrival_time_20',
            'bins': 150,
            'xlim': None,
            'selection': is_true_laser_sync,
            'gaussian_fit': False,
            'histogram_plot': True,
            'event_plot': False,
        },
        {
            'name': 'True_arrival_time_80',
            'variable': 'arrival_time_80',
            'bins': 150,
            'xlim': None,
            'selection': is_true_laser_sync,
            'gaussian_fit': False,
            'histogram_plot': True,
            'event_plot': False,
        },
    ]

    correlations_to_plot = [
        {
            'name': 'laser_sync_arrival_vs_signal_fall_amplitude',
            'x': 'laser_sync_arrival',
            'y': 'signal_fall_amplitude',
            'bins': 100,
            'selection': None,
        },
        {
            'name': 'laser_sync_arrival_vs_event_interval',
            'x': 'laser_sync_arrival',
            'y': 'event_interval',
            'bins': 100,
            'selection': None,
        },
    ]
    return analysis_variables, correlations_to_plot


def analyze_single_file(filename, args, analysis_variables, correlations_to_plot, stage='stage2_statistics'):
    print(f"\n{'='*70}")
    print(f"Processing: {filename}")
    print(f"{'='*70}")

    events, metadata, summary = read_file(filename)

    if not events:
        print(f"  Warning: No event data found in {filename}")
        return None

    print(f"  Total events: {len(events)}")

    if summary:
        print(f"\n  Summary Statistics from Stage 1:")
        print(f"    Count Rate: {summary.get('count_rate', 0):.2f} Hz")

    base_name = Path(filename).stem
    output_dir = determine_output_directory(filename, args.output_dir, stage=stage)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Computing statistics for all variables:")
    variable_statistics = {}

    for var in analysis_variables:
        stats = compute_variable_statistics(
            events, var['variable'], bins=var.get('bins', 100), selection=var.get('selection')
        )
        if var['gaussian_fit'] and stats is not None:
            selection = var.get('selection')
            def _passes_selection(event):
                if callable(selection):
                    return selection(event)
                if isinstance(selection, str):
                    return eval(selection, {}, event)
                return True
            fit_values = [
                event[var['variable']] for event in events
                if var['variable'] in event and _passes_selection(event)
            ]
            if fit_values:
                stats['gaussian_fit'] = fit_gaussian_to_histogram(
                    np.array(fit_values), stats, var.get('bins', 100)
                )
        if stats:
            variable_statistics[var['name']] = stats

    time_window_s = summary.get('total_time', 1.0)
    key_vars = calculate_key_variables(
        events,
        signal_selection=is_true_laser_sync,
        dark_selection=is_dark_laser_sync,
        time_window_s=time_window_s,
        source_rate=int(metadata.get('Repetition rate (kHz)', 1.0)) * 1e3
    )
    for k, v in key_vars.items():
        variable_statistics[k] = v

    output_statistics = {
        'input_file': filename,
        'metadata': metadata,
        'variable_statistics': variable_statistics,
        'analysis_timestamp': datetime.now().isoformat(),
    }

    # Create json_stats subfolder
    json_stats_dir = os.path.join(output_dir, 'json_stats')
    os.makedirs(json_stats_dir, exist_ok=True)
    stats_output_path = os.path.join(json_stats_dir, f"statistics_{base_name}.json")
    with open(stats_output_path, 'w') as f:
        json.dump(output_statistics, f, indent=2)
    print(f"\n  ✓ Statistics saved to: json_stats/statistics_{base_name}.json")

    if args.no_plots:
        print(f"  Skipping plots (--no-plots enabled)")
        return output_dir

    print(f"\n  Plotting histograms:")
    for var in analysis_variables:
        if not var.get('histogram_plot', False):
            continue
        try:
            stats = variable_statistics.get(var['name'])
            var_hist_dir = os.path.join(output_dir, var['name'])
            os.makedirs(var_hist_dir, exist_ok=True)
            fig, _ = plot_variable_histogram(
                events, var['variable'], filename, metadata,
                bins=var['bins'], stats_dict=stats, xlim=var['xlim'], selection=var['selection'], title_name=var['name'])
            if fig:
                output_path = os.path.join(var_hist_dir, f"{var['name']}_histogram_{base_name}.png")
                fig.savefig(output_path, dpi=150)
                plt.close(fig)
        except Exception as e:
            print(f"  Warning: Could not plot histogram for '{var['name']}' - {e}")

    print(f"\n  Plotting variable vs event number:")
    for var in analysis_variables:
        if not var.get('event_plot', False):
            continue
        try:
            fig = plot_variable_vs_event(events, var['variable'], filename, metadata)
            if fig:
                output_path = os.path.join(output_dir, f"{var['variable']}_vs_event_{base_name}.png")
                fig.savefig(output_path, dpi=150)
                plt.close(fig)
        except Exception as e:
            print(f"  Warning: Could not plot event plot for '{var['variable']}' - {e}")

    print(f"\n  Plotting 2D correlations:")
    # Create correlations subfolder
    corr_dir = os.path.join(output_dir, 'correlations')
    os.makedirs(corr_dir, exist_ok=True)
    for corr in correlations_to_plot:
        try:
            bins = corr.get('bins', 100)
            selection = corr.get('selection', None)
            filtered_events = [event for event in events if selection(event)] if selection else events
            fig = plot_2d_correlation(filtered_events, corr['x'], corr['y'], filename, metadata, bins=bins)
            if fig:
                output_path = os.path.join(corr_dir, f"correlation_{corr['x']}_vs_{corr['y']}_{base_name}.png")
                fig.savefig(output_path, dpi=150)
                plt.close(fig)
        except Exception as e:
            print(f"  Warning: Could not create correlation plot ({corr['x']} vs {corr['y']}) - {e}")

    print(f"\n  Completed processing: {filename}")
    return output_dir


def analyze_all(args, analysis_variables, correlations_to_plot, stage='stage2_statistics'):
    last_output_dir = None
    for filename in args.in_filenames:
        last_output_dir = analyze_single_file(filename, args, analysis_variables, correlations_to_plot, stage=stage)
    print(f"\n{'='*70}")
    print("Stage 2 Analysis Complete")
    if last_output_dir:
        print(f"Outputs saved to: {last_output_dir}")
    print(f"{'='*70}\n")

def expected_outputs_for_file(input_file, args, analysis_variables, correlations_to_plot, stage='stage2_statistics'):
    output_dir = determine_output_directory(input_file, args.output_dir, stage=stage)
    base_name = Path(input_file).stem
    outputs = [os.path.join(output_dir, 'json_stats', f"statistics_{base_name}.json")]
    if args.no_plots:
        return output_dir, outputs
    for var in analysis_variables:
        if var.get('histogram_plot', False):
            outputs.append(os.path.join(output_dir, var['name'], f"{var['name']}_histogram_{base_name}.png"))
        if var.get('event_plot', False):
            outputs.append(os.path.join(output_dir, f"{var['variable']}_vs_event_{base_name}.png"))
    corr_dir = os.path.join(output_dir, 'correlations')
    for corr in correlations_to_plot:
        outputs.append(os.path.join(corr_dir, f"correlation_{corr['x']}_vs_{corr['y']}_{base_name}.png"))
    return output_dir, outputs


def outputs_missing_for_file(input_file, args, analysis_variables, correlations_to_plot, stage='stage2_statistics'):
    output_dir, expected_outputs = expected_outputs_for_file(input_file, args, analysis_variables, correlations_to_plot, stage=stage)
    missing = [path for path in expected_outputs if not os.path.exists(path)]
    for path in missing:
        print(f"Missing: {path}")
    return output_dir, missing


def reset_outputs_for_inputs(args, analysis_variables, correlations_to_plot, stage='stage2_statistics'):
    for input_file in args.in_filenames:
        output_dir, _ = expected_outputs_for_file(input_file, args, analysis_variables, correlations_to_plot, stage=stage)
        if output_dir in ('.', '', '/'):
            print(f"Refusing to remove outputs in unsafe directory: {output_dir}")
            continue
        if os.path.exists(output_dir):
            print(f"Removing outputs in {output_dir} ...")
            shutil.rmtree(output_dir)
        else:
            print(f"Output directory {output_dir} does not exist.")


def scan_outputs_for_inputs(args, analysis_variables, correlations_to_plot, stage='stage2_statistics'):
    missing_all = []
    for input_file in args.in_filenames:
        _, missing = outputs_missing_for_file(input_file, args, analysis_variables, correlations_to_plot, stage=stage)
        missing_all.extend(missing)
    if not missing_all:
        print("All expected outputs are present for all input files.")
    return missing_all


def update_outputs_for_inputs(args, analysis_variables, correlations_to_plot, stage='stage2_statistics'):
    for input_file in args.in_filenames:
        output_dir, missing = outputs_missing_for_file(input_file, args, analysis_variables, correlations_to_plot, stage=stage)
        if not missing:
            print(f"Outputs already present for {input_file} in {output_dir}")
            continue
        print(f"Regenerating outputs for {input_file} ...")
        analyze_single_file(input_file, args, analysis_variables, correlations_to_plot, stage=stage)
    print("Update complete.")


def restart_outputs_for_inputs(args, analysis_variables, correlations_to_plot, stage='stage2_statistics'):
    reset_outputs_for_inputs(args, analysis_variables, correlations_to_plot, stage=stage)
    analyze_all(args, analysis_variables, correlations_to_plot, stage=stage)


def main():
    args = parser.parse_args()
    analysis_variables, correlations_to_plot = get_analysis_config()
    stage = 'stage2_statistics'

    if args.reset:
        reset_outputs_for_inputs(args, analysis_variables, correlations_to_plot, stage=stage)
        return
    if args.scan:
        scan_outputs_for_inputs(args, analysis_variables, correlations_to_plot, stage=stage)
        return
    if args.update:
        update_outputs_for_inputs(args, analysis_variables, correlations_to_plot, stage=stage)
        return
    if args.restart:
        restart_outputs_for_inputs(args, analysis_variables, correlations_to_plot, stage=stage)
        return
    if not args.reset and not args.scan and not args.update and not args.restart:
        analyze_all(args, analysis_variables, correlations_to_plot, stage=stage)

if __name__ == "__main__":
    main()
