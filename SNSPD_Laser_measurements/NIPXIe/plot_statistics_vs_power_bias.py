import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path


def _val_err(entry):
    """Extract (value, error) from dict stats or scalar values.
    
    For dict entries (e.g., from compute_variable_statistics):
        Returns (mean, sem) if available
    For scalar entries (e.g., from calculate_key_variables):
        Returns (value, nan)
    """
    if isinstance(entry, dict):
        val = entry.get('mean', entry.get('mu', entry.get('value', np.nan)))
        err = entry.get('sem', entry.get('std', np.nan))
        return val, err
    return entry, np.nan


# Variables to compare vs bias (for each power)
VARS_VS_BIAS = [
    ('signal_count_rate', 'Signal Rate (Hz)'),
    ('count_rate', 'Count Rate (Hz)'),
    ('dark_count_rate', 'Dark Count Rate (Hz)'),
    ('efficiency', 'Efficiency'),
    ('True_signal_fall_amplitude', 'True Signal Fall Amplitude (mV)'),
    ('Dark_signal_fall_amplitude', 'Dark Signal Fall Amplitude (mV)'),
]

# Variables to compare vs power (for each bias)
VARS_VS_POWER = [
    ('signal_count_rate', 'Signal Rate (Hz)'),
    ('count_rate', 'Count Rate (Hz)'),
    ('dark_count_rate', 'Dark Count Rate (Hz)'),
    ('efficiency', 'Efficiency'),
    ('True_signal_fall_amplitude', 'True Signal Fall Amplitude (mV)'),
    ('Dark_signal_fall_amplitude', 'Dark Signal Fall Amplitude (mV)'),
]

# Bias voltages to include in multi plots (set to None to plot all)
BIASES_TO_PLOT = [68,70,74]  # e.g., [1600, 1650, 1700] or None for all

def plot_statistics_vs_bias(power, power_data, output_dir):
    bias_voltages = [d['bias_voltage'] for d in power_data]
    outdir = os.path.join(output_dir, "bias")
    os.makedirs(outdir, exist_ok=True)
    for var, label in VARS_VS_BIAS:
        var_dir = os.path.join(outdir, var)
        os.makedirs(var_dir, exist_ok=True)
        pairs = [_val_err(d.get(var, np.nan)) for d in power_data]
        values = [p[0] for p in pairs]
        errors = [p[1] for p in pairs]
        plt.figure(figsize=(8, 5))
        plt.errorbar(bias_voltages, values, yerr=errors, marker='o', linestyle=':', label=label)
        plt.xlabel('Bias Voltage (mV)')
        plt.ylabel(label)
        plt.title(f'{label} vs Bias Voltage ({power} nW)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        fname = os.path.join(var_dir, f'{var}_vs_bias_{power}nW.png')
        plt.savefig(fname, dpi=150)
        plt.close()

def plot_statistics_vs_power(bias, bias_data, output_dir, loglog_fit_range=None):
    powers = [d['power'] for d in bias_data if d['power'] is not None]
    outdir = os.path.join(output_dir, "power")
    os.makedirs(outdir, exist_ok=True)
    for var, label in VARS_VS_POWER:
        var_dir = os.path.join(outdir, var)
        os.makedirs(var_dir, exist_ok=True)
        pairs = [_val_err(d.get(var, np.nan)) for d in bias_data if d['power'] is not None]
        values = [p[0] for p in pairs]
        errors = [p[1] for p in pairs]
        # Linear plot
        plt.figure(figsize=(8, 5))
        plt.errorbar(powers, values, yerr=errors, marker='o', linestyle=':', label=label)
        plt.xlabel('Power (nW)')
        plt.ylabel(label)
        plt.title(f'{label} vs Power ({bias} mV)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        fname = os.path.join(var_dir, f'{var}_vs_power_{bias}mV.png')
        plt.savefig(fname, dpi=150)
        plt.close()
        # Log-log plot with fit for count_rate and signal_count_rate
        # Filter out non-positive values for log scale
        powers_arr = np.array(powers)
        values_arr = np.array(values)
        errors_arr = np.array(errors)
        positive_mask = (powers_arr > 0) & (values_arr > 0) & np.isfinite(values_arr)
        if not np.any(positive_mask):
            continue  # Skip log-log plot if no positive values
        powers_pos = powers_arr[positive_mask]
        values_pos = values_arr[positive_mask]
        errors_pos = errors_arr[positive_mask]
        
        plt.figure(figsize=(8, 5))
        plt.errorbar(powers_pos, values_pos, yerr=errors_pos, marker='o', linestyle=':', label=label)
        plt.xlabel('Power (nW)')
        plt.ylabel(label)
        plt.title(f'{label} vs Power (log-log, {bias} mV)')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which='both', ls=':')
        fit_label = None
        if var in ('count_rate', 'signal_count_rate'):
            fit_powers = powers_pos
            fit_values = values_pos
            # Apply loglog_fit_range if provided
            if loglog_fit_range is not None and len(loglog_fit_range) == 2:
                fit_min, fit_max = loglog_fit_range
                fit_range_mask = (fit_powers >= fit_min) & (fit_powers <= fit_max)
                fit_powers = fit_powers[fit_range_mask]
                fit_values = fit_values[fit_range_mask]
            if len(fit_powers) >= 2:
                log_p = np.log(fit_powers)
                log_v = np.log(fit_values)
                coeffs = np.polyfit(log_p, log_v, 1)
                n = coeffs[0]
                logA = coeffs[1]
                fit_label = f"Fit: n={n:.2f}"
                p_fit = np.linspace(min(fit_powers), max(fit_powers), 100)
                v_fit = np.exp(logA) * p_fit**n
                plt.plot(p_fit, v_fit, 'k--', label=fit_label)
        plt.legend()
        fname_log = os.path.join(var_dir, f'{var}_vs_power_{bias}mV_loglog.png')
        plt.savefig(fname_log, dpi=150)
        plt.close()

def plot_multi_statistics_vs_bias(power_groups, output_dir):
    bias_set = set()
    outdir = os.path.join(output_dir, "multi")
    os.makedirs(outdir, exist_ok=True)
    for pdata in power_groups.values():
        bias_set.update([d['bias_voltage'] for d in pdata])
    bias_list = sorted(bias_set)
    for var, label in VARS_VS_BIAS:
        var_dir = os.path.join(outdir, var)
        os.makedirs(var_dir, exist_ok=True)
        plt.figure(figsize=(8, 5))
        for power, pdata in power_groups.items():
            bias_vals = [d['bias_voltage'] for d in pdata]
            pairs = [_val_err(d.get(var, np.nan)) for d in pdata]
            values = [p[0] for p in pairs]
            errors = [p[1] for p in pairs]
            plt.errorbar(bias_vals, values, yerr=errors, marker='o', linestyle=':', label=f'{power} nW')
        plt.xlabel('Bias Voltage (mV)')
        plt.ylabel(label)
        plt.title(f'{label} vs Bias Voltage (all powers)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        fname = os.path.join(var_dir, f'{var}_vs_bias_multi.png')
        plt.savefig(fname, dpi=150)
        plt.close()

def plot_multi_statistics_vs_power(bias_groups, output_dir, loglog_fit_range=None):
    power_set = set()
    outdir = os.path.join(output_dir, "multi")
    os.makedirs(outdir, exist_ok=True)
    for bdata in bias_groups.values():
        power_set.update([d['power'] for d in bdata if d['power'] is not None])
    power_list = sorted(power_set)
    for var, label in VARS_VS_POWER:
        var_dir = os.path.join(outdir, var)
        os.makedirs(var_dir, exist_ok=True)
        # Linear plot
        plt.figure(figsize=(8, 5))
        for bias, bdata in bias_groups.items():
            if BIASES_TO_PLOT is not None and bias not in BIASES_TO_PLOT:
                continue
            power_vals = [d['power'] for d in bdata if d['power'] is not None]
            pairs = [_val_err(d.get(var, np.nan)) for d in bdata if d['power'] is not None]
            values = [p[0] for p in pairs]
            errors = [p[1] for p in pairs]
            plt.errorbar(power_vals, values, yerr=errors, marker='o', linestyle=':', label=f'{bias} mV')
        plt.xlabel('Power (nW)')
        plt.ylabel(label)
        plt.title(f'{label} vs Power (all biases)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        fname = os.path.join(var_dir, f'{var}_vs_power_multi.png')
        plt.savefig(fname, dpi=150)
        plt.close()
        # Log-log plot with fit for count_rate and signal_count_rate
        plt.figure(figsize=(8, 5))
        has_positive_data = False
        for bias, bdata in bias_groups.items():
            if BIASES_TO_PLOT is not None and bias not in BIASES_TO_PLOT:
                continue
            power_vals = np.array([d['power'] for d in bdata if d['power'] is not None])
            values = np.array([_val_err(d.get(var, np.nan))[0] for d in bdata if d['power'] is not None])
            errors = np.array([_val_err(d.get(var, np.nan))[1] for d in bdata if d['power'] is not None])
            
            # Filter out non-positive values for log scale
            positive_mask = (power_vals > 0) & (values > 0) & np.isfinite(values)
            if not np.any(positive_mask):
                continue
            
            has_positive_data = True
            power_vals_pos = power_vals[positive_mask]
            values_pos = values[positive_mask]
            errors_pos = errors[positive_mask]
            
            plt.errorbar(power_vals_pos, values_pos, yerr=errors_pos, marker='o', linestyle=':', label=f'{bias} mV')
            if var in ('count_rate', 'signal_count_rate'):
                fit_powers = power_vals_pos
                fit_values = values_pos
                # Apply loglog_fit_range if provided
                if loglog_fit_range is not None and len(loglog_fit_range) == 2:
                    fit_min, fit_max = loglog_fit_range
                    fit_range_mask = (fit_powers >= fit_min) & (fit_powers <= fit_max)
                    fit_powers = fit_powers[fit_range_mask]
                    fit_values = fit_values[fit_range_mask]
                if len(fit_powers) >= 2:
                    log_p = np.log(fit_powers)
                    log_v = np.log(fit_values)
                    coeffs = np.polyfit(log_p, log_v, 1)
                    n = coeffs[0]
                    logA = coeffs[1]
                    fit_label = f"Fit {bias}mV: n={n:.2f}"
                    p_fit = np.linspace(min(fit_powers), max(fit_powers), 100)
                    v_fit = np.exp(logA) * p_fit**n
                    plt.plot(p_fit, v_fit, '--', label=fit_label)
        
        if not has_positive_data:
            plt.close()
            continue  # Skip saving if no positive data to plot
        
        plt.xlabel('Power (nW)')
        plt.ylabel(label)
        plt.title(f'{label} vs Power (log-log, all biases)')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which='both', ls=':')
        plt.legend()
        fname_log = os.path.join(var_dir, f'{var}_vs_power_multi_loglog.png')
        plt.savefig(fname_log, dpi=150)
        plt.close()


def plot_laser_sync_histogram_stack_vs_power(bias_data, output_dir):
    """
    Create stacked histograms of True_laser_sync_time for power sweep.
    
    Args:
        bias_data: List of measurement data dicts grouped by bias voltage
        output_dir: Output directory for plots
    """
    from analyze_events import LASER_SYNC_LOWER_LIMIT, LASER_SYNC_UPPER_LIMIT, is_true_laser_sync
    
    # Filter data with valid power values
    valid_data = [d for d in bias_data if d.get('power') is not None]
    if not valid_data:
        return
    
    # Sort by power
    valid_data.sort(key=lambda x: x['power'])
    
    # Extract powers and bias voltage
    powers = [d['power'] for d in valid_data]
    bias = valid_data[0].get('bias_voltage', 'Unknown')
    
    # Read event data and extract True_laser_sync_time histograms
    histograms = []
    labels = []
    
    for d in valid_data:
        # Find the corresponding Stage 1 event JSON file
        stats_file = d.get('file_path')
        if not stats_file:
            continue
            
        # Determine the Stage 1 events file path
        # Stage 2 statistics files are in: .../stage2_statistics/.../json_stats/statistics_*_analysis.json
        # Stage 1 files are in: .../stage1_events/.../*_analysis.json (without 'statistics_' prefix and not in json_stats/)
        stats_path = Path(stats_file)
        
        # Check if this is a Stage 2 file (in stage2_statistics directory)
        if 'stage2_statistics' in str(stats_path):
            # Construct Stage 1 path by:
            # 1. Replace stage2_statistics with stage1_events
            # 2. Remove '/json_stats/' subdirectory
            # 3. Remove 'statistics_' prefix from filename
            event_path_str = str(stats_path).replace('stage2_statistics', 'stage1_events')
            # Remove the json_stats subdirectory part
            event_path_str = event_path_str.replace('/json_stats', '')
            # Get the filename without 'statistics_' prefix
            event_filename = stats_path.name.replace('statistics_', '')
            event_path = Path(event_path_str).parent / event_filename
        else:
            # Already a Stage 1 file
            event_path = stats_path
        
        if not event_path.exists():
            continue
        
        # Read event data
        try:
            with open(event_path, 'r') as f:
                event_data = json.load(f)
            
            # Extract True_laser_sync events
            events = event_data.get('events', [])
            
            true_sync_times = [
                event['laser_sync_arrival'] 
                for event in events 
                if 'laser_sync_arrival' in event and is_true_laser_sync(event)
            ]
            
            if true_sync_times:
                histograms.append(true_sync_times)
                labels.append(f"{d['power']:.1f} nW")
        except Exception as e:
            print(f"    Warning: Error reading {event_path.name}: {e}")
            continue
    
    if not histograms:
        print(f"No histogram data found for bias {bias} mV")
        return
    
    # Create stacked histogram plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use consistent bins across all histograms
    bins = np.linspace(LASER_SYNC_LOWER_LIMIT, LASER_SYNC_UPPER_LIMIT, 100)
    
    # Plot stacked histograms with offset for visibility
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(histograms)))
    
    for i, (hist_data, label) in enumerate(zip(histograms, labels)):
        counts, bin_edges = np.histogram(hist_data, bins=bins)
        # Normalize to counts per bin
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, counts, label=label, color=colors[i], linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Laser Sync Time (ns)', fontsize=12)
    ax.set_ylabel('Counts', fontsize=12)
    ax.set_title(f'True Laser Sync Time Distribution vs Power (Bias: {bias} mV)', fontsize=14, fontweight='bold')
    ax.set_xlim(LASER_SYNC_LOWER_LIMIT, LASER_SYNC_UPPER_LIMIT)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Save plot
    outdir = os.path.join(output_dir, "power", "laser_sync_histograms")
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f'laser_sync_histogram_stack_vs_power_{bias}mV.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved stacked histogram: {fname}")
