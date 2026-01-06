import numpy as np
import matplotlib.pyplot as plt
import os


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
