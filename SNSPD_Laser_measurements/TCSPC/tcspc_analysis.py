#!/usr/bin/env python3
"""
Shared analysis functions for dark count subtraction and power law fitting.
Used by both read_phu.py and create_combined_plot.py to ensure consistency.
"""

import numpy as np
from scipy import stats


def extract_oot_pre_dark_counts(hist, resolution_s, signal_width_ns, acq_time_s):
    """
    Extract OOT_pre (0-60 ns) dark count for a single measurement.
    
    Parameters:
    -----------
    hist : np.ndarray
        Histogram data (counts per bin)
    resolution_s : float
        Time resolution per bin (seconds, typically 4e-12 s)
    signal_width_ns : float
        Signal window width in nanoseconds (typically 4 ns)
    acq_time_s : float
        Acquisition time in seconds
    
    Returns:
    --------
    float
        OOT_pre dark count rate (counts/s)
    """
    oot_bin_min = int(0 * 1e-9 / resolution_s)
    oot_bin_max = int(60.0 * 1e-9 / resolution_s)
    oot_width = 60.0  # ns
    
    counts_oot = int(np.sum(hist[oot_bin_min:oot_bin_max]))
    oot_pre_dark = (counts_oot / oot_width) * signal_width_ns / acq_time_s
    
    return oot_pre_dark


def subtract_dark_counts(counts_arr, dark_counts_arr, method='per_measurement'):
    """
    Subtract dark counts from signal counts.
    
    Parameters:
    -----------
    counts_arr : np.ndarray
        Array of raw count rates (counts/s)
    dark_counts_arr : np.ndarray
        Array of dark count rates (counts/s), one per measurement
    method : str
        'per_measurement': subtract individual dark for each point
        'mean': subtract mean dark count from all points
    
    Returns:
    --------
    np.ndarray
        Dark-corrected count rates (counts/s), clipped to minimum of 0.1
    float
        Mean dark count rate used for reporting
    """
    if method == 'per_measurement':
        # Subtract per-measurement dark counts
        counts_corrected = np.maximum(counts_arr - dark_counts_arr, 0.1)
        dark_count_rate = np.mean(dark_counts_arr)
    elif method == 'mean':
        # Subtract mean dark count from all points
        mean_dark = np.mean(dark_counts_arr)
        counts_corrected = np.maximum(counts_arr - mean_dark, 0.1)
        dark_count_rate = mean_dark
    else:
        raise ValueError(f"Unknown dark subtraction method: {method}")
    
    return counts_corrected, dark_count_rate


def fit_power_law(powers_arr, counts_arr, fit_max_uw):
    """
    Perform power law fit on log-log data in low-power region.
    
    Parameters:
    -----------
    powers_arr : np.ndarray
        Array of laser powers (µW)
    counts_arr : np.ndarray
        Array of dark-corrected count rates (counts/s)
    fit_max_uw : float
        Maximum power for fit region (µW)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'slope': power law exponent (n)
        - 'intercept': log intercept
        - 'std_err': uncertainty in slope
        - 'chi2_ndf': reduced chi-squared goodness of fit
        - 'fit_mask': boolean mask of points used in fit
        - 'fit_powers': powers of fitted points
        - 'fit_counts': counts of fitted points
        - 'fit_line': fitted count values
    """
    # Apply fit mask: power <= fit_max
    fit_mask = powers_arr <= fit_max_uw
    
    if np.sum(fit_mask) < 2:
        raise ValueError(f"Not enough data points in fit range (< 2 points at power <= {fit_max_uw} µW)")
    
    powers_fit = powers_arr[fit_mask]
    counts_fit = counts_arr[fit_mask]
    
    # Perform log-log linear regression
    log_powers_fit = np.log10(powers_fit)
    log_counts_fit = np.log10(counts_fit)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_powers_fit, log_counts_fit)
    
    # Calculate fitted values
    log_fit_line = slope * log_powers_fit + intercept
    fit_line = 10**log_fit_line
    
    # Calculate chi-squared with Poisson errors in linear space
    # χ² = Σ [(y_obs - y_fit)² / σ²], where σ = √y for Poisson statistics
    sigma = np.sqrt(counts_fit)
    chi2 = np.sum(((counts_fit - fit_line) / sigma) ** 2)
    ndf = len(counts_fit) - 2  # 2 parameters (slope, intercept)
    chi2_ndf = chi2 / ndf if ndf > 0 else np.nan
    
    return {
        'slope': slope,
        'intercept': intercept,
        'std_err': std_err,
        'chi2_ndf': chi2_ndf,
        'fit_mask': fit_mask,
        'fit_powers': powers_fit,
        'fit_counts': counts_fit,
        'fit_line': fit_line,
    }


def print_fit_summary(slope, std_err, chi2_ndf, intercept, dark_count_rate=None):
    """
    Print summary of fit results.
    
    Parameters:
    -----------
    slope : float
        Power law exponent
    std_err : float
        Uncertainty in slope
    chi2_ndf : float
        Reduced chi-squared
    intercept : float
        Log intercept
    dark_count_rate : float, optional
        Dark count rate used for correction
    """
    print(f"\n=== Fit Results ===")
    if dark_count_rate is not None:
        print(f"Dark count (OOT_pre): {dark_count_rate:.2f} cts/s")
    print(f"Power law exponent: n = {slope:.4f} ± {std_err:.4f}")
    print(f"Chi²/ndf: {chi2_ndf:.4f}")
    print(f"Fit equation: Count Rate = {10**intercept:.2f} × Power^{slope:.4f}")
