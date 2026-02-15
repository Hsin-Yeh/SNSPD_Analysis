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


def extract_oot_mid_dark_counts(hist, resolution_s, signal_width_ns, acq_time_s):
    """
    Extract OOT_mid (80-100 ns) dark count for a single measurement.
    
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
        OOT_mid dark count rate (counts/s)
    """
    oot_bin_min = int(80.0 * 1e-9 / resolution_s)
    oot_bin_max = int(100.0 * 1e-9 / resolution_s)
    oot_width = 20.0  # ns
    
    counts_oot = int(np.sum(hist[oot_bin_min:oot_bin_max]))
    oot_mid_dark = (counts_oot / oot_width) * signal_width_ns / acq_time_s
    
    return oot_mid_dark


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


def fit_power_law(powers_arr, counts_arr, fit_max_uw, measurement_time=None):
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
    measurement_time : float, optional
        Measurement time in seconds (for proper error calculation)
    
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
    
    # Calculate chi-squared in log space
    # For rates R = N/t: σ_R = √N/t = √(R/t)
    # Then σ_log10(R) = σ_R / (R * ln(10)) = √(R/t) / (R * ln(10))
    if measurement_time is not None:
        sigma_log = np.sqrt(counts_fit / measurement_time) / (counts_fit * np.log(10))
    else:
        # Fallback: assume sqrt(rate) if time unknown (less accurate)
        sigma_log = np.sqrt(counts_fit) / (counts_fit * np.log(10))
    chi2 = np.sum(((log_counts_fit - log_fit_line) / sigma_log) ** 2)
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


def print_chi2_explanation(acq_time_s=None):
    """
    Print an explanation of chi-squared calculation for log-log fits.

    Notes:
    - The fit is performed in log-log space to get slope/intercept.
    - The chi-squared is computed in log space using log10(y).
    - We use σ_log10(R) = σ_R / (R * ln(10)) where R is count rate.
    - For rates: σ_R = √N/t = √(R/t) where N is photon count, t is measurement time.
    - This gives: σ_log10(R) = √(R/t) / (R * ln(10)) = 1 / (√(R*t) * ln(10))
    """
    print("\n=== Chi² Calculation Summary (Log-Log Fits) ===")
    print("Fit: log10(count rate) vs log10(power) to obtain slope/intercept.")
    print("Chi²/ndf is computed in log space:")
    print("  χ² = Σ[(log10(R_obs) - log10(R_fit))² / σ_log10(R)²]")
    print("  For count rates R = N/t (N = photon counts, t = measurement time):")
    print("    σ_R = √N/t = √(R/t)  [Poisson error on rate]")
    print("    σ_log10(R) = σ_R / (R * ln(10)) = √(R/t) / (R * ln(10))")
    if acq_time_s is not None:
        print(f"Acquisition time: {acq_time_s:.3f} s")
        print("For rates, the Poisson error should be:")
        print("  σ_rate = √(y * t) / t = √(y / t)")
        print("Since t > 1 s, σ_rate is smaller than √y, so χ²/ndf would increase.")
    print("If χ²/ndf still looks small, σ may still be conservative for rates.")
