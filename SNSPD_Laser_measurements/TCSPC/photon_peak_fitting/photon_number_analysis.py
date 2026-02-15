#!/usr/bin/env python3
"""
Photon number analysis for TCSPC histograms.

This module performs multi-peak Gaussian fitting to identify photon number states
and matches them to Poisson statistics.
"""

import numpy as np
from scipy import signal, optimize, stats
import matplotlib.pyplot as plt
from pathlib import Path


def find_two_main_peaks(hist, resolution_s, t_min_ns=70, t_max_ns=85):
    """
    Find the two main peaks in TOA histogram (likely due to laser sync shifting).
    
    Parameters:
    -----------
    hist : np.ndarray
        Histogram data (counts per bin)
    resolution_s : float
        Time resolution in seconds
    t_min_ns : float
        Minimum time for search region (ns)
    t_max_ns : float
        Maximum time for search region (ns)
    
    Returns:
    --------
    dict with:
        'peak1_center': center bin of first peak
        'peak2_center': center bin of second peak
        'peak1_counts': total counts in first peak
        'peak2_counts': total counts in second peak
        'peak1_range': (start_bin, end_bin) for peak 1
        'peak2_range': (start_bin, end_bin) for peak 2
    """
    # Convert time window to bins
    bin_min = int(t_min_ns * 1e-9 / resolution_s)
    bin_max = int(t_max_ns * 1e-9 / resolution_s)
    
    # Extract region of interest
    hist_roi = hist[bin_min:bin_max]
    
    # Smooth histogram to find main peaks
    window_size = max(5, len(hist_roi) // 100)
    if window_size % 2 == 0:
        window_size += 1
    hist_smooth = signal.savgol_filter(hist_roi, window_size, 3)
    
    # Find peaks with prominence to get the two strongest
    peaks, properties = signal.find_peaks(hist_smooth, prominence=np.max(hist_smooth)*0.1, width=10)
    
    if len(peaks) < 2:
        print(f"Warning: Found only {len(peaks)} main peaks, expected 2")
        if len(peaks) == 1:
            # Use the one peak found and estimate second peak location
            peak1_idx = peaks[0]
            # Assume peaks are separated by ~half the range
            peak2_idx = min(len(hist_roi) - 1, peak1_idx + len(hist_roi) // 2)
        else:
            # No peaks found, use quartile positions
            peak1_idx = len(hist_roi) // 4
            peak2_idx = 3 * len(hist_roi) // 4
    else:
        # Get the two most prominent peaks
        prominence_order = np.argsort(properties['prominences'])[::-1]
        peak_indices = peaks[prominence_order[:2]]
        peak1_idx, peak2_idx = sorted(peak_indices)
    
    # Define boundaries between peaks (use valley between them)
    if peak1_idx < peak2_idx:
        valley_region = hist_smooth[peak1_idx:peak2_idx]
        if len(valley_region) > 0:
            valley_idx = peak1_idx + np.argmin(valley_region)
        else:
            valley_idx = (peak1_idx + peak2_idx) // 2
    else:
        valley_idx = (peak1_idx + peak2_idx) // 2
    
    # Calculate counts in each peak
    peak1_counts = np.sum(hist_roi[:valley_idx])
    peak2_counts = np.sum(hist_roi[valley_idx:])
    
    return {
        'peak1_center': bin_min + peak1_idx,
        'peak2_center': bin_min + peak2_idx,
        'peak1_counts': int(peak1_counts),
        'peak2_counts': int(peak2_counts),
        'peak1_range': (bin_min, bin_min + valley_idx),
        'peak2_range': (bin_min + valley_idx, bin_max),
        'valley_bin': bin_min + valley_idx,
    }


def gaussian(x, amplitude, mean, std):
    """Gaussian function for fitting."""
    return amplitude * np.exp(-0.5 * ((x - mean) / std) ** 2)


def multi_gaussian(x, *params):
    """Sum of multiple Gaussian functions."""
    n_peaks = len(params) // 3
    result = np.zeros_like(x, dtype=float)
    for i in range(n_peaks):
        amplitude = params[3*i]
        mean = params[3*i + 1]
        std = params[3*i + 2]
        result += gaussian(x, amplitude, mean, std)
    return result


def fit_photon_number_peaks(hist, resolution_s, peak_range, n_photons=5, min_peak_sep_ns=0.3):
    """
    Fit multiple Gaussian peaks within a main peak region to identify photon number states.
    
    Parameters:
    -----------
    hist : np.ndarray
        Histogram data
    resolution_s : float
        Time resolution in seconds
    peak_range : tuple (start_bin, end_bin)
        Range of bins to analyze
    n_photons : int
        Expected number of photon peaks to fit
    min_peak_sep_ns : float
        Minimum separation between peaks in nanoseconds
    
    Returns:
    --------
    dict with:
        'peaks': list of dicts, each containing:
            - 'photon_number': assigned photon number (1, 2, 3, ...)
            - 'mean_bin': center position (bin)
            - 'mean_ns': center position (ns)
            - 'std_ns': standard deviation (ns)
            - 'amplitude': peak amplitude
            - 'counts': integrated counts in peak
        'fit_quality': dict with chi2, r2, etc.
        'fit_params': raw fit parameters
        'fit_curve': fitted histogram values
    """
    start_bin, end_bin = peak_range
    hist_roi = hist[start_bin:end_bin]
    bins_roi = np.arange(len(hist_roi))
    
    # Smooth for peak finding
    if len(hist_roi) < 10:
        return None
    
    window_size = min(7, len(hist_roi) // 4)
    if window_size % 2 == 0:
        window_size += 1
    if window_size < 3:
        window_size = 3
        
    hist_smooth = signal.savgol_filter(hist_roi, window_size, min(2, window_size-1))
    
    # Find peaks
    min_peak_sep_bins = int(min_peak_sep_ns * 1e-9 / resolution_s)
    peaks_idx, properties = signal.find_peaks(
        hist_smooth,
        distance=max(1, min_peak_sep_bins),
        prominence=np.max(hist_smooth)*0.05,
        width=1
    )
    
    if len(peaks_idx) == 0:
        print("  Warning: No peaks found in region")
        return None
    
    # Limit to n_photons strongest peaks
    if len(peaks_idx) > n_photons:
        prominence_order = np.argsort(properties['prominences'])[::-1]
        peaks_idx = np.sort(peaks_idx[prominence_order[:n_photons]])
    
    # Initial parameter estimates for multi-Gaussian fit
    initial_params = []
    bounds_lower = []
    bounds_upper = []
    
    for peak_idx in peaks_idx:
        # Amplitude
        amplitude = hist_smooth[peak_idx]
        initial_params.extend([amplitude, peak_idx, 5])  # amplitude, mean, std
        
        # Bounds
        bounds_lower.extend([amplitude*0.1, max(0, peak_idx-10), 0.5])
        bounds_upper.extend([amplitude*10, min(len(hist_roi)-1, peak_idx+10), 50])
    
    # Perform multi-peak Gaussian fit
    try:
        popt, pcov = optimize.curve_fit(
            multi_gaussian,
            bins_roi,
            hist_roi,
            p0=initial_params,
            bounds=(bounds_lower, bounds_upper),
            maxfev=5000
        )
        
        # Calculate fit quality
        fit_curve = multi_gaussian(bins_roi, *popt)
        residuals = hist_roi - fit_curve
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((hist_roi - np.mean(hist_roi))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate chi-squared
        # Assume Poisson errors: sigma = sqrt(counts)
        sigma = np.sqrt(np.maximum(hist_roi, 1))
        chi2 = np.sum((residuals / sigma)**2)
        ndf = len(hist_roi) - len(popt)
        chi2_ndf = chi2 / ndf if ndf > 0 else np.inf
        
    except Exception as e:
        print(f"  Warning: Fit failed - {e}")
        return None
    
    # Extract individual peak parameters
    n_peaks = len(popt) // 3
    peaks_list = []
    
    for i in range(n_peaks):
        amplitude = popt[3*i]
        mean_bin = popt[3*i + 1]
        std_bin = popt[3*i + 2]
        
        # Convert to physical units
        mean_ns = (start_bin + mean_bin) * resolution_s * 1e9
        std_ns = std_bin * resolution_s * 1e9
        
        # Integrate Gaussian to get total counts
        # Integral of Gaussian = amplitude * std * sqrt(2*pi)
        counts = amplitude * std_bin * np.sqrt(2 * np.pi)
        
        peaks_list.append({
            'mean_bin': start_bin + mean_bin,
            'mean_ns': float(mean_ns),
            'std_ns': float(std_ns),
            'amplitude': float(amplitude),
            'counts': float(counts),
        })
    
    # Sort peaks by TOA position (mean_ns) in ascending order
    peaks_list.sort(key=lambda x: x['mean_ns'])
    
    # Assign photon numbers in reverse order: largest TOA = n=1, smallest TOA = highest n
    # This is physically correct: single photons arrive later
    for i, peak in enumerate(peaks_list):
        peak['photon_number'] = n_peaks - i
    
    return {
        'peaks': peaks_list,
        'fit_quality': {
            'chi2': float(chi2),
            'ndf': int(ndf),
            'chi2_ndf': float(chi2_ndf),
            'r2': float(r2),
        },
        'fit_params': popt.tolist(),
        'fit_curve': fit_curve.tolist(),
        'bins_roi': bins_roi.tolist(),
        'hist_roi': hist_roi.tolist(),
    }


def match_poisson_distribution(peak_counts, max_n=10):
    """
    Match observed peak counts to Poisson distribution P(n|μ) to extract mean photon number μ.
    
    P(n|μ) = (μ^n * e^(-μ)) / n!
    
    Parameters:
    -----------
    peak_counts : list or dict
        Counts for each photon number peak
        If dict, keys should be photon numbers (1, 2, 3, ...)
        If list, assumes sequential photon numbers starting from 1
    max_n : int
        Maximum photon number to consider
    
    Returns:
    --------
    dict with:
        'mu': fitted mean photon number
        'mu_std': uncertainty in mu
        'poisson_probs': expected Poisson probabilities
        'observed_probs': observed probabilities (normalized counts)
        'chi2': chi-squared goodness of fit
        'chi2_ndf': reduced chi-squared
    """
    # Convert to dict if list
    if isinstance(peak_counts, list):
        peak_counts = {i+1: count for i, count in enumerate(peak_counts)}
    
    # Extract photon numbers and counts
    photon_numbers = sorted([k for k in peak_counts.keys() if k > 0])
    counts = np.array([peak_counts[n] for n in photon_numbers])
    
    if len(counts) < 2:
        print("Warning: Need at least 2 photon number peaks for Poisson fitting")
        return None
    
    # Normalize to probabilities
    total_counts = np.sum(counts)
    if total_counts == 0:
        return None
    
    observed_probs = counts / total_counts
    
    # Fit to Poisson distribution
    # We need to find μ that minimizes chi-squared
    def poisson_pmf(n, mu):
        """Poisson probability mass function."""
        if mu <= 0:
            return 0
        return stats.poisson.pmf(n, mu)
    
    def chi2_objective(mu):
        """Chi-squared between observed and Poisson distribution."""
        expected_probs = np.array([poisson_pmf(n, mu) for n in photon_numbers])
        # Use observed counts for error estimate (sigma = sqrt(N))
        sigma = np.sqrt(counts) / total_counts
        sigma = np.maximum(sigma, 1/total_counts)  # Minimum uncertainty
        chi2 = np.sum(((observed_probs - expected_probs) / sigma)**2)
        return chi2
    
    # Initial guess: use weighted mean of photon numbers
    mu_initial = np.sum(np.array(photon_numbers) * observed_probs)
    
    # Optimize
    try:
        result = optimize.minimize_scalar(
            chi2_objective,
            bounds=(0.01, 20),
            method='bounded'
        )
        mu_fit = result.x
        chi2_min = result.fun
        
        # Estimate uncertainty in mu using chi2 curvature
        # Very rough estimate: delta_mu ~ sqrt(2/chi2_curvature)
        delta_mu = 0.01
        chi2_plus = chi2_objective(mu_fit + delta_mu)
        chi2_minus = chi2_objective(mu_fit - delta_mu)
        curvature = (chi2_plus + chi2_minus - 2*chi2_min) / (delta_mu**2)
        
        if curvature > 0:
            mu_std = np.sqrt(2 / curvature)
        else:
            mu_std = 0.1 * mu_fit  # Fallback: 10% uncertainty
        
    except Exception as e:
        print(f"Warning: Poisson fitting failed - {e}")
        mu_fit = mu_initial
        mu_std = 0.1 * mu_initial
        chi2_min = chi2_objective(mu_fit)
    
    # Calculate expected Poisson probabilities
    poisson_probs = {n: poisson_pmf(n, mu_fit) for n in photon_numbers}
    
    # Calculate chi2/ndf
    ndf = len(photon_numbers) - 1  # 1 parameter (mu)
    chi2_ndf = chi2_min / ndf if ndf > 0 else np.inf
    
    return {
        'mu': float(mu_fit),
        'mu_std': float(mu_std),
        'poisson_probs': {int(k): float(v) for k, v in poisson_probs.items()},
        'observed_probs': {int(photon_numbers[i]): float(observed_probs[i]) for i in range(len(photon_numbers))},
        'chi2': float(chi2_min),
        'ndf': int(ndf),
        'chi2_ndf': float(chi2_ndf),
    }


def plot_simple_histogram(hist, resolution_s, t_min_ns, t_max_ns, power_uw, output_path, 
                         block_id=None, acq_time_s=None, cut_time_ns=None):
    """
    Plot simple histogram without fitting for a single power level.
    Uses same binning and style as read_phu.py histogram plots.
    
    Parameters:
    -----------
    hist : np.ndarray
        Full histogram data
    resolution_s : float
        Time resolution in seconds
    t_min_ns : float
        Minimum time for plot window (ns)
    t_max_ns : float
        Maximum time for plot window (ns)
    power_uw : float
        Laser power in µW
    output_path : Path or str
        Where to save the plot
    block_id : int, optional
        Block/trace ID from acquisition
    acq_time_s : float, optional
        Acquisition time in seconds
    cut_time_ns : float, optional
        Time value to draw a vertical cut line (ns)
    """
    from pathlib import Path
    
    # Convert time window to bins
    bin_min = int(t_min_ns * 1e-9 / resolution_s)
    bin_max = int(t_max_ns * 1e-9 / resolution_s)
    
    # Extract region of interest
    if bin_max > len(hist):
        bin_max = len(hist)
    
    hist_roi = hist[bin_min:bin_max]
    
    # Create time axis using native binning (same as read_phu.py)
    time_bins = np.arange(len(hist_roi)) * resolution_s * 1e9 + (bin_min * resolution_s * 1e9)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot histogram as line (same style as read_phu.py)
    ax.plot(time_bins, hist_roi, linewidth=1.2, alpha=0.85, color='steelblue')
    
    # Add cut line if specified
    if cut_time_ns is not None:
        ax.axvline(cut_time_ns, color='red', linestyle='--', linewidth=2.0, 
                   label=f'Cut at {cut_time_ns:.2f} ns', alpha=0.7)
        ax.legend(fontsize=11, loc='upper right')
    
    # Labels and title
    ax.set_xlabel('TOA (Time of Arrival) (ns)', fontsize=13, weight='bold')
    ax.set_ylabel('Counts', fontsize=13, weight='bold')
    
    title = f'TOA Histogram - Power: {power_uw:.4f} µW'
    if block_id is not None:
        title += f' (Block {block_id})'
    if acq_time_s is not None:
        title += f'\nAcquisition time: {acq_time_s:.1f} s'
    ax.set_title(title, fontsize=14, weight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add statistics text box
    total_counts = int(np.sum(hist_roi))
    max_counts = int(np.max(hist_roi))
    mean_time = np.average(time_bins, weights=hist_roi) if np.sum(hist_roi) > 0 else 0
    
    stats_text = f'Total counts: {total_counts:,}\n'
    stats_text += f'Peak counts: {max_counts:,}\n'
    stats_text += f'Mean TOA: {mean_time:.2f} ns\n'
    stats_text += f'Time resolution: {resolution_s*1e12:.2f} ps'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_separated_histograms(hist, resolution_s, t_min_ns, t_max_ns, power_uw, output_path, 
                              cut_time_ns=76.2, block_id=None, acq_time_s=None,
                              photon_ref_pos=None):
    """
    Plot two separated histograms by a cut time value with photon number peaks marked.
    Uses reference positions to assign photon numbers correctly.
    
    Parameters:
    -----------
    hist : np.ndarray
        Full histogram data
    resolution_s : float
        Time resolution in seconds
    t_min_ns : float
        Minimum time for plot window (ns)
    t_max_ns : float
        Maximum time for plot window (ns)
    power_uw : float
        Laser power in µW
    output_path : Path or str
        Where to save the plot
    cut_time_ns : float
        Time value to separate histograms (ns)
    block_id : int, optional
        Block/trace ID from acquisition
    acq_time_s : float, optional
        Acquisition time in seconds
    photon_ref_pos : dict, optional
        Reference photon positions {n: position_ns} for matching peaks
    """
    from pathlib import Path
    from scipy.signal import find_peaks
    
    # Default reference positions from block curve 9 (3.86 µW)
    if photon_ref_pos is None:
        photon_ref_pos = {
            3: 76.2760,
            2: 76.3960,
            1: 76.5360
        }
    
    # Convert time window to bins
    bin_min = int(t_min_ns * 1e-9 / resolution_s)
    bin_max = int(t_max_ns * 1e-9 / resolution_s)
    cut_bin = int(cut_time_ns * 1e-9 / resolution_s)
    
    # Extract region of interest
    if bin_max > len(hist):
        bin_max = len(hist)
    
    hist_roi = hist[bin_min:bin_max]
    
    # Create time axis using native binning
    time_bins = np.arange(len(hist_roi)) * resolution_s * 1e9 + (bin_min * resolution_s * 1e9)
    
    # Split histogram at cut
    cut_bin_roi = cut_bin - bin_min
    hist_before = hist_roi[:cut_bin_roi]
    hist_after = hist_roi[cut_bin_roi:]
    time_before = time_bins[:cut_bin_roi]
    time_after = time_bins[cut_bin_roi:]
    
    # Find peaks in both regions using sensitive detection
    peaks_before, _ = find_peaks(hist_before, 
                                prominence=np.max(hist_before)*0.02,
                                distance=20)
    
    peaks_after, _ = find_peaks(hist_after, 
                               prominence=np.max(hist_after)*0.05,
                               distance=30)
    
    # Assign photon numbers for peaks
    # For after region (shifted peak): use reference positions
    photon_map_after = {}
    for peak_idx in peaks_after:
        peak_time = time_after[peak_idx]
        # Find closest reference photon number
        distances = {n: abs(peak_time - pos) for n, pos in photon_ref_pos.items()}
        closest_n = min(distances, key=distances.get)
        photon_map_after[peak_idx] = closest_n
    
    # For before region (main peak): assign in reverse order by position
    # Higher photon numbers at earlier times
    photon_map_before = {}
    if len(peaks_before) > 0:
        # Sort peaks by index position
        sorted_peaks = sorted(enumerate(peaks_before), key=lambda x: x[1])
        # Assign highest n to earliest peak
        for rank, (orig_idx, peak_idx) in enumerate(sorted_peaks):
            n = len(peaks_before) - rank
            photon_map_before[peak_idx] = n
    
    # Create figure with 3 subplots (full + two separated)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Full histogram with cut line
    ax = axes[0]
    ax.plot(time_bins, hist_roi, linewidth=1.2, alpha=0.85, color='steelblue', label='Full data')
    ax.axvline(cut_time_ns, color='red', linestyle='--', linewidth=2.0, 
               label=f'Cut at {cut_time_ns:.2f} ns', alpha=0.7)
    ax.set_xlabel('TOA (ns)', fontsize=12, weight='bold')
    ax.set_ylabel('Counts', fontsize=12, weight='bold')
    ax.set_title(f'Full Histogram with Cut Line', fontsize=13, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Before cut with photon number peaks marked
    ax = axes[1]
    ax.plot(time_before, hist_before, linewidth=1.2, alpha=0.85, color='darkgreen')
    
    # Mark and label photon number peaks in before region
    colors = ['blue', 'purple', 'orange', 'brown', 'pink', 'cyan']
    for i, peak_idx in enumerate(peaks_before):
        peak_time = time_before[peak_idx]
        peak_height = hist_before[peak_idx]
        n = photon_map_before.get(peak_idx, i+1)
        ax.plot(peak_time, peak_height, 'x', color=colors[i % len(colors)], 
               markersize=12, markeredgewidth=2)
        # Label with photon number
        ax.text(peak_time, peak_height + 50, f'n={n}', ha='center', fontsize=10, 
               weight='bold', color=colors[i % len(colors)])
    
    counts_before = int(np.sum(hist_before))
    ax.set_xlabel('TOA (ns)', fontsize=12, weight='bold')
    ax.set_ylabel('Counts', fontsize=12, weight='bold')
    ax.set_title(f'Before Cut (TOA < {cut_time_ns:.2f} ns): {counts_before:,} counts - Main Peak', 
                 fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: After cut with photon number peaks marked
    ax = axes[2]
    ax.plot(time_after, hist_after, linewidth=1.2, alpha=0.85, color='darkred')
    
    # Mark and label photon number peaks in after region
    for i, peak_idx in enumerate(peaks_after):
        peak_time = time_after[peak_idx]
        peak_height = hist_after[peak_idx]
        n = photon_map_after.get(peak_idx, len(peaks_after) - i)
        ax.plot(peak_time, peak_height, 'x', color=colors[i % len(colors)], 
               markersize=12, markeredgewidth=2)
        # Label with photon number
        ax.text(peak_time, peak_height + 20, f'n={n}', ha='center', fontsize=10, 
               weight='bold', color=colors[i % len(colors)])
    
    counts_after = int(np.sum(hist_after))
    ax.set_xlabel('TOA (ns)', fontsize=12, weight='bold')
    ax.set_ylabel('Counts', fontsize=12, weight='bold')
    ax.set_title(f'After Cut (TOA >= {cut_time_ns:.2f} ns): {counts_after:,} counts - Shifted Peak', 
                 fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle(f'TOA Histogram Separated at {cut_time_ns:.2f} ns - Power: {power_uw:.4f} µW', 
                 fontsize=14, weight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return {
        'counts_before': counts_before,
        'counts_after': counts_after,
        'ratio': counts_before / counts_after if counts_after > 0 else 0,
        'peaks_before': peaks_before,
        'peaks_after': peaks_after,
        'photon_map_before': photon_map_before,
        'photon_map_after': photon_map_after,
        'time_before': time_before,
        'time_after': time_after
    }


def plot_histogram_with_fits(hist, resolution_s, peak_analysis, power_uw, output_path):
    """
    Plot histogram with fitted peaks for a single power level.
    
    Parameters:
    -----------
    hist : np.ndarray
        Full histogram data
    resolution_s : float
        Time resolution
    peak_analysis : dict
        Results from fit_photon_number_peaks
    power_uw : float
        Laser power in µW
    output_path : Path or str
        Where to save the plot
    """
    if peak_analysis is None:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Get data
    bins_roi = np.array(peak_analysis['bins_roi'])
    hist_roi = np.array(peak_analysis['hist_roi'])
    fit_curve = np.array(peak_analysis['fit_curve'])
    peaks = peak_analysis['peaks']
    
    # Convert bins to time (ns)
    start_bin = int(peaks[0]['mean_bin'] - bins_roi[0])  # Reconstruct start bin
    time_roi_ns = (start_bin + bins_roi) * resolution_s * 1e9
    
    # Top panel: Histogram with total fit
    ax1.bar(time_roi_ns, hist_roi, width=resolution_s*1e9, alpha=0.6, color='lightblue', edgecolor='blue', label='Data')
    ax1.plot(time_roi_ns, fit_curve, 'r-', linewidth=2, label=f'Total fit (χ²/ndf={peak_analysis["fit_quality"]["chi2_ndf"]:.2f})')
    
    # Plot individual Gaussian components
    for peak in peaks:
        mean_ns = peak['mean_ns']
        std_ns = peak['std_ns']
        amplitude = peak['amplitude']
        photon_num = peak['photon_number']
        
        # Generate Gaussian curve
        time_range = time_roi_ns
        gaussian_curve = gaussian(
            (time_range - mean_ns) / (resolution_s * 1e9),
            amplitude,
            0,
            std_ns / (resolution_s * 1e9)
        )
        
        ax1.plot(time_range, gaussian_curve, '--', linewidth=1.5, alpha=0.7,
                label=f'n={photon_num}: μ={mean_ns:.2f}ns, σ={std_ns:.2f}ns')
    
    ax1.set_xlabel('Time (ns)', fontsize=12)
    ax1.set_ylabel('Counts', fontsize=12)
    ax1.set_title(f'Photon Number Peak Fitting - Power: {power_uw:.3f} µW', fontsize=14, weight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Residuals
    residuals = hist_roi - fit_curve
    ax2.bar(time_roi_ns, residuals, width=resolution_s*1e9, alpha=0.6, color='gray', edgecolor='black')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel('Time (ns)', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Fit Residuals', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
