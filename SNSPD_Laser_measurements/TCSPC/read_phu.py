#!/usr/bin/env python3
"""
Read PicoQuant .phu files (PicoHarp Unified format)

Based on PicoQuant PHU file format specification.
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats

def read_phu_file(filepath):
    """Read a .phu file and extract header information and data."""
    
    with open(filepath, 'rb') as f:
        # Read magic string (8 bytes)
        magic = f.read(8).decode('ascii').rstrip('\0')
        print(f"Magic: {magic}")
        
        # Read version (8 bytes)
        version = f.read(8).decode('ascii').rstrip('\0')
        print(f"Version: {version}")
        
        # Read header data
        header = {}
        
        # Read variable-length header tags
        while True:
            # Read tag identifier (32 chars)
            tag_ident = f.read(32).decode('ascii').rstrip('\0')
            
            if not tag_ident:
                break
            
            # Read tag index (int32)
            tag_idx = struct.unpack('<i', f.read(4))[0]
            
            # Read tag type (int32)
            tag_type = struct.unpack('<i', f.read(4))[0]
            
            # Read tag value based on type
            if tag_type == 0xFFFF0001:  # tyEmpty8
                tag_value = struct.unpack('<q', f.read(8))[0]
            elif tag_type == 0x00000008:  # tyBool8
                tag_value = bool(struct.unpack('<q', f.read(8))[0])
            elif tag_type == 0x10000008:  # tyInt8
                tag_value = struct.unpack('<q', f.read(8))[0]
            elif tag_type == 0x11000008:  # tyBitSet64
                tag_value = struct.unpack('<Q', f.read(8))[0]
            elif tag_type == 0x12000008:  # tyColor8
                tag_value = struct.unpack('<Q', f.read(8))[0]
            elif tag_type == 0x20000008:  # tyFloat8
                tag_value = struct.unpack('<d', f.read(8))[0]
            elif tag_type == 0x21000008:  # tyTDateTime
                tag_value = struct.unpack('<d', f.read(8))[0]
            elif tag_type == 0x2001FFFF:  # tyFloat8Array
                array_size = struct.unpack('<q', f.read(8))[0]
                tag_value = struct.unpack(f'<{array_size}d', f.read(8 * array_size))
            elif tag_type == 0x4001FFFF:  # tyAnsiString
                string_size = struct.unpack('<q', f.read(8))[0]
                tag_value = f.read(string_size).decode('ascii', errors='ignore').rstrip('\0')
            elif tag_type == 0x4002FFFF:  # tyWideString
                string_size = struct.unpack('<q', f.read(8))[0]
                tag_value = f.read(string_size * 2).decode('utf-16-le', errors='ignore').rstrip('\0')
            elif tag_type == 0xFFFFFFFF:  # tyBinaryBlob
                blob_size = struct.unpack('<q', f.read(8))[0]
                tag_value = f"<binary data: {blob_size} bytes>"
                f.read(blob_size)  # Skip the blob
            else:
                # Unknown type - read 8 bytes and continue
                tag_value = f"<unknown type 0x{tag_type:08X}>"
                f.read(8)
            
            # Store in header dict
            if tag_idx == -1:
                header[tag_ident] = tag_value
            else:
                if tag_ident not in header:
                    header[tag_ident] = {}
                header[tag_ident][tag_idx] = tag_value
            
            # Break if we hit the end marker
            if tag_ident == 'Header_End':
                break
        
        # Print header info
        print("\n=== Header Information ===")
        for key, value in sorted(header.items()):
            if isinstance(value, dict):
                print(f"{key}:")
                for idx, val in sorted(value.items()):
                    print(f"  [{idx}]: {val}")
            else:
                print(f"{key}: {value}")
        
        # Read histogram data if present
        if 'HistResDscr_DataOffset' in header and 'HistResDscr_HistogramBins' in header:
            print("\n=== Reading Histogram Data ===")
            
            num_curves = header.get('HistoResult_NumberOfCurves', 0)
            print(f"Number of curves: {num_curves}")
            
            histograms = []
            for i in range(num_curves):
                if i in header['HistResDscr_DataOffset'] and i in header['HistResDscr_HistogramBins']:
                    offset = header['HistResDscr_DataOffset'][i]
                    num_bins = header['HistResDscr_HistogramBins'][i]
                    
                    # Seek to data offset
                    current_pos = f.tell()
                    f.seek(offset)
                    
                    # Read histogram data (32-bit unsigned integers)
                    hist_data = struct.unpack(f'<{num_bins}I', f.read(4 * num_bins))
                    histograms.append(np.array(hist_data))
                    
                    # Return to end of header
                    f.seek(current_pos)
                    
                    print(f"  Curve {i}: {num_bins} bins, total counts = {sum(hist_data)}")
            
            return header, histograms
        
        return header, None

def load_power_data(power_file_path):
    """Load power data from Attenuation file (Block# -> Power in uW)."""
    power_data = {}
    
    if not Path(power_file_path).exists():
        return power_data
    
    try:
        with open(power_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    try:
                        block_id = int(parts[0])
                        power_uw = float(parts[1])
                        power_data[block_id] = power_uw
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Warning: Could not load power data from {power_file_path}: {e}")
    
    return power_data

def plot_count_rate_vs_power(header, histograms, power_data, output_dir, time_window_ns=(75.0, 80.0), bias_voltage=None):
    """Create a plot showing count rate vs power with time window cut on histograms."""
    if not power_data or len(power_data) == 0:
        print("No power data available for count rate vs power plot")
        return
    
    if not histograms:
        print("No histogram data available for count rate vs power plot")
        return
    
    curve_indices = header.get('HistResDscr_CurveIndex', {})
    acq_time_ms = header.get('MeasDesc_AcquisitionTime', 10000)  # Default 10 seconds
    acq_time_s = acq_time_ms / 1000.0
    resolution_s = header.get('MeasDesc_Resolution', 4e-12)  # Default 4 ps
    
    # Convert time window from ns to bin indices
    t_min_ns, t_max_ns = time_window_ns
    bin_min = int(t_min_ns * 1e-9 / resolution_s)
    bin_max = int(t_max_ns * 1e-9 / resolution_s)
    
    print(f"Applying time window cut: {t_min_ns:.1f}-{t_max_ns:.1f} ns (bins {bin_min}-{bin_max})")
    print(f"Acquisition time per curve: {acq_time_s:.3f} s\n")
    
    # Collect data points
    powers = []
    output_counts = []
    estimated_dark_counts = []  # Dark count estimate from 0-10 ns region
    estimated_dark_counts_early = []  # Dark count estimate from 0-60 ns region
    estimated_dark_counts_late = []   # Dark count estimate from >100 ns region
    block_ids = []
    dark_count_rate = None  # For Block 0 (dark count)
    
    # Define out-of-time regions (avoiding >80 ns which may have signal contamination)
    signal_width_ns = t_max_ns - t_min_ns  # 5.4 ns
    
    oot_regions = {
        '0-10ns': (0, 10.0),
        '0-60ns': (0, 60.0),
        '100-200ns': (100.0, 200.0)
    }
    
    for i in range(header.get('HistoResult_NumberOfCurves', 0)):
        block_id = curve_indices.get(i, None)
        if block_id is not None and block_id in power_data:
            # Get histogram for this curve
            if i < len(histograms):
                hist = histograms[i]
                
                # Extract counts in signal time window
                if bin_max <= len(hist):
                    counts_in_window = int(np.sum(hist[bin_min:bin_max]))
                    # Calculate count rate in signal window
                    count_rate = counts_in_window / acq_time_s
                    
                    # Extract counts from multiple out-of-time regions
                    for region_name, (t_start, t_end) in oot_regions.items():
                        oot_bin_min = int(t_start * 1e-9 / resolution_s)
                        oot_bin_max = int(t_end * 1e-9 / resolution_s)
                        oot_width = t_end - t_start
                        
                        counts_oot = int(np.sum(hist[oot_bin_min:oot_bin_max]))
                        estimated_dark = (counts_oot / oot_width) * signal_width_ns / acq_time_s
                        
                        if region_name == '0-10ns' and block_id != 0:
                            estimated_dark_counts.append(estimated_dark)
                        elif region_name == '0-60ns' and block_id != 0:
                            estimated_dark_counts_early.append(estimated_dark)
                        elif region_name == '100-200ns' and block_id != 0:
                            estimated_dark_counts_late.append(estimated_dark)
                    
                    # Separate dark count data (Block 0 at 0 µW)
                    if block_id == 0:
                        dark_count_rate = count_rate
                    else:
                        powers.append(power_data[block_id])
                        output_counts.append(count_rate)
                        block_ids.append(block_id)
    
    if len(powers) == 0:
        print("No matching power-count data points")
        return
    
    # Convert to numpy arrays for fitting
    powers_arr = np.array(powers)
    counts_arr = np.array(output_counts)
    estimated_dark_arr = np.array(estimated_dark_counts)
    estimated_dark_arr_early = np.array(estimated_dark_counts_early)
    estimated_dark_arr_late = np.array(estimated_dark_counts_late)
    
    # Handle dark count subtraction
    print(f"\n=== Dark Count Subtraction ===")
    if dark_count_rate is not None:
        print(f"Using Block 0 dark count: {dark_count_rate:.2f} cts/s")
        counts_arr_corrected = counts_arr - dark_count_rate
    else:
        # No Block 0, estimate dark count from first histogram's full spectrum
        print("No Block 0 found, using full histogram estimate from lowest power measurement")
        first_hist = histograms[0]
        total_counts = int(np.sum(first_hist))
        total_rate = total_counts / acq_time_s
        total_bins = len(first_hist)
        total_time_range_ns = total_bins * resolution_s * 1e9
        dark_count_rate = total_rate * (signal_width_ns / total_time_range_ns)
        print(f"Estimated dark count rate: {dark_count_rate:.2f} cts/s (from full histogram)")
        counts_arr_corrected = counts_arr - dark_count_rate
    
    # Print statistics
    print(f"\nSignal correction (before - after):")
    print(f"  Before subtraction: mean = {np.mean(counts_arr):.1f} cts/s, std = {np.std(counts_arr):.1f} cts/s")
    print(f"  After subtraction: mean = {np.mean(counts_arr_corrected):.1f} cts/s, std = {np.std(counts_arr_corrected):.1f} cts/s")
    
    # Define fit range: low power region
    # Auto-detect reasonable fit range based on data
    if powers_arr.min() > 0.5:  # If lowest power > 0.5 µW, use wider range
        fit_min = 0
        fit_max = min(powers_arr.max(), 20.0)  # Use all data up to 20 µW
        print(f"Using extended fit range for high-power data")
    else:
        fit_min = 0  # No lower limit
        fit_max = 2e-1  # Upper limit: 0.2 µW
    
    # Filter data for fit range
    fit_mask = (powers_arr >= fit_min) & (powers_arr <= fit_max)
    powers_fit = powers_arr[fit_mask]
    counts_fit = counts_arr[fit_mask]
    
    print(f"\n=== Fit Range: {fit_min:.2e} - {fit_max:.2e} µW ===")
    print(f"Data points in fit range: {np.sum(fit_mask)}/{len(powers_arr)}")
    
    if np.sum(fit_mask) < 2:
        print("Not enough data points in fit range")
        return
    
    # Use the corrected counts for fitting
    counts_fit = counts_arr_corrected[fit_mask]
    
    # Perform linear fit on log-log data in the selected range
    # log(counts) = slope * log(power) + intercept
    # where slope = n (power law exponent)
    log_powers_fit = np.log10(powers_fit)
    log_counts_fit = np.log10(counts_fit)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_powers_fit, log_counts_fit)
    
    def chi2_ndf(y_obs, y_fit, n_params=2):
        """Calculate reduced chi-squared with Poisson errors in linear space.
        
        Even though we fit in log-log space, we evaluate χ² in linear space
        using the actual count rates to properly assess goodness-of-fit.
        χ² = Σ [(y_obs - y_fit)² / σ²]
        where σ = √y for Poisson statistics
        """
        ndf = len(y_obs) - n_params
        if ndf <= 0:
            return np.nan
        
        # Chi-squared with Poisson errors in linear space
        # Poisson: σ = √N
        sigma = np.sqrt(y_obs)
        chi2 = np.sum(((y_obs - y_fit) / sigma) ** 2)
        return chi2 / ndf
    
    # Calculate fitted values for fit range only
    log_fit_line = slope * log_powers_fit + intercept
    fit_counts = 10**log_fit_line
    chi2_ndf_main = chi2_ndf(counts_fit, fit_counts)
    
    print(f"\n=== Linear Fit Results (log-log, with out-of-time dark count correction) ===")
    print(f"Power law exponent (n): {slope:.4f} ± {std_err:.4f}")
    print(f"chi^2/ndf: {chi2_ndf_main:.6f}")
    print(f"Intercept (log scale): {intercept:.4f}")
    print(f"Fit equation: Count Rate = {10**intercept:.2f} × Power^{slope:.4f}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot: Output Rate vs Power (with time window cut)
    ax.scatter(powers, counts_arr, s=110, alpha=0.9, facecolors='white', edgecolors='black', linewidth=1.6,
               marker='o', label='Signal (original)')
    ax.scatter(powers, counts_arr_corrected, s=100, alpha=0.7, color='blue', edgecolors='black', linewidth=1.5,
               label='Signal (dark-corrected)')
    
    # Plot linear fit only over the fit range
    ax.plot(powers_fit, fit_counts, color='darkgreen', linewidth=3.2, alpha=0.95,
            label=f'Fit (low-power): n={slope:.3f}±{std_err:.3f}, χ²/ndf={chi2_ndf_main:.4f}')
    
    # Add dark count line if available
    if dark_count_rate is not None:
        # Plot as horizontal dotted line spanning the power range
        power_range = [min(powers) * 0.5, max(powers) * 2]  # Extend slightly for visibility
        ax.plot([power_range[0], power_range[1]], [dark_count_rate, dark_count_rate], 
               'r--', linewidth=2.5, label=f'Dark (Block 0: {dark_count_rate:.1f} cts/s)', alpha=0.8)
        ax.set_xlim(power_range[0], power_range[1])
    
    ax.set_xlabel('Laser Power (µW)', fontsize=12)
    ax.set_ylabel(f'Count Rate in {t_min_ns}-{t_max_ns} ns window (cts/s)', fontsize=12)
    ax.set_title(f'SNSPD Output Rate vs Laser Power (Original Data)\n(Time window: {t_min_ns:.1f}-{t_max_ns:.1f} ns)', fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=10)
    
    # Create second figure: Dark count comparison (percentage relative to Block 0)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    
    # Dark count estimate methods comparison (percentage relative to Block 0)
    oot_0_60 = np.array(estimated_dark_counts_early)
    oot_100_200 = np.array(estimated_dark_counts_late)
    
    # Calculate percentage deviations
    pct_0_60 = 100 * (oot_0_60 - dark_count_rate) / dark_count_rate
    pct_100_200 = 100 * (oot_100_200 - dark_count_rate) / dark_count_rate
    
    ax2.scatter(powers_arr, pct_0_60, s=80, alpha=0.7, color='green', edgecolors='darkgreen', linewidth=1.5,
                label='OOT_pre (0-60 ns)', marker='^')
    ax2.scatter(powers_arr, pct_100_200, s=80, alpha=0.7, color='purple', edgecolors='indigo', linewidth=1.5,
                label='OOT_post (100-200 ns)', marker='s')
    
    # Reference line at 0% (Block 0)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2.5, label='Dark baseline (0%)', alpha=0.8)
    
    ax2.set_xlabel('Laser Power (µW)', fontsize=12)
    ax2.set_ylabel('Deviation from Block 0 (%)', fontsize=12)
    ax2.set_title('Dark Count Methods Comparison\n(% Deviation from Block 0)', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.axhspan(-5, 5, alpha=0.1, color='green', label='±5% range')
    ax2.legend(loc='best', fontsize=10)
    ax2.set_ylim(-120, 30)
    
    plt.tight_layout()
    
    # Create third figure: Effect of dark count subtraction methods on fit
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 7))
    
    # Prepare different dark count subtraction scenarios for low-power region only
    powers_lowp = powers_arr[fit_mask]
    counts_lowp = counts_arr[fit_mask]
    
    # Method 1: No dark count subtraction
    log_powers_no_dark = np.log10(powers_lowp)
    log_counts_no_dark = np.log10(counts_lowp)
    slope_no_dark, intercept_no_dark, r_no_dark, _, std_err_no_dark = stats.linregress(log_powers_no_dark, log_counts_no_dark)
    log_fit_no_dark = slope_no_dark * log_powers_no_dark + intercept_no_dark
    fit_no_dark = 10**log_fit_no_dark
    chi2_ndf_no_dark = chi2_ndf(counts_lowp, fit_no_dark)
    
    # Method 2: Block 0 subtraction (current/recommended)
    counts_lowp_block0 = counts_lowp - dark_count_rate
    log_counts_block0 = np.log10(counts_lowp_block0)
    slope_block0, intercept_block0, r_block0, _, std_err_block0 = stats.linregress(log_powers_no_dark, log_counts_block0)
    log_fit_block0 = slope_block0 * log_powers_no_dark + intercept_block0
    fit_block0 = 10**log_fit_block0
    chi2_ndf_block0 = chi2_ndf(counts_lowp_block0, fit_block0)
    
    # Method 3: OOT 0-60 ns subtraction
    oot_0_60_lowp = estimated_dark_arr_early[fit_mask]
    counts_lowp_oot_0_60 = counts_lowp - oot_0_60_lowp
    log_counts_oot_0_60 = np.log10(counts_lowp_oot_0_60)
    slope_oot_0_60, intercept_oot_0_60, r_oot_0_60, _, std_err_oot_0_60 = stats.linregress(log_powers_no_dark, log_counts_oot_0_60)
    log_fit_oot_0_60 = slope_oot_0_60 * log_powers_no_dark + intercept_oot_0_60
    fit_oot_0_60 = 10**log_fit_oot_0_60
    chi2_ndf_oot_0_60 = chi2_ndf(counts_lowp_oot_0_60, fit_oot_0_60)
    
    # Method 4: OOT 100-200 ns subtraction
    oot_late_lowp = estimated_dark_arr_late[fit_mask]
    counts_lowp_oot_late = counts_lowp - oot_late_lowp
    log_counts_oot_late = np.log10(counts_lowp_oot_late)
    slope_oot_late, intercept_oot_late, r_oot_late, _, std_err_oot_late = stats.linregress(log_powers_no_dark, log_counts_oot_late)
    log_fit_oot_late = slope_oot_late * log_powers_no_dark + intercept_oot_late
    fit_oot_late = 10**log_fit_oot_late
    chi2_ndf_oot_late = chi2_ndf(counts_lowp_oot_late, fit_oot_late)
    
    # Plot data and fits
    ax3.scatter(powers_lowp, counts_lowp, s=120, alpha=0.4, color='gray', edgecolors='black', 
                linewidth=1.5, label='Original data (no correction)', zorder=1)
    ax3.scatter(powers_lowp, counts_lowp_block0, s=120, alpha=0.7, color='darkblue', edgecolors='navy', 
                linewidth=1.5, label='Dark subtracted', zorder=5)
    ax3.scatter(powers_lowp, counts_lowp_oot_0_60, s=80, alpha=0.5, color='green', edgecolors='darkgreen', 
                linewidth=1.5, marker='d', label='OOT_pre subtracted', zorder=3)
    ax3.scatter(powers_lowp, counts_lowp_oot_late, s=80, alpha=0.5, color='purple', edgecolors='indigo', 
                linewidth=1.5, marker='s', label='OOT_post subtracted', zorder=4)
    
    # Plot fit lines
    ax3.plot(powers_lowp, fit_no_dark, 'k--', linewidth=2.5, alpha=0.6,
             label=f'No corr: n={slope_no_dark:.3f}±{std_err_no_dark:.3f}')
    ax3.plot(powers_lowp, fit_block0, 'b-', linewidth=3, 
             label=f'Dark: n={slope_block0:.3f}±{std_err_block0:.3f}')
    ax3.plot(powers_lowp, fit_oot_0_60, 'green', linewidth=2.5, linestyle='-.', alpha=0.8,
             label=f'OOT_pre: n={slope_oot_0_60:.3f}±{std_err_oot_0_60:.3f}')
    ax3.plot(powers_lowp, fit_oot_late, 'purple', linewidth=2.5, linestyle=':', alpha=0.8,
             label=f'OOT_post: n={slope_oot_late:.3f}±{std_err_oot_late:.3f}')
    
    ax3.set_xlabel('Laser Power (µW)', fontsize=13)
    ax3.set_ylabel('Count Rate (cts/s)', fontsize=13)
    ax3.set_title(f'Effect of Dark Count Subtraction on Power Law Fit\n(Low-Power Region: < {fit_max:.2e} µW)', 
                  fontsize=13, weight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    
    # Create fourth figure: Difference between OOT regions vs power (percent)
    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
    diff_oot_pct = 100 * (oot_0_60 - oot_100_200) / dark_count_rate
    ax4.scatter(powers_arr, diff_oot_pct, s=90, alpha=0.8, color='teal', edgecolors='black', linewidth=1.2)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax4.set_xlabel('Laser Power (µW)', fontsize=12)
    ax4.set_ylabel('OOT_pre - OOT_post (% of Dark)', fontsize=12)
    ax4.set_title('Difference Between OOT Regions vs Power', fontsize=12, weight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    print(f"\nRegion 1: 0-60 ns (broad pre-signal baseline)")
    print(f"  Mean: {np.mean(oot_0_60):.2f} cts/s")
    print(f"  Std Dev: {np.std(oot_0_60):.2f} cts/s")
    print(f"  Min: {np.min(oot_0_60):.2f} cts/s")
    print(f"  Max: {np.max(oot_0_60):.2f} cts/s")
    
    if dark_count_rate is not None:
        print(f"\nBlock 0 baseline (0 µW): {dark_count_rate:.2f} cts/s")
        print(f"\nComparison vs Block 0:")
        print(f"  0-60 ns:   {np.mean(oot_0_60) - dark_count_rate:+.2f} cts/s ({100*(np.mean(oot_0_60) - dark_count_rate)/dark_count_rate:+.1f}%)")
        print(f"\nBlock 0 baseline (0 µW): {dark_count_rate:.2f} cts/s")
        mean_diff = np.mean(estimated_dark_arr) - dark_count_rate
        mean_pct = 100 * mean_diff / dark_count_rate
        print(f"Difference from baseline (mean): {mean_diff:.2f} cts/s ({mean_pct:+.1f}%)")
        
        # Per-measurement percentage comparison
        pct_diff = 100 * (estimated_dark_arr - dark_count_rate) / dark_count_rate
        print(f"\nPer-measurement OOT vs Block 0 comparison:")
        print(f"  Mean percentage difference: {np.mean(pct_diff):+.1f}%")
        print(f"  Std Dev: {np.std(pct_diff):.1f}%")
        print(f"  Min: {np.min(pct_diff):+.1f}%")
        print(f"  Max: {np.max(pct_diff):+.1f}%")
        
        # Detailed comparison table
        print(f"\nDetailed OOT Dark Count vs Block 0 Baseline:")
        print(f"{'Power (µW)':<12} {'OOT Est (cts/s)':<18} {'Block 0 (cts/s)':<18} {'Diff (cts/s)':<15} {'Diff (%)':<10}")
        print(f"{'-'*73}")
        for i in range(len(powers_arr)):
            oot_val = estimated_dark_arr[i]
            diff_val = oot_val - dark_count_rate
            diff_pct = 100 * diff_val / dark_count_rate
            print(f"{powers_arr[i]:<12.4e} {oot_val:<18.2f} {dark_count_rate:<18.2f} {diff_val:<15.2f} {diff_pct:+.1f}%")
    
    plt.tight_layout()
    
    # Save figures
    output_path = output_dir / f"1_count_rate_vs_power_original_{t_min_ns:.1f}-{t_max_ns:.1f}ns.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Plot 1 saved: {output_path}")
    
    output_path2 = output_dir / f"2_dark_analysis_comparison_{t_min_ns:.1f}-{t_max_ns:.1f}ns.png"
    fig2.savefig(output_path2, dpi=200, bbox_inches='tight')
    print(f"✓ Plot 2 saved: {output_path2}")
    
    output_path3 = output_dir / f"3_dark_subtraction_methods_fit_{t_min_ns:.1f}-{t_max_ns:.1f}ns.png"
    fig3.savefig(output_path3, dpi=200, bbox_inches='tight')
    print(f"✓ Plot 3 saved: {output_path3}")
    
    output_path4 = output_dir / f"4_oot_region_difference_{t_min_ns:.1f}-{t_max_ns:.1f}ns.png"
    fig4.savefig(output_path4, dpi=200, bbox_inches='tight')
    print(f"✓ Plot 4 saved: {output_path4}")
    
    # Print summary
    print(f"\n=== DARK COUNT ANALYSIS SUMMARY ===")
    print(f"Signal window: {t_min_ns:.1f}-{t_max_ns:.1f} ns ({signal_width_ns:.1f} ns width)")
    print(f"Fit range: {fit_min:.2e} - {fit_max:.2e} µW ({np.sum(fit_mask)} points)")
    print(f"\nRegion 1: OOT_pre (0-60 ns)")
    print(f"  Mean: {np.mean(oot_0_60):.2f} cts/s | Deviation: {100*(np.mean(oot_0_60) - dark_count_rate)/dark_count_rate:+.1f}%")
    print(f"\nDark Baseline (Block 0): {dark_count_rate:.2f} cts/s")
    print(f"\n=== POWER LAW FIT COMPARISON ===")
    print(f"No dark correction:    n = {slope_no_dark:.4f} ± {std_err_no_dark:.4f}, chi^2/ndf = {chi2_ndf_no_dark:.4f}")
    print(f"Dark subtraction:      n = {slope_block0:.4f} ± {std_err_block0:.4f}, chi^2/ndf = {chi2_ndf_block0:.4f} [RECOMMENDED]")
    print(f"OOT_pre subtract:      n = {slope_oot_0_60:.4f} ± {std_err_oot_0_60:.4f}, chi^2/ndf = {chi2_ndf_oot_0_60:.4f}")
    print(f"OOT_post subtract:     n = {slope_oot_late:.4f} ± {std_err_oot_late:.4f}, chi^2/ndf = {chi2_ndf_oot_late:.4f}")
    print(f"\nNote: All OOT methods scale their respective time windows to the {signal_width_ns:.1f} ns signal window")
    print(f"  OOT_pre (0-60 ns)   -> 60 ns scaled to {signal_width_ns:.1f} ns")
    print(f"  OOT_post (100-200 ns) -> 100 ns scaled to {signal_width_ns:.1f} ns")
    print(f"\nConclusion: Using Dark (Block 0: {dark_count_rate:.2f} cts/s) for dark count subtraction")
    
    plt.close('all')  # Close all figures to prevent hanging
    

    return fig

def get_curve_names(header, filepath=None):
    """Extract or generate meaningful curve names.
    
    Uses HistResDscr_CurveIndex (the Trc/Block ID from DAQ software) as primary label,
    then adds acquisition time for uniqueness.
    """
    
    num_curves = header.get('HistoResult_NumberOfCurves', 0)
    curve_indices = header.get('HistResDscr_CurveIndex', {})
    times = header.get('HistResDscr_TimeOfRecording', {})
    
    names = {}
    for i in range(num_curves):
        # Get the block/trace ID from DAQ software
        block_id = curve_indices.get(i, None)
        
        # Get the acquisition time
        if i in times:
            time_val = times[i]
            from datetime import datetime, timedelta
            try:
                base_date = datetime(1900, 1, 1)
                record_time = base_date + timedelta(days=time_val)
                time_str = record_time.strftime('%H:%M:%S')
            except:
                time_str = "unknown"
        else:
            time_str = "unknown"
        
        # Format name with both block ID and time
        if block_id is not None:
            names[i] = f"Block {block_id:3d}: {time_str}"
        else:
            names[i] = f"Curve {i:2d}: {time_str}"
    
    return names

def print_count_rates_summary(header, group_by=5, filepath=None, power_data=None):
    """Print a summary table of input/output rates for all curves.
    
    Args:
        group_by: Number of curves per group (default 5)
        power_data: Dict mapping block ID to power in uW
    """
    print("\n=== Count Rates Summary with Power ===")
    
    if power_data:
        print(f"{'Curve':<12} {'Power (uW)':<12} {'Input Rate':<15} {'Output Rate':<15} {'Sync Rate':<15} {'Integral':<12}")
        print("-" * 92)
    else:
        print(f"{'Curve':<12} {'Input Rate':<15} {'Output Rate':<15} {'Sync Rate':<15} {'Integral':<12}")
        print("-" * 72)
    
    num_curves = header.get('HistoResult_NumberOfCurves', 0)
    
    input_rates = header.get('HistResDscr_InputRate', {})
    output_rates = header.get('HistResDscr_HistCountRate', {})
    sync_rates = header.get('HistResDscr_SyncRate', {})
    integrals = header.get('HistResDscr_IntegralCount', {})
    
    curve_names = get_curve_names(header, filepath)
    curve_indices = header.get('HistResDscr_CurveIndex', {})
    
    for i in range(num_curves):
        input_rate = input_rates.get(i, 0)
        output_rate = output_rates.get(i, 0)
        sync_rate = sync_rates.get(i, 0)
        integral = integrals.get(i, 0)
        
        curve_label = curve_names.get(i, f"Curve {i}")
        
        # Get block ID and power
        block_id = curve_indices.get(i, None)
        power_val = power_data.get(block_id, None) if power_data and block_id else None
        
        if power_data and power_val is not None:
            print(f"{curve_label:<12} {power_val:<12.4f} {input_rate:<15.0f} {output_rate:<15.0f} {sync_rate:<15.0f} {integral:<12.0f}")
        else:
            print(f"{curve_label:<12} {input_rate:<15.0f} {output_rate:<15.0f} {sync_rate:<15.0f} {integral:<12.0f}")
        
        # Print separator every N curves
        if (i + 1) % group_by == 0 and i < num_curves - 1:
            if power_data:
                print("-" * 92)
            else:
                print("-" * 72)
    
    return input_rates, output_rates

def main():
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python3 read_phu.py <file.phu> [--plot] [--group-by N] [--output-dir DIR]")
        sys.exit(1)
    
    filepath = Path(sys.argv[1])
    plot_flag = '--plot' in sys.argv or '-p' in sys.argv
    
    # Parse output directory parameter
    custom_output_dir = None
    if '--output-dir' in sys.argv:
        idx = sys.argv.index('--output-dir')
        if idx + 1 < len(sys.argv):
            custom_output_dir = Path(sys.argv[idx + 1])
            custom_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse group-by parameter
    group_by = 5  # Default
    for i, arg in enumerate(sys.argv):
        if arg == '--group-by' and i + 1 < len(sys.argv):
            try:
                group_by = int(sys.argv[i + 1])
            except:
                pass
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    print(f"Reading: {filepath}\n")
    header, histograms = read_phu_file(filepath)
    
    # Try to find and load corresponding power data from Attenuation folder
    power_data = {}
    
    # Look for power file in the workspace Attenuation folder
    workspace_attenuation = Path(__file__).parent.parent / "Attenuation" / "Rotation_10MHz_5degrees_data_20260205.txt"
    
    if workspace_attenuation.exists():
        print(f"Found power data file: {workspace_attenuation.name}")
        power_data = load_power_data(workspace_attenuation)
        print(f"Loaded power data for {len(power_data)} block IDs\n")
    
    # Print count rates summary
    print_count_rates_summary(header, group_by=group_by, filepath=filepath, power_data=power_data)
    
    if histograms is not None:
        print(f"\nTotal curves loaded: {len(histograms)}")
        
        # Get resolution for x-axis
        resolution = header.get('MeasDesc_Resolution', 4e-12)  # Default 4 ps
        
        # Get curve names with block IDs and load power data
        curve_names = get_curve_names(header, filepath)
        curve_indices = header.get('HistResDscr_CurveIndex', {})
        
        # Load power data for legend
        power_data_for_plot = {}
        workspace_attenuation = Path(__file__).parent.parent / "Attenuation" / "Rotation_10MHz_5degrees_data_20260205.txt"
        if workspace_attenuation.exists():
            power_data_for_plot = load_power_data(workspace_attenuation)
        
        # Define signal region (time window cut)
        t_min_ns, t_max_ns = 75.0, 79.0
        bin_min = int(t_min_ns * 1e-9 / resolution)
        bin_max = int(t_max_ns * 1e-9 / resolution)
        
        # Plot all histograms on same frame
        num_curves = len(histograms)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, hist in enumerate(histograms):
            # Create time axis in nanoseconds - zoomed to signal region
            time_bins_full = np.arange(len(hist)) * resolution * 1e9  # Convert to ns
            time_bins = time_bins_full[bin_min:bin_max]
            hist_zoomed = hist[bin_min:bin_max]
            
            # Get count rate if available
            count_rate = header.get('HistResDscr_HistCountRate', {}).get(i, 'N/A')
            
            # Skip empty curves
            if sum(hist_zoomed) == 0:
                continue
            
            # Get block ID and power for label (without time)
            block_id = curve_indices.get(i, None)
            power_val = power_data_for_plot.get(block_id, None) if block_id else None
            
            # Create label with Power and Count Rate (no block number in legend)
            if block_id is not None and power_val is not None:
                label = f"{power_val:.4f} µW ({count_rate} cts/s)"
            elif block_id is not None:
                label = f"Block {block_id} ({count_rate} cts/s)"
            else:
                label = f"Curve {i} ({count_rate} cts/s)"
            
            # Plot with label (zoomed to signal region)
            ax.plot(time_bins, hist_zoomed, linewidth=1.0, alpha=0.7, label=label)
        
        ax.set_xlabel('Time (ns)', fontsize=12)
        ax.set_ylabel('Counts', fontsize=12)
        ax.set_title(f'TCSPC Histograms (Signal Region: {t_min_ns:.1f}-{t_max_ns:.1f} ns)', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        
        plt.tight_layout()
        
        # Save linear-scale histogram
        plot_output_dir = custom_output_dir if custom_output_dir else filepath.parent
        output_path = plot_output_dir / f"0_histograms_linear_{t_min_ns:.1f}-{t_max_ns:.1f}ns.png"
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Linear histogram saved: {output_path}")
        
        # Create log-scale version
        fig_log, ax_log = plt.subplots(figsize=(14, 8))
        
        for i, hist in enumerate(histograms):
            # Create time axis in nanoseconds - zoomed to signal region
            time_bins_full = np.arange(len(hist)) * resolution * 1e9  # Convert to ns
            time_bins = time_bins_full[bin_min:bin_max]
            hist_zoomed = hist[bin_min:bin_max]
            
            # Get count rate if available
            count_rate = header.get('HistResDscr_HistCountRate', {}).get(i, 'N/A')
            
            # Skip empty curves
            if sum(hist_zoomed) == 0:
                continue
            
            # Get block ID and power for label (without time)
            block_id = curve_indices.get(i, None)
            power_val = power_data_for_plot.get(block_id, None) if block_id else None
            
            # Create label with Power and Count Rate (no block number in legend)
            if block_id is not None and power_val is not None:
                label = f"{power_val:.4f} µW ({count_rate} cts/s)"
            elif block_id is not None:
                label = f"Block {block_id} ({count_rate} cts/s)"
            else:
                label = f"Curve {i} ({count_rate} cts/s)"
            
            # Plot with label (zoomed to signal region)
            ax_log.plot(time_bins, hist_zoomed, linewidth=1.0, alpha=0.7, label=label)
        
        ax_log.set_xlabel('Time (ns)', fontsize=12)
        ax_log.set_ylabel('Counts', fontsize=12)
        ax_log.set_title(f'TCSPC Histograms - Log Scale (Signal Region: {t_min_ns:.1f}-{t_max_ns:.1f} ns)', fontsize=14, weight='bold')
        ax_log.set_yscale('log')
        ax_log.grid(True, alpha=0.3, which='both')
        ax_log.legend(loc='upper right', fontsize=7, ncol=2)
        
        plt.tight_layout()
        
        output_path_log = plot_output_dir / f"0_histograms_log_{t_min_ns:.1f}-{t_max_ns:.1f}ns.png"
        fig_log.savefig(output_path_log, dpi=200, bbox_inches='tight')
        print(f"✓ Log histogram saved: {output_path_log}")
        plt.close(fig_log)
        
        # Create count rate vs power plot
        plot_output_dir = custom_output_dir if custom_output_dir else filepath.parent
        if power_data_for_plot:
            plot_count_rate_vs_power(header, histograms, power_data_for_plot, plot_output_dir)
        
        if plot_flag:
            plt.show()
        else:
            print("Use --plot flag to display the plot interactively")
    else:
        print("\nNo histogram data found in file")

if __name__ == '__main__':
    main()
