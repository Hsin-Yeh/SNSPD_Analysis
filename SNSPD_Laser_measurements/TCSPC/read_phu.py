#!/usr/bin/env python3
"""
Read PicoQuant .phu files (PicoHarp Unified format)

Based on PicoQuant PHU file format specification.
"""

import struct
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats
from tcspc_config import T_MIN_NS, T_MAX_NS, SIGNAL_WIDTH_NS, FIT_MAX_UW, OUTPUT_DIR_INDIVIDUAL
from tcspc_analysis import (
    extract_oot_pre_dark_counts,
    subtract_dark_counts,
    fit_power_law,
    print_fit_summary,
    print_chi2_explanation,
)
from results_generator import write_bias_results, append_to_history_log

def read_phu_file(filepath, verbose=True):
    """Read a .phu file and extract header information and data."""
    
    with open(filepath, 'rb') as f:
        # Read magic string (8 bytes)
        magic = f.read(8).decode('ascii').rstrip('\0')
        if verbose:
            print(f"Magic: {magic}")
        
        # Read version (8 bytes)
        version = f.read(8).decode('ascii').rstrip('\0')
        if verbose:
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
        if verbose:
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
            if verbose:
                print("\n=== Reading Histogram Data ===")
            
            num_curves = header.get('HistoResult_NumberOfCurves', 0)
            if verbose:
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
                    
                    if verbose:
                        print(f"  Curve {i}: {num_bins} bins, total counts = {sum(hist_data)}")
            
            return header, histograms
        
        return header, None


def get_block_count_rate_from_file(block0_file, block0_block, time_window_ns):
    """Return count rate for a specific block from an external PHU file."""
    if block0_file is None or block0_block is None:
        return None

    block0_path = Path(block0_file)
    if not block0_path.exists():
        print(f"Warning: Block-0 reference file not found: {block0_path}")
        return None

    header, histograms = read_phu_file(block0_path, verbose=False)
    if histograms is None:
        print(f"Warning: No histogram data in block-0 reference file: {block0_path}")
        return None

    curve_indices = header.get('HistResDscr_CurveIndex', {})
    curve_idx = None
    for idx, block_id in curve_indices.items():
        if block_id == block0_block:
            curve_idx = idx
            break

    if curve_idx is None or curve_idx >= len(histograms):
        print(f"Warning: Block {block0_block} not found in reference file: {block0_path}")
        return None

    resolution_s = header.get('MeasDesc_Resolution', 4e-12)
    acq_time_ms = header.get('MeasDesc_AcquisitionTime', 10000)
    acq_time_s = acq_time_ms / 1000.0
    t_min_ns, t_max_ns = time_window_ns
    bin_min = int(t_min_ns * 1e-9 / resolution_s)
    bin_max = int(t_max_ns * 1e-9 / resolution_s)

    hist = histograms[curve_idx]
    if bin_max > len(hist):
        return None

    counts_in_window = int(np.sum(hist[bin_min:bin_max]))
    return counts_in_window / acq_time_s

def load_power_data(power_file_path):
    """Load power data from Attenuation file (Angle -> Power in nW)."""
    power_data = {}
    
    if not Path(power_file_path).exists():
        return power_data
    
    try:
        with open(power_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Split by any whitespace (tabs or spaces)
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        # Convert angle to integer (round float angles)
                        angle_deg = int(round(float(parts[0])))
                        # Power in nW (keep as float, convert from scientific notation)
                        power_nw = float(parts[1])
                        power_data[angle_deg] = power_nw
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Warning: Could not load power data from {power_file_path}: {e}")
    
    return power_data

def plot_count_rate_vs_power(header, histograms, power_data, output_dir, time_window_ns=None, bias_voltage=None,
                             block0_ref_file=None, block0_ref_block=None, debug=False, source_file=None):
    """Create a plot showing count rate vs power with time window cut on histograms."""
    if time_window_ns is None:
        time_window_ns = (T_MIN_NS, T_MAX_NS)
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
    
    if debug:
        print(f"Applying time window cut: {t_min_ns:.1f}-{t_max_ns:.1f} ns (bins {bin_min}-{bin_max})")
        print(f"Acquisition time per curve: {acq_time_s:.3f} s\n")
    
    # Collect data points
    powers = []
    output_counts = []
    window_counts = []
    estimated_dark_counts = []  # Dark count estimate from 0-10 ns region
    estimated_dark_counts_early = []  # Dark count estimate from 0-60 ns region
    estimated_dark_counts_mid = []    # Dark count estimate from 80-100 ns region
    estimated_dark_counts_late = []   # Dark count estimate from >100 ns region
    block_ids = []
    dark_count_rate = None  # For 0µW (dark count)
    
    # Define out-of-time regions
    signal_width_ns = t_max_ns - t_min_ns  # 5.4 ns
    
    oot_regions = {
        '0-10ns': (0, 10.0),
        '0-60ns': (0, 60.0),
        '80-100ns': (80.0, 100.0),
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
                    # Calculate count rate in TOA window
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
                        elif region_name == '80-100ns' and block_id != 0:
                            estimated_dark_counts_mid.append(estimated_dark)
                        elif region_name == '100-200ns' and block_id != 0:
                            estimated_dark_counts_late.append(estimated_dark)
                    
                    # Separate dark count data (0µW at 0 µW)
                    if block_id == 0:
                        dark_count_rate = count_rate
                    else:
                        powers.append(power_data[block_id])
                        output_counts.append(count_rate)
                        window_counts.append(counts_in_window)
                        block_ids.append(block_id)
    
    # If 0µW is missing, try external reference file
    if dark_count_rate is None and block0_ref_file is not None and block0_ref_block is not None:
        external_dark = get_block_count_rate_from_file(block0_ref_file, block0_ref_block, time_window_ns)
        if external_dark is not None:
            dark_count_rate = external_dark
            print(f"Using external 0µW reference: {block0_ref_file} (block {block0_ref_block})")
            print(f"External 0µW count rate: {dark_count_rate:.2f} cts/s")

    if len(powers) == 0:
        print("No matching power-count data points")
        return
    
    # Convert to numpy arrays for fitting
    powers_arr = np.array(powers)
    counts_arr = np.array(output_counts)
    window_counts_arr = np.array(window_counts)
    estimated_dark_arr_early = np.array(estimated_dark_counts_early)
    estimated_dark_arr_mid = np.array(estimated_dark_counts_mid)
    estimated_dark_arr_late = np.array(estimated_dark_counts_late)
    
    # Handle dark count subtraction using shared function
    print(f"\n=== Dark Count Subtraction ===")
    print("Using OOT_{mid} (80-100 ns) dark count method")
    
    # Subtract per-measurement dark counts
    if len(estimated_dark_arr_mid) > 0:
        counts_arr_corrected, dark_count_rate_used = subtract_dark_counts(
            counts_arr, estimated_dark_arr_mid, method='per_measurement'
        )
        print(f"OOT_{{mid}} dark count (per-measurement subtraction): {dark_count_rate_used:.2f} cts/s (mean)")
    else:
        print("Warning: No OOT_{mid} dark data available")
        counts_arr_corrected = counts_arr
        dark_count_rate_used = None
    
    # Print statistics
    print(f"\nSignal correction (before - after):")
    print(f"  Before subtraction: mean = {np.mean(counts_arr):.1f} cts/s, std = {np.std(counts_arr):.1f} cts/s")
    print(f"  After subtraction: mean = {np.mean(counts_arr_corrected):.1f} cts/s, std = {np.std(counts_arr_corrected):.1f} cts/s")
    
    # Fit power law using shared function
    try:
        fit_results = fit_power_law(powers_arr, counts_arr_corrected, FIT_MAX_UW, measurement_time=acq_time_s)
        slope = fit_results['slope']
        intercept = fit_results['intercept']
        std_err = fit_results['std_err']
        chi2_ndf_main = fit_results['chi2_ndf']
        fit_mask = fit_results['fit_mask']
        powers_fit = fit_results['fit_powers']
        counts_fit = fit_results['fit_counts']
        fit_counts = fit_results['fit_line']
        fit_range_used = f"0 - {FIT_MAX_UW:.2e} µW"
        
        print(f"\n=== Fit Range: {fit_range_used} ===")
        print(f"Data points in fit range: {np.sum(fit_mask)}/{len(powers_arr)}")
        
        # Calculate responsivity at 100 nW
        responsivity_100nw = (10**intercept) * (0.1**slope)  # 0.1 µW = 100 nW
    except ValueError as e:
        # If fit fails due to insufficient points, try fallback range 1-3 µW
        print(f"\n⚠️ Initial fit failed: {e}")
        print(f"\n=== Attempting fallback fit: 1 - 3 µW ===")
        
        fallback_min = 1.0  # µW
        fallback_max = 3.0  # µW
        fallback_mask = (powers_arr >= fallback_min) & (powers_arr <= fallback_max)
        
        if np.sum(fallback_mask) < 2:
            print(f"❌ Fallback fit also failed: only {np.sum(fallback_mask)} points in range {fallback_min}-{fallback_max} µW")
            # Still do photon analysis even if fit fails
            print(f"\nSkipping power-law fitting, proceeding with photon number analysis only...")
            
            # Initialize with dummy values
            slope = np.nan
            intercept = np.nan
            std_err = np.nan
            chi2_ndf_main = np.nan
            fit_range_used = "fitting failed"
            fit_mask = np.zeros(len(powers_arr), dtype=bool)
            powers_fit = np.array([])
            counts_fit = np.array([])
            fit_counts = np.array([])
            
            return fig if 'fig' in locals() else None

        
        # Perform fit with fallback range
        powers_fallback = powers_arr[fallback_mask]
        counts_fallback = counts_arr_corrected[fallback_mask]
        
        log_powers_fallback = np.log10(powers_fallback)
        log_counts_fallback = np.log10(counts_fallback)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_powers_fallback, log_counts_fallback)
        
        log_fit_line = slope * log_powers_fallback + intercept
        fit_counts_fallback = 10**log_fit_line
        
        # Calculate chi-squared
        sigma_log = np.sqrt(counts_fallback / acq_time_s) / (counts_fallback * np.log(10))
        chi2 = np.sum(((log_counts_fallback - log_fit_line) / sigma_log) ** 2)
        ndf = len(counts_fallback) - 2
        chi2_ndf_main = chi2 / ndf if ndf > 0 else np.nan
        
        # Update variables for plotting
        fit_mask = fallback_mask
        powers_fit = powers_fallback
        counts_fit = counts_fallback
        fit_counts = fit_counts_fallback
        fit_range_used = f"{fallback_min:.1f} - {fallback_max:.1f} µW"
        
        print(f"✓ Fallback fit successful: {np.sum(fallback_mask)} points in range {fallback_min}-{fallback_max} µW")
        
        print_fit_summary(slope, std_err, chi2_ndf_main, intercept, dark_count_rate_used)
        print_chi2_explanation(acq_time_s=acq_time_s)
        
        # Calculate responsivity at 100 nW
        responsivity_100nw = (10**intercept) * (0.1**slope)  # 0.1 µW = 100 nW
        
        # Generate structured result files for tracking and documentation
        try:
            
            # Prepare results dictionary for output
            results_dict = {
                'slope': slope,
                'std_err': std_err,
                'chi2_ndf': chi2_ndf_main,
                'intercept': intercept,
                'dark_count_oot': dark_count_rate_used if dark_count_rate_used else 0,
                'fit_max_uw': FIT_MAX_UW,
                'fit_range_used': fit_range_used,
                'fit_points': int(np.sum(fit_mask)),
                'total_points': len(powers_arr),
                'power_range': (np.min(powers_arr[powers_arr > 0]), np.max(powers_arr)),
                'responsivity_100nw': responsivity_100nw,
                'acq_time_s': acq_time_s
            }
            
            # Generate output directory path and bias string
            bias_str = bias_voltage if bias_voltage else "unknown"
            output_base_dir = os.path.join(str(output_dir), 'results')
            os.makedirs(output_base_dir, exist_ok=True)
            
            # Write results to file
            write_bias_results(output_base_dir, bias_str, results_dict)
            append_to_history_log(output_base_dir, bias_str, results_dict)
            
            print(f"\n✓ Results saved to {output_base_dir}/")
            
        except Exception as e:
            print(f"Warning: Could not generate result files: {e}")
        
    except ValueError as e:
        print(f"Error during fitting: {e}")
        return

    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert powers to array and filter positive values for plotting (log scale requires positive values)
    powers_arr_plot = np.array(powers)
    pos_mask = (counts_arr_corrected > 0) & (counts_arr > 0) & (powers_arr_plot > 0)
    powers_pos = powers_arr_plot[pos_mask]
    counts_orig_pos = counts_arr[pos_mask]
    counts_corr_pos = counts_arr_corrected[pos_mask]
    errors_arr = np.sqrt(window_counts_arr) / acq_time_s
    errors_pos = errors_arr[pos_mask]
    
    # Plot: Output Rate vs Power (with time window cut)
    ax.errorbar(
        powers_pos,
        counts_orig_pos,
        yerr=errors_pos,
        fmt='o',
        markersize=5.0,
        alpha=0.9,
        markerfacecolor='white',
        markeredgecolor='black',
        markeredgewidth=0.9,
        ecolor='black',
        elinewidth=1.2,
        capsize=2,
        label='Data (original)',
    )
    ax.errorbar(
        powers_pos,
        counts_corr_pos,
        yerr=errors_pos,
        fmt='o',
        markersize=5.0,
        alpha=0.7,
        color='blue',
        markeredgecolor='black',
        markeredgewidth=0.9,
        ecolor='blue',
        elinewidth=1.0,
        capsize=2,
        label='Data (dark-corrected)',
    )
    
    # Plot linear fit only over the fit range
    powers_fit_pos = powers_fit[powers_fit > 0]
    if len(powers_fit_pos) > 0:
        fit_counts_fit_pos = 10**(slope * np.log10(powers_fit_pos) + intercept)
        ax.plot(powers_fit_pos, fit_counts_fit_pos, color='darkgreen', linewidth=3.2, alpha=0.95,
                label=f'Fit ({fit_range_used}): n={slope:.3f}±{std_err:.3f}, χ²/ndf={chi2_ndf_main:.4f}')
    
    # Add dark count line if available
    if dark_count_rate is not None:
        # Plot as horizontal dotted line spanning the power range
        power_range = [min(powers) * 0.5, max(powers) * 2]  # Extend slightly for visibility
        ax.plot([power_range[0], power_range[1]], [dark_count_rate, dark_count_rate], 
                'r--', linewidth=2.5, label=f'Dark (0µW: {dark_count_rate:.1f} cts/s)', alpha=0.8)
        ax.set_xlim(power_range[0], power_range[1])
    
    ax.set_xlabel('Laser Power (µW)', fontsize=12)
    ax.set_ylabel(f'Count Rate in TOA window ({t_min_ns:.1f}-{t_max_ns:.1f} ns) (cts/s)', fontsize=12)
    # ax.set_title(f'Count Rate vs Laser Power\n(TOA window: {t_min_ns:.0f}-{t_max_ns:.1f} ns)', fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    legend_title = f'{bias_voltage}' if bias_voltage else None
    ax.legend(loc='upper left', fontsize=10, title=legend_title, title_fontsize=11)
    
    # Create second figure: Dark count comparison (percentage relative to 0µW)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    
    # Dark count estimate methods comparison (percentage relative to 0µW)
    oot_0_60 = np.array(estimated_dark_counts_early)
    oot_80_100 = np.array(estimated_dark_counts_mid)
    oot_100_200 = np.array(estimated_dark_counts_late)
    
    # Calculate percentage deviations (with protection for zero or missing dark count)
    if dark_count_rate is not None and dark_count_rate > 0:
        pct_0_60 = (oot_0_60 - dark_count_rate) / dark_count_rate
        pct_80_100 = (oot_80_100 - dark_count_rate) / dark_count_rate
        pct_100_200 = (oot_100_200 - dark_count_rate) / dark_count_rate
    else:
        pct_0_60 = np.zeros_like(oot_0_60)
        pct_80_100 = np.zeros_like(oot_80_100)
        pct_100_200 = np.zeros_like(oot_100_200)
    
    ax2.scatter(powers_arr, pct_0_60, s=80, alpha=0.7, color='green', edgecolors='darkgreen', linewidth=1.5,
                label=r'OOT$_{pre}$', marker='^')
    ax2.scatter(powers_arr, pct_80_100, s=80, alpha=0.7, color='orange', edgecolors='darkorange', linewidth=1.5,
                label=r'OOT$_{mid}$', marker='o')
    ax2.scatter(powers_arr, pct_100_200, s=80, alpha=0.7, color='purple', edgecolors='indigo', linewidth=1.5,
                label=r'OOT$_{post}$', marker='s')
    
    # Reference line at 0% (0µW)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2.5, label='0µW', alpha=0.8)
    
    ax2.set_xlabel('Laser Power (µW)', fontsize=12)
    ax2.set_ylabel('(OOT - 0µW)/0µW', fontsize=12)
    # ax2.set_title('Dark Count Methods Comparison', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.axhspan(-0.05, 0.05, alpha=0.1, color='green', label='±5% range')
    legend_title = f'{bias_voltage}' if bias_voltage else None
    ax2.legend(loc='best', fontsize=10, title=legend_title, title_fontsize=11)
    
    plt.tight_layout()
    
    # Create third figure: Effect of dark count subtraction methods on fit
    # Only create if we have valid data for comparison
    try:
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 7))
        
        # Helper function for chi-square calculation
        def calc_chi2_ndf(y_obs, y_fit, n_params=2):
            """Calculate reduced chi-squared in log space using proper rate errors."""
            ndf = len(y_obs) - n_params
            if ndf <= 0:
                return np.nan
            log_obs = np.log10(y_obs)
            log_fit = np.log10(y_fit)
            # For rates: σ_log = √(rate/t) / (rate * ln(10))
            sigma_log = np.sqrt(y_obs / acq_time_s) / (y_obs * np.log(10))
            chi2 = np.sum(((log_obs - log_fit) / sigma_log) ** 2)
            return chi2 / ndf
        
        # Prepare different dark count subtraction scenarios for low-power region only
        powers_lowp = powers_arr[fit_mask]
        counts_lowp = counts_arr[fit_mask]
        window_counts_lowp = window_counts_arr[fit_mask]
        sigma_lowp = np.sqrt(window_counts_lowp) / acq_time_s
        
        # Method 1: No dark count subtraction
        log_powers_no_dark = np.log10(powers_lowp)
        log_counts_no_dark = np.log10(counts_lowp)
        slope_no_dark, intercept_no_dark, r_no_dark, _, std_err_no_dark = stats.linregress(log_powers_no_dark, log_counts_no_dark)
        log_fit_no_dark = slope_no_dark * log_powers_no_dark + intercept_no_dark
        fit_no_dark = 10**log_fit_no_dark
        chi2_ndf_no_dark = calc_chi2_ndf(counts_lowp, fit_no_dark)
        
        # Method 2: 0µW subtraction (only if 0µW is available)
        if dark_count_rate is not None:
            counts_lowp_block0 = np.maximum(counts_lowp - dark_count_rate, 0.1)  # Minimum 0.1 cts/s
            log_counts_block0 = np.log10(counts_lowp_block0)
            slope_block0, intercept_block0, r_block0, _, std_err_block0 = stats.linregress(log_powers_no_dark, log_counts_block0)
            log_fit_block0 = slope_block0 * log_powers_no_dark + intercept_block0
            fit_block0 = 10**log_fit_block0
            chi2_ndf_block0 = calc_chi2_ndf(counts_lowp_block0, fit_block0)
        else:
            # No 0µW available - set to None
            slope_block0 = std_err_block0 = chi2_ndf_block0 = None
            counts_lowp_block0 = fit_block0 = None
        
        # Method 3: OOT 0-60 ns subtraction
        oot_0_60_lowp = estimated_dark_arr_early[fit_mask]
        counts_lowp_oot_0_60 = np.maximum(counts_lowp - oot_0_60_lowp, 0.1)  # Minimum 0.1 cts/s
        log_counts_oot_0_60 = np.log10(counts_lowp_oot_0_60)
        slope_oot_0_60, intercept_oot_0_60, r_oot_0_60, _, std_err_oot_0_60 = stats.linregress(log_powers_no_dark, log_counts_oot_0_60)
        log_fit_oot_0_60 = slope_oot_0_60 * log_powers_no_dark + intercept_oot_0_60
        fit_oot_0_60 = 10**log_fit_oot_0_60
        chi2_ndf_oot_0_60 = calc_chi2_ndf(counts_lowp_oot_0_60, fit_oot_0_60)
        
        # Method 4: OOT 80-100 ns subtraction
        oot_mid_lowp = estimated_dark_arr_mid[fit_mask]
        counts_lowp_oot_mid = np.maximum(counts_lowp - oot_mid_lowp, 0.1)  # Minimum 0.1 cts/s
        log_counts_oot_mid = np.log10(counts_lowp_oot_mid)
        slope_oot_mid, intercept_oot_mid, r_oot_mid, _, std_err_oot_mid = stats.linregress(log_powers_no_dark, log_counts_oot_mid)
        log_fit_oot_mid = slope_oot_mid * log_powers_no_dark + intercept_oot_mid
        fit_oot_mid = 10**log_fit_oot_mid
        chi2_ndf_oot_mid = calc_chi2_ndf(counts_lowp_oot_mid, fit_oot_mid)
        
        # Method 5: OOT 100-200 ns subtraction
        oot_late_lowp = estimated_dark_arr_late[fit_mask]
        counts_lowp_oot_late = np.maximum(counts_lowp - oot_late_lowp, 0.1)  # Minimum 0.1 cts/s
        log_counts_oot_late = np.log10(counts_lowp_oot_late)
        slope_oot_late, intercept_oot_late, r_oot_late, _, std_err_oot_late = stats.linregress(log_powers_no_dark, log_counts_oot_late)
        log_fit_oot_late = slope_oot_late * log_powers_no_dark + intercept_oot_late
        fit_oot_late = 10**log_fit_oot_late
        chi2_ndf_oot_late = calc_chi2_ndf(counts_lowp_oot_late, fit_oot_late)
        
        # Plot data and fits (with statistical error bars)
        ax3.errorbar(
            powers_lowp,
            counts_lowp,
            yerr=sigma_lowp,
            fmt='o',
            markersize=4.5,
            alpha=0.4,
            color='gray',
            markeredgecolor='black',
            markeredgewidth=0.8,
            ecolor='gray',
            elinewidth=1.0,
            capsize=2,
            label='data (no correction)',
            zorder=1,
        )
        if dark_count_rate is not None:
            ax3.errorbar(
                powers_lowp,
                counts_lowp_block0,
                yerr=sigma_lowp,
                fmt='o',
                markersize=4.5,
                alpha=0.7,
                color='darkblue',
                markeredgecolor='navy',
                markeredgewidth=0.8,
                ecolor='darkblue',
                elinewidth=1.0,
                capsize=2,
                label='0µW subtracted',
                zorder=5,
            )
        ax3.errorbar(
            powers_lowp,
            counts_lowp_oot_0_60,
            yerr=sigma_lowp,
            fmt='d',
            markersize=4.5,
            alpha=0.5,
            color='green',
            markeredgecolor='darkgreen',
            markeredgewidth=0.8,
            ecolor='green',
            elinewidth=1.0,
            capsize=2,
            label=r'OOT$_{pre}$ (0-60ns) sub',
            zorder=3,
        )
        ax3.errorbar(
            powers_lowp,
            counts_lowp_oot_mid,
            yerr=sigma_lowp,
            fmt='^',
            markersize=4.5,
            alpha=0.5,
            color='orange',
            markeredgecolor='darkorange',
            markeredgewidth=0.8,
            ecolor='orange',
            elinewidth=1.0,
            capsize=2,
            label=r'OOT$_{mid}$ (80-100ns) sub',
            zorder=4,
        )
        ax3.errorbar(
            powers_lowp,
            counts_lowp_oot_late,
            yerr=sigma_lowp,
            fmt='s',
            markersize=4.5,
            alpha=0.5,
            color='purple',
            markeredgecolor='indigo',
            markeredgewidth=0.8,
            ecolor='purple',
            elinewidth=1.0,
            capsize=2,
            label=r'OOT$_{post}$ (100-200ns) sub',
            zorder=5,
        )
        
        # Plot fit lines
        ax3.plot(powers_lowp, fit_no_dark, 'k--', linewidth=2.5, alpha=0.6,
             label=f'No corr: n={slope_no_dark:.3f}±{std_err_no_dark:.3f}, χ²/ndf={chi2_ndf_no_dark:.3f}')
        if dark_count_rate is not None:
            ax3.plot(powers_lowp, fit_block0, 'b-', linewidth=3, 
                     label=f'0µW: n={slope_block0:.3f}±{std_err_block0:.3f}, χ²/ndf={chi2_ndf_block0:.3f}')
        ax3.plot(powers_lowp, fit_oot_0_60, 'green', linewidth=2.5, linestyle='-.', alpha=0.8,
             label=fr'OOT$_{{pre}}$ (0-60ns): n={slope_oot_0_60:.3f}±{std_err_oot_0_60:.3f}, χ²/ndf={chi2_ndf_oot_0_60:.3f}')
        ax3.plot(powers_lowp, fit_oot_mid, 'orange', linewidth=2.5, linestyle=(0, (3, 1, 1, 1)), alpha=0.8,
             label=fr'OOT$_{{mid}}$ (80-100ns): n={slope_oot_mid:.3f}±{std_err_oot_mid:.3f}, χ²/ndf={chi2_ndf_oot_mid:.3f}')
        ax3.plot(powers_lowp, fit_oot_late, 'purple', linewidth=2.5, linestyle=':', alpha=0.8,
             label=fr'OOT$_{{post}}$ (100-200ns): n={slope_oot_late:.3f}±{std_err_oot_late:.3f}, χ²/ndf={chi2_ndf_oot_late:.3f}')
        
        ax3.set_xlabel('Laser Power (µW)', fontsize=13)
        ax3.set_ylabel('Count Rate (cts/s)', fontsize=13)
        ax3.set_title(f'Dark Count Subtraction Comparison\n(Low-Power Region: < {FIT_MAX_UW:.2e} µW)', 
                  fontsize=13, weight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        legend_title = f'{bias_voltage}' if bias_voltage else None
        ax3.legend(loc='upper left', fontsize=10, framealpha=0.95, title=legend_title, title_fontsize=11)
        
        plt.tight_layout()
    except Exception as e:
        print(f"Warning: Could not create comparison figure: {e}")
        if 'fig3' in locals():
            plt.close(fig3)
    
    # Create fourth figure: Difference between OOT regions vs power (percent)
    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
    if dark_count_rate is not None and dark_count_rate > 0:
        diff_oot_pct = 100 * (oot_0_60 - oot_80_100) / dark_count_rate
    else:
        diff_oot_pct = oot_0_60 - oot_80_100
    ax4.scatter(powers_arr, diff_oot_pct, s=90, alpha=0.8, color='teal', edgecolors='black', linewidth=1.2)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax4.set_xlabel('Laser Power (µW)', fontsize=12)
    ax4.set_ylabel(r'(OOT$_{pre}$ - OOT$_{mid}$) / 0µW', fontsize=12)
    # ax4.set_title(r'Difference Between OOT$_{pre}$ and OOT$_{mid}$ vs Power', fontsize=12, weight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    # Add bias voltage as text in upper right since this plot doesn't have a legend
    if bias_voltage:
        ax4.text(0.98, 0.95, f'{bias_voltage}', transform=ax4.transAxes, fontsize=12, weight='bold',
                 verticalalignment='top', horizontalalignment='right')
    
    print(f"\nRegion 1: 0-60 ns (broad pre-signal baseline)")
    print(f"  Mean: {np.mean(oot_0_60):.2f} cts/s")
    print(f"  Std Dev: {np.std(oot_0_60):.2f} cts/s")
    print(f"  Min: {np.min(oot_0_60):.2f} cts/s")
    print(f"  Max: {np.max(oot_0_60):.2f} cts/s")
    
    if dark_count_rate is not None:
        print(f"\n0µW baseline (0 µW): {dark_count_rate:.2f} cts/s")
        print(f"\nComparison vs 0µW:")
        print(f"  0-60 ns:   {np.mean(oot_0_60) - dark_count_rate:+.2f} cts/s ({100*(np.mean(oot_0_60) - dark_count_rate)/dark_count_rate:+.1f}%)")
        print(f"\n0µW baseline (0 µW): {dark_count_rate:.2f} cts/s")
        mean_diff = np.mean(estimated_dark_arr_early) - dark_count_rate
        mean_pct = 100 * mean_diff / dark_count_rate
        print(f"Difference from baseline (mean): {mean_diff:.2f} cts/s ({mean_pct:+.1f}%)")
        
        # Per-measurement percentage comparison
        pct_diff = 100 * (estimated_dark_arr_early - dark_count_rate) / dark_count_rate
        print(f"\nPer-measurement OOT vs 0µW comparison:")
        print(f"  Mean percentage difference: {np.mean(pct_diff):+.1f}%")
        print(f"  Std Dev: {np.std(pct_diff):.1f}%")
        print(f"  Min: {np.min(pct_diff):+.1f}%")
        print(f"  Max: {np.max(pct_diff):+.1f}%")
        
        # Detailed comparison table
        print(f"\nDetailed OOT Dark Count vs 0µW Baseline:")
        print(f"{'Power (µW)':<12} {'OOT Est (cts/s)':<18} {'0µW (cts/s)':<18} {'Diff (cts/s)':<15} {'Diff (%)':<10}")
        print(f"{'-'*73}")
        for i in range(len(powers_arr)):
            oot_val = estimated_dark_arr_early[i]
            diff_val = oot_val - dark_count_rate
            diff_pct = 100 * diff_val / dark_count_rate
            print(f"{powers_arr[i]:<12.4e} {oot_val:<18.2f} {dark_count_rate:<18.2f} {diff_val:<15.2f} {diff_pct:+.1f}%")
    
    plt.tight_layout()
    
    # Create fifth figure: Histograms for selected power levels in signal region
    # Select up to 25 power levels evenly distributed
    power_indices_to_plot = []
    if len(powers_arr) > 0:
        if len(powers_arr) <= 25:
            # If we have 25 or fewer points, show all
            power_indices_to_plot = list(range(len(powers_arr)))
        else:
            # Select 25 evenly distributed indices
            power_indices_to_plot = np.linspace(0, len(powers_arr) - 1, 25, dtype=int).tolist()
    
    if power_indices_to_plot:
        num_subplots = len(power_indices_to_plot)
        # Use 5 columns, calculate rows needed
        ncols = 5
        nrows = (num_subplots + ncols - 1) // ncols
        fig5, axes5 = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
        
        # Flatten axes array for easier indexing
        if nrows == 1:
            axes5_flat = axes5 if ncols == 1 else axes5.flatten()
        else:
            axes5_flat = axes5.flatten()
        
        # Get time axis for x-axis labels (in nanoseconds)
        time_axis_ns = np.arange(len(histograms[0])) * resolution_s * 1e9
        
        # Cut directly to signal region (no buffer)
        bin_min_plot = int(t_min_ns * 1e-9 / resolution_s)
        bin_max_plot = int(t_max_ns * 1e-9 / resolution_s)
        bin_min_plot = max(0, bin_min_plot)
        bin_max_plot = min(len(histograms[0]), bin_max_plot)
        
        for ax_idx, power_idx in enumerate(power_indices_to_plot):
            if power_idx < len(histograms):
                hist = histograms[power_idx]
                power_val = powers_arr[power_idx] if power_idx < len(powers_arr) else 0
                
                # Plot histogram (signal region only)
                ax = axes5_flat[ax_idx]
                time_region = time_axis_ns[bin_min_plot:bin_max_plot]
                hist_region = hist[bin_min_plot:bin_max_plot]
                
                ax.plot(time_region, hist_region, linewidth=1.5, color='blue', alpha=0.7)
                ax.fill_between(time_region, hist_region, alpha=0.3, color='blue')
                
                ax.set_xlabel('Time (ns)', fontsize=9)
                if ax_idx % ncols == 0:
                    ax.set_ylabel('Counts', fontsize=9)
                ax.set_title(f'{power_val:.2e} µW', fontsize=10, weight='bold')
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
        
        # Hide unused subplots
        for ax_idx in range(len(power_indices_to_plot), len(axes5_flat)):
            axes5_flat[ax_idx].set_visible(False)
        
        fig5.suptitle(f'Time-of-Arrival Histograms - Signal Region (Bias: {bias_voltage})', fontsize=13, weight='bold', y=1.00)
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
    
    if power_indices_to_plot:
        output_path5 = output_dir / f"5_histograms_power_comparison_{t_min_ns:.1f}-{t_max_ns:.1f}ns.png"
        fig5.savefig(output_path5, dpi=200, bbox_inches='tight')
        print(f"✓ Plot 5 saved: {output_path5}")

    # Save analysis summary to JSON for combined plotting
    try:
        def _safe_float(val):
            try:
                return float(val)
            except Exception:
                return None

        results_dict_json = {
            'slope': _safe_float(slope),
            'std_err': _safe_float(std_err),
            'chi2_ndf': _safe_float(chi2_ndf_main),
            'intercept': _safe_float(intercept),
            'dark_count_oot': _safe_float(dark_count_rate_used) if dark_count_rate_used is not None else None,
            'fit_max_uw': _safe_float(FIT_MAX_UW),
            'fit_range_used': fit_range_used,
            'fit_points': int(np.sum(fit_mask)),
            'total_points': int(len(powers_arr)),
            'power_range': [
                _safe_float(np.min(powers_arr[powers_arr > 0])),
                _safe_float(np.max(powers_arr)),
            ],
            'responsivity_100nw': _safe_float(responsivity_100nw),
            'acq_time_s': _safe_float(acq_time_s),
        }

        analysis_summary = {
            'bias_voltage': bias_voltage,
            'source_file': str(source_file) if source_file else None,
            'output_dir': str(output_dir),
            'acquisition_time_s': _safe_float(acq_time_s),
            'time_window_ns': {
                't_min': _safe_float(t_min_ns),
                't_max': _safe_float(t_max_ns),
                'width': _safe_float(signal_width_ns),
            },
            'dark_subtraction': {
                'method': 'OOT_pre per_measurement',
                'dark_count_rate_used': _safe_float(dark_count_rate_used) if dark_count_rate_used is not None else None,
                'dark_count_rate_block0': _safe_float(dark_count_rate) if dark_count_rate is not None else None,
            },
            'fit': {
                'slope': _safe_float(slope),
                'intercept': _safe_float(intercept),
                'std_err': _safe_float(std_err),
                'chi2_ndf': _safe_float(chi2_ndf_main),
                'fit_max_uw': _safe_float(FIT_MAX_UW),
                'fit_range_used': fit_range_used,
                'fit_mask': fit_mask.tolist(),
                'fit_powers': powers_fit.tolist(),
            },
            'results': results_dict_json,
            'plot_paths': {
                'plot1': str(output_path),
                'plot2': str(output_path2),
                'plot3': str(output_path3),
                'plot4': str(output_path4),
            },
            'plot1_data': {
                'powers_uw': powers_pos.tolist(),
                'counts_original': counts_orig_pos.tolist(),
                'counts_corrected': counts_corr_pos.tolist(),
                'errors': errors_pos.tolist(),
            },
            'plot2_data': {
                'powers_uw': powers_arr.tolist(),
                'pct_oot_pre': pct_0_60.tolist(),
                'pct_oot_mid': pct_80_100.tolist(),
                'pct_oot_post': pct_100_200.tolist(),
            },
            'plot3_data': {
                'no_dark': {
                    'slope': _safe_float(locals().get('slope_no_dark')),
                    'std_err': _safe_float(locals().get('std_err_no_dark')),
                    'chi2_ndf': _safe_float(locals().get('chi2_ndf_no_dark')),
                },
                'block0': {
                    'slope': _safe_float(locals().get('slope_block0')),
                    'std_err': _safe_float(locals().get('std_err_block0')),
                    'chi2_ndf': _safe_float(locals().get('chi2_ndf_block0')),
                },
                'oot_pre': {
                    'slope': _safe_float(locals().get('slope_oot_0_60')),
                    'std_err': _safe_float(locals().get('std_err_oot_0_60')),
                    'chi2_ndf': _safe_float(locals().get('chi2_ndf_oot_0_60')),
                },
                'oot_mid': {
                    'slope': _safe_float(locals().get('slope_oot_mid')),
                    'std_err': _safe_float(locals().get('std_err_oot_mid')),
                    'chi2_ndf': _safe_float(locals().get('chi2_ndf_oot_mid')),
                },
                'oot_post': {
                    'slope': _safe_float(locals().get('slope_oot_late')),
                    'std_err': _safe_float(locals().get('std_err_oot_late')),
                    'chi2_ndf': _safe_float(locals().get('chi2_ndf_oot_late')),
                },
            },
            'plot4_data': {
                'powers_uw': powers_arr.tolist(),
                'diff_oot_pct': diff_oot_pct.tolist(),
            },
        }

        json_path = output_dir / 'analysis_summary.json'
        with open(json_path, 'w') as f:
            json.dump(analysis_summary, f, indent=2)
        print(f"✓ JSON summary saved: {json_path}")
    except Exception as e:
        print(f"Warning: Could not save JSON summary: {e}")
    
    # Print summary
    print(f"\n=== DARK COUNT ANALYSIS SUMMARY ===")
    print(f"TOA window: {t_min_ns:.1f}-{t_max_ns:.1f} ns ({signal_width_ns:.1f} ns width)")
    print(f"Fit range: 0 - {FIT_MAX_UW:.2e} µW ({np.sum(fit_mask)} points)")
    print("\nRegion 1: OOT_{pre} (0-60 ns)")
    print(f"  Mean: {np.mean(oot_0_60):.2f} cts/s")
    if dark_count_rate is not None:
        print(f"  Deviation from 0µW: {100*(np.mean(oot_0_60) - dark_count_rate)/dark_count_rate:+.1f}%")
        print(f"\nDark Baseline (0µW): {dark_count_rate:.2f} cts/s")
    else:
        print(f"  Note: No 0µW (dark count) measurement available in this sweep")
    print(f"\n=== POWER LAW FIT COMPARISON ===")
    print(f"No dark correction:    n = {slope_no_dark:.4f} ± {std_err_no_dark:.4f}, chi^2/ndf = {chi2_ndf_no_dark:.4f}")
    if dark_count_rate is not None:
        print(f"0µW subtraction:       n = {slope_block0:.4f} ± {std_err_block0:.4f}, chi^2/ndf = {chi2_ndf_block0:.4f}")
    print(f"OOT_{{pre}} subtract:      n = {slope_oot_0_60:.4f} ± {std_err_oot_0_60:.4f}, chi^2/ndf = {chi2_ndf_oot_0_60:.4f}")
    print(f"OOT_{{mid}} subtract:      n = {slope_oot_mid:.4f} ± {std_err_oot_mid:.4f}, chi^2/ndf = {chi2_ndf_oot_mid:.4f} [MAIN FIT - matches combined plot]")
    print(f"OOT_{{post}} subtract:     n = {slope_oot_late:.4f} ± {std_err_oot_late:.4f}, chi^2/ndf = {chi2_ndf_oot_late:.4f}")
    print(f"\nNote: All OOT methods scale their respective time windows to the {signal_width_ns:.1f} ns TOA window")
    print(f"  OOT_{{pre}} (0-60 ns)    -> 60 ns scaled to {signal_width_ns:.1f} ns")
    print(f"  OOT_{{mid}} (80-100 ns)  -> 20 ns scaled to {signal_width_ns:.1f} ns")
    print(f"  OOT_{{post}} (100-200 ns) -> 100 ns scaled to {signal_width_ns:.1f} ns")
    print(f"\nMain analysis uses: OOT_{{mid}} per-measurement subtraction (chi^2/ndf = {chi2_ndf_oot_mid:.4f})")
    print(f"This comparison figure shows alternative methods for reference")
    
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
    import re
    
    if len(sys.argv) < 2:
        print("Usage: python3 read_phu.py <file.phu> [--plot] [--group-by N] [--output-dir DIR] [--block0-file PATH] [--block0-block ID] [--debug]")
        sys.exit(1)
    
    filepath = Path(sys.argv[1])
    plot_flag = '--plot' in sys.argv or '-p' in sys.argv
    debug_flag = '--debug' in sys.argv
    
    # Parse output directory parameter
    custom_output_dir = None
    if '--output-dir' in sys.argv:
        idx = sys.argv.index('--output-dir')
        if idx + 1 < len(sys.argv):
            custom_output_dir = Path(sys.argv[idx + 1])
            custom_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Extract bias voltage from filename (e.g., "70mV" from "SMSPD_3_2-7_500kHz_70mV_20260205_0122.phu")
        filename = filepath.stem
        bias_match = re.search(r'(\d+mV)', filename)
        if bias_match:
            bias_voltage = bias_match.group(1)
            # Create output directory: ~/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/70mV/
            custom_output_dir = OUTPUT_DIR_INDIVIDUAL / bias_voltage
            custom_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Fallback: use filename as folder name
            custom_output_dir = OUTPUT_DIR_INDIVIDUAL / filename
            custom_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse group-by parameter
    group_by = 5  # Default
    for i, arg in enumerate(sys.argv):
        if arg == '--group-by' and i + 1 < len(sys.argv):
            try:
                group_by = int(sys.argv[i + 1])
            except:
                pass
    
    # Parse external block-0 reference (optional)
    block0_ref_file = None
    block0_ref_block = None
    if '--block0-file' in sys.argv:
        idx = sys.argv.index('--block0-file')
        if idx + 1 < len(sys.argv):
            block0_ref_file = sys.argv[idx + 1]
    if '--block0-block' in sys.argv:
        idx = sys.argv.index('--block0-block')
        if idx + 1 < len(sys.argv):
            try:
                block0_ref_block = int(sys.argv[idx + 1])
            except ValueError:
                block0_ref_block = None

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    if debug_flag:
        print(f"Reading: {filepath}\n")
    header, histograms = read_phu_file(filepath, verbose=debug_flag)
    
    # Try to find and load corresponding power data from Attenuation folder
    power_data = {}
    
    # Look for power file in the workspace Attenuation folder (using 1-degree interpolated data)
    workspace_attenuation = Path(__file__).parent.parent / "Attenuation" / "Rotation_10MHz_1degree_data_20260205.txt"
    
    if workspace_attenuation.exists():
        if debug_flag:
            print(f"Found power data file: {workspace_attenuation.name}")
        power_data = load_power_data(workspace_attenuation)
        if debug_flag:
            print(f"Loaded power data for {len(power_data)} angle points\n")
    
    # Print count rates summary
    print_count_rates_summary(header, group_by=group_by, filepath=filepath, power_data=power_data)
    
    if histograms is not None:
        if debug_flag:
            print(f"\nTotal curves loaded: {len(histograms)}")
        
        # Get resolution for x-axis
        resolution = header.get('MeasDesc_Resolution', 4e-12)  # Default 4 ps
        
        # Get curve names with block IDs and load power data
        curve_names = get_curve_names(header, filepath)
        curve_indices = header.get('HistResDscr_CurveIndex', {})
        
        # Load power data for legend (using 1-degree interpolated data)
        power_data_for_plot = {}
        workspace_attenuation = Path(__file__).parent.parent / "Attenuation" / "Rotation_10MHz_1degree_data_20260205.txt"
        if workspace_attenuation.exists():
            power_data_for_plot = load_power_data(workspace_attenuation)
        
        # Plot full-range histograms (linear + log)
        plot_output_dir = custom_output_dir if custom_output_dir else filepath.parent
        time_bins_full = np.arange(len(histograms[0])) * resolution * 1e9  # Convert to ns
        full_start_ns = 0.0
        full_end_ns = time_bins_full[-1] if len(time_bins_full) > 0 else 0.0

        fig_full, ax_full = plt.subplots(figsize=(14, 8))

        for i, hist in enumerate(histograms):
            # Skip empty curves
            if np.sum(hist) == 0:
                continue

            # Get count rate if available
            count_rate = header.get('HistResDscr_HistCountRate', {}).get(i, 'N/A')

            # Get block ID and power for label (without time)
            block_id = curve_indices.get(i, None)
            power_val = power_data_for_plot.get(block_id, None) if block_id is not None else None

            # Create label with Power and Count Rate (no block number in legend)
            if block_id == 0:
                label = f"0µW ({count_rate} cts/s)"
            elif block_id is not None and power_val is not None:
                label = f"{power_val:.4f} µW ({count_rate} cts/s)"
            elif block_id is not None:
                label = f"Block {block_id} ({count_rate} cts/s)"
            else:
                label = f"Curve {i} ({count_rate} cts/s)"

            ax_full.plot(time_bins_full, hist, linewidth=1.0, alpha=0.7, label=label)

        ax_full.set_xlabel('TOA (Time of Arrival) (ns)', fontsize=12)
        ax_full.set_ylabel('Counts', fontsize=12)
        ax_full.set_title(
            f'TCSPC Histograms (Full TOA Range: {full_start_ns:.1f}-{full_end_ns:.1f} ns)',
            fontsize=14,
            weight='bold',
        )
        ax_full.grid(True, alpha=0.3)
        ax_full.legend(loc='upper right', fontsize=7, ncol=2)

        plt.tight_layout()

        output_path_full = plot_output_dir / "0_histograms_full_linear.png"
        fig_full.savefig(output_path_full, dpi=200, bbox_inches='tight')
        print(f"\n✓ Full-range linear histogram saved: {output_path_full}")

        fig_full_log, ax_full_log = plt.subplots(figsize=(14, 8))
        for i, hist in enumerate(histograms):
            if np.sum(hist) == 0:
                continue

            count_rate = header.get('HistResDscr_HistCountRate', {}).get(i, 'N/A')
            block_id = curve_indices.get(i, None)
            power_val = power_data_for_plot.get(block_id, None) if block_id is not None else None

            if block_id == 0:
                label = f"0µW ({count_rate} cts/s)"
            elif block_id is not None and power_val is not None:
                label = f"{power_val:.4f} µW ({count_rate} cts/s)"
            elif block_id is not None:
                label = f"Block {block_id} ({count_rate} cts/s)"
            else:
                label = f"Curve {i} ({count_rate} cts/s)"

            ax_full_log.plot(time_bins_full, hist, linewidth=1.0, alpha=0.7, label=label)

        ax_full_log.set_xlabel('TOA (Time of Arrival) (ns)', fontsize=12)
        ax_full_log.set_ylabel('Counts', fontsize=12)
        ax_full_log.set_title(
            f'TCSPC Histograms - Log Scale (Full TOA Range: {full_start_ns:.1f}-{full_end_ns:.1f} ns)',
            fontsize=14,
            weight='bold',
        )
        ax_full_log.set_yscale('log')
        ax_full_log.grid(True, alpha=0.3, which='both')
        ax_full_log.legend(loc='upper right', fontsize=7, ncol=2)

        plt.tight_layout()

        output_path_full_log = plot_output_dir / "0_histograms_full_log.png"
        fig_full_log.savefig(output_path_full_log, dpi=200, bbox_inches='tight')
        print(f"✓ Full-range log histogram saved: {output_path_full_log}")
        plt.close(fig_full_log)
        plt.close(fig_full)

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
            if block_id == 0:
                label = f"0µW ({count_rate} cts/s)"
            elif block_id is not None and power_val is not None:
                label = f"{power_val:.4f} µW ({count_rate} cts/s)"
            elif block_id is not None:
                label = f"Block {block_id} ({count_rate} cts/s)"
            else:
                label = f"Curve {i} ({count_rate} cts/s)"
            
            # Plot with label (zoomed to signal region)
            ax.plot(time_bins, hist_zoomed, linewidth=1.0, alpha=0.7, label=label)
        
        ax.set_xlabel('TOA (Time of Arrival) (ns)', fontsize=12)
        ax.set_ylabel('Counts', fontsize=12)
        ax.set_title(f'TCSPC Histograms (TOA window: {t_min_ns:.1f}-{t_max_ns:.1f} ns)', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        
        plt.tight_layout()
        
        # Save linear-scale histogram
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
            if block_id == 0:
                label = f"0µW ({count_rate} cts/s)"
            elif block_id is not None and power_val is not None:
                label = f"{power_val:.4f} µW ({count_rate} cts/s)"
            elif block_id is not None:
                label = f"Block {block_id} ({count_rate} cts/s)"
            else:
                label = f"Curve {i} ({count_rate} cts/s)"
            
            # Plot with label (zoomed to signal region)
            ax_log.plot(time_bins, hist_zoomed, linewidth=1.0, alpha=0.7, label=label)
        
        ax_log.set_xlabel('TOA (Time of Arrival) (ns)', fontsize=12)
        ax_log.set_ylabel('Counts', fontsize=12)
        ax_log.set_title(f'TCSPC Histograms - Log Scale (TOA window: {t_min_ns:.1f}-{t_max_ns:.1f} ns)', fontsize=14, weight='bold')
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
            # Extract bias voltage from filename for result file naming
            filename = filepath.stem
            bias_match = re.search(r'(\d+mV)', filename)
            bias_voltage = bias_match.group(1) if bias_match else None
            
            plot_count_rate_vs_power(
                header,
                histograms,
                power_data_for_plot,
                plot_output_dir,
                time_window_ns=(T_MIN_NS, T_MAX_NS),
                bias_voltage=bias_voltage,
                block0_ref_file=block0_ref_file,
                block0_ref_block=block0_ref_block,
                debug=debug_flag,
                source_file=filepath,
            )
        
        if plot_flag:
            plt.show()
        else:
            print("Use --plot flag to display the plot interactively")
    else:
        print("\nNo histogram data found in file")

if __name__ == '__main__':
    main()
