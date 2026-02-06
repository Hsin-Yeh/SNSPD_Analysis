#!/usr/bin/env python3
"""
Analyze SNSPD response vs bias voltage at fixed laser power.

For each measurement block:
- Block number encodes bias voltage (in mV)
- Extract count rate in signal window (75-79 ns)
- Estimate dark count from OOT region (0-60 ns) using shared functions
- Plot corrected count rate vs bias voltage
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import struct
from pathlib import Path
import json
import sys
import re

from tcspc_analysis import extract_oot_pre_dark_counts, subtract_dark_counts
from tcspc_config import OUTPUT_DIR_BIAS_SWEEP, T_MIN_NS, T_MAX_NS


def read_phu_file_simple(filepath):
    """Read a .phu file and extract header information and histogram data."""
    
    with open(filepath, 'rb') as f:
        # Read magic string (8 bytes)
        magic = f.read(8).decode('ascii').rstrip('\0')
        version = f.read(8).decode('ascii').rstrip('\0')
        
        # Read header data
        header = {}
        
        # Read variable-length header tags
        while True:
            tag_ident = f.read(32).decode('ascii').rstrip('\0')
            
            if not tag_ident:
                break
            
            tag_idx = struct.unpack('<i', f.read(4))[0]
            tag_type = struct.unpack('<i', f.read(4))[0]
            
            # Parse tag value based on type
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
                f.read(blob_size)
            else:
                tag_value = f"<unknown type 0x{tag_type:08X}>"
                f.read(8)
            
            if tag_idx == -1:
                header[tag_ident] = tag_value
            else:
                if tag_ident not in header:
                    header[tag_ident] = {}
                header[tag_ident][tag_idx] = tag_value
        
        # Read histogram data
        num_curves = header.get('HistoResult_NumberOfCurves', 0)
        bits_per_bin = header.get('HistoResult_BitsPerBin', 32)
        
        histograms = []
        for curve_idx in range(num_curves):
            num_bins = 65536
            
            if bits_per_bin == 32:
                hist_bytes = f.read(num_bins * 4)
                if len(hist_bytes) == num_bins * 4:
                    hist_data = struct.unpack(f'<{num_bins}I', hist_bytes)
                else:
                    hist_data = [0] * num_bins
            else:
                hist_bytes = f.read(num_bins * 8)
                if len(hist_bytes) == num_bins * 8:
                    hist_data = struct.unpack(f'<{num_bins}Q', hist_bytes)
                else:
                    hist_data = [0] * num_bins
            
            histograms.append(np.array(hist_data, dtype=np.uint64))
    
    return header, histograms


def analyze_bias_sweep(filepath, time_window_ns=None, output_dir=None):
    """
    Analyze SNSPD response vs bias voltage.
    
    Parameters:
    -----------
    filepath : str
        Path to .phu file
    time_window_ns : tuple, optional
        Signal extraction window in nanoseconds (t_min, t_max)
        If None, uses T_MIN_NS and T_MAX_NS from config
    output_dir : str or Path, optional
        Output directory for plots and data
    """
    
    # Use config defaults if not specified
    if time_window_ns is None:
        time_window_ns = (T_MIN_NS, T_MAX_NS)
    
    # Extract power and angle from filename
    filename = Path(filepath).name
    
    power_match = re.search(r'(\d+(?:\.\d+)?)nW', filename)
    angle_match = re.search(r'(\d+)degrees', filename)
    
    if power_match:
        laser_power_nW = float(power_match.group(1))
    else:
        laser_power_nW = 0
    
    if angle_match:
        laser_angle_deg = int(angle_match.group(1))
    else:
        laser_angle_deg = None
    
    # Set output directory
    if output_dir is None:
        # Extract power from filename and create folder
        if power_match:
            power_str = power_match.group(0)  # e.g., "99nW"
            output_dir = OUTPUT_DIR_BIAS_SWEEP / power_str
        else:
            output_dir = OUTPUT_DIR_BIAS_SWEEP / "unknown"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read file
    print(f"Reading: {filepath}\n")
    header, histograms = read_phu_file_simple(filepath)
    
    # Extract parameters
    acq_time_ms = header.get('MeasDesc_AcquisitionTime', 30000)
    acq_time_s = acq_time_ms / 1000.0
    resolution_s = header.get('MeasDesc_Resolution', 4e-12)
    
    t_min_ns, t_max_ns = time_window_ns
    bin_min = int(t_min_ns * 1e-9 / resolution_s)
    bin_max = int(t_max_ns * 1e-9 / resolution_s)
    signal_width_ns = t_max_ns - t_min_ns
    
    print(f"Signal window: {t_min_ns:.1f}-{t_max_ns:.1f} ns (width: {signal_width_ns:.1f} ns)")
    print(f"Acquisition time: {acq_time_s:.3f} s")
    print(f"Resolution: {resolution_s*1e12:.1f} ps\n")
    
    # Extract data for each block (bias voltage)
    bias_voltages = []
    count_rates = []
    dark_estimates = []
    
    num_curves = header.get('HistoResult_NumberOfCurves', 0)
    curve_indices = header.get('HistResDscr_CurveIndex', {})
    
    print(f"{'Block':<6} {'Bias (mV)':<12} {'Signal (cts/s)':<16} {'Dark OOT (cts/s)':<18} {'Net (cts/s)':<16}")
    print("-" * 80)
    
    for block_idx in range(num_curves):
        if block_idx >= len(histograms):
            continue
        
        # Get bias voltage from curve index
        bias_mV = curve_indices.get(block_idx, None) if isinstance(curve_indices, dict) else None
        
        # Skip if no bias voltage found
        if bias_mV is None:
            continue
        
        hist = histograms[block_idx]
        
        # Extract signal counts
        if bin_max <= len(hist):
            signal_counts = int(np.sum(hist[bin_min:bin_max]))
            signal_rate = signal_counts / acq_time_s
        else:
            signal_rate = 0
        
        # Use shared function to extract dark counts from OOT_pre region (0-60 ns)
        dark_counts_per_measurement = extract_oot_pre_dark_counts(
            hist, resolution_s, signal_width_ns, acq_time_s
        )
        dark_rate = dark_counts_per_measurement / acq_time_s
        
        # For dark count file (0 nW), the signal IS the dark count
        if laser_power_nW == 0:
            net_rate = 0  # No signal in dark-only measurement
        else:
            # Corrected count rate
            net_rate = signal_rate - dark_rate
        
        bias_voltages.append(bias_mV)
        count_rates.append(net_rate)
        dark_estimates.append(dark_rate)
        
        print(f"{block_idx:<6} {bias_mV:<12.0f} {signal_rate:<16.2f} {dark_rate:<18.2f} {net_rate:<16.2f}")
    
    print()
    
    # Convert to numpy arrays
    bias_arr = np.array(bias_voltages, dtype=np.float32)
    counts_arr = np.array(count_rates, dtype=np.float32)
    dark_arr = np.array(dark_estimates, dtype=np.float32)
    
    # For dark count file, use dark_arr as the main count rate
    if laser_power_nW == 0:
        counts_arr = dark_arr
    
    # Filter out zero or negative values for analysis
    valid_mask = counts_arr > 0
    bias_valid = bias_arr[valid_mask]
    counts_valid = counts_arr[valid_mask]
    
    if len(bias_valid) < 2:
        print("ERROR: Not enough valid data points")
        return
    
    # Normalize by saturation (maximum count rate)
    max_count = np.max(counts_valid)
    counts_arr_normalized = counts_arr / max_count
    dark_arr_normalized = dark_arr / max_count
    counts_normalized = counts_valid / max_count
    
    # ========== PLOT 1: Count Rate vs Bias ==========
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.scatter(bias_arr, counts_arr_normalized, s=120, alpha=0.8, color='blue', edgecolors='navy', 
                linewidth=1.5, label='SNSPD output (normalized)')
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Saturation')
    
    ax1.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax1.set_ylabel(f'Normalized Count Rate (arb. units)', fontsize=12)
    
    # Build title based on available info
    title_parts = ['SNSPD Output vs Bias Voltage (Normalized)']
    subtitle_parts = []
    if laser_power_nW > 0:
        subtitle_parts.append(f'{laser_power_nW:.0f} nW')
    else:
        subtitle_parts.append('Dark (0 nW)')
    if laser_angle_deg is not None:
        subtitle_parts.append(f'{laser_angle_deg}°')
    
    if subtitle_parts:
        title = '\n'.join([title_parts[0], f"({', '.join(subtitle_parts)})"])
    else:
        title = title_parts[0]
    
    ax1.set_title(title, fontsize=13, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_ylim([-0.05, 1.15])
    
    plot1_file = output_dir / f"1_snspd_vs_bias_normalized_{t_min_ns:.1f}-{t_max_ns:.1f}ns.png"
    plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot 1 saved: {plot1_file}")
    plt.close()
    
    # ========== PLOT 2: Breakdown (Signal, Dark, Net) ==========
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: stacked bar (normalized)
    ax2a.bar(bias_arr, dark_arr_normalized, label='Dark (OOT-est)', color='red', alpha=0.6, width=1.5)
    ax2a.bar(bias_arr, counts_arr_normalized, bottom=dark_arr_normalized, label='Net signal', color='blue', alpha=0.7, width=1.5)
    ax2a.axhline(y=1.0, color='darkred', linestyle='--', linewidth=2, alpha=0.5, label='Saturation level')
    
    ax2a.set_xlabel('Bias Voltage (mV)', fontsize=11)
    ax2a.set_ylabel('Normalized Count Rate', fontsize=11)
    ax2a.set_title('Signal Breakdown (Normalized)', fontsize=12, weight='bold')
    ax2a.legend(fontsize=10)
    ax2a.grid(True, alpha=0.3, axis='y')
    
    # Right: ratio
    total_counts = counts_arr + dark_arr
    dark_fraction = np.divide(dark_arr, total_counts, where=total_counts > 0, 
                              out=np.zeros_like(dark_arr))
    
    ax2b.scatter(bias_arr, dark_fraction * 100, s=120, alpha=0.8, color='red', 
                 edgecolors='darkred', linewidth=1.5)
    ax2b.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='5% threshold')
    
    ax2b.set_xlabel('Bias Voltage (mV)', fontsize=11)
    ax2b.set_ylabel('Dark Fraction (%)', fontsize=11)
    ax2b.set_title('Dark Count Contribution', fontsize=12, weight='bold')
    ax2b.grid(True, alpha=0.3)
    ax2b.legend(fontsize=10)
    ax2b.set_ylim([0, max(dark_fraction * 100) * 1.2])
    
    plt.tight_layout()
    plot2_file = output_dir / f"2_signal_breakdown_{t_min_ns:.1f}-{t_max_ns:.1f}ns.png"
    plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot 2 saved: {plot2_file}")
    plt.close()
    
    # ========== ANALYSIS: Threshold and plateau regions ==========
    # Find threshold region (where response starts rising sharply)
    if len(counts_valid) >= 3:
        # Calculate slope between consecutive points
        slopes = np.diff(counts_valid) / np.diff(bias_valid)
        
        # Estimate threshold and saturation regions
        threshold_idx = np.argmax(slopes)  # Steepest slope region
        threshold_bias = bias_valid[threshold_idx]
        threshold_count = counts_valid[threshold_idx]
        
        max_count = np.max(counts_valid)
        saturation_level = 0.95 * max_count
        saturation_indices = np.where(counts_valid >= saturation_level)[0]
        
        if len(saturation_indices) > 0:
            saturation_start_bias = bias_valid[saturation_indices[0]]
        else:
            saturation_start_bias = None
        
        print("=== BIAS SWEEP ANALYSIS ===")
        print(f"Threshold region (max slope): ~{threshold_bias:.0f} mV")
        print(f"Maximum output rate: {max_count:.1f} cts/s")
        if saturation_start_bias:
            print(f"Saturation region (>95% max): {saturation_start_bias:.0f} mV onwards")
    
    # ========== PLOT 3: Log scale (to see low-bias behavior) ==========
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    ax3.semilogy(bias_arr, counts_arr_normalized, 'o-', color='blue', markersize=8, linewidth=2, 
                 label='SNSPD output (normalized)')
    ax3.semilogy(bias_arr, dark_arr_normalized, 's--', color='red', markersize=6, linewidth=1.5, 
                 alpha=0.7, label='Dark (OOT, normalized)')
    
    ax3.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax3.set_ylabel('Normalized Count Rate [log scale]', fontsize=12)
    
    # Build title
    title_parts = ['SNSPD vs Bias - Log Scale (Normalized)']
    subtitle_parts = []
    if laser_power_nW > 0:
        subtitle_parts.append(f'{laser_power_nW:.0f} nW')
    else:
        subtitle_parts.append('Dark (0 nW)')
    if laser_angle_deg is not None:
        subtitle_parts.append(f'{laser_angle_deg}°')
    
    if subtitle_parts:
        title = '\n'.join([title_parts[0], f"({', '.join(subtitle_parts)})"])
    else:
        title = title_parts[0]
    
    ax3.set_title(title, fontsize=13, weight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(fontsize=11)
    
    plot3_file = output_dir / f"3_snspd_vs_bias_log_{t_min_ns:.1f}-{t_max_ns:.1f}ns.png"
    plt.savefig(plot3_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot 3 saved: {plot3_file}")
    plt.close()
    
    # Save summary to JSON
    summary = {
        'measurement': Path(filepath).name,
        'laser_power_nW': laser_power_nW,
        'laser_angle_deg': laser_angle_deg,
        'signal_window_ns': [t_min_ns, t_max_ns],
        'acquisition_time_s': acq_time_s,
        'saturation_count_rate_cts_s': float(max_count),
        'bias_voltages_mV': bias_voltages,
        'signal_rates_cts_s': [float(r) for r in counts_arr],
        'signal_rates_normalized': [float(r) for r in counts_arr_normalized],
        'dark_rates_cts_s': [float(d) for d in dark_arr],
        'dark_rates_normalized': [float(d) for d in dark_arr_normalized],
        'max_output_rate_cts_s': float(np.max(counts_arr)),
        'output_files': [
            str(plot1_file.name),
            str(plot2_file.name),
            str(plot3_file.name),
        ]
    }
    
    summary_file = output_dir / "bias_sweep_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved: {summary_file}")
    
    print(f"\n✓ Analysis complete. Output saved to: {output_dir}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_bias_sweep.py <phu_file> [--output-dir <dir>]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    output_dir = None
    
    if '--output-dir' in sys.argv:
        idx = sys.argv.index('--output-dir')
        if idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]
    
    analyze_bias_sweep(filepath, output_dir=output_dir)
