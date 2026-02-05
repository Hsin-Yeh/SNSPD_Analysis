#!/usr/bin/env python3
"""
Analyze the difference between signal-window-only vs full-histogram dark count estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import the PHU reader from analyze_bias_sweep
sys.path.insert(0, str(Path(__file__).parent))
from analyze_bias_sweep import read_phu_file_simple


def analyze_dark_statistics(phu_file, signal_window_ns=(75.0, 79.0)):
    """
    Compare dark count estimation methods and calculate statistical errors.
    """
    
    print(f"Analyzing: {phu_file}\n")
    
    # Read file
    header, histograms = read_phu_file_simple(phu_file)
    
    # Extract parameters
    acq_time_ms = header.get('MeasDesc_AcquisitionTime', 30000)
    acq_time_s = acq_time_ms / 1000.0
    resolution_s = header.get('MeasDesc_Resolution', 4e-12)
    resolution_ps = resolution_s * 1e12
    
    # Calculate signal window bins
    t_min_ns, t_max_ns = signal_window_ns
    bin_min = int(t_min_ns * 1e-9 / resolution_s)
    bin_max = int(t_max_ns * 1e-9 / resolution_s)
    signal_width_ns = t_max_ns - t_min_ns
    
    print(f"Acquisition time: {acq_time_s:.1f} s")
    print(f"Time resolution: {resolution_ps:.1f} ps")
    print(f"Signal window: {t_min_ns:.1f}-{t_max_ns:.1f} ns (width: {signal_width_ns:.1f} ns)")
    print(f"Number of bias points: {len(histograms)}\n")
    
    # Analyze a few representative bias voltages
    curve_indices = header.get('HistResDscr_CurveIndex', {})
    
    print("="*90)
    print(f"{'Bias':<8} {'Signal Window':<30} {'Full Histogram':<30} {'Ratio':<10}")
    print(f"{'(mV)':<8} {'Counts | Rate (cts/s) | σ':<30} {'Counts | Rate (cts/s) | σ':<30} {'Full/Signal':<10}")
    print("="*90)
    
    # Track data for plotting
    bias_voltages = []
    signal_rates = []
    signal_errors = []
    full_rates = []
    full_errors = []
    
    for block_idx in range(len(histograms)):
        if block_idx >= len(histograms):
            continue
            
        bias_mV = curve_indices.get(block_idx, None) if isinstance(curve_indices, dict) else None
        if bias_mV is None:
            continue
        
        hist = histograms[block_idx]
        
        # Method 1: Signal window only
        if bin_max <= len(hist):
            signal_counts = int(np.sum(hist[bin_min:bin_max]))
            signal_rate = signal_counts / acq_time_s
            signal_error = np.sqrt(signal_counts) / acq_time_s  # Poisson error
        else:
            signal_counts = 0
            signal_rate = 0
            signal_error = 0
        
        # Method 2: Full histogram
        total_counts = int(np.sum(hist))
        total_rate = total_counts / acq_time_s
        total_error = np.sqrt(total_counts) / acq_time_s  # Poisson error
        
        # Calculate ratio
        ratio = total_rate / signal_rate if signal_rate > 0 else 0
        
        bias_voltages.append(bias_mV)
        signal_rates.append(signal_rate)
        signal_errors.append(signal_error)
        full_rates.append(total_rate)
        full_errors.append(total_error)
        
        # Print only non-zero and representative values
        if signal_counts > 10 or total_counts > 100:
            print(f"{bias_mV:<8.0f} {signal_counts:>6} | {signal_rate:>7.2f} | {signal_error:>5.2f}     "
                  f"{total_counts:>6} | {total_rate:>7.1f} | {total_error:>5.1f}     {ratio:>5.1f}x")
    
    print("="*90)
    
    # Calculate histogram properties
    total_time_range_ns = len(histograms[0]) * resolution_s * 1e9
    print(f"\nHistogram properties:")
    print(f"  Total bins: {len(histograms[0])}")
    print(f"  Total time range: {total_time_range_ns:.1f} ns")
    print(f"  Signal window / Total range: {signal_width_ns:.1f} / {total_time_range_ns:.1f} = {signal_width_ns/total_time_range_ns:.4f}")
    print(f"  Expected ratio (if uniform): {total_time_range_ns/signal_width_ns:.1f}x")
    
    # Statistical analysis at highest dark count
    max_idx = np.argmax(full_rates)
    max_bias = bias_voltages[max_idx]
    max_signal_rate = signal_rates[max_idx]
    max_signal_error = signal_errors[max_idx]
    max_full_rate = full_rates[max_idx]
    max_full_error = full_errors[max_idx]
    actual_ratio = max_full_rate / max_signal_rate if max_signal_rate > 0 else 0
    
    print(f"\nAt maximum dark count ({max_bias:.0f} mV):")
    print(f"  Signal window method: {max_signal_rate:.2f} ± {max_signal_error:.2f} cts/s ({max_signal_error/max_signal_rate*100:.1f}% error)")
    print(f"  Full histogram method: {max_full_rate:.1f} ± {max_full_error:.1f} cts/s ({max_full_error/max_full_rate*100:.1f}% error)")
    print(f"  Actual ratio: {actual_ratio:.1f}x")
    
    # Calculate if difference is statistically significant
    # Difference in rates
    rate_diff = max_full_rate - max_signal_rate * actual_ratio
    # Combined error (for scaled comparison)
    combined_error = np.sqrt((max_full_error)**2 + (max_signal_error * actual_ratio)**2)
    
    print(f"\nStatistical significance:")
    print(f"  Expected full rate (if uniform): {max_signal_rate * (total_time_range_ns/signal_width_ns):.1f} cts/s")
    print(f"  Observed full rate: {max_full_rate:.1f} cts/s")
    print(f"  Difference: {max_full_rate - max_signal_rate * (total_time_range_ns/signal_width_ns):.1f} cts/s")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    bias_arr = np.array(bias_voltages)
    
    # Plot 1: Both methods with error bars
    ax1.errorbar(bias_arr, signal_rates, yerr=signal_errors, marker='s', color='blue', 
                 markersize=6, label='Signal window only (75-79 ns)', capsize=3, linewidth=1.5)
    ax1.errorbar(bias_arr, full_rates, yerr=full_errors, marker='o', color='red', 
                 markersize=6, label='Full histogram', capsize=3, linewidth=1.5, alpha=0.7)
    
    ax1.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax1.set_ylabel('Dark Count Rate (cts/s)', fontsize=12)
    ax1.set_title('Dark Count Rate: Signal Window vs Full Histogram', fontsize=13, weight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ratio
    ratios = np.array(full_rates) / np.array(signal_rates)
    ratios[np.array(signal_rates) == 0] = 0
    
    ax2.plot(bias_arr, ratios, marker='d', color='green', markersize=6, linewidth=1.5)
    ax2.axhline(y=total_time_range_ns/signal_width_ns, color='orange', linestyle='--', 
                linewidth=2, label=f'Expected ratio (uniform): {total_time_range_ns/signal_width_ns:.1f}x')
    
    ax2.set_xlabel('Bias Voltage (mV)', fontsize=12)
    ax2.set_ylabel('Ratio (Full / Signal Window)', fontsize=12)
    ax2.set_title('Ratio of Full Histogram to Signal Window Dark Counts', fontsize=13, weight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = Path(phu_file).parent / "dark_count_statistics_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_file}")
    
    plt.close()


if __name__ == "__main__":
    phu_file = "/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_0nW_20260205_0518.phu"
    
    if len(sys.argv) > 1:
        phu_file = sys.argv[1]
    
    analyze_dark_statistics(phu_file)
