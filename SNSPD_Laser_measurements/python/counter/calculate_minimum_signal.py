#!/usr/bin/env python3
"""
Calculate minimum detectable signal above dark count fluctuations
Determines what count difference is statistically significant
"""

import numpy as np
from pathlib import Path
import re
from datetime import datetime

def parse_filename(filename):
    """Extract power and timestamp from filename"""
    match = re.search(r'(\d+)nW_(\d{8}_\d{4})\.txt', filename)
    if match:
        power_nw = int(match.group(1))
        timestamp_str = match.group(2)
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M')
        return power_nw, timestamp
    return None, None

def read_counter_file(filepath):
    """Read counter data file and extract bias voltage and count rates"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data_lines = lines[1:]
    
    bias_voltages = []
    count_rates = []
    count_errors = []
    
    for line in data_lines:
        parts = line.strip().split('\t')
        if len(parts) < 7:
            continue
            
        bias_voltage = float(parts[0])
        measurements = [float(x) for x in parts[6:]]
        
        median_count_rate = np.median(measurements)
        std_error = np.std(measurements) / np.sqrt(len(measurements))
        
        bias_voltages.append(bias_voltage)
        count_rates.append(median_count_rate)
        count_errors.append(std_error)
    
    return np.array(bias_voltages), np.array(count_rates), np.array(count_errors)

def main():
    # Data directory
    data_dir = Path('/Users/ya/Documents/Projects/SNSPD/SNSPD_data/SMSPD_3/Counter_sweep_power_3/2-7/6K/0nW')
    
    # Get latest 8 files
    all_files = sorted(data_dir.glob('*.txt'), key=lambda x: x.stat().st_mtime, reverse=True)
    latest_files = all_files[:8]
    
    dark_files = []
    for filepath in latest_files:
        power, timestamp = parse_filename(filepath.name)
        if power == 0 and timestamp is not None:
            dark_files.append((filepath, timestamp))
    
    dark_files.sort(key=lambda x: x[1])
    
    # Analyze all bias voltages
    first_bias, _, _ = read_counter_file(dark_files[0][0])
    
    print("="*80)
    print("MINIMUM DETECTABLE SIGNAL ANALYSIS")
    print("="*80)
    print(f"Based on {len(dark_files)} dark count measurements over 7.12 hours")
    print("\nFor each bias voltage, calculating:")
    print("  - Statistical error (from counting statistics)")
    print("  - Temporal fluctuation (real system variation)")
    print("  - Minimum detectable signal (various confidence levels)")
    print("="*80)
    
    results = []
    
    for bias_idx in range(len(first_bias)):
        bias_val = first_bias[bias_idx]
        
        rates = []
        stat_errors = []
        
        for filepath, timestamp in dark_files:
            bias_voltages, count_rates, count_errors = read_counter_file(filepath)
            if bias_idx < len(count_rates):
                rates.append(count_rates[bias_idx])
                stat_errors.append(count_errors[bias_idx])
        
        rates = np.array(rates)
        stat_errors = np.array(stat_errors)
        
        median_rate = np.median(rates)
        mean_stat_error = np.mean(stat_errors)
        temporal_std = np.std(rates)
        
        # Minimum detectable signals at different confidence levels
        # Using temporal fluctuation as the uncertainty
        signal_1sigma = temporal_std  # 68% confidence
        signal_2sigma = 2 * temporal_std  # 95% confidence
        signal_3sigma = 3 * temporal_std  # 99.7% confidence
        signal_5sigma = 5 * temporal_std  # "discovery" level
        
        # Same but using statistical error only (ignoring temporal drift)
        stat_1sigma = mean_stat_error
        stat_2sigma = 2 * mean_stat_error
        stat_3sigma = 3 * mean_stat_error
        
        results.append({
            'bias': bias_val,
            'median': median_rate,
            'temporal_std': temporal_std,
            'stat_error': mean_stat_error,
            'signal_1sigma': signal_1sigma,
            'signal_2sigma': signal_2sigma,
            'signal_3sigma': signal_3sigma,
            'signal_5sigma': signal_5sigma,
            'stat_3sigma': stat_3sigma
        })
    
    # Print results for selected bias voltages
    selected_indices = [0, len(results)//4, len(results)//2, 3*len(results)//4, -1]
    
    for idx in selected_indices:
        if idx >= len(results):
            continue
            
        r = results[idx]
        
        print(f"\n{'='*80}")
        print(f"Bias Voltage: {r['bias']*1000:.1f} mV")
        print(f"{'='*80}")
        print(f"Median dark count:           {r['median']:10.1f} counts/s")
        print(f"Statistical error (SE):      {r['stat_error']:10.1f} counts/s")
        print(f"Temporal fluctuation (SD):   {r['temporal_std']:10.1f} counts/s")
        print(f"Fluctuation/Statistical:     {r['temporal_std']/r['stat_error']:10.1f}×")
        print(f"\n--- Minimum Detectable Signal (above dark counts) ---")
        print(f"\nUsing TEMPORAL fluctuation (real-world scenario):")
        print(f"  1σ (68% CL):   {r['signal_1sigma']:10.1f} counts/s  ({r['signal_1sigma']/r['median']*100:6.2f}% of dark)")
        print(f"  2σ (95% CL):   {r['signal_2sigma']:10.1f} counts/s  ({r['signal_2sigma']/r['median']*100:6.2f}% of dark)")
        print(f"  3σ (99.7% CL): {r['signal_3sigma']:10.1f} counts/s  ({r['signal_3sigma']/r['median']*100:6.2f}% of dark)")
        print(f"  5σ (discovery):{r['signal_5sigma']:10.1f} counts/s  ({r['signal_5sigma']/r['median']*100:6.2f}% of dark)")
        
        print(f"\nUsing STATISTICAL error only (ideal case, ignoring drift):")
        print(f"  3σ:            {r['stat_3sigma']:10.1f} counts/s  ({r['stat_3sigma']/r['median']*100:6.2f}% of dark)")
        
        print(f"\n→ Recommendation: Signal should exceed {r['signal_3sigma']:.0f} counts/s")
        print(f"  (3σ above temporal fluctuation = {r['signal_3sigma']/r['median']*100:.1f}% of dark count)")
    
    print(f"\n{'='*80}")
    print("SUMMARY RECOMMENDATIONS")
    print("="*80)
    print("\nDue to significant temporal drift, use TEMPORAL fluctuation (not stat error)")
    print("for determining minimum detectable signal.\n")
    
    # Find typical bias voltage (mid-range)
    mid_idx = len(results)//2
    r_mid = results[mid_idx]
    
    print(f"At typical bias ({r_mid['bias']*1000:.1f} mV, ~{r_mid['median']:.0f} counts/s dark):")
    print(f"  Conservative (3σ): Signal must exceed {r_mid['signal_3sigma']:.0f} counts/s")
    print(f"  Very conservative (5σ): Signal must exceed {r_mid['signal_5sigma']:.0f} counts/s")
    print(f"\nAnything below ~{r_mid['signal_2sigma']:.0f} counts/s (2σ) is likely")
    print(f"indistinguishable from dark count fluctuations.")
    
    # Create summary table
    print(f"\n{'='*80}")
    print("QUICK REFERENCE TABLE (3σ threshold)")
    print("="*80)
    print(f"{'Bias (mV)':>12} {'Dark (c/s)':>12} {'3σ Threshold':>15} {'% of Dark':>12}")
    print("-"*80)
    
    for r in results[::len(results)//10 or 1]:  # Show ~10 rows
        if r['median'] > 0:
            print(f"{r['bias']*1000:12.1f} {r['median']:12.1f} {r['signal_3sigma']:15.1f} {r['signal_3sigma']/r['median']*100:12.1f}%")
    
    print("="*80)

if __name__ == "__main__":
    main()
