#!/usr/bin/env python3
"""
Plot individual measurements vs index for dark count file
Each measurement is 10 seconds integration time
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_counter_file_individual(filepath):
    """Read counter data file and get individual measurements for each bias"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data_lines = lines[1:]
    
    measurements_dict = {}
    
    for line in data_lines:
        parts = line.strip().split('\t')
        if len(parts) < 7:
            continue
            
        bias_voltage = float(parts[0])
        time_totalize = float(parts[4])  # Integration time per measurement
        measurements = np.array([float(x) for x in parts[6:]])
        
        # Convert to rate (counts/s)
        rates = measurements / time_totalize
        
        measurements_dict[bias_voltage] = {
            'rates': rates,
            'time_per_sample': time_totalize
        }
    
    return measurements_dict

def main():
    # File path
    filepath = Path('/Users/ya/Documents/Projects/SNSPD/SNSPD_data/SMSPD_3/Counter_sweep_power_3/2-7/6K/0nW/SMSPD_3_2-7_0nW_20251213_0915.txt')
    
    print("="*80)
    print(f"Analyzing: {filepath.name}")
    print("="*80)
    
    # Read data
    data = read_counter_file_individual(filepath)
    
    bias_voltages = sorted(data.keys())
    n_bias = len(bias_voltages)
    
    print(f"Number of bias voltages: {n_bias}")
    print(f"Time per measurement: {data[bias_voltages[0]]['time_per_sample']:.0f} seconds")
    print(f"Number of measurements per bias: {len(data[bias_voltages[0]]['rates'])}")
    
    # Create subplots - one for each bias voltage
    n_cols = 2
    n_rows = (n_bias + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
    axes = axes.flatten() if n_bias > 1 else [axes]
    
    for idx, bias_val in enumerate(bias_voltages):
        rates = data[bias_val]['rates']
        indices = np.arange(len(rates))
        
        ax = axes[idx]
        
        # Plot as scatter points only (no line)
        ax.plot(indices, rates, 'o', markersize=6, alpha=0.7, color='steelblue')
        
        # Add statistics
        median_rate = np.median(rates)
        std_rate = np.std(rates)
        
        ax.axhline(y=median_rate, color='r', linestyle='--', linewidth=1.5, 
                  alpha=0.7, label=f'Median: {median_rate:.1f} c/s')
        ax.axhline(y=median_rate + std_rate, color='orange', linestyle=':', 
                  linewidth=1.5, alpha=0.5, label=f'±1σ: {std_rate:.1f} c/s')
        ax.axhline(y=median_rate - std_rate, color='orange', linestyle=':', 
                  linewidth=1.5, alpha=0.5)
        
        ax.set_xlabel('Measurement Index', fontsize=11)
        ax.set_ylabel('Count Rate (counts/s)', fontsize=11)
        ax.set_title(f'Bias: {bias_val*1000:.1f} mV', fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        
        # Print summary
        print(f"\nBias {bias_val*1000:.1f} mV:")
        print(f"  Median: {median_rate:.1f} c/s")
        print(f"  Std:    {std_rate:.1f} c/s")
        print(f"  Min:    {np.min(rates):.1f} c/s")
        print(f"  Max:    {np.max(rates):.1f} c/s")
        print(f"  CV:     {std_rate/median_rate*100:.2f}%")
    
    # Hide unused subplots
    for idx in range(n_bias, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Dark Count Measurements vs Index - {filepath.name}\\n(Each point = 10 second measurement)', 
                fontsize=14, weight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    output_dir = Path('output/SMSPD_3')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'dark_count_0915_time_series.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n\nPlot saved to: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    main()
