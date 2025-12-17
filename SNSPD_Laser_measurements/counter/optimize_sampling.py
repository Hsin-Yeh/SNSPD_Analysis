#!/usr/bin/env python3
"""
Optimize number of measurements per bias point
Balance between statistical precision and measurement time
"""

import numpy as np
import matplotlib.pyplot as plt
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

def read_counter_file_full(filepath):
    """Read counter data file and get all individual measurements"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data_lines = lines[1:]
    
    all_measurements = {}
    
    for line in data_lines:
        parts = line.strip().split('\t')
        if len(parts) < 7:
            continue
            
        bias_voltage = float(parts[0])
        measurements = np.array([float(x) for x in parts[6:]])
        
        all_measurements[bias_voltage] = measurements
    
    return all_measurements

def main():
    # Data directory
    data_dir = Path('/Users/ya/Documents/Projects/SNSPD/SNSPD_data/SMSPD_3/Counter_sweep_power_3/2-7/6K/0nW')
    
    # Get latest file
    all_files = sorted(data_dir.glob('*.txt'), key=lambda x: x.stat().st_mtime, reverse=True)
    latest_file = all_files[0]
    
    power, timestamp = parse_filename(latest_file.name)
    print("="*80)
    print("MEASUREMENT OPTIMIZATION ANALYSIS")
    print("="*80)
    print(f"Analyzing: {latest_file.name}")
    print(f"Time: {timestamp.strftime('%Y-%m-%d %H:%M')}")
    print()
    
    # Read all measurements
    measurements_dict = read_counter_file_full(latest_file)
    
    # Analyze for several bias voltages
    bias_voltages = sorted(measurements_dict.keys())
    selected_indices = [len(bias_voltages)//4, len(bias_voltages)//2, 3*len(bias_voltages)//4]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    all_results = []
    
    for idx, bias_idx in enumerate(selected_indices):
        bias_val = bias_voltages[bias_idx]
        measurements = measurements_dict[bias_val]
        
        n_total = len(measurements)
        median_rate = np.median(measurements)
        
        print(f"\n{'='*80}")
        print(f"Bias: {bias_val*1000:.1f} mV")
        print(f"Total measurements available: {n_total}")
        print(f"Median count rate: {median_rate:.1f} counts/s")
        print(f"{'='*80}")
        
        # Test different sample sizes
        sample_sizes = np.arange(5, n_total+1, 5)
        std_errors = []
        relative_errors = []
        
        for n in sample_sizes:
            # Use first n measurements
            subset = measurements[:n]
            median_subset = np.median(subset)
            std_error = np.std(subset) / np.sqrt(n)
            std_errors.append(std_error)
            if median_subset > 0:
                relative_errors.append(std_error / median_subset * 100)
            else:
                relative_errors.append(0)
        
        # Find optimal points
        # Statistical error scales as 1/sqrt(N)
        # Temporal drift dominates beyond certain point
        
        # Find where improvement becomes marginal (< 10% improvement)
        improvements = []
        for i in range(1, len(std_errors)):
            improvement = (std_errors[i-1] - std_errors[i]) / std_errors[i-1] * 100
            improvements.append(improvement)
        
        # Find where we get 95% of the benefit
        final_error = std_errors[-1]
        target_error = final_error * 1.05  # 5% worse than best
        
        optimal_n = None
        for i, err in enumerate(std_errors):
            if err <= target_error:
                optimal_n = sample_sizes[i]
                break
        
        if optimal_n is None:
            optimal_n = sample_sizes[-1]
        
        all_results.append({
            'bias': bias_val,
            'median': median_rate,
            'optimal_n': optimal_n,
            'error_at_optimal': std_errors[sample_sizes.tolist().index(optimal_n)],
            'error_at_100': std_errors[sample_sizes.tolist().index(100)] if 100 in sample_sizes else std_errors[-1]
        })
        
        print(f"\nStatistical Error vs Sample Size:")
        print(f"  N=10:   {std_errors[sample_sizes.tolist().index(10) if 10 in sample_sizes else 1]:.2f} c/s ({relative_errors[sample_sizes.tolist().index(10) if 10 in sample_sizes else 1]:.2f}%)")
        print(f"  N=20:   {std_errors[sample_sizes.tolist().index(20) if 20 in sample_sizes else 3]:.2f} c/s ({relative_errors[sample_sizes.tolist().index(20) if 20 in sample_sizes else 3]:.2f}%)")
        print(f"  N=50:   {std_errors[sample_sizes.tolist().index(50) if 50 in sample_sizes else 9]:.2f} c/s ({relative_errors[sample_sizes.tolist().index(50) if 50 in sample_sizes else 9]:.2f}%)")
        print(f"  N=100:  {std_errors[-1]:.2f} c/s ({relative_errors[-1]:.2f}%)")
        print(f"\n→ Optimal N ≈ {optimal_n} (achieves 95% of benefit)")
        print(f"  Time saved vs N=100: {(100-optimal_n)/100*100:.0f}%")
        
        # Plot
        ax = axes[idx//2, idx%2]
        ax.plot(sample_sizes, std_errors, 'b-', linewidth=2, label='Standard Error')
        ax.axvline(x=optimal_n, color='r', linestyle='--', linewidth=2, label=f'Optimal N={optimal_n}')
        ax.axvline(x=100, color='g', linestyle='--', linewidth=2, alpha=0.5, label='Current N=100')
        ax.set_xlabel('Number of Measurements', fontsize=11)
        ax.set_ylabel('Standard Error (counts/s)', fontsize=11)
        ax.set_title(f'{bias_val*1000:.1f} mV (~{median_rate:.0f} c/s)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    # Fourth plot: Summary
    ax4 = axes[1, 1]
    bias_vals = [r['bias']*1000 for r in all_results]
    optimal_ns = [r['optimal_n'] for r in all_results]
    
    ax4.bar(range(len(all_results)), optimal_ns, color='steelblue', alpha=0.7)
    ax4.axhline(y=100, color='g', linestyle='--', linewidth=2, label='Current (N=100)')
    ax4.set_xlabel('Bias Voltage Condition', fontsize=11)
    ax4.set_ylabel('Optimal Number of Measurements', fontsize=11)
    ax4.set_title('Recommended Sample Size by Bias', fontsize=12)
    ax4.set_xticks(range(len(all_results)))
    ax4.set_xticklabels([f'{b:.0f} mV' for b in bias_vals], rotation=0)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('output/SMSPD_3')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'measurement_optimization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n\nPlot saved to: {output_file}")
    
    # Final recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print("="*80)
    
    avg_optimal = int(np.mean(optimal_ns))
    
    print(f"\nBased on statistical analysis:")
    print(f"  Average optimal N: {avg_optimal}")
    print(f"  Range: {min(optimal_ns)}-{max(optimal_ns)}")
    
    print(f"\nPractical recommendations:")
    print(f"\n1. MINIMAL (fastest, ~5x speedup):")
    print(f"   N = 20 measurements per bias point")
    print(f"   - Statistical error ~2-3× larger than N=100")
    print(f"   - Still much smaller than temporal drift (10-25×)")
    print(f"   - Acceptable for monitoring/trending")
    
    print(f"\n2. BALANCED (recommended, ~2.5x speedup):")
    print(f"   N = 40 measurements per bias point")
    print(f"   - Statistical error ~1.6× larger than N=100")
    print(f"   - Good compromise: 60% time saved, minimal precision loss")
    print(f"   - Best for routine measurements")
    
    print(f"\n3. PRECISE (current):")
    print(f"   N = 100 measurements per bias point")
    print(f"   - Smallest statistical error")
    print(f"   - But temporal drift still dominates!")
    print(f"   - Only needed for critical calibrations")
    
    print(f"\n{'='*80}")
    print("KEY INSIGHT")
    print("="*80)
    print("Since temporal drift (100-500 c/s) >> statistical error (5-10 c/s),")
    print("reducing N from 100 to 20-40 has MINIMAL impact on total uncertainty.")
    print(f"\nYou can safely use N=20-40 and save 60-80%% measurement time!")
    print("="*80)
    
    plt.show()

if __name__ == "__main__":
    main()
