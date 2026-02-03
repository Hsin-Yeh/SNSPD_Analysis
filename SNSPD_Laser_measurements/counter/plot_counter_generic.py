#!/usr/bin/env python3
"""
Generic counter data plotter - works with different measurement folders
Usage: python plot_counter_generic.py <data_folder> [--bias 68,70,72] [--powers all/369,446,534]
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plot_style import setup_hep_style

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import argparse

from counter_utils import (
    parse_power_timestamp,
    read_counter_file_median,
    find_latest_files,
    find_closest_dark_file,
    get_available_bias_voltages_from_file,
)


def select_bias_voltages(bias_spec, available_biases):
    """
    Select bias voltages based on specification.
    
    Args:
        bias_spec: Can be:
            - "all" or -1: Use all available bias voltages
            - Percentage string like "20%" or "50%": Select percentage of bias voltages evenly spaced
            - Comma-separated values like "66,68,70,72,74": Specific bias voltages
        available_biases: List of available bias voltages
    
    Returns:
        List of selected bias voltages
    """
    if isinstance(bias_spec, str):
        if bias_spec.lower() == "all" or bias_spec == "-1":
            return available_biases
        elif bias_spec.endswith('%'):
            # Percentage-based selection
            try:
                percentage = float(bias_spec.rstrip('%'))
                n_total = len(available_biases)
                n_select = max(1, int(n_total * percentage / 100))
                # Select evenly spaced indices
                if n_select >= n_total:
                    return available_biases
                indices = np.linspace(0, n_total - 1, n_select, dtype=int)
                return [available_biases[i] for i in indices]
            except ValueError:
                print(f"Warning: Invalid percentage '{bias_spec}', using default")
                return [66, 68, 70, 72, 74]
        else:
            # Comma-separated values
            return [float(b.strip()) for b in bias_spec.split(',')]
    elif bias_spec == -1:
        return available_biases
    else:
        return bias_spec


def generate_bias_analysis_plots(power_files, target_biases_mv, output_dir, measurement_name, dark_files=None):
    """
    Generate histograms and event vs time plots for each bias voltage.
    Includes dark count data (0nW) in the plots.
    
    Args:
        power_files: Dict of {power_nw: filepath}
        target_biases_mv: List of target bias voltages (mV)
        output_dir: Path object for output directory
        measurement_name: Name of the measurement
        dark_files: List of (filepath, timestamp) tuples for dark count files (optional)
    
    Returns:
        Dict mapping {bias_mv: {'lower_stats': {power: stats}, 'higher_stats': {power: stats}}}
    """
    bias_gaussian_stats = {}
    
    # Create subdirectories for plots
    histogram_dir = output_dir / 'histograms'
    histogram_dir.mkdir(parents=True, exist_ok=True)
    
    event_number_dir = output_dir / 'event_number_plots'
    event_number_dir.mkdir(parents=True, exist_ok=True)
    
    for bias_mv in target_biases_mv:
        print(f"  Generating plots for bias voltage {bias_mv} mV...")
        
        # Collect all measurements for this bias voltage across all powers (including dark counts)
        measurements_by_power = {}
        
        # Process signal files
        for power, filepath in sorted(power_files.items()):
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                # Skip header
                data_lines = lines[1:]
                
                for line in data_lines:
                    parts = line.strip().split()
                    if len(parts) < 8:
                        continue
                    
                    target_voltage = float(parts[0])
                    # Check if this is the bias voltage we're looking for
                    if abs(target_voltage * 1000 - bias_mv) < 1:  # Within 1 mV tolerance
                        # measurements start from index 7 (8th column)
                        measurements = [float(x) for x in parts[7:]]
                        if measurements:
                            measurements_by_power[power] = measurements
                        break
            except Exception as e:
                print(f"    Warning: Could not read {filepath}: {e}")
                continue
        
        # Process dark count files (0nW)
        if dark_files:
            for dark_filepath, _ in dark_files:
                try:
                    with open(dark_filepath, 'r') as f:
                        lines = f.readlines()
                    
                    # Skip header
                    data_lines = lines[1:]
                    
                    for line in data_lines:
                        parts = line.strip().split()
                        if len(parts) < 8:
                            continue
                        
                        target_voltage = float(parts[0])
                        # Check if this is the bias voltage we're looking for
                        if abs(target_voltage * 1000 - bias_mv) < 1:  # Within 1 mV tolerance
                            # measurements start from index 7 (8th column)
                            measurements = [float(x) for x in parts[7:]]
                            if measurements:
                                measurements_by_power[0] = measurements  # 0 nW for dark counts
                            break
                except Exception as e:
                    print(f"    Warning: Could not read dark file {dark_filepath}: {e}")
                    continue
        
        if not measurements_by_power:
            print(f"  No data found for bias voltage {bias_mv} mV, skipping")
            continue
        
        # Combine all measurements for reference
        all_measurements = np.concatenate([np.array(measurements_by_power[power]) 
                                          for power in sorted(measurements_by_power.keys())])
        
        # Initialize statistics dictionaries
        lower_stats = {}
        higher_stats = {}
        power_fits = {}  # Store fit parameters for each power
        
        # Try to import k-means
        try:
            from sklearn.cluster import KMeans
            use_kmeans = True
        except ImportError:
            use_kmeans = False
        
        # Fit separate Gaussians for each power level
        for power in sorted(measurements_by_power.keys()):
            measurements = np.array(measurements_by_power[power])
            
            # Skip first 10 measurements to allow for settling
            if len(measurements) > 10:
                measurements = measurements[10:]
            
            try:
                if use_kmeans:
                    # Use k-means to find 2 clusters and their centers
                    measurements_2d = measurements.reshape(-1, 1)
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    kmeans.fit(measurements_2d)
                    
                    # Get cluster centers sorted
                    cluster_centers = sorted(kmeans.cluster_centers_.flatten())
                    mean_lower_fit = cluster_centers[0]
                    mean_higher_fit = cluster_centers[1]
                    
                    # Count points in each cluster
                    lower_mask = kmeans.labels_ == np.argmin(kmeans.cluster_centers_)
                    higher_mask = kmeans.labels_ == np.argmax(kmeans.cluster_centers_)
                    n_lower = np.sum(lower_mask)
                    n_higher = np.sum(higher_mask)
                    n_total = len(measurements)
                    
                    # Check if this is a single distribution:
                    # 1. One cluster center is negative (invalid for count rates) - priority check
                    # 2. One cluster is too small (< 10% of total points)
                    # 3. Centers are too close (< 20% relative distance to the positive mean)
                    
                    # Calculate relative distance using the valid (positive) mean
                    if mean_lower_fit >= 0:
                        relative_distance = abs(mean_higher_fit - mean_lower_fit) / max(abs(mean_lower_fit), 1.0)
                    else:
                        # If lower mean is negative, use higher mean for relative distance
                        relative_distance = abs(mean_higher_fit - mean_lower_fit) / max(abs(mean_higher_fit), 1.0)
                    
                    cluster_size_ratio = min(n_lower, n_higher) / n_total
                    
                    # Priority: negative cluster center indicates outliers
                    is_single_distribution = (mean_lower_fit < 0 or 
                                             cluster_size_ratio < 0.1 or 
                                             relative_distance < 0.2)
                    
                    # Debug output for problematic cases
                    if mean_lower_fit < 0 or cluster_size_ratio < 0.1:
                        print(f"    {power} nW: Single distribution detected (lower_mean={mean_lower_fit:.1f}, higher_mean={mean_higher_fit:.1f}, " + 
                              f"rel_dist={relative_distance:.3f}, cluster_ratio={cluster_size_ratio:.3f}, n_lower={n_lower}, n_higher={n_higher})")
                    
                    if is_single_distribution:
                        # Single distribution - use same mean for both lower and higher
                        single_mean = np.mean(measurements)
                        mean_lower_fit = single_mean
                        mean_higher_fit = single_mean
                        lower_measurements_final = measurements
                        higher_measurements_final = measurements
                    else:
                        # Two distributions - separate based on cluster assignment
                        lower_mask = kmeans.labels_ == np.argmin(kmeans.cluster_centers_)
                        higher_mask = kmeans.labels_ == np.argmax(kmeans.cluster_centers_)
                        
                        lower_measurements_final = measurements[lower_mask]
                        higher_measurements_final = measurements[higher_mask]
                    
                    # Calculate cutoff as midpoint
                    cutoff = (mean_lower_fit + mean_higher_fit) / 2
                    
                    # If cutoff is negative, treat as single distribution
                    if cutoff < 0 and not is_single_distribution:
                        print(f"    {power} nW: Negative cutoff detected ({cutoff:.1f}), treating as single distribution")
                        is_single_distribution = True
                        single_mean = np.mean(measurements)
                        mean_lower_fit = single_mean
                        mean_higher_fit = single_mean
                        cutoff = single_mean
                        lower_measurements_final = measurements
                        higher_measurements_final = measurements
                    
                    gaussian_fit_available = False  # No Gaussian fit, just k-means centers
                    
                    # Store fit parameters for plotting
                    power_fits[power] = {
                        'cutoff': cutoff,
                        'mean_lower': mean_lower_fit,
                        'mean_higher': mean_higher_fit,
                        'gaussian_fit_available': gaussian_fit_available,
                        'is_single_distribution': is_single_distribution
                    }
                    
                else:
                    # Fallback: use median split with Gaussian fits
                    initial_cutoff = np.median(measurements)
                    lower_measurements = measurements[measurements < initial_cutoff]
                    higher_measurements = measurements[measurements >= initial_cutoff]
                    
                    def single_gaussian(x, amp, mean, sigma):
                        return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
                    
                    # Fit lower distribution
                    if len(lower_measurements) > 2:
                        hist_lower, bins_lower = np.histogram(lower_measurements, bins=30, density=True)
                        bin_centers_lower = (bins_lower[:-1] + bins_lower[1:]) / 2
                        
                        mean_lower = np.mean(lower_measurements)
                        sigma_lower = np.std(lower_measurements)
                        amp_lower = 1.0 / (sigma_lower * np.sqrt(2 * np.pi))
                        
                        try:
                            popt_lower, _ = curve_fit(single_gaussian, bin_centers_lower, hist_lower,
                                                     p0=[amp_lower, mean_lower, sigma_lower],
                                                     maxfev=5000)
                            amp_lower, mean_lower_fit, sigma_lower_fit = popt_lower
                        except:
                            mean_lower_fit = mean_lower
                            sigma_lower_fit = sigma_lower
                            amp_lower = 1.0 / (sigma_lower * np.sqrt(2 * np.pi))
                    else:
                        mean_lower_fit = np.mean(lower_measurements) if len(lower_measurements) > 0 else 0
                        sigma_lower_fit = np.std(lower_measurements) if len(lower_measurements) > 0 else 1
                        amp_lower = 1.0 / (sigma_lower_fit * np.sqrt(2 * np.pi))
                    
                    # Fit higher distribution
                    if len(higher_measurements) > 2:
                        hist_higher, bins_higher = np.histogram(higher_measurements, bins=30, density=True)
                        bin_centers_higher = (bins_higher[:-1] + bins_higher[1:]) / 2
                        
                        mean_higher = np.mean(higher_measurements)
                        sigma_higher = np.std(higher_measurements)
                        amp_higher = 1.0 / (sigma_higher * np.sqrt(2 * np.pi))
                        
                        try:
                            popt_higher, _ = curve_fit(single_gaussian, bin_centers_higher, hist_higher,
                                                      p0=[amp_higher, mean_higher, sigma_higher],
                                                      maxfev=5000)
                            amp_higher, mean_higher_fit, sigma_higher_fit = popt_higher
                        except:
                            mean_higher_fit = mean_higher
                            sigma_higher_fit = sigma_higher
                            amp_higher = 1.0 / (sigma_higher * np.sqrt(2 * np.pi))
                    else:
                        mean_higher_fit = np.mean(higher_measurements) if len(higher_measurements) > 0 else 0
                        sigma_higher_fit = np.std(higher_measurements) if len(higher_measurements) > 0 else 1
                        amp_higher = 1.0 / (sigma_higher_fit * np.sqrt(2 * np.pi))
                    
                    # Calculate cutoff for this power level
                    cutoff = (mean_lower_fit + mean_higher_fit) / 2
                    
                    # If cutoff is negative, treat as single distribution
                    if cutoff < 0:
                        print(f"    {power} nW: Negative cutoff detected ({cutoff:.1f}), treating as single distribution")
                        single_mean = np.mean(measurements)
                        mean_lower_fit = single_mean
                        mean_higher_fit = single_mean
                        cutoff = single_mean
                        lower_measurements_final = measurements
                        higher_measurements_final = measurements
                    else:
                        # Separate measurements using the power-specific cutoff
                        lower_measurements_final = measurements[measurements < cutoff]
                        higher_measurements_final = measurements[measurements >= cutoff]
                    gaussian_fit_available = True
                
                if len(lower_measurements_final) > 0:
                    lower_stats[power] = {
                        'mean': np.mean(lower_measurements_final),
                        'std': np.std(lower_measurements_final),
                        'count': len(lower_measurements_final)
                    }
                
                if len(higher_measurements_final) > 0:
                    higher_stats[power] = {
                        'mean': np.mean(higher_measurements_final),
                        'std': np.std(higher_measurements_final),
                        'count': len(higher_measurements_final)
                    }
                
                # Check if this is effectively a single distribution
                relative_distance = abs(mean_higher_fit - mean_lower_fit) / max(abs(mean_lower_fit), 1.0)
                is_single_distribution = relative_distance < 0.2
                
                # Store fit parameters for plotting
                power_fits[power] = {
                    'cutoff': cutoff,
                    'mean_lower': mean_lower_fit,
                    'mean_higher': mean_higher_fit,
                    'gaussian_fit_available': gaussian_fit_available,
                    'is_single_distribution': is_single_distribution
                }
                
                if gaussian_fit_available and not use_kmeans:
                    power_fits[power].update({
                        'sigma_lower': sigma_lower_fit,
                        'amp_lower': amp_lower,
                        'sigma_higher': sigma_higher_fit,
                        'amp_higher': amp_higher
                    })
                
            except Exception as e:
                print(f"    Warning: Clustering/fit failed for {power} nW: {e}")
                # Fallback to simple median
                cutoff = np.median(measurements)
                
                # If cutoff is negative, treat as single distribution
                if cutoff < 0:
                    print(f"    {power} nW: Negative cutoff detected ({cutoff:.1f}), treating as single distribution")
                    single_mean = np.mean(measurements)
                    cutoff = single_mean
                    lower_measurements_final = measurements
                    higher_measurements_final = measurements
                else:
                    lower_measurements_final = measurements[measurements < cutoff]
                    higher_measurements_final = measurements[measurements >= cutoff]
                
                if len(lower_measurements_final) > 0:
                    lower_stats[power] = {
                        'mean': np.mean(lower_measurements_final),
                        'std': np.std(lower_measurements_final),
                        'count': len(lower_measurements_final)
                    }
                
                if len(higher_measurements_final) > 0:
                    higher_stats[power] = {
                        'mean': np.mean(higher_measurements_final),
                        'std': np.std(higher_measurements_final),
                        'count': len(higher_measurements_final)
                    }
                
                # Check if effectively single distribution
                mean_lower = np.mean(lower_measurements_final) if len(lower_measurements_final) > 0 else 0
                mean_higher = np.mean(higher_measurements_final) if len(higher_measurements_final) > 0 else 0
                relative_distance = abs(mean_higher - mean_lower) / max(abs(mean_lower), 1.0)
                is_single_distribution = relative_distance < 0.2
                
                power_fits[power] = {
                    'cutoff': cutoff,
                    'gaussian_fit_available': False,
                    'is_single_distribution': is_single_distribution
                }
        
        # Create individual histogram plots for each power
        for power in sorted(measurements_by_power.keys()):
            fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
            
            measurements = np.array(measurements_by_power[power])
            ax_hist.hist(measurements, bins=30, alpha=0.6, color='blue', edgecolor='black')
            
            # Determine if single distribution
            is_single_dist = False
            if power in power_fits:
                is_single_dist = power_fits[power].get('is_single_distribution', False)
            
            # Get fit parameters for this power
            if power in power_fits and power_fits[power]['gaussian_fit_available']:
                def single_gaussian(x, amp, mean, sigma):
                    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
                
                fit_params = power_fits[power]
                cutoff = fit_params['cutoff']
                
                x_plot = np.linspace(measurements.min(), measurements.max(), 300)
                g_lower = single_gaussian(x_plot, fit_params['amp_lower'], fit_params['mean_lower'], fit_params['sigma_lower'])
                g_higher = single_gaussian(x_plot, fit_params['amp_higher'], fit_params['mean_higher'], fit_params['sigma_higher'])
                
                # Convert to match histogram density
                bin_width = 30  # approximate from histogram bins
                g_lower = g_lower * bin_width
                g_higher = g_higher * bin_width
                
                ax_hist.plot(x_plot, g_lower, 'g-', linewidth=2, label=f'Lower Gaussian (μ={fit_params["mean_lower"]:.1f})')
                ax_hist.plot(x_plot, g_higher, 'm-', linewidth=2, label=f'Higher Gaussian (μ={fit_params["mean_higher"]:.1f})')
                ax_hist.axvline(x=cutoff, color='orange', linestyle='--', linewidth=2, label=f'Cutoff ({cutoff:.1f} c/s)')
            else:
                cutoff = power_fits[power]['cutoff'] if power in power_fits else 500
                ax_hist.axvline(x=cutoff, color='red', linestyle='--', linewidth=2, label=f'Threshold ({cutoff} c/s)')
            
            ax_hist.set_xlabel('Count Rate (counts/s)')
            ax_hist.set_ylabel('Frequency')
            
            # Set title with single/dual distribution indicator
            if is_single_dist:
                ax_hist.set_title(f'{measurement_name} - {power} nW at {bias_mv} mV Bias (Single Distribution)', fontweight='bold', color='darkred')
            else:
                ax_hist.set_title(f'{measurement_name} - {power} nW at {bias_mv} mV Bias')
            
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)
            
            hist_file = histogram_dir / f'{measurement_name}_histogram_bias_{bias_mv}mV_power_{power}nW.png'
            fig_hist.tight_layout()
            fig_hist.savefig(hist_file, dpi=300, bbox_inches='tight')
            print(f"    Histogram saved to: {hist_file}")
            plt.close(fig_hist)
        
        # Print statistics for this bias voltage
        print(f"    Lower distribution statistics:")
        for power in sorted(lower_stats.keys()):
            stats = lower_stats[power]
            print(f"      {power} nW: mean={stats['mean']:.1f}, std={stats['std']:.1f}, n={stats['count']}")
        print(f"    Higher distribution statistics:")
        for power in sorted(higher_stats.keys()):
            stats = higher_stats[power]
            print(f"      {power} nW: mean={stats['mean']:.1f}, std={stats['std']:.1f}, n={stats['count']}")
        
        # Store statistics for this bias voltage
        bias_gaussian_stats[bias_mv] = {
            'lower_stats': lower_stats,
            'higher_stats': higher_stats,
            'power_fits': power_fits
        }
        
        # Create individual event vs number plots for each power
        for power in sorted(measurements_by_power.keys()):
            measurements = measurements_by_power[power]
            if len(measurements) == 0:
                continue
            
            # Skip first 10 measurements to match histogram/mean calculation
            if len(measurements) > 10:
                measurements_plot = measurements[10:]
            else:
                measurements_plot = measurements
            
            event_indices = np.arange(len(measurements_plot))
            
            fig_event, ax_event = plt.subplots(figsize=(12, 6))
            ax_event.plot(event_indices, measurements_plot, marker='o', linestyle='-', 
                         color='blue', alpha=0.7, markersize=4, linewidth=1)
            
            ax_event.set_xlabel('Event Number')
            ax_event.set_ylabel('Count Rate (counts/s)')
            ax_event.set_title(f'{measurement_name} - Count Rate vs Event Number at {bias_mv} mV, {power} nW')
            ax_event.grid(True, alpha=0.3)
            ax_event.minorticks_on()
            
            event_file = event_number_dir / f'{measurement_name}_event_number_bias_{bias_mv}mV_power_{power}nW.png'
            fig_event.tight_layout()
            fig_event.savefig(event_file, dpi=300, bbox_inches='tight')
        
        # Create combined figures with histograms and event number plots (2 rows, 6 columns each)
        powers_sorted = sorted([p for p in measurements_by_power.keys() if measurements_by_power[p]])
        n_powers = len(powers_sorted)
        
        # Create figures in groups of 6 powers
        fig_num = 1
        for start_idx in range(0, n_powers, 6):
            end_idx = min(start_idx + 6, n_powers)
            powers_group = powers_sorted[start_idx:end_idx]
            n_cols = len(powers_group)
            
            fig_combined, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 10))
            
            # Handle case of fewer powers than columns (ensure 2D array)
            if n_cols == 1:
                axes = axes.reshape(2, 1)
            elif n_cols < 6:
                # Add columns for consistency with empty subplots
                pass
            
            for idx, power in enumerate(powers_group):
                measurements = np.array(measurements_by_power[power])
                
                # Skip first 10 measurements
                if len(measurements) > 10:
                    measurements_plot = measurements[10:]
                else:
                    measurements_plot = measurements
                
                # Top row: Histograms
                ax_hist = axes[0, idx]
                ax_hist.hist(measurements_plot, bins=30, alpha=0.6, color='blue', edgecolor='black')
                ax_hist.set_xlabel('Count Rate (counts/s)', fontsize=9)
                ax_hist.set_ylabel('Frequency', fontsize=9)
                ax_hist.set_title(f'{power} nW', fontsize=10, fontweight='bold')
                ax_hist.grid(True, alpha=0.3)
                ax_hist.tick_params(labelsize=8)
                
                # Add fit information if available
                if power in power_fits and power_fits[power].get('gaussian_fit_available', False):
                    fit_params = power_fits[power]
                    cutoff = fit_params['cutoff']
                    # Add vertical line at cutoff
                    ax_hist.axvline(cutoff, color='red', linestyle='--', linewidth=2, 
                                   label=f'Cutoff: {cutoff:.1f}')
                    ax_hist.legend(fontsize=8)
                
                # Bottom row: Event number plots
                ax_event = axes[1, idx]
                event_indices = np.arange(len(measurements_plot))
                ax_event.plot(event_indices, measurements_plot, marker='o', linestyle='-', 
                            color='green', alpha=0.7, markersize=3, linewidth=1)
                ax_event.set_xlabel('Event Number', fontsize=9)
                ax_event.set_ylabel('Count Rate (counts/s)', fontsize=9)
                ax_event.set_title(f'{power} nW', fontsize=10, fontweight='bold')
                ax_event.grid(True, alpha=0.3)
                ax_event.tick_params(labelsize=8)
                ax_event.minorticks_on()
            
            fig_combined.text(0.02, 0.98, 'Histograms', fontsize=12, fontweight='bold', 
                            transform=fig_combined.transFigure, va='top')
            fig_combined.text(0.02, 0.49, 'Event Numbers', fontsize=12, fontweight='bold', 
                            transform=fig_combined.transFigure, va='top')

            
            fig_combined.suptitle(f'{measurement_name} - {bias_mv} mV Bias (Powers {start_idx+1}-{end_idx})', 
                                 fontsize=14, fontweight='bold', y=0.995)
            fig_combined.tight_layout()
            
            # Save with suffix if multiple figures
            if n_powers > 6:
                combined_file = histogram_dir / f'{measurement_name}_combined_bias_{bias_mv}mV_part{fig_num}.png'
            else:
                combined_file = histogram_dir / f'{measurement_name}_combined_bias_{bias_mv}mV.png'
            
            fig_combined.savefig(combined_file, dpi=300, bbox_inches='tight')
            print(f"    Combined histogram & event number plot saved to: {combined_file}")
            plt.close(fig_combined)
            
            fig_num += 1
        
        # Also create combined event vs time plot (index based)
        fig_time, ax_time = plt.subplots(figsize=(14, 5))
        for power in sorted(measurements_by_power.keys()):
            measurements = measurements_by_power[power]
            event_indices = np.arange(len(measurements))
            ax_time.plot(event_indices, measurements, marker='o', linestyle='-', 
                        label=f'{power} nW', alpha=0.7, markersize=3)
        
        ax_time.set_xlabel('Event Number')
        ax_time.set_ylabel('Count Rate (counts/s)')
        ax_time.set_title(f'{measurement_name} - Count Rate vs Time at {bias_mv} mV Bias')
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
        
        time_file = histogram_dir / f'{measurement_name}_time_series_bias_{bias_mv}mV.png'
        fig_time.tight_layout()
        fig_time.savefig(time_file, dpi=300, bbox_inches='tight')
        print(f"    Time series plot saved to: {time_file}")
        plt.close(fig_time)
    
    return bias_gaussian_stats


def main():
    # Setup HEP plotting style
    setup_hep_style()
    
    parser = argparse.ArgumentParser(description='Plot counter data from a measurement folder')
    parser.add_argument('data_folder', type=str, help='Path to data folder (e.g., /path/to/SMSPD_data/SMSPD_3/test/2-7/6K)')
    parser.add_argument('--bias', type=str, default='66,68,70,72,74', 
                        help='Bias voltages: "all"/-1 for all, "20%%"/"50%%" for percentage, or comma-separated values in mV (default: 66,68,70,72,74)')
    parser.add_argument('--powers', type=str, default='all', 
                        help='Power levels: "all" for all available, or comma-separated values in nW (default: all)')
    parser.add_argument('--dark-subtract-mode', type=str, default='closest', 
                        help='Dark count subtraction method: "closest" (closest in time) or "latest" (latest file) (default: closest)')
    parser.add_argument('--remove-lowest', type=int, default=0, help='Number of lowest power points to remove (default: 0)')
    parser.add_argument('--tolerance', type=float, default=0.5, help='Bias voltage tolerance in mV (default: 1.5)')
    parser.add_argument('--linear-fit', type=str, default='false', help='Enable linear fit: "true" or "false" (default: false)')
    parser.add_argument('--fit-range', type=str, default='all', help='Fit range: "all" or "min_power,max_power" in nW (default: all)')
    parser.add_argument('--fit-line-range', type=str, default='all', help='Fit line display range: "all" or "min_power,max_power" in nW (default: all)')
    parser.add_argument('--loglog', type=str, default='false', help='Use log-log scale for power plot: "true" or "false" (default: false)')
    parser.add_argument('--yaxis-scale', type=str, default='auto', help='Y-axis scale: "auto" or "min,max" (default: auto)')
    parser.add_argument('--measurement-name', type=str, default=None, help='Measurement name for output (default: derived from folder path)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_folder)
    if not data_dir.exists():
        print(f"Error: Data folder {data_dir} does not exist!")
        return
    
    # Find latest files for each power
    power_files, dark_files = find_latest_files(data_dir)
    
    if not power_files:
        print(f"No signal data files found in {data_dir}")
        return
    
    # Get available bias voltages from the first data file
    first_file = next(iter(power_files.values()))
    available_biases = get_available_bias_voltages_from_file(first_file)
    print(f"Available bias voltages: {available_biases} mV")
    
    # Parse target bias voltages based on specification
    target_biases_mv = select_bias_voltages(args.bias, available_biases)
    print(f"Selected bias voltages: {target_biases_mv} mV")
    
    # Filter power levels based on --powers argument
    if args.powers.lower() != 'all':
        try:
            selected_powers = [float(p.strip()) for p in args.powers.split(',')]
            # Filter power_files to only include selected powers
            power_files = {p: f for p, f in power_files.items() if p in selected_powers}
            print(f"Selected powers: {selected_powers} nW")
        except ValueError:
            print(f"Warning: Invalid power specification '{args.powers}', using all powers")
    
    if not power_files:
        print(f"No signal data files found in {data_dir}")
        return
    
    print(f"Processing data from: {data_dir}")
    print(f"Found {len(power_files)} power levels with data")
    for power, filepath in sorted(power_files.items()):
        print(f"  {power} nW: {filepath.name}")
    
    if dark_files:
        print(f"Found {len(dark_files)} dark count files")
    
    # Get measurement name from argument or derive from path
    if args.measurement_name:
        measurement_name = args.measurement_name
    else:
        # Extract measurement name from path hierarchy
        # Structure: /SMSPD_3/{measurement}/2-7/6K/ or /SMSPD_3/{measurement}/6K/
        if data_dir.name == '6K':
            if data_dir.parent.name == '2-7':
                # Path: .../measurement/2-7/6K
                measurement_name = data_dir.parent.parent.name
            else:
                # Path: .../measurement/6K
                measurement_name = data_dir.parent.name
        else:
            measurement_name = data_dir.name
    
    # Create output directory based on measurement name
    output_root = Path('~/SNSPD_analyzed_output/counter').expanduser()
    output_dir = output_root / measurement_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create three separate figures
    fig_bias_raw, ax_bias_raw = plt.subplots(figsize=(10, 7))
    fig_bias_dark, ax_bias_dark = plt.subplots(figsize=(10, 7))
    fig_power, ax_power = plt.subplots(figsize=(10, 7))
    
    # Colors for different powers - use distinguishable colors
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(power_files))))
    
    # Data structure for second plot
    row_data = {}
    
    # Use the latest (last) dark count file for plotting
    latest_dark_file = None
    if dark_files:
        latest_dark_file = max(dark_files, key=lambda x: x[1])[0]
    
    for idx, (power, filepath) in enumerate(sorted(power_files.items())):
        print(f"\nProcessing {power} nW from {filepath.name}")
        
        # Select dark count file based on mode
        _, signal_timestamp = parse_power_timestamp(filepath.name)
        dark_file_for_subtraction = None
        
        if args.dark_subtract_mode.lower() == 'latest':
            # Use latest dark count file for all signals
            dark_file_for_subtraction = latest_dark_file
            if dark_file_for_subtraction:
                print(f"  Using latest dark count file: {dark_file_for_subtraction.name}")
        else:
            # Use closest dark count file (default)
            if dark_files and signal_timestamp:
                dark_file_for_subtraction = find_closest_dark_file(signal_timestamp, dark_files)
                if dark_file_for_subtraction:
                    print(f"  Using closest dark count file: {dark_file_for_subtraction.name}")
        
        # Read signal data
        bias_voltages, count_rates = read_counter_file_median(filepath)
        
        # Read dark counts from the selected dark file
        signal_dark_rates = {}
        if dark_file_for_subtraction:
            dark_bias, dark_rates = read_counter_file_median(dark_file_for_subtraction)
            for bias, rate in zip(dark_bias, dark_rates):
                signal_dark_rates[bias] = rate
        
        # Subtract dark counts for plot 1 only
        if signal_dark_rates:
            dark_bias_array = np.array(list(signal_dark_rates.keys()))
            dark_rates_array = np.array(list(signal_dark_rates.values()))
            dark_interp = np.interp(bias_voltages, dark_bias_array, dark_rates_array)
            count_rates_subtracted = count_rates - dark_interp
        else:
            count_rates_subtracted = count_rates
        
        # Plot raw count vs bias
        ax_bias_raw.plot(bias_voltages * 1000, count_rates, 'o',
                label=f'{power} nW', color=colors[idx], alpha=0.9)
        
        # Plot dark subtracted count vs bias
        ax_bias_dark.plot(bias_voltages * 1000, count_rates_subtracted, 'o',
                label=f'{power} nW', color=colors[idx], alpha=0.9)
        
        # Collect data for second plot - key by actual bias voltage in mV
        for bias, rate in zip(bias_voltages, count_rates):
            bias_mv = bias * 1000  # Convert to mV for consistent keying
            if bias_mv not in row_data:
                row_data[bias_mv] = {}
            row_data[bias_mv][power] = rate
    
    # Configure raw bias plot
    ax_bias_raw.set_xlabel('Bias Voltage (mV)')
    ax_bias_raw.set_ylabel('Count Rate (counts/s)')
    ax_bias_raw.set_title(f'{measurement_name} (Raw)', fontsize=18, fontweight='bold')
    ax_bias_raw.legend(loc='best', frameon=True)
    ax_bias_raw.set_ylim(bottom=0)
    ax_bias_raw.minorticks_on()
    
    # Configure dark subtracted bias plot
    ax_bias_dark.set_xlabel('Bias Voltage (mV)')
    ax_bias_dark.set_ylabel('Count Rate (counts/s)')
    ax_bias_dark.set_title(f'{measurement_name} (Dark Subtracted)', fontsize=18, fontweight='bold')
    ax_bias_dark.legend(loc='best', frameon=True)
    ax_bias_dark.set_ylim(bottom=0)
    ax_bias_dark.minorticks_on()
    
    # Plot dark counts on both bias plots
    if latest_dark_file:
        dark_bias, dark_rates = read_counter_file_median(latest_dark_file)
        ax_bias_raw.plot(dark_bias * 1000, dark_rates, 's:', color='dimgray', 
                alpha=0.7, label='Dark Count')
        ax_bias_raw.legend(loc='best', frameon=True)
        ax_bias_dark.plot(dark_bias * 1000, dark_rates, 's:', color='dimgray', 
                alpha=0.7, label='Dark Count')
        ax_bias_dark.legend(loc='best', frameon=True)
    
    # Plot count vs power for selected bias voltages
    colors_biases = plt.cm.rainbow(np.linspace(0, 1, len(target_biases_mv)))
    color_idx = 0
    
    # Sort bias voltages and filter to those with multiple power points
    all_bias_mv = sorted(row_data.keys())
    valid_bias_mv = [bias_mv for bias_mv in all_bias_mv if len(row_data[bias_mv]) > 1]
    
    if valid_bias_mv:
        for bias_mv in valid_bias_mv:
            # Check if this bias is close to any target
            matched = False
            for target in target_biases_mv:
                if abs(bias_mv - target) <= args.tolerance:
                    matched = True
                    break
            
            if not matched:
                continue
            
            power_rate_pairs = sorted(row_data[bias_mv].items())
            powers = np.array([p for p, rate in power_rate_pairs])
            rates = np.array([rate for p, rate in power_rate_pairs])
            
            # Filter out non-positive rates and very small values for log-log plots
            if args.loglog.lower() == 'true':
                # For log-log plots, remove zeros and very small values
                min_threshold = 1e-3  # Minimum threshold for log scale
                valid_mask = (rates > min_threshold) & (powers > min_threshold)
            else:
                # For linear plots, just remove non-positive rates
                valid_mask = rates > 0
            
            powers_valid = powers[valid_mask]
            rates_valid = rates[valid_mask]
            
            if len(powers_valid) < 2:
                continue
            
            plot_color = colors_biases[color_idx % len(colors_biases)]
            
            # Perform linear fit if enabled
            if args.linear_fit.lower() == 'true' and len(powers_valid) >= 2:
                # Determine fit range
                if args.fit_range.lower() == 'all':
                    fit_mask = np.ones(len(powers_valid), dtype=bool)
                else:
                    try:
                        fit_min, fit_max = map(float, args.fit_range.split(','))
                        fit_mask = (powers_valid >= fit_min) & (powers_valid <= fit_max)
                    except:
                        print(f"Warning: Invalid fit range '{args.fit_range}', using all data")
                        fit_mask = np.ones(len(powers_valid), dtype=bool)
                
                powers_fit = powers_valid[fit_mask]
                rates_fit = rates_valid[fit_mask]
                
                if len(powers_fit) >= 2:
                    try:
                        # Check if log-log scale is enabled for power-law fit
                        if args.loglog.lower() == 'true':
                            # Power-law fit: Rate = A * Power^n
                            # In log space: log(Rate) = log(A) + n*log(Power)
                            log_powers_fit = np.log(powers_fit)
                            log_rates_fit = np.log(rates_fit)
                            
                            coeffs, cov = np.polyfit(log_powers_fit, log_rates_fit, 1, cov=True)
                            n = coeffs[0]  # power exponent
                            log_A = coeffs[1]
                            A = np.exp(log_A)
                            n_err = np.sqrt(cov[0, 0])
                            log_A_err = np.sqrt(cov[1, 1])
                            A_err = A * log_A_err  # Error propagation for A
                            
                            # Calculate chi-squared in log space
                            log_fit_rates = n * log_powers_fit + log_A
                            log_residuals = log_rates_fit - log_fit_rates
                            
                            # Uncertainty in log space
                            log_uncertainties = 1.0 / np.sqrt(np.abs(rates_fit))  # Approximate
                            log_uncertainties[log_uncertainties == 0] = 1
                            
                            chi_squared = np.sum((log_residuals / log_uncertainties)**2)
                            ndf = len(powers_fit) - 2
                            chi2_ndf = chi_squared / ndf if ndf > 0 else 0
                            
                            # Plot data points with power-law fit label
                            # label_combined = f'{bias_mv:.1f}mV: n={n:.2f}±{n_err:.2f}, χ²/ndf={chi2_ndf:.2f}'
                            label_combined = f'{bias_mv:.1f}mV: n={n:.2f}±{n_err:.2f}'
                            ax_power.plot(powers_valid, rates_valid, 'o', 
                                    label=label_combined, color=plot_color, alpha=0.9)
                            
                            print(f"  Bias {bias_mv:.1f} mV power-law fit:")
                            print(f"    Rate = ({A:.2e} ± {A_err:.2e}) * Power^({n:.3f} ± {n_err:.3f})")
                            print(f"    χ²/ndf = {chi_squared:.2f}/{ndf} = {chi2_ndf:.3f}")
                            
                        else:
                            # Linear fit: Rate = slope * Power + intercept using polyfit with covariance
                            coeffs, cov = np.polyfit(powers_fit, rates_fit, 1, cov=True)
                            slope = coeffs[0]
                            intercept = coeffs[1]
                            slope_err = np.sqrt(cov[0, 0])
                            intercept_err = np.sqrt(cov[1, 1])
                            
                            # Calculate chi-squared and ndf
                            fit_rates = slope * powers_fit + intercept
                            residuals = rates_fit - fit_rates
                            
                            # Use Poisson uncertainty: sigma = sqrt(counts)
                            # For count rates, we approximate uncertainty as sqrt(rate)
                            uncertainties = np.sqrt(np.abs(rates_fit))
                            uncertainties[uncertainties == 0] = 1  # Avoid division by zero
                            
                            chi_squared = np.sum((residuals / uncertainties)**2)
                            ndf = len(powers_fit) - 2  # number of data points - number of parameters
                            chi2_ndf = chi_squared / ndf if ndf > 0 else 0
                            
                            # Plot data points with combined label (compact format)
                            label_combined = f'{bias_mv:.1f}mV: {slope:.2f}±{slope_err:.2f}, χ²/ndf={chi2_ndf:.2f}'
                            ax_power.plot(powers_valid, rates_valid, 'o', 
                                    label=label_combined, color=plot_color, alpha=0.9)
                            
                            print(f"  Bias {bias_mv:.1f} mV linear fit:")
                            print(f"    Rate = ({slope:.3f} ± {slope_err:.3f}) * Power + ({intercept:.1f} ± {intercept_err:.1f})")
                            print(f"    χ²/ndf = {chi_squared:.2f}/{ndf} = {chi2_ndf:.3f}")
                        
                        # Plot fit line with specified range
                        if args.fit_line_range.lower() == 'all':
                            power_fit_min = powers_valid.min()
                            power_fit_max = powers_valid.max()
                        else:
                            try:
                                line_min, line_max = map(float, args.fit_line_range.split(','))
                                power_fit_min = max(line_min, powers_valid.min())
                                power_fit_max = min(line_max, powers_valid.max())
                            except:
                                print(f"Warning: Invalid fit line range '{args.fit_line_range}', using all data")
                                power_fit_min = powers_valid.min()
                                power_fit_max = powers_valid.max()
                        
                        power_fit_line = np.linspace(power_fit_min, power_fit_max, 100)
                        
                        # Calculate fit line based on fit type
                        if args.loglog.lower() == 'true':
                            # Power-law: Rate = A * Power^n
                            rate_fit_line = A * power_fit_line**n
                        else:
                            # Linear: Rate = slope * Power + intercept
                            rate_fit_line = slope * power_fit_line + intercept
                        
                        ax_power.plot(power_fit_line, rate_fit_line, '--', color=plot_color, 
                                alpha=0.7)
                        
                    except Exception as e:
                        print(f"  Fit failed for {bias_mv:.1f} mV: {e}")
                        # Plot data only if fit fails
                        ax_power.plot(powers_valid, rates_valid, 'o', 
                                label=f'{bias_mv:.1f} mV', color=plot_color, alpha=0.9)
            else:
                # Plot data points without fit
                ax_power.plot(powers_valid, rates_valid, 'o', 
                        label=f'{bias_mv:.1f} mV', color=plot_color, alpha=0.9)
            
            color_idx += 1
    
    # Add dark count reference lines if available (use latest dark file)
    # Filter out very small dark count values (< 1) to avoid extending y-axis range
    if latest_dark_file:
        dark_bias, dark_rates = read_counter_file_median(latest_dark_file)
        dark_dict = {bias * 1000: rate for bias, rate in zip(dark_bias, dark_rates) if rate >= 1.0}
        
        # Show dark count line for each selected bias voltage (no legend labels)
        for target in target_biases_mv:
            for bias_mv, dark_rate in dark_dict.items():
                if abs(bias_mv - target) <= args.tolerance:
                    ax_power.axhline(y=dark_rate, color='dimgray', linestyle=':', 
                                     alpha=0.5)
                    break
    
    # Configure power plot
    ax_power.set_xlabel('Optical Power (nW)')
    ax_power.set_ylabel('Count Rate (counts/s)')
    ax_power.set_title(f'{measurement_name} (Power Dependence)', fontsize=18, fontweight='bold')
    # Dynamic legend: more columns for more entries, smaller font, tighter spacing
    num_entries = len([h for h in ax_power.get_legend_handles_labels()[0]])
    if num_entries > 10:
        legend_ncol = 4
        legend_fontsize = 11
    elif num_entries > 6:
        legend_ncol = 3
        legend_fontsize = 12
    else:
        legend_ncol = 2
        legend_fontsize = 13
    
    # Position legend based on plot type
    if args.loglog.lower() == 'true':
        legend_loc = 'lower right'
        legend_bbox = (1, 0)
    else:
        legend_loc = 'upper left'
        legend_bbox = (0, 1)
    
    ax_power.legend(fontsize=legend_fontsize, loc=legend_loc, ncol=legend_ncol, 
              frameon=True, bbox_to_anchor=legend_bbox,
              columnspacing=0.8, handlelength=1.5, handletextpad=0.5)
    
    # Enable minor ticks
    ax_power.minorticks_on()
    
    # Apply log-log scale if requested
    if args.loglog.lower() == 'true':
        ax_power.set_xscale('log')
        ax_power.set_yscale('log')
    else:
        ax_power.set_ylim(bottom=0)
    
    # Apply custom y-axis scale if specified
    if args.yaxis_scale.lower() != 'auto':
        try:
            ymin, ymax = map(float, args.yaxis_scale.split(','))
            ax_power.set_ylim(ymin, ymax)
        except:
            print(f"Warning: Invalid y-axis scale '{args.yaxis_scale}', using auto")
    
    # Save all three figures individually
    fig_bias_raw.tight_layout()
    output_file_raw = output_dir / f'{measurement_name}_count_vs_bias_raw.png'
    fig_bias_raw.savefig(output_file_raw, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file_raw}")
    plt.close(fig_bias_raw)
    
    fig_bias_dark.tight_layout()
    output_file_dark = output_dir / f'{measurement_name}_count_vs_bias_dark_subtracted.png'
    fig_bias_dark.savefig(output_file_dark, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file_dark}")
    plt.close(fig_bias_dark)
    
    fig_power.tight_layout()
    output_file_power = output_dir / f'{measurement_name}_count_vs_power.png'
    fig_power.savefig(output_file_power, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file_power}")
    plt.close(fig_power)
    
    # Generate histograms and event vs time plots for each bias voltage
    print(f"\nGenerating histograms and event vs time plots...")
    bias_gaussian_stats = generate_bias_analysis_plots(power_files, target_biases_mv, output_dir, measurement_name, dark_files)
    
    # Create power plots using the fitted Gaussian means
    print(f"\nGenerating count vs power plots using fitted means...")
    
    # Plot lower distribution means vs power
    if bias_gaussian_stats:
        fig_power_lower, ax_power_lower = plt.subplots(figsize=(10, 7))
        fig_power_higher, ax_power_higher = plt.subplots(figsize=(10, 7))
        
        colors_biases = plt.cm.rainbow(np.linspace(0, 1, len(target_biases_mv)))
        color_idx = 0
        
        for bias_mv in sorted(bias_gaussian_stats.keys()):
            stats_dict = bias_gaussian_stats[bias_mv]
            lower_stats = stats_dict['lower_stats']
            higher_stats = stats_dict['higher_stats']
            power_fits = stats_dict.get('power_fits', {})
            
            plot_color = colors_biases[color_idx % len(colors_biases)]
            
            # Plot lower distribution means
            if lower_stats:
                powers_lower = np.array(sorted(lower_stats.keys()))
                means_lower = np.array([lower_stats[p]['mean'] for p in sorted(lower_stats.keys())])
                # Filter out zero or negative values for log-log plot
                mask_lower = (powers_lower > 0) & (means_lower > 0)
                ax_power_lower.plot(powers_lower[mask_lower], means_lower[mask_lower], 'o', 
                                   label=f'{bias_mv:.1f} mV', color=plot_color, alpha=0.9, markersize=6)
                
                # Linear fit below 2500nW
                fit_below_mask = (powers_lower < 2500) & (powers_lower > 0) & (means_lower > 0)
                if np.sum(fit_below_mask) >= 2:
                    fit_powers_below = powers_lower[fit_below_mask]
                    fit_means_below = means_lower[fit_below_mask]
                    log_powers_below = np.log10(fit_powers_below)
                    log_means_below = np.log10(fit_means_below)
                    slope_below, intercept_below = np.polyfit(log_powers_below, log_means_below, 1)
                    fit_line_powers_below = np.logspace(np.log10(fit_powers_below.min()), np.log10(fit_powers_below.max()), 100)
                    fit_line_means_below = 10**(slope_below * np.log10(fit_line_powers_below) + intercept_below)
                    ax_power_lower.plot(fit_line_powers_below, fit_line_means_below, ':', 
                                      color=plot_color, alpha=0.6, linewidth=2,
                                      label=f'{bias_mv:.1f} mV <2500nW (n={slope_below:.2f})')
                
                # Linear fit from 2500nW to 10000nW
                fit_mask = (powers_lower >= 2500) & (powers_lower <= 10000) & (powers_lower > 0) & (means_lower > 0)
                if np.sum(fit_mask) >= 2:  # Need at least 2 points for fit
                    fit_powers = powers_lower[fit_mask]
                    fit_means = means_lower[fit_mask]
                    # Linear fit in log-log space: log(y) = slope * log(x) + intercept
                    log_powers = np.log10(fit_powers)
                    log_means = np.log10(fit_means)
                    slope, intercept = np.polyfit(log_powers, log_means, 1)
                    # Plot fit line
                    fit_line_powers = np.logspace(np.log10(fit_powers.min()), np.log10(fit_powers.max()), 100)
                    fit_line_means = 10**(slope * np.log10(fit_line_powers) + intercept)
                    ax_power_lower.plot(fit_line_powers, fit_line_means, '--', 
                                      color=plot_color, alpha=0.6, linewidth=2,
                                      label=f'{bias_mv:.1f} mV 2500-10000nW (n={slope:.2f})')
            
            # Plot higher distribution means
            if higher_stats:
                powers_higher = np.array(sorted(higher_stats.keys()))
                means_higher = np.array([higher_stats[p]['mean'] for p in sorted(higher_stats.keys())])
                # Filter out zero or negative values for log-log plot
                mask_higher = (powers_higher > 0) & (means_higher > 0)
                ax_power_higher.plot(powers_higher[mask_higher], means_higher[mask_higher], 's', 
                                    label=f'{bias_mv:.1f} mV', color=plot_color, alpha=0.9, markersize=6)
                
                # Linear fit below 2500nW
                fit_below_mask = (powers_higher < 2500) & (powers_higher > 0) & (means_higher > 0)
                if np.sum(fit_below_mask) >= 2:
                    fit_powers_below = powers_higher[fit_below_mask]
                    fit_means_below = means_higher[fit_below_mask]
                    log_powers_below = np.log10(fit_powers_below)
                    log_means_below = np.log10(fit_means_below)
                    slope_below, intercept_below = np.polyfit(log_powers_below, log_means_below, 1)
                    fit_line_powers_below = np.logspace(np.log10(fit_powers_below.min()), np.log10(fit_powers_below.max()), 100)
                    fit_line_means_below = 10**(slope_below * np.log10(fit_line_powers_below) + intercept_below)
                    ax_power_higher.plot(fit_line_powers_below, fit_line_means_below, ':', 
                                       color=plot_color, alpha=0.6, linewidth=2,
                                       label=f'{bias_mv:.1f} mV <2500nW (n={slope_below:.2f})')
                
                # Linear fit from 2500nW to 10000nW
                fit_mask = (powers_higher >= 2500) & (powers_higher <= 10000) & (powers_higher > 0) & (means_higher > 0)
                if np.sum(fit_mask) >= 2:  # Need at least 2 points for fit
                    fit_powers = powers_higher[fit_mask]
                    fit_means = means_higher[fit_mask]
                    # Linear fit in log-log space: log(y) = slope * log(x) + intercept
                    log_powers = np.log10(fit_powers)
                    log_means = np.log10(fit_means)
                    slope, intercept = np.polyfit(log_powers, log_means, 1)
                    # Plot fit line
                    fit_line_powers = np.logspace(np.log10(fit_powers.min()), np.log10(fit_powers.max()), 100)
                    fit_line_means = 10**(slope * np.log10(fit_line_powers) + intercept)
                    ax_power_higher.plot(fit_line_powers, fit_line_means, '--', 
                                       color=plot_color, alpha=0.6, linewidth=2,
                                       label=f'{bias_mv:.1f} mV 2500-10000nW (n={slope:.2f})')
            
            color_idx += 1
        
        # Configure lower distribution plot with log-log scale
        ax_power_lower.set_xscale('log')
        ax_power_lower.set_yscale('log')
        ax_power_lower.set_xlabel('Optical Power (nW)')
        ax_power_lower.set_ylabel('Mean Count Rate (counts/s)')
        ax_power_lower.set_title(f'{measurement_name} - Lower Distribution Mean vs Power (Log-Log)', fontsize=14, fontweight='bold')
        ax_power_lower.legend(loc='best', frameon=True)
        ax_power_lower.grid(True, alpha=0.3, which='both')
        ax_power_lower.minorticks_on()
        
        # Configure higher distribution plot with log-log scale
        ax_power_higher.set_xscale('log')
        ax_power_higher.set_yscale('log')
        ax_power_higher.set_xlabel('Optical Power (nW)')
        ax_power_higher.set_ylabel('Mean Count Rate (counts/s)')
        ax_power_higher.set_title(f'{measurement_name} - Higher Distribution Mean vs Power (Log-Log)', fontsize=14, fontweight='bold')
        ax_power_higher.legend(loc='best', frameon=True)
        ax_power_higher.grid(True, alpha=0.3, which='both')
        ax_power_higher.minorticks_on()
        
        # Save lower distribution plot
        fig_power_lower.tight_layout()
        output_file_lower = output_dir / f'{measurement_name}_fitted_mean_vs_power_lower.png'
        fig_power_lower.savefig(output_file_lower, dpi=300, bbox_inches='tight')
        print(f"Lower distribution plot saved to: {output_file_lower}")
        plt.close(fig_power_lower)
        
        # Save higher distribution plot
        fig_power_higher.tight_layout()
        output_file_higher = output_dir / f'{measurement_name}_fitted_mean_vs_power_higher.png'
        fig_power_higher.savefig(output_file_higher, dpi=300, bbox_inches='tight')
        print(f"Higher distribution plot saved to: {output_file_higher}")
        plt.close(fig_power_higher)
    
    plt.close('all')

if __name__ == "__main__":
    main()
