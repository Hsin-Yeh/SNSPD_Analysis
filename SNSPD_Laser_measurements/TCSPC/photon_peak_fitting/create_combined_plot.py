#!/usr/bin/env python3
"""
Create combined plots by reading per-bias JSON summaries.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tcspc_config import OUTPUT_DIR_COMBINED, OUTPUT_DIR_POWER_SWEEP, AUTO_COLORS, AUTO_MARKERS


def discover_json_files(base_dir: Path):
    """Find all analysis_summary.json files under power sweep outputs."""
    return sorted(base_dir.glob("*/analysis_summary.json"))


def assign_colors_markers(biases):
    """Auto-assign colors and markers to bias voltages."""
    colors = {}
    markers = {}
    for i, bias in enumerate(biases):
        colors[bias] = AUTO_COLORS[i % len(AUTO_COLORS)]
        markers[bias] = AUTO_MARKERS[i % len(AUTO_MARKERS)]
    return colors, markers


def load_json_data(json_files):
    """Load analysis summaries keyed by bias voltage."""
    all_data = {}
    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            bias = data.get('bias_voltage') or json_path.parent.name
            all_data[bias] = data
        except Exception as e:
            print(f"Warning: Could not read {json_path}: {e}")
    return all_data


def main():
    output_dir = OUTPUT_DIR_COMBINED
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = discover_json_files(OUTPUT_DIR_POWER_SWEEP)
    if not json_files:
        print(f"Error: No analysis_summary.json files found in {OUTPUT_DIR_POWER_SWEEP}")
        return

    all_data = load_json_data(json_files)
    if not all_data:
        print("Error: No valid JSON summaries could be loaded")
        return

    sorted_biases = sorted(all_data.keys(), key=lambda x: int(x.replace('mV', '')))
    colors, markers = assign_colors_markers(sorted_biases)

    print(f"Loaded JSON summaries for biases: {sorted_biases}")

    sample_bias = sorted_biases[0]
    sample = all_data[sample_bias]
    time_window = sample.get('time_window_ns', {})
    t_min = time_window.get('t_min', 0)
    t_max = time_window.get('t_max', 0)

    # Combined Plot 1: Count vs Power (log-log) with fit
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    for bias in sorted_biases:
        data = all_data[bias]
        plot1 = data.get('plot1_data', {})
        fit = data.get('fit', {})

        powers = np.array(plot1.get('powers_uw', []))
        counts = np.array(plot1.get('counts_corrected', []))
        errors = np.array(plot1.get('errors', []))

        if len(powers) == 0:
            continue

        color = colors[bias]
        marker = markers[bias]

        ax1.errorbar(
            powers,
            counts,
            yerr=errors if len(errors) == len(counts) else None,
            fmt=marker,
            markersize=7.0,
            alpha=0.75,
            color=color,
            markeredgecolor='black',
            markeredgewidth=1.1,
            ecolor=color,
            elinewidth=1.0,
            capsize=2,
            label=f'{bias} data',
            zorder=5,
        )

        if 'fit_powers' in fit and 'slope' in fit and 'intercept' in fit:
            fit_powers = np.array(fit.get('fit_powers', []))
            if len(fit_powers) > 0:
                slope = fit.get('slope')
                intercept = fit.get('intercept')
                std_err = fit.get('std_err', 0.0)
                chi2_ndf = fit.get('chi2_ndf', 0.0)
                fit_range = fit.get('fit_range_used', 'fit')
                fit_line = 10 ** (slope * np.log10(fit_powers) + intercept)
                ax1.plot(
                    fit_powers,
                    fit_line,
                    color=color,
                    linewidth=2.8,
                    linestyle='-',
                    alpha=0.85,
                    label=f'{bias}: n={slope:.3f}±{std_err:.3f}, χ²={chi2_ndf:.2f}',
                    zorder=3,
                )

    ax1.set_xlabel('Laser Power (µW)', fontsize=14, weight='bold')
    ax1.set_ylabel('Count Rate (cts/s)', fontsize=14, weight='bold')
    ax1.set_title(
        f'Count vs Power: Combined Plot 1 (Dark-corrected)\n(TOA window: {t_min:.1f}-{t_max:.1f} ns)',
        fontsize=15,
        weight='bold',
    )
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95, ncol=2)
    ax1.tick_params(labelsize=11)
    plt.tight_layout()

    output_path_plot1 = output_dir / 'combined_plot1_loglog.png'
    fig1.savefig(output_path_plot1, dpi=300, bbox_inches='tight')
    print(f"✓ Combined Plot 1 saved: {output_path_plot1}")
    plt.close(fig1)

    # Combined Plot 1 (Linear)
    fig1_lin, ax1_lin = plt.subplots(figsize=(12, 8))
    for bias in sorted_biases:
        data = all_data[bias]
        plot1 = data.get('plot1_data', {})

        powers = np.array(plot1.get('powers_uw', []))
        counts = np.array(plot1.get('counts_corrected', []))
        errors = np.array(plot1.get('errors', []))

        if len(powers) == 0:
            continue

        color = colors[bias]
        marker = markers[bias]

        ax1_lin.errorbar(
            powers,
            counts,
            yerr=errors if len(errors) == len(counts) else None,
            fmt=marker,
            markersize=7.0,
            alpha=0.75,
            color=color,
            markeredgecolor='black',
            markeredgewidth=1.1,
            ecolor=color,
            elinewidth=1.0,
            capsize=2,
            label=f'{bias}',
            zorder=5,
        )

    ax1_lin.set_xlabel('Laser Power (µW)', fontsize=14, weight='bold')
    ax1_lin.set_ylabel('Count Rate (cts/s)', fontsize=14, weight='bold')
    ax1_lin.set_title(
        f'Count vs Power: Combined Plot 1 (Linear)\n(TOA window: {t_min:.1f}-{t_max:.1f} ns)',
        fontsize=15,
        weight='bold',
    )
    ax1_lin.grid(True, alpha=0.3)
    ax1_lin.legend(loc='upper left', fontsize=10, framealpha=0.95, ncol=2)
    ax1_lin.tick_params(labelsize=11)
    plt.tight_layout()

    output_path_plot1_lin = output_dir / 'combined_plot1_linear.png'
    fig1_lin.savefig(output_path_plot1_lin, dpi=300, bbox_inches='tight')
    print(f"✓ Combined Plot 1 (Linear) saved: {output_path_plot1_lin}")
    plt.close(fig1_lin)

    # Combined Plot 2: Dark count comparison (OOT_mid only - used for dark correction)
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    for bias in sorted_biases:
        data = all_data[bias]
        plot2 = data.get('plot2_data', {})

        powers = np.array(plot2.get('powers_uw', []))
        pct_mid = np.array(plot2.get('pct_oot_mid', []))

        if len(powers) == 0:
            print(f"Warning: No plot2_data for {bias}")
            print(f"  Available keys: {list(plot2.keys())}")
            continue
        
        if len(pct_mid) == 0:
            print(f"Warning: No pct_oot_mid for {bias}")
            print(f"  Available keys: {list(plot2.keys())}")
            continue

        color = colors[bias]
        marker = markers[bias]

        ax2.scatter(
            powers,
            pct_mid,
            s=70,
            alpha=0.7,
            color=color,
            edgecolors='black',
            linewidth=1.0,
            marker=marker,
            label=f'{bias} OOT_mid',
            zorder=5,
        )

    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2.0, alpha=0.8)
    ax2.set_xlabel('Laser Power (µW)', fontsize=14, weight='bold')
    ax2.set_ylabel('(OOT - 0µW)/0µW', fontsize=14, weight='bold')
    ax2.set_title('Dark Count Comparison: Combined Plot 2', fontsize=15, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.legend(loc='best', fontsize=9, framealpha=0.95, ncol=2)
    ax2.tick_params(labelsize=11)
    
    # Add bias voltage text in top right corner, outside frame
    bias_text = "\n".join(sorted_biases)
    ax2.text(1.02, 0.98, bias_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()

    output_path_plot2 = output_dir / 'combined_plot2_dark_comparison.png'
    fig2.savefig(output_path_plot2, dpi=300, bbox_inches='tight')
    print(f"✓ Combined Plot 2 saved: {output_path_plot2}")
    plt.close(fig2)

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Power Law Fits from JSON")
    print(f"{'='*80}")
    print(f"{'Bias':<10} {'Dark (cts/s)':<15} {'Exponent (n)':<25} {'Chi^2/ndf':<15}")
    print(f"{'-'*80}")
    for bias in sorted_biases:
        data = all_data[bias]
        results = data.get('results', {})
        print(
            f"{bias:<10} "
            f"{results.get('dark_count_oot', float('nan')):<15.2f} "
            f"{results.get('slope', float('nan')):.4f} ± {results.get('std_err', float('nan')):.4f}       "
            f"{results.get('chi2_ndf', float('nan')):<15.4f}"
        )
    print(f"{'='*80}\n")

    print(f"All outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
