#!/usr/bin/env python3
"""
Refit all blocks with 4-Gaussian model (same as before), but expose
mean/sigma limits explicitly for easy tuning.
- Tail cut: 75.7 ns
- Plot range: 76.0 ns
- Mean bounds: block145 means ± MEAN_TOL_NS
- Sigma bounds: [SIGMA_MIN_PS, SIGMA_MAX_PS]
"""

import subprocess
import pickle
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Extract histograms in subprocess with timeout
code = """
import sys
sys.path.insert(0, '/Users/ya/Documents/Projects/SNSPD/SNSPD_Analysis/SNSPD_Laser_measurements/TCSPC')
import pickle
from read_phu import read_phu_file

phu_file = '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/sweep_power/SMSPD_3_2-7_500kHz_73mV_20260205_1213.phu'
header, histograms = read_phu_file(phu_file, verbose=False)

with open('/tmp/snspd_histograms.pkl', 'wb') as f:
    pickle.dump((header, histograms), f)

print('OK', flush=True)
"""

print("Loading histograms...")
try:
    result = subprocess.run(['/usr/bin/python3', '-c', code],
                            capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)
except subprocess.TimeoutExpired:
    print("ERROR: Subprocess timed out")
    sys.exit(1)

with open('/tmp/snspd_histograms.pkl', 'rb') as f:
    header, histograms = pickle.load(f)

print(f"✓ Loaded {len(histograms)} histograms\n")

# Multi-Gaussian function

def multi_gaussian(x, *params):
    result = np.zeros_like(x, dtype=float)
    n_peaks = len(params) // 3
    for i in range(n_peaks):
        mu = params[3 * i]
        sigma = params[3 * i + 1]
        A = params[3 * i + 2]
        result += A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    return result


def single_gaussian(x, mu, sigma, A):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


# Reference means from block 160 (better fit quality than block145)
REF_MEANS_NS = np.array([75.2323506141056, 75.35237035114503, 75.48166022478044, 75.64799999999998])
REF_SIGMAS_NS = np.array([16.92754629146084, 37.24041993238192, 49.4937827839825, 70.58657951667115]) / 1000.0  # Convert ps to ns

# Limit settings
MEAN_TOL_FRACTION = 0.05  # ±5% (very tight constraint)
SIGMA_TOL_FRACTION = 0.30  # ±30%

# Load powers
with open('/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/73mV/analysis_summary.json', 'r') as f:
    analysis_summary = json.load(f)
powers_uw = analysis_summary.get('plot1_data', {}).get('powers_uw', [])

resolution_s = header.get('MeasDesc_Resolution', 4e-12)
curve_indices = header.get('HistResDscr_CurveIndex', {})

# Parameters
T_MIN_NS = 75.0
T_MAX_NS = 79.0
cut_time_ns = 76.2
TAIL_CUT_NS = 75.7
PLOT_MAX_NS = 76.0

results_all = {}
block_ids = []
block_160_params = None  # Will store fitted parameters from block 160

print("=" * 80)
print("FITTING BLOCKS < 200: 4-Gaussian with block 160 as reference")
print(f"Using block 160 fitted parameters as initial guess")
print(f"Mean bounds: ±{MEAN_TOL_FRACTION*100:.1f}% of block 160 fitted means (VERY TIGHT)")
print(f"Sigma bounds: ±{SIGMA_TOL_FRACTION*100:.1f}% of block 160 fitted sigmas")
print(f"Tail cut: 75.70ns (blocks ≤160), 75.75ns (blocks >160)")
print("=" * 80)

# First pass: fit block 160 to get reference parameters
print("\nFitting block 160 for reference parameters...")
for block_idx in range(len(histograms)):
    hist = histograms[block_idx]
    block_id = curve_indices.get(block_idx, block_idx)
    
    if block_id != 160:
        continue
    
    power_idx = sum(1 for i in range(block_idx) if curve_indices.get(i, i) != 0)
    power_uw = powers_uw[power_idx] if power_idx < len(powers_uw) else 0.0
    
    # Set tail cut: 75.75ns for blocks > 160, 75.70ns otherwise
    tail_cut_ns = 75.75 if block_id > 160 else 75.70
    
    # Extract ROI
    bin_min = int(T_MIN_NS * 1e-9 / resolution_s)
    bin_max = int(T_MAX_NS * 1e-9 / resolution_s)
    if bin_max > len(hist):
        bin_max = len(hist)
    
    hist_roi = hist[bin_min:bin_max]
    cut_bin = int(cut_time_ns * 1e-9 / resolution_s)
    tail_cut_bin = int(tail_cut_ns * 1e-9 / resolution_s)
    cut_bin_roi = cut_bin - bin_min
    tail_cut_bin_roi = tail_cut_bin - bin_min
    
    hist_before = hist_roi[:cut_bin_roi]
    time_bins_full = np.arange(len(hist_roi)) * resolution_s * 1e9 + (bin_min * resolution_s * 1e9)
    time_before = time_bins_full[:cut_bin_roi]
    
    hist_trimmed = hist_before[:tail_cut_bin_roi]
    time_trimmed = time_before[:tail_cut_bin_roi]
    
    # Initial parameters
    initial_params = []
    bounds_lower = []
    bounds_upper = []
    
    A_guesses = []
    for mu_ref in REF_MEANS_NS:
        idx_closest = np.argmin(np.abs(time_trimmed - mu_ref))
        A_guesses.append(max(hist_trimmed[idx_closest], 100))
    
    for i, (mu_ref, sigma_ref) in enumerate(zip(REF_MEANS_NS, REF_SIGMAS_NS)):
        sigma_init_ns = sigma_ref
        A_init = A_guesses[i]
        
        mean_tol = mu_ref * MEAN_TOL_FRACTION
        sigma_min = sigma_ref * (1 - SIGMA_TOL_FRACTION)
        sigma_max = sigma_ref * (1 + SIGMA_TOL_FRACTION)
        
        initial_params.extend([mu_ref, sigma_init_ns, A_init])
        bounds_lower.extend([mu_ref - mean_tol, sigma_min, 0.0])
        bounds_upper.extend([mu_ref + mean_tol, sigma_max, A_init * 3.0])
    
    try:
        popt, pcov = curve_fit(
            multi_gaussian,
            time_trimmed,
            hist_trimmed,
            p0=initial_params,
            bounds=(bounds_lower, bounds_upper),
            maxfev=50000,
        )
        
        # Extract block 160 parameters
        block_160_params = {
            'means': [popt[3*i] for i in range(4)],
            'sigmas': [popt[3*i+1] for i in range(4)],
            'amplitudes': [popt[3*i+2] for i in range(4)]
        }
        
        # Calculate R²
        y_fit = multi_gaussian(time_trimmed, *popt)
        ss_res = np.sum((hist_trimmed - y_fit) ** 2)
        ss_tot = np.sum((hist_trimmed - np.mean(hist_trimmed)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"✓ Block 160 FIT COMPLETE")
        print(f"  Power: {power_uw:.4f} µW")
        print(f"  R²: {r_squared:.6f}")
        print(f"  Tail cut: {tail_cut_ns:.2f} ns")
        print(f"  Means (ns): {[f'{m:.4f}' for m in block_160_params['means']]}")
        print(f"  Sigmas (ps): {[f'{s*1000:.2f}' for s in block_160_params['sigmas']]}")
        print(f"  Amplitudes: {[f'{a:.0f}' for a in block_160_params['amplitudes']]}")
    except Exception as e:
        print(f"✗ Failed to fit block 160: {e}")

# Second pass: fit all blocks < 200 using block 160 parameters
print("\n" + "=" * 80)
print("Fitting blocks < 200 using block 160 as reference...")
print("=" * 80 + "\n")

# Fit all blocks < 200
for block_idx in range(len(histograms)):
    hist = histograms[block_idx]
    block_id = curve_indices.get(block_idx, block_idx)

    if block_id == 0:
        continue
    
    # Only fit blocks < 200
    if block_id >= 200:
        continue
    
    # Skip block 160 (already fitted)
    if block_id == 160:
        block_ids.append(block_id)
        continue

    block_ids.append(block_id)

    power_idx = sum(1 for i in range(block_idx) if curve_indices.get(i, i) != 0)
    power_uw = powers_uw[power_idx] if power_idx < len(powers_uw) else 0.0

    # Set tail cut: 75.75ns for blocks > 160, 75.70ns otherwise (same as first pass)
    tail_cut_ns = 75.75 if block_id > 160 else 75.70

    # Extract ROI
    bin_min = int(T_MIN_NS * 1e-9 / resolution_s)
    bin_max = int(T_MAX_NS * 1e-9 / resolution_s)

    if bin_max > len(hist):
        bin_max = len(hist)

    hist_roi = hist[bin_min:bin_max]
    cut_bin = int(cut_time_ns * 1e-9 / resolution_s)
    tail_cut_bin = int(tail_cut_ns * 1e-9 / resolution_s)
    cut_bin_roi = cut_bin - bin_min
    tail_cut_bin_roi = tail_cut_bin - bin_min

    # Extract main peak
    hist_before = hist_roi[:cut_bin_roi]
    time_bins_full = np.arange(len(hist_roi)) * resolution_s * 1e9 + (bin_min * resolution_s * 1e9)
    time_before = time_bins_full[:cut_bin_roi]

    # Trim at tail_cut
    hist_trimmed = hist_before[:tail_cut_bin_roi]
    time_trimmed = time_before[:tail_cut_bin_roi]

    # Initial parameters using block 160 as reference
    initial_params = []
    bounds_lower = []
    bounds_upper = []

    if block_160_params is None:
        print(f"ERROR: Block 160 parameters not available, skipping block {block_id}")
        continue

    # Use block 160 fitted parameters as initial guess with ±30% tolerance
    for i in range(4):
        mu_ref = block_160_params['means'][i]
        sigma_ref = block_160_params['sigmas'][i]
        A_ref = block_160_params['amplitudes'][i]
        
        # Use reference parameters as initial guess
        mu_init = mu_ref
        sigma_init = sigma_ref
        # Estimate amplitude from histogram near this mean
        idx_closest = np.argmin(np.abs(time_trimmed - mu_ref))
        A_init = max(hist_trimmed[idx_closest], 100)
        
        # Calculate bounds: ±1% mean, ±5% sigma
        mean_tol = mu_ref * MEAN_TOL_FRACTION
        sigma_min = sigma_ref * (1 - SIGMA_TOL_FRACTION)
        sigma_max = sigma_ref * (1 + SIGMA_TOL_FRACTION)
        
        initial_params.extend([mu_init, sigma_init, A_init])
        bounds_lower.extend([mu_ref - mean_tol, sigma_min, 0.0])
        bounds_upper.extend([mu_ref + mean_tol, sigma_max, A_init * 5.0])

    # Fit
    try:
        popt, pcov = curve_fit(
            multi_gaussian,
            time_trimmed,
            hist_trimmed,
            p0=initial_params,
            bounds=(bounds_lower, bounds_upper),
            maxfev=50000,
        )

        fit_results = {}
        for i in range(len(REF_MEANS_NS)):
            mu = popt[3 * i]
            sigma = popt[3 * i + 1]
            A = popt[3 * i + 2]
            fit_results[i] = {"mu": mu, "sigma": sigma, "A": A}

        y_pred = multi_gaussian(time_trimmed, *popt)
        ss_res = np.sum((hist_trimmed - y_pred) ** 2)
        ss_tot = np.sum((hist_trimmed - np.mean(hist_trimmed)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        results_all[block_id] = {
            "power": power_uw,
            "r_squared": r_squared,
            "params": fit_results,
            "popt": popt,
            "pcov": pcov,
            "time_trimmed": time_trimmed,
            "hist_trimmed": hist_trimmed,
        }

        sigmas = [fit_results[i]["sigma"] * 1000 for i in range(4)]
        print(
            f"Block {block_id:3d} @ {power_uw:7.4f} µW | σ: {sigmas[0]:6.2f}, {sigmas[1]:6.2f}, {sigmas[2]:6.2f}, {sigmas[3]:6.2f} ps | R²: {r_squared:.6f}"
        )

    except Exception as e:
        print(f"Block {block_id:3d} @ {power_uw:7.4f} µW | FIT FAILED: {str(e)[:40]}")
        continue

print("\n" + "=" * 80)
print(f"Successfully fitted {len(results_all)} blocks")
print("=" * 80)

# Generate plots for each block
output_dir = Path('/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/73mV/peak_fits_75p7_limits')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nGenerating plots in: {output_dir}")

for block_id in sorted(results_all.keys()):
    data = results_all[block_id]
    popt = data["popt"]
    time_trimmed = data["time_trimmed"]
    hist_trimmed = data["hist_trimmed"]
    power_uw = data["power"]
    r_squared = data["r_squared"]
    
    # Set tail cut based on block number
    tail_cut_ns = 75.8 if block_id > 160 else TAIL_CUT_NS

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    plot_end_bin = np.searchsorted(time_trimmed, PLOT_MAX_NS)
    if plot_end_bin == 0:
        plot_end_bin = len(time_trimmed)

    time_plot = time_trimmed[:plot_end_bin]
    hist_plot = hist_trimmed[:plot_end_bin]
    y_pred_plot = multi_gaussian(time_plot, *popt)

    ax1.plot(time_plot, hist_plot, "o-", markersize=3, linewidth=1.2,
             label="Data", color="steelblue", alpha=0.8)
    ax1.plot(time_plot, y_pred_plot, "-", linewidth=2.5, label="Multi-Gaussian Fit",
             color="red", alpha=0.9)

    colors_comp = plt.cm.Set2(np.linspace(0, 1, 4))
    for i in range(4):
        mu = popt[3 * i]
        sigma = popt[3 * i + 1]
        A = popt[3 * i + 2]
        component = single_gaussian(time_plot, mu, sigma, A)
        n_label = 4 - i
        ax1.plot(time_plot, component, "--", linewidth=1.8, alpha=0.5,
                 color=colors_comp[i], label=f"n={n_label}")

    ax1.axvline(tail_cut_ns, color="orange", linestyle=":", linewidth=2, alpha=0.7,
                label=f"Tail cut ({tail_cut_ns} ns)")

    # Add fit results text box
    fit_text = f"Fit Parameters:\n"
    for i in range(4):
        mu = popt[3 * i]
        sigma = popt[3 * i + 1] * 1000  # ps
        A = popt[3 * i + 2]
        n_label = 4 - i
        fit_text += f"n={n_label}: μ={mu:.4f}ns, σ={sigma:.1f}ps, A={A:.0f}\n"
    ax1.text(0.02, 0.98, fit_text, transform=ax1.transAxes, fontsize=8,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel("TOA (ns)", fontsize=12, weight="bold")
    ax1.set_ylabel("Counts", fontsize=12, weight="bold")
    ax1.set_title(f"Block {block_id} @ {power_uw:.4f} µW | R² = {r_squared:.6f}",
                  fontsize=13, weight="bold")
    ax1.legend(fontsize=9, loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([time_plot[0], PLOT_MAX_NS])

    residuals = hist_plot - y_pred_plot
    ax2.plot(time_plot, residuals, "o-", markersize=3, linewidth=1.2,
             color="darkgreen", alpha=0.7)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.5)
    ax2.axvline(tail_cut_ns, color="orange", linestyle=":", linewidth=2, alpha=0.7)
    ax2.set_xlabel("TOA (ns)", fontsize=12, weight="bold")
    ax2.set_ylabel("Residuals", fontsize=12, weight="bold")
    ax2.set_title("Fit Residuals", fontsize=11, weight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([time_plot[0], PLOT_MAX_NS])

    plt.tight_layout()
    output_file = output_dir / f"block_{block_id:03d}_fit_75p7_limits.png"
    fig.savefig(output_file, dpi=120, bbox_inches="tight")
    plt.close(fig)

print(f"✓ Generated {len(results_all)} plots")

# Save summary
summary_file = output_dir / "fit_results_summary.json"
summary_dict = {}
for block_id in sorted(results_all.keys()):
    data = results_all[block_id]
    summary_dict[str(block_id)] = {
        "power_uw": data["power"],
        "r_squared": data["r_squared"],
        "sigmas_ps": [data["params"][i]["sigma"] * 1000 for i in range(4)],
        "mus_ns": [data["params"][i]["mu"] for i in range(4)],
        "amplitudes": [data["params"][i]["A"] for i in range(4)],
    }

with open(summary_file, "w") as f:
    json.dump(summary_dict, f, indent=2)

print(f"✓ Saved summary to: {summary_file}")

# Generate HTML output
html_file = output_dir / "fit_results.html"
print(f"\nGenerating HTML summary: {html_file}")

html_content = """<!DOCTYPE html>
<html>
<head>
    <title>SNSPD TCSPC Fit Results - 4-Gaussian Model</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        h1 { color: #333; text-align: center; }
        .info { background: white; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .block { margin: 30px 0; background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .block h2 { margin-top: 0; color: #2c5aa0; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .good { color: green; font-weight: bold; }
        .warning { color: orange; font-weight: bold; }
        .bad { color: red; font-weight: bold; }
    </style>
</head>
<body>
    <h1>SNSPD TCSPC Peak Fitting Results</h1>
    <div class="info">
        <h2>Fit Parameters</h2>
        <ul>
            <li><strong>Model:</strong> 4-Gaussian (n=4, 3, 2, 1)</li>
            <li><strong>Tail cut:</strong> 75.7 ns</li>
            <li><strong>Plot range:</strong> 75.0 - 76.0 ns</li>
            <li><strong>Mean bounds:</strong> Block145 means ± 30 ps</li>
            <li><strong>Sigma bounds:</strong> 15 - 120 ps</li>
            <li><strong>Total blocks fitted:</strong> """ + str(len(results_all)) + """</li>
        </ul>
    </div>
"""

# Summary table
html_content += """    <div class="info">
        <h2>Summary Table</h2>
        <table>
            <tr>
                <th>Block</th>
                <th>Power (µW)</th>
                <th>n=4 (ps)</th>
                <th>n=3 (ps)</th>
                <th>n=2 (ps)</th>
                <th>n=1 (ps)</th>
                <th>R²</th>
            </tr>
"""

for block_id in sorted(results_all.keys()):
    data = results_all[block_id]
    sigmas = [data["params"][i]["sigma"] * 1000 for i in range(4)]
    r2 = data["r_squared"]
    
    # Color code R²
    if r2 > 0.99:
        r2_class = "good"
    elif r2 > 0.95:
        r2_class = "warning"
    else:
        r2_class = "bad"
    
    html_content += f"""            <tr>
                <td>{block_id}</td>
                <td>{data['power']:.4f}</td>
                <td>{sigmas[0]:.2f}</td>
                <td>{sigmas[1]:.2f}</td>
                <td>{sigmas[2]:.2f}</td>
                <td>{sigmas[3]:.2f}</td>
                <td class="{r2_class}">{r2:.6f}</td>
            </tr>
"""

html_content += """        </table>
    </div>
"""

# Individual block plots
for block_id in sorted(results_all.keys()):
    data = results_all[block_id]
    plot_file = f"block_{block_id:03d}_fit_75p7_limits.png"
    
    html_content += f"""    <div class="block">
        <h2>Block {block_id} - {data['power']:.4f} µW (R² = {data['r_squared']:.6f})</h2>
        <img src="{plot_file}" alt="Block {block_id} fit">
    </div>
"""

html_content += """</body>
</html>
"""

with open(html_file, 'w') as f:
    f.write(html_content)

print(f"✓ Saved HTML summary to: {html_file}")

# ============================================================================
# PARAMETER VS POWER ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING PARAMETER VS POWER PLOTS")
print("=" * 80)

# Extract parameters for plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 14))

# Sort by power for cleaner plots
sorted_blocks = sorted(results_all.items(), key=lambda x: x[1]['power'])
powers = [item[1]['power'] for item in sorted_blocks]

# Arrays for each photon number
means_n = [[], [], [], []]
sigmas_n = [[], [], [], []]
amplitudes_n = [[], [], [], []]

for block_id, data in sorted_blocks:
    for i in range(4):
        means_n[i].append(data['params'][i]['mu'])
        sigmas_n[i].append(data['params'][i]['sigma'] * 1000)  # ps
        amplitudes_n[i].append(data['params'][i]['A'])

# Plot 1: Mean position vs power
ax = axes[0]
colors = ['purple', 'blue', 'green', 'red']
for i in range(4):
    n_label = 4 - i
    ax.plot(powers, means_n[i], 'o-', label=f'n={n_label}', color=colors[i], markersize=5, alpha=0.7)
ax.set_xlabel('Power (µW)', fontsize=12, weight='bold')
ax.set_ylabel('Mean Position (ns)', fontsize=12, weight='bold')
ax.set_title('Peak Mean Position vs Optical Power', fontsize=13, weight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Plot 2: Sigma vs power
ax = axes[1]
for i in range(4):
    n_label = 4 - i
    ax.plot(powers, sigmas_n[i], 'o-', label=f'n={n_label}', color=colors[i], markersize=5, alpha=0.7)
ax.set_xlabel('Power (µW)', fontsize=12, weight='bold')
ax.set_ylabel('Sigma (ps)', fontsize=12, weight='bold')
ax.set_title('Peak Width vs Optical Power', fontsize=13, weight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Plot 3: Amplitude vs power
ax = axes[2]
for i in range(4):
    n_label = 4 - i
    ax.plot(powers, amplitudes_n[i], 'o-', label=f'n={n_label}', color=colors[i], markersize=5, alpha=0.7)
ax.set_xlabel('Power (µW)', fontsize=12, weight='bold')
ax.set_ylabel('Amplitude (counts)', fontsize=12, weight='bold')
ax.set_title('Peak Amplitude vs Optical Power', fontsize=13, weight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

plt.tight_layout()
params_plot_file = output_dir / 'parameters_vs_power.png'
fig.savefig(params_plot_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"✓ Saved parameters vs power plot: {params_plot_file}")

# ============================================================================
# POISSON MEAN ESTIMATION
# ============================================================================

print("\n" + "=" * 80)
print("ESTIMATING POISSON MEAN FROM PEAK AMPLITUDES")
print("=" * 80)

from scipy.optimize import minimize

def poisson_prob(n, lam):
    """Poisson probability P(n | lambda)"""
    from math import factorial, exp
    return (lam**n / factorial(n)) * exp(-lam)

def estimate_poisson_mean(amplitudes):
    """
    Estimate Poisson mean from observed peak amplitudes.
    amplitudes: [A_n4, A_n3, A_n2, A_n1]
    Returns: estimated lambda
    """
    # Normalize amplitudes
    total = sum(amplitudes)
    if total == 0:
        return 0.0
    
    observed_probs = [A / total for A in amplitudes]
    
    # Minimize chi-square between observed and Poisson distribution
    def objective(lam):
        if lam <= 0:
            return 1e10
        predicted_probs = [poisson_prob(n, lam) for n in [4, 3, 2, 1]]
        # Normalize predicted (in case higher n contribute)
        pred_sum = sum(predicted_probs)
        if pred_sum == 0:
            return 1e10
        predicted_probs = [p / pred_sum for p in predicted_probs]
        
        chi_sq = sum((obs - pred)**2 / (pred + 1e-10) 
                     for obs, pred in zip(observed_probs, predicted_probs))
        return chi_sq
    
    # Initial guess: approximate from n=1 amplitude
    result = minimize(objective, x0=2.0, bounds=[(0.1, 10.0)], method='L-BFGS-B')
    return result.x[0] if result.success else 0.0

def estimate_poisson_error_mc(popt, pcov, n_samples=500):
    """
    Estimate error in Poisson mean using Monte Carlo sampling.
    
    Parameters:
    - popt: fitted parameters [mu0, sigma0, A0, mu1, sigma1, A1, ...]
    - pcov: covariance matrix from fit
    - n_samples: number of Monte Carlo samples
    
    Returns:
    - lambda_mean: mean Poisson parameter
    - lambda_std: standard deviation of Poisson parameter
    """
    try:
        # Check if pcov is valid
        if pcov is None or np.any(np.isinf(pcov)) or np.any(np.isnan(pcov)):
            return None, None
        
        # Extract amplitude parameters and their covariances
        # Amplitudes are at indices 2, 5, 8, 11 (every third parameter starting from 2)
        amp_indices = [2, 5, 8, 11]
        amps_opt = [popt[i] for i in amp_indices]
        
        # Extract covariance submatrix for amplitudes
        cov_amps = pcov[np.ix_(amp_indices, amp_indices)]
        
        # Check if covariance matrix is positive definite
        eigenvals = np.linalg.eigvals(cov_amps)
        if np.any(eigenvals <= 0):
            # Use diagonal approximation if not positive definite
            cov_amps = np.diag(np.diag(cov_amps))
        
        # Generate samples from multivariate normal
        amp_samples = np.random.multivariate_normal(amps_opt, cov_amps, n_samples)
        
        # Ensure all amplitudes are positive
        amp_samples = np.abs(amp_samples)
        
        # Estimate lambda for each sample
        lambda_samples = []
        for sample in amp_samples:
            lam = estimate_poisson_mean(list(sample))
            if lam > 0:  # Only keep valid estimates
                lambda_samples.append(lam)
        
        if len(lambda_samples) < 10:
            return None, None
        
        return np.mean(lambda_samples), np.std(lambda_samples)
    
    except Exception as e:
        return None, None

poisson_means = []
poisson_errors = []
poisson_powers = []

for block_id, data in sorted_blocks:
    r2 = data['r_squared']
    if r2 < 0.5:  # Skip poor fits
        continue
    
    amplitudes = [data['params'][i]['A'] for i in range(4)]  # n=4,3,2,1
    lam = estimate_poisson_mean(amplitudes)
    
    # Estimate error using Monte Carlo
    popt = data['popt']
    pcov = data.get('pcov', None)
    lam_mean, lam_err = estimate_poisson_error_mc(popt, pcov, n_samples=500)
    
    poisson_means.append(lam)
    poisson_errors.append(lam_err if lam_err is not None else 0.0)
    poisson_powers.append(data['power'])

print(f"Estimated Poisson mean for {len(poisson_means)} blocks (R² > 0.5)")

# Plot Poisson mean vs power with error bars
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Convert to arrays
poisson_powers_arr = np.array(poisson_powers)
poisson_means_arr = np.array(poisson_means)
poisson_errors_arr = np.array(poisson_errors)

# Plot with error bars
ax.errorbar(poisson_powers_arr, poisson_means_arr, yerr=poisson_errors_arr,
            fmt='o', markersize=6, linewidth=1.5, capsize=4,
            color='darkblue', alpha=0.7, label='Estimated λ ± σ')
ax.plot(poisson_powers_arr, poisson_means_arr, '-', linewidth=1,
        color='darkblue', alpha=0.3)

ax.set_xlabel('Optical Power (µW)', fontsize=12, weight='bold')
ax.set_ylabel('Poisson Mean (λ)', fontsize=12, weight='bold')
ax.set_title('Estimated Poisson Mean vs Optical Power\n(from photon-number peak amplitudes with MC error estimation)', 
             fontsize=13, weight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.legend(fontsize=11)

# Print statistics on errors
valid_errors = [e for e in poisson_errors if e > 0]
if len(valid_errors) > 0:
    avg_rel_error = np.mean([poisson_errors[i] / poisson_means[i] 
                             for i in range(len(poisson_means)) 
                             if poisson_errors[i] > 0])
    print(f"Average relative error: {avg_rel_error*100:.2f}%")
    print(f"Valid error estimates: {len(valid_errors)}/{len(poisson_errors)}")

# Add linear fit on log-log scale
if len(poisson_powers) > 5:
    from scipy.stats import linregress
    log_powers = np.log10(poisson_powers)
    log_means = np.log10(poisson_means)
    slope, intercept, r_value, p_value, std_err = linregress(log_powers, log_means)
    
    # Plot fit line
    fit_powers = np.logspace(np.log10(min(poisson_powers)), np.log10(max(poisson_powers)), 100)
    fit_means = 10**(slope * np.log10(fit_powers) + intercept)
    ax.plot(fit_powers, fit_means, '--', linewidth=2, color='red', alpha=0.7, 
            label=f'Power-law fit: slope={slope:.3f}, R²={r_value**2:.4f}')
    ax.legend(fontsize=11)
    
    print(f"Power-law fit: λ ∝ Power^{slope:.3f} (R² = {r_value**2:.4f})")

plt.tight_layout()
poisson_plot_file = output_dir / 'poisson_mean_vs_power.png'
fig.savefig(poisson_plot_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"✓ Saved Poisson mean vs power plot: {poisson_plot_file}")

# Save Poisson data
poisson_data = {
    'powers_uw': poisson_powers,
    'poisson_means': poisson_means,
    'poisson_errors': poisson_errors
}
poisson_file = output_dir / 'poisson_analysis.json'
with open(poisson_file, 'w') as f:
    json.dump(poisson_data, f, indent=2)
print(f"✓ Saved Poisson analysis data: {poisson_file}")

# Also save detailed table
poisson_table = output_dir / 'poisson_analysis.txt'
with open(poisson_table, 'w') as f:
    f.write("# Poisson Mean Photon Number Estimation\n")
    f.write("# Columns: Power(µW), Lambda, Error, Relative_Error(%)\n")
    f.write(f"{'Power(µW)':>12} {'Lambda':>12} {'Error':>12} {'Rel.Err(%)':>12}\n")
    for i in range(len(poisson_powers)):
        rel_err = (poisson_errors[i] / poisson_means[i] * 100) if poisson_means[i] > 0 else 0
        f.write(f"{poisson_powers[i]:12.6f} {poisson_means[i]:12.6f} {poisson_errors[i]:12.6f} {rel_err:12.2f}\n")
print(f"✓ Saved Poisson table: {poisson_table}")

print("\n✓ All analysis complete!")

# ============================================================================
# COMBINED MULTI-FRAME PLOT FOR BLOCKS < 200
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING COMBINED PLOT FOR BLOCKS < 200")
print("=" * 80)

# Filter blocks < 200
blocks_lt200 = {bid: data for bid, data in results_all.items() if bid < 200}
print(f"Found {len(blocks_lt200)} blocks with ID < 200")

if len(blocks_lt200) > 0:
    # Sort by block ID
    sorted_blocks_lt200 = sorted(blocks_lt200.items(), key=lambda x: x[0])
    
    # Determine grid size (5 columns)
    n_plots = len(sorted_blocks_lt200)
    n_cols = 5
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure
    fig = plt.figure(figsize=(5 * 4, n_rows * 3))
    
    colors_comp = ['purple', 'blue', 'green', 'red']  # n=4,3,2,1
    
    for plot_idx, (block_id, data) in enumerate(sorted_blocks_lt200):
        ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1)
        
        # Get data
        time_trimmed = data['time_trimmed']
        hist_trimmed = data['hist_trimmed']
        popt = data['popt']
        power = data['power']
        r2 = data['r_squared']
        
        # Plot histogram
        ax.plot(time_trimmed, hist_trimmed, 'o', markersize=3, color='lightblue', 
                alpha=0.6, label='Data')
        
        # Plot total fit
        y_fit = multi_gaussian(time_trimmed, *popt)
        ax.plot(time_trimmed, y_fit, '-', linewidth=2, color='red', alpha=0.8, label='Total fit')
        
        # Plot individual components
        for i in range(4):
            mu = popt[3*i]
            sigma = popt[3*i+1]
            A = popt[3*i+2]
            y_comp = single_gaussian(time_trimmed, mu, sigma, A)
            n_label = 4 - i
            ax.plot(time_trimmed, y_comp, '--', linewidth=1.5, color=colors_comp[i], 
                    alpha=0.7, label=f'n={n_label}')
        
        # Formatting
        ax.set_xlabel('Time (ns)', fontsize=9)
        ax.set_ylabel('Counts', fontsize=9)
        ax.set_title(f'Block {block_id}: {power:.4f} µW (R²={r2:.4f})', 
                    fontsize=10, weight='bold')
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        
        # Legend only on first plot
        if plot_idx == 0:
            ax.legend(fontsize=7, loc='upper right')
    
    plt.tight_layout()
    combined_file = output_dir / 'combined_fits_blocks_lt200.png'
    fig.savefig(combined_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved combined plot: {combined_file}")
else:
    print("No blocks with ID < 200 found")

print("\n✓ Combined plot generation complete!")
