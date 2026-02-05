import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.stats import poisson
import sys
from pathlib import Path

# Add parent directory to path to import plot_style
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import setup_hep_style

# Apply HEP plotting style
setup_hep_style()

# Avoid reds in color cycle to keep fit overlays distinct
plt.rcParams["axes.prop_cycle"] = cycler(color=[
  "#1f77b4",  # blue
  "#2ca02c",  # green
  "#9467bd",  # purple
  "#8c564b",  # brown
  "#17becf",  # cyan
  "#7f7f7f",  # gray
  "#bcbd22",  # olive
  "#e377c2",  # pink/magenta
])


def count_rate_from_nbar(nbar, rep_rate_hz, n, dcr=0):
    """Pure n-photon detector model with dark count rate (DCR).
    
    Mathematical form:
      P(click | n_threshold) = 1 - CDF(n-1, μ) = Σ_{k=n}^∞ (μ^k/k!) * e^(-μ)
      R_count = rep_rate * P(click) + DCR
    
    where μ = nbar (mean photons per pulse), DCR is background dark counts (counts/s)
    """
    mu = nbar  # mean photons per pulse
    p_click = 1 - poisson.cdf(n - 1, mu)
    return rep_rate_hz * p_click + dcr


def count_rate_mixed_sensitivity(nbar, rep_rate_hz, p1, p2):
    """Mixed-sensitivity detector: p1 prob for 1-photon, p2 prob for 2+ photons.
    
    Mathematical form:
      P(click | mixed) = p1 * P(k=1) + p2 * P(k≥2)
                       = p1 * (μ * e^(-μ)) + p2 * (1 - e^(-μ) - μ*e^(-μ))
      R_count = rep_rate * P(click | mixed)
    
    where:
      P(k=0) = e^(-μ)
      P(k=1) = μ * e^(-μ)
      P(k≥2) = 1 - P(k=0) - P(k=1)
      μ = nbar
      p1 = probability of detecting 1 photon (typically 0-1)
      p2 = probability of detecting 2+ photons (typically 0-1)
    """
    mu = nbar
    p_0 = np.exp(-mu)
    p_1 = mu * np.exp(-mu)
    p_2plus = 1 - p_0 - p_1
    p_click = p1 * p_1 + p2 * p_2plus
    return rep_rate_hz * p_click


def count_rate_three_level_sensitivity(nbar, rep_rate_hz, p1, p2, p3, dcr=0):
    """Three-level sensitivity detector: separate probs for 1, 2, and 3+ photons with DCR.
    
    Mathematical form:
      P(click | 3-level) = p1 * P(k=1) + p2 * P(k=2) + p3 * P(k≥3)
                         = p1 * (μ * e^(-μ)) + p2 * (μ²/2 * e^(-μ)) + p3 * P(k≥3)
      R_count = rep_rate * P(click | 3-level) + DCR
    
    where:
      P(k=0) = e^(-μ)
      P(k=1) = μ * e^(-μ)
      P(k=2) = μ²/2 * e^(-μ)
      P(k≥3) = 1 - P(k=0) - P(k=1) - P(k=2)
      μ = nbar
      p1 = probability of detecting 1 photon (typically 0-1)
      p2 = probability of detecting 2 photons (typically 0-1)
      p3 = probability of detecting 3+ photons (typically 0-1)
      DCR = dark count rate background (counts/s)
    """
    mu = nbar
    p_0 = np.exp(-mu)
    p_1 = mu * np.exp(-mu)
    p_2 = (mu**2 / 2) * np.exp(-mu)
    p_3plus = 1 - p_0 - p_1 - p_2
    p_click = p1 * p_1 + p2 * p_2 + p3 * p_3plus
    return rep_rate_hz * p_click + dcr


def fit_power_law(nbar, count_rate, nbar_min=1e-4, nbar_max=1e-1):
    mask = (nbar > nbar_min) & (nbar < nbar_max)
    x = np.log10(nbar[mask])
    y = np.log10(count_rate[mask])
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def autoscale_y(ax, margin=0.1):
  """Auto-scale y on log axes with a relative margin."""
  ymin, ymax = np.inf, -np.inf
  for line in ax.get_lines():
    ydata = line.get_ydata()
    if ydata is None:
      continue
    ydata = np.atleast_1d(ydata)
    finite = np.isfinite(ydata)
    if not np.any(finite):
      continue
    yvals = ydata[finite]
    yvals = yvals[yvals > 0]  # ignore non-positive for log scale
    if yvals.size == 0:
      continue
    ymin = min(ymin, np.min(yvals))
    ymax = max(ymax, np.max(yvals))
  if ymin <= 0 or ymax <= 0 or ymin >= ymax:
    return
  span = ymax / ymin
  ax.set_ylim(ymin / (span**margin), ymax * (span**margin))


rep_rate = 10e3  # laser repetition rate (Hz) -> 10 kHz
n_values = [1, 2, 3]
nbar = np.logspace(-6, 2, 500)  # average photons per pulse (extended down to 1e-6)

# Fitting parameters for two regions
fit_low_min = 1e-5   # x << 1 region: lower limit
fit_low_max = 1e-3   # x << 1 region: upper limit
fit_mid_min = 1e-1   # x ~ 1 region: lower limit
fit_mid_max = 1e0    # x ~ 1 region: upper limit
fit_color = "red"    # color for fitted trend lines
photon_colors = {1: "red", 2: "purple", 3: "green"}

# Multi-photon sensitivity configurations (three examples)
multi_photon_configs = [
  (0.2, 0.9, 1.0),  # baseline: modest 1-photon, mid 2-photon, full 3+
  (0.01, 0.8, 1.0),  # higher 1- and 2-photon sensitivity
  (0, 0.4, 1.0),  # near-single-photon sensitive with strong 2+
]

# Dark Count Rate (DCR) in counts/s - typical SNSPD DCR ranges from 10s to 1000s Hz
dcr_values = [100, 500]  # Two DCR scenarios: 100 Hz and 500 Hz

output_dir = Path(__file__).parent / "simulation_output"
output_dir.mkdir(exist_ok=True)

# Figure 1: Only pure photons (no fitting curves, slopes from x << 1)
fig1, ax1 = plt.subplots(figsize=(7, 6))
dcr_per_n = {1: 2000, 2: 1000, 3: 800}  # Hz per detector order
for n in n_values:
  R_count_no_dcr = count_rate_from_nbar(nbar, rep_rate, n, dcr=0)
  slope_low, _ = fit_power_law(nbar, R_count_no_dcr, fit_low_min, fit_low_max)
  dcr_n = dcr_per_n.get(n, 0)
  R_count_with_dcr = count_rate_from_nbar(nbar, rep_rate, n, dcr=dcr_n)
  ax1.loglog(
    nbar,
    R_count_with_dcr,
    label=f"{n}-photon (slope {slope_low:.2f}, DCR={dcr_n} Hz)",
    linestyle='-',
    alpha=0.8,
    color=photon_colors.get(n, None),
  )
ax1.set_xlabel("Average photons per pulse")
ax1.set_ylabel("Count rate (s$^{-1}$)")
ax1.set_xlim(1e-3, 2e1)
ax1.grid(True, which="both")
ax1.legend(fontsize=9)
autoscale_y(ax1)
fig1.tight_layout()

# Figure 2: Multi-photon cases (no fitting, includes pure photon curves)
fig2, ax2 = plt.subplots(figsize=(7, 6))
for n in n_values:
  R_count = count_rate_from_nbar(nbar, rep_rate, n, dcr=0)
  ax2.loglog(
    nbar,
    R_count,
    label=f"Pure {n}-photon",
    linestyle='--',
    alpha=0.7,
    color=photon_colors.get(n, None),
  )
for p1, p2, p3 in multi_photon_configs:
  R_count_multi = count_rate_three_level_sensitivity(nbar, rep_rate, p1, p2, p3, dcr=0)
  label_multi = f"Multi-photon ({int(p1*100)}% 1ph, {int(p2*100)}% 2ph, {int(p3*100)}% 3+)"
  ax2.loglog(nbar, R_count_multi, label=label_multi, linestyle='-', alpha=0.9, linewidth=2)
ax2.set_xlabel("Average photons per pulse")
ax2.set_ylabel("Count rate (s$^{-1}$)")
ax2.set_xlim(1e-3, 2e1)
ax2.grid(True, which="both")
ax2.legend(fontsize=9)
autoscale_y(ax2)
fig2.tight_layout()

# Figure 3: Fit to the curve at x~1 (pure + multi)
fig3, ax3 = plt.subplots(figsize=(7, 6))
for n in n_values:
  R_count = count_rate_from_nbar(nbar, rep_rate, n, dcr=0)
  slope_mid, intercept_mid = fit_power_law(nbar, R_count, fit_mid_min, fit_mid_max)
  label = f"Pure {n}-photon (slope: {slope_mid:.2f})"
  ax3.loglog(
    nbar,
    R_count,
    label=label,
    linestyle='-',
    alpha=0.8,
    color=photon_colors.get(n, None),
  )

for p1, p2, p3 in multi_photon_configs:
  R_count_multi = count_rate_three_level_sensitivity(nbar, rep_rate, p1, p2, p3, dcr=0)
  slope_mid, intercept_mid = fit_power_law(nbar, R_count_multi, fit_mid_min, fit_mid_max)
  label_multi = f"Multi ({int(p1*100)}%/ {int(p2*100)}%/ {int(p3*100)}%) slope: {slope_mid:.2f}"
  ax3.loglog(nbar, R_count_multi, label=label_multi, linestyle='-', alpha=0.8)
  nbar_fit_mid = np.logspace(np.log10(fit_mid_min), np.log10(fit_mid_max), 50)
  R_fit_mid = 10**intercept_mid * nbar_fit_mid**slope_mid
  ax3.loglog(nbar_fit_mid, R_fit_mid, linestyle='--', color=fit_color, alpha=0.8, linewidth=2)

ax3.set_xlabel("Average photons per pulse")
ax3.set_ylabel("Count rate (s$^{-1}$)")
ax3.set_xlim(1e-3, 2e1)
ax3.grid(True, which="both")
ax3.legend(fontsize=9)
autoscale_y(ax3, margin=0.02)
bottom, top = ax3.get_ylim()
ax3.set_ylim(max(bottom, 1e0), top)
fig3.tight_layout()

# Figure 4: Fit to the curve at x<<1 (pure + multi)
fig4, ax4 = plt.subplots(figsize=(7, 6))
for n in n_values:
  R_count = count_rate_from_nbar(nbar, rep_rate, n, dcr=0)
  slope_low, intercept_low = fit_power_law(nbar, R_count, fit_low_min, fit_low_max)
  label = f"Pure {n}-photon (slope: {slope_low:.2f})"
  ax4.loglog(
    nbar,
    R_count,
    label=label,
    linestyle='-',
    alpha=0.8,
    color=photon_colors.get(n, None),
  )

for p1, p2, p3 in multi_photon_configs:
  R_count_multi = count_rate_three_level_sensitivity(nbar, rep_rate, p1, p2, p3, dcr=0)
  slope_low, intercept_low = fit_power_law(nbar, R_count_multi, fit_low_min, fit_low_max)
  label_multi = f"Multi ({int(p1*100)}%/ {int(p2*100)}%/ {int(p3*100)}%) slope: {slope_low:.2f}"
  ax4.loglog(nbar, R_count_multi, label=label_multi, linestyle='-', alpha=0.8)
  nbar_fit_low = np.logspace(np.log10(fit_low_min), np.log10(fit_low_max), 50)
  R_fit_low = 10**intercept_low * nbar_fit_low**slope_low
  ax4.loglog(nbar_fit_low, R_fit_low, linestyle='--', color=fit_color, alpha=0.8, linewidth=2)

ax4.set_xlabel("Average photons per pulse")
ax4.set_ylabel("Count rate (s$^{-1}$)")
ax4.set_xlim(1e-3, 2e1)
ax4.grid(True, which="both")
ax4.legend(fontsize=9)
autoscale_y(ax4)
fig4.tight_layout()

fig1.savefig(output_dir / "pure_photon.png", dpi=200)
fig2.savefig(output_dir / "multi_photon.png", dpi=200)
fig3.savefig(output_dir / "fit_mid.png", dpi=200)
fig4.savefig(output_dir / "fit_low.png", dpi=200)

# Figure 5: DCR effect on pure photon detectors
fig5, ax5 = plt.subplots(figsize=(7, 6))
for n in n_values:
  R_count_no_dcr = count_rate_from_nbar(nbar, rep_rate, n, dcr=0)
  ax5.loglog(
    nbar,
    R_count_no_dcr,
    label=f"Pure {n}-photon (no DCR)",
    linestyle='-',
    alpha=0.9,
    color=photon_colors.get(n, None),
  )
  
  # Add with DCR
  for dcr in dcr_values:
    R_count_with_dcr = count_rate_from_nbar(nbar, rep_rate, n, dcr=dcr)
    ax5.loglog(
      nbar,
      R_count_with_dcr,
      label=f"Pure {n}-photon (DCR={dcr}Hz)",
      linestyle='--',
      alpha=0.7,
      color=photon_colors.get(n, None),
    )

ax5.set_xlabel("Average photons per pulse")
ax5.set_ylabel("Count rate (s$^{-1}$)")
ax5.set_xlim(1e-3, 2e1)
ax5.grid(True, which="both")
ax5.legend(fontsize=8)
autoscale_y(ax5)
fig5.tight_layout()
fig5.savefig(output_dir / "dcr_effect_pure.png", dpi=200)

# Figure 6: DCR effect on multi-photon detectors
fig6, ax6 = plt.subplots(figsize=(7, 6))
p1, p2, p3 = multi_photon_configs[0]  # Use first config
R_count_no_dcr = count_rate_three_level_sensitivity(nbar, rep_rate, p1, p2, p3, dcr=0)
ax6.loglog(nbar, R_count_no_dcr, label=f"Multi-photon (no DCR)", linestyle='-', alpha=0.9, linewidth=2)

for dcr in dcr_values:
  R_count_with_dcr = count_rate_three_level_sensitivity(nbar, rep_rate, p1, p2, p3, dcr=dcr)
  ax6.loglog(nbar, R_count_with_dcr, label=f"Multi-photon (DCR={dcr}Hz)", linestyle='--', alpha=0.8, linewidth=2)

ax6.set_xlabel("Average photons per pulse")
ax6.set_ylabel("Count rate (s$^{-1}$)")
ax6.set_xlim(1e-3, 2e1)
ax6.grid(True, which="both")
ax6.legend(fontsize=9)
autoscale_y(ax6)
fig6.tight_layout()
fig6.savefig(output_dir / "dcr_effect_multi.png", dpi=200)
fig3.savefig(output_dir / "fit_mid.png", dpi=200)
fig4.savefig(output_dir / "fit_low.png", dpi=200)

# Figure 7: Poisson distribution histogram with mu=0.1
fig7, ax7 = plt.subplots(figsize=(8, 6))
mu = 0.1
k_max = 5
k = np.arange(0, k_max + 1)
pmf = poisson.pmf(k, mu)

ax7.bar(k, pmf, width=0.6, alpha=0.7, color='steelblue', edgecolor='black')
ax7.set_xlabel('Number of Photons (k)')
ax7.set_ylabel('Probability P(k; μ=0.1)')
ax7.set_title(f'Poisson Distribution (μ = {mu})')
ax7.set_xticks(k)
ax7.grid(True, axis='y', alpha=0.3)

# Add text annotations for each bar
for ki, prob in zip(k, pmf):
    ax7.text(ki, prob + 0.005, f'{prob:.4f}', ha='center', va='bottom', fontsize=9)

ax7.set_ylim(0, max(pmf) * 1.15)
fig7.tight_layout()
fig7.savefig(output_dir / "poisson_distribution_mu01.png", dpi=200)

plt.show()