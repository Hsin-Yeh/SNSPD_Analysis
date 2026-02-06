#!/usr/bin/env python3
"""Test consistency between read_phu and create_combined_plot chi-square values."""

from read_phu import read_phu_file, load_power_data
from tcspc_analysis import extract_oot_pre_dark_counts, subtract_dark_counts, fit_power_law
from tcspc_config import T_MIN_NS, T_MAX_NS, SIGNAL_WIDTH_NS, FIT_MAX_UW, POWER_DATA_FILE
import numpy as np

# Load 70mV file
filepath = '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_70mV_20260205_0122.phu'
power_data = load_power_data(POWER_DATA_FILE)
header, histograms = read_phu_file(filepath)

# Extract data (mimicking create_combined_plot.py logic)
curve_indices = header.get('HistResDscr_CurveIndex', {})
acq_time_s = header.get('MeasDesc_AcquisitionTime', 10000) / 1000.0
resolution_s = header.get('MeasDesc_Resolution', 4e-12)

bin_min = int(T_MIN_NS * 1e-9 / resolution_s)
bin_max = int(T_MAX_NS * 1e-9 / resolution_s)

powers, counts, oot_darks = [], [], []
for i, hist in enumerate(histograms):
    block_id = curve_indices.get(i, None)
    if block_id is None or block_id not in power_data or block_id == 0:
        continue
    count_rate = int(np.sum(hist[bin_min:bin_max])) / acq_time_s
    oot_dark = extract_oot_pre_dark_counts(hist, resolution_s, SIGNAL_WIDTH_NS, acq_time_s)
    powers.append(power_data[block_id])
    counts.append(count_rate)
    oot_darks.append(oot_dark)

powers_arr = np.array(powers)
counts_arr = np.array(counts)
oot_darks_arr = np.array(oot_darks)

counts_corrected, dark_rate = subtract_dark_counts(counts_arr, oot_darks_arr, method='per_measurement')
fit_results = fit_power_law(powers_arr, counts_corrected, FIT_MAX_UW)

print(f'70mV Analysis Results:')
print(f'='*60)
print(f'Total data points: {len(powers_arr)}')
print(f'Fit points (power <= {FIT_MAX_UW} µW): {np.sum(fit_results["fit_mask"])}')
print(f'Dark count rate: {dark_rate:.10f} cts/s')
print(f'Power law exponent: {fit_results["slope"]:.10f} ± {fit_results["std_err"]:.10f}')
print(f'Chi²/ndf (full precision): {fit_results["chi2_ndf"]:.15f}')
print(f'Chi²/ndf (4 decimals): {fit_results["chi2_ndf"]:.4f}')
print(f'Chi²/ndf (6 decimals): {fit_results["chi2_ndf"]:.6f}')
