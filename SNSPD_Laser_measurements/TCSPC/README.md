# TCSPC Analysis Scripts

Analysis tools for Time-Correlated Single Photon Counting (TCSPC) measurements of SNSPDs.

## Directory Structure

```
TCSPC/
├── README.md                          # This file
├── tcspc_config.py                    # Central configuration (paths, bias settings, fit parameters)
├── tcspc_analysis.py                  # Core analysis functions (dark count subtraction, power law fitting)
├── read_phu.py                        # PicoHarp .phu file parser and analyzer
├── process_all_bias.py                # Batch processing script for all bias voltages
├── create_combined_plot.py            # Generate comparison plots across biases
├── analyze_bias_sweep.py              # Individual bias voltage analysis
├── analyze_dark_count_statistics.py   # Dark count characterization
├── compare_bias_sweeps.py             # Comparison utilities
├── compare_power_sweeps.py            # Power sweep comparison utilities
├── run_power_comparison.sh            # Shell script for batch comparisons
├── docs/                              # Documentation and summaries
│   ├── INTEGRATION_SUMMARY.md         # Integration guide for analysis workflow
│   └── INTERPOLATED_DATA_READY.txt    # Power data interpolation notes
├── utils/                             # Utility scripts
│   └── inject_block0_from_0nW.py      # Block 0 injection tool (alternative to external reference)
└── archive/                           # Old test files and debug scripts
```

## Quick Start

### 1. Configuration

Edit [tcspc_config.py](tcspc_config.py) to set:
- **Data file paths**: `BIAS_FILES` dictionary with paths to .phu files
- **Bias settings**: Colors and markers for each bias in `BIAS_SETTINGS`
- **Analysis parameters**: Signal window (`T_MIN_NS`, `T_MAX_NS`), fit range (`FIT_MAX_UW`)
- **Power calibration**: Path to rotation-power mapping file (`POWER_DATA_FILE`)

### 2. Process All Biases

```bash
python3 process_all_bias.py
```

This will:
- Process each bias voltage sweep individually
- Use external Block 0 reference (0nW file) for dark count measurements
- Generate individual plots for each bias
- Save results to `~/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/{bias}/`

### 3. Create Combined Plots

```bash
python3 create_combined_plot.py
```

This will:
- Load all processed bias data
- Generate comparison plots (log-log, linear, saturation views)
- Use adaptive fit range for biases with limited low-power data
- Save to `~/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/combined/`

## Key Features

### External Block 0 Reference System

Power sweeps don't include Block 0 (dark count) measurements. The analysis uses an external reference file (`SMSPD_3_2-7_500kHz_0nW_20260205_0518.phu`) containing dark counts at different bias voltages:

```python
block0_reference_blocks = {
    '66mV': 66,
    '70mV': 70,
    '73mV': 73,
    '74mV': 74,
    '78mV': 78
}
```

### Adaptive Fit Range

For biases with insufficient low-power data (e.g., 66mV starts at 14.33 µW), the combined plot script automatically expands the fit range to include at least 5 data points for power law fitting.

### Dark Count Subtraction Methods

- **OOT_pre**: Uses out-of-time (0-60 ns) region from each measurement (default)
- **Block 0**: Uses dedicated dark count measurement (via external reference)
- **Per-measurement**: Individual dark subtraction for each power point

## Analysis Workflow

1. **Data Collection**: PicoHarp .phu files with power sweeps at different biases
2. **Power Calibration**: Rotation angle → laser power mapping (1° resolution, 0-355°)
3. **Histogram Extraction**: Parse .phu files, extract histograms for each block
4. **Dark Subtraction**: OOT_pre method with external Block 0 reference
5. **Count Rate Calculation**: Integrate signal window (75-79 ns, 4 ns width)
6. **Power Law Fitting**: Linear fit in log-log space for low-power region (≤ 0.3 µW)
7. **Visualization**: Individual and combined plots with fit lines

## Output Structure

```
~/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/
├── 66mV/
│   ├── power_sweep_loglog_66mV.png
│   ├── power_sweep_linear_66mV.png
│   └── power_sweep_saturation_66mV.png
├── 70mV/
│   └── ...
├── 73mV/
│   └── ...
├── 74mV/
│   └── ...
├── 78mV/
│   └── ...
└── combined/
    ├── combined_power_sweep_loglog.png
    ├── combined_power_sweep_linear.png
    └── combined_power_sweep_saturation.png
```

## Command-Line Tools

### read_phu.py

Direct analysis of individual .phu files:

```bash
python3 read_phu.py <file.phu> [--debug] [--block0-file <ref.phu>] [--block0-block <id>]
```

Options:
- `--debug`: Show verbose output (header info, histogram details)
- `--block0-file`: Path to external Block 0 reference file
- `--block0-block`: Block ID to use from reference file

## Configuration Reference

### tcspc_config.py

```python
# Signal window (ns)
T_MIN_NS = 75.0
T_MAX_NS = 79.0

# Fit parameters
FIT_MAX_UW = 0.3  # Maximum power for low-power linear fit (µW)

# Bias settings
BIAS_SETTINGS = {
    '66mV': {'color': 'purple', 'marker': 'D'},
    '70mV': {'color': 'blue', 'marker': 'o'},
    '73mV': {'color': 'cyan', 'marker': 'v'},
    '74mV': {'color': 'green', 'marker': 's'},
    '78mV': {'color': 'red', 'marker': '^'}
}
```

## Dependencies

- numpy
- matplotlib
- scipy
- pathlib

## Notes

- All timestamps use PicoHarp 300 resolution (4 ps default)
- Power calibration uses linear interpolation between measured angles
- Chi-squared calculated with Poisson errors (σ = √counts)
- Adaptive fit range activates when < 2 points in default range
