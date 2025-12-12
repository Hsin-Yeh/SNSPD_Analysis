# NIPXIe - SNSPD Analysis Package

A comprehensive Python package for analyzing Superconducting Nanowire Single-Photon Detector (SNSPD) data from NI PXIe systems. Process TDMS files and generate publication-quality plots.

## Features

- **TDMS File Analysis**: Process raw TDMS files to extract pulse characteristics
- **Pulse Characterization**: Calculate pulse amplitude, rise/fall times, trigger times, and more
- **Statistical Analysis**: Compute signal rates, dark count rates, detection efficiency with proper error propagation
- **Flexible Plotting**: Generate various plots for analyzing detector performance
- **Event-by-Event Analysis**: Track individual pulse properties across measurements

## Installation

### Requirements

- Python 3.7 or higher
- Required packages (see `requirements.txt`):
  - numpy
  - pandas
  - matplotlib
  - scipy
  - nptdms

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The package is ready to use directly from this directory.

## Usage

### 1. Analyzing TDMS Files

The main analysis script is `SelfTrigger.py`. It processes TDMS files and generates JSON output with pulse statistics.

#### Basic Usage

```bash
python SelfTrigger.py <path_to_tdms_file_or_directory>
```

#### Command-Line Options

- `--method [spline|simple]`: Choose trigger time calculation method
  - `spline` (default): Accurate method using cubic spline interpolation (slower)
  - `simple`: Fast approximation using threshold crossing (10-50x faster)

- `--display`: Show matplotlib plots for each pulse (useful for debugging)

- `--trigger_check_min <value>`: Minimum trigger check value (default: 196)

- `--trigger_check_max <value>`: Maximum trigger check value (default: 198)

#### Examples

```bash
# Analyze a single TDMS file with spline method
python SelfTrigger.py /path/to/data.tdms

# Analyze with fast simple method
python SelfTrigger.py /path/to/data.tdms --method simple

# Analyze all TDMS files in a directory
python SelfTrigger.py /path/to/data_directory/

# Show plots for debugging
python SelfTrigger.py /path/to/data.tdms --display

# Use custom trigger check range
python SelfTrigger.py /path/to/data.tdms --trigger_check_min 195 --trigger_check_max 199
```

#### Output Format

The analysis generates JSON files in an output directory (auto-detected as `SNSPD_analyzed_json` parallel to the input data directory). Each JSON file contains:

```json
{
  "metadata": {
    "tdms_file": "...",
    "bias_current_uA": 700,
    "bias_voltage_mV": 145,
    "power_nW": 10000,
    "angle_degrees": 0,
    "wavelength_nm": 515,
    "timestamp": "20250611_025719",
    "trigger_method": "spline"
  },
  "summary_statistics": {
    "total_time": 10.0,
    "total_events": 5000,
    "signal_rate": 450.5,
    "signal_rate_error": 0.67,
    "dark_count_rate": 49.5,
    "dark_count_rate_error": 0.22,
    "efficiency": 0.0451,
    "efficiency_error": 0.0003,
    "ptp_mean": 0.0123,
    "ptp_std": 0.0015,
    "rise_amplitude_mean": 0.0089,
    "rise_amplitude_std": 0.0011,
    "rise_time_mean": 2.34e-9,
    "rise_time_std": 0.15e-9,
    "fall_time_mean": 3.56e-9,
    "fall_time_std": 0.22e-9
  },
  "total_events": 5000,
  "event_by_event_data": [
    {
      "event_number": 0,
      "ptp": 0.0125,
      "rise_amplitude": 0.0091,
      "rise_time": 2.35e-9,
      "fall_time": 3.58e-9,
      "trigger_time": 1.234e-6
    },
    ...
  ]
}
```

#### Key Variables

- **Signal Rate**: Count rate with trigger check in [196, 198] (signal + accidentals)
- **Dark Count Rate**: Count rate with trigger check outside [196, 198]
- **Efficiency**: `signal_rate / SOURCE_RATE` where `SOURCE_RATE = 1E7` (10 MHz laser)
- **PTP (Peak-to-Peak)**: `abs(pulse_max - pulse_min)`
- **Rise Amplitude**: `abs(baseline - pulse_min)`
- **Rise Time**: Time from 10% to 90% of rising edge
- **Fall Time**: Time from 90% to 10% of falling edge

#### Error Calculations

- **Rate Errors**: Poisson statistics `sqrt(N) / time`
- **Efficiency Error**: Binomial statistics `sqrt(p * (1-p) / n_pulses)` where `n_pulses = SOURCE_RATE * total_time`

### 2. Plotting Results

The package includes several plotting scripts for visualizing analysis results.

#### Plot Pulse Characteristics

Plot peak-to-peak amplitude and/or rise amplitude vs bias current or optical power.

```bash
python plot_pulse_characteristics.py <path_to_json_directory> [options]
```

**Options:**
- `--mode [all|vs_bias|vs_power|ptp|amplitude]`: Select plot type
  - `all` (default): Generate all 4 plots
  - `vs_bias`: Plot vs bias current (both variables)
  - `vs_power`: Plot vs power (both variables)
  - `ptp`: Plot only PTP variable (vs bias and power)
  - `amplitude`: Plot only rise amplitude (vs bias and power)
  
- `--variable [ptp|rise_amplitude|both]`: Select which variable to plot (default: both)

- `--output_dir <path>`: Output directory for plots (default: input directory)

**Examples:**
```bash
# Generate all 4 plots
python plot_pulse_characteristics.py /path/to/SNSPD_analyzed_json/

# Plot only vs bias current
python plot_pulse_characteristics.py /path/to/SNSPD_analyzed_json/ --mode vs_bias

# Plot only PTP (both vs bias and vs power)
python plot_pulse_characteristics.py /path/to/SNSPD_analyzed_json/ --mode ptp

# Plot only rise amplitude vs power
python plot_pulse_characteristics.py /path/to/SNSPD_analyzed_json/ --mode vs_power --variable rise_amplitude
```

**Output Files:**
- `pulse_ptp_vs_bias.png`
- `pulse_ptp_vs_power.png`
- `rise_amplitude_vs_bias.png`
- `rise_amplitude_vs_power.png`

#### Plot Count Rates vs Bias

Plot signal rate, dark count rate, and efficiency vs bias current.

```bash
python plot_rates_vs_bias_v2.py <path_to_json_directory> [options]
```

**Options:**
- `--mode [single|multi]`: Plot mode
  - `single`: One plot per power level
  - `multi`: Combined plot with all power levels
  
- `--output_dir <path>`: Output directory for plots

**Examples:**
```bash
# Single plots for each power
python plot_rates_vs_bias_v2.py /path/to/SNSPD_analyzed_json/ --mode single

# Combined multi-power plot
python plot_rates_vs_bias_v2.py /path/to/SNSPD_analyzed_json/ --mode multi
```

#### Plot Count Rates vs Power

Plot signal rate, dark count rate, and efficiency vs optical power.

```bash
python plot_rates_vs_power.py <path_to_json_directory> [options]
```

**Options:**
- `--mode [single|multi]`: Plot mode
- `--output_dir <path>`: Output directory

**Examples:**
```bash
# Single plots for each bias
python plot_rates_vs_power.py /path/to/SNSPD_analyzed_json/ --mode single

# Combined multi-bias plot
python plot_rates_vs_power.py /path/to/SNSPD_analyzed_json/ --mode multi
```

#### Plot Event-by-Event Data

Plot histograms and scatter plots of individual pulse properties.

```bash
python plot_event_data.py <path_to_json_file> [options]
```

**Options:**
- `--output_dir <path>`: Output directory for plots

**Example:**
```bash
python plot_event_data.py /path/to/SNSPD_analyzed_json/sample.json
```

**Output:**
- Histograms of PTP, rise amplitude, rise/fall times
- Scatter plots showing correlations

#### Generate All Plots

Convenience script to generate all plot types at once.

```bash
python plot_all.py <path_to_json_directory> [options]
```

**Options:**
- `--output_dir <path>`: Output directory for all plots

**Example:**
```bash
python plot_all.py /path/to/SNSPD_analyzed_json/
```

This generates:
- All pulse characteristic plots
- All rate vs bias plots
- All rate vs power plots

## Workflow Example

Complete analysis workflow from raw data to plots:

```bash
# Step 1: Analyze TDMS files
python SelfTrigger.py /path/to/SNSPD_rawdata/

# Step 2: Generate all plots
python plot_all.py /path/to/SNSPD_analyzed_json/

# Step 3: Generate specific plots as needed
python plot_pulse_characteristics.py /path/to/SNSPD_analyzed_json/ --mode vs_bias
python plot_event_data.py /path/to/SNSPD_analyzed_json/specific_file.json
```

## File Naming Convention

TDMS files should follow this naming pattern for automatic parameter extraction:
```
<DeviceID>_<Mode>_<Wavelength>_<Power>nW_<Angle>degrees_<Current>uA_<Voltage>mV_<Timestamp>.tdms
```

Example:
```
SMSPD_NbTiN_2025Jun_3-4_pristine_Pulse_515_10000nW_0degrees_700uA_145mV_20250611_025719.tdms
```

## Package Structure

```
NIPXIe/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── __init__.py                        # Package initialization
├── SelfTrigger.py                     # Main analysis script
├── plot_pulse_characteristics.py      # Plot PTP and rise amplitude
├── plot_rates_vs_bias_v2.py          # Plot rates vs bias
├── plot_rates_vs_power.py            # Plot rates vs power
├── plot_event_data.py                # Plot event histograms
├── plot_all.py                       # Generate all plots
├── plot_utils.py                     # Common plotting utilities
└── utils/                            # Utility modules
    ├── __init__.py                   # Utils package initialization
    ├── Timing_Analyzer.py            # Timing analysis utilities
    ├── tdmsUtils.py                  # TDMS file utilities
    ├── osUtils.py                    # OS/path utilities
    └── plotUtilscopy.py              # Additional plotting utilities
```

## Notes

- **Trigger Cut**: Signal events are defined as having trigger_check values between 196-198 (configurable)
- **Source Rate**: Default laser repetition rate is 10 MHz (1E7 Hz)
- **Trigger Methods**: 
  - Spline method is more accurate but slower
  - Simple method is 10-50x faster with slightly reduced accuracy
- **Memory**: For large datasets, consider processing in batches
- **Output**: JSON format allows easy integration with other analysis tools

## Troubleshooting

**No JSON files generated:**
- Check that TDMS files follow the naming convention
- Verify input path is correct
- Check file permissions

**Import errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version ≥ 3.7

**Plotting errors:**
- Check that JSON files contain required fields
- Ensure matplotlib backend is configured correctly

## Contributing

For bug reports or feature requests, please contact the development team.

## Version

Current version: 1.0.0
