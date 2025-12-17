# NIPXIe - TDMS Waveform Analysis

Event-by-event analysis of single-photon detection waveforms from NI PXIe digitizer TDMS files.

## Features

- **Event Detection**: Automatic pulse detection with configurable thresholds
- **Pulse Characterization**: Rise time, fall time, amplitude, FWHM analysis
- **Timing Analysis**: Jitter measurement, trigger delay characterization
- **Efficiency Calculation**: Detection efficiency vs bias voltage/optical power
- **Dark Count Analysis**: Background count rate characterization
- **Single Event Extraction**: Save individual events for detailed inspection
- **HEP-Style Plotting**: Professional publication-quality plots

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Main Scripts

### SelfTrigger.py - Core Analysis

Process TDMS files to extract pulse characteristics.

- `--display_report`, `-p`: Display matplotlib plots for each pulse (useful for debugging)

- `--debug_report`, `-b`: Enable debug output messages

- `--outputDir <path>`, `-d`: Output directory (default: ./Stats)

- `--report <N>`, `-r`: Report progress every N events (default: 1000)

- `--subset <N>`, `-s`: Process only first N events (default: -1 for all)

- `--checkSingleEvent <N>`, `-c`: Analyze only a specific event number (default: -1)

#### Examples

```bash
# Analyze a single TDMS file with spline method
python3 SelfTrigger.py /path/to/data.tdms

# Analyze with fast simple method
python3 SelfTrigger.py /path/to/data.tdms --trigger_method simple

# Analyze all TDMS files in a directory

**Usage:**
```bash
# Analyze single file
python SelfTrigger.py /path/to/file.tdms

# Analyze directory recursively
python SelfTrigger.py /path/to/directory --recursive

# Extract single event without full analysis
python SelfTrigger.py /path/to/file.tdms --save_single_event 5

# Show plots for debugging
python SelfTrigger.py /path/to/file.tdms --display_report

# Process subset of events
python SelfTrigger.py /path/to/file.tdms --subset 100

# Custom output directory with debug mode
python SelfTrigger.py /path/to/file.tdms -d ./output -b
```

**Output Files:**
- `*_analysis.json`: Event-by-event data and summary statistics
- `*_meta.json`: Metadata and analysis parameters
- `*_event<N>.json`: Single event waveform (with --save_single_event)

### SelfTrigger_plot.py - Diagnostic Plots

Generate diagnostic plots for pulse characteristics.

**Usage:**
```bash
python SelfTrigger_plot.py /path/to/analysis.json
```

**Output:** 28 plots including pulse amplitude, rise/fall time distributions, timing jitter, and correlation plots.

### plot_rates_vs_power.py - Power Dependence

Plot detection rates vs optical power for multiple bias voltages.

**Usage:**
```bash
python plot_rates_vs_power.py --input_dir /path/to/analyzed_json --output_dir ./plots
```

### plot_rates_vs_bias_v2.py - Bias Dependence

Plot detection rates vs bias voltage for multiple power levels.

**Usage:**
```bash
python plot_rates_vs_bias_v2.py --input_dir /path/to/analyzed_json --output_dir ./plots
```

### plot_all.py - Unified Plotting

Generate all standard plots with one command.

**Usage:**
```bash
python plot_all.py --input_dir /path/to/analyzed_json --output_dir ./plots --recursive
```

### plot_multiple_events.py - Event Visualization

Visualize multiple single-event waveforms together.

**Usage:**
```bash
python plot_multiple_events.py /path/to/directory/with/event_jsons
```

## Analysis Workflow

1. **Collect Data**: Acquire TDMS files from NI PXIe digitizer
2. **Run Analysis**: `python SelfTrigger.py /path/to/rawdata --recursive`
3. **Generate Plots**: `python plot_all.py --input_dir /path/to/analyzed_json --output_dir ./plots`

## Output JSON Structure

### Analysis JSON (`*_analysis.json`)
```json
{
  "metadata": {
    "bias_current_uA": 700,
    "bias_voltage_mV": 145,
    "power_nW": 10000,
    "wavelength_nm": 515,
    "trigger_method": "spline"
  },
  "summary_statistics": {
    "total_events": 5000,
    "signal_rate": 450.5,
    "signal_rate_error": 0.67,
    "dark_count_rate": 49.5,
    "efficiency": 0.0451,
    "efficiency_error": 0.0003
  },
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
python3 plot_pulse_characteristics.py <path_to_json_directory> [options]
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
python3 plot_pulse_characteristics.py /path/to/SNSPD_analyzed_json/

# Plot only vs bias current
python3 plot_pulse_characteristics.py /path/to/SNSPD_analyzed_json/ --mode vs_bias

# Plot only PTP (both vs bias and vs power)
python3 plot_pulse_characteristics.py /path/to/SNSPD_analyzed_json/ --mode ptp

# Plot only rise amplitude vs power
python3 plot_pulse_characteristics.py /path/to/SNSPD_analyzed_json/ --mode vs_power --variable rise_amplitude
```

**Output Files:**
- `pulse_ptp_vs_bias.png`
- `pulse_ptp_vs_power.png`
- `rise_amplitude_vs_bias.png`
- `rise_amplitude_vs_power.png`

#### Plot Count Rates vs Bias

Plot signal rate, dark count rate, and efficiency vs bias current.

```bash
python3 plot_rates_vs_bias_v2.py <path_to_json_directory> [options]
```

**Options:**
- `--mode [single|multi]`: Plot mode
  - `single`: One plot per power level
  - `multi`: Combined plot with all power levels
  
- `--output_dir <path>`: Output directory for plots

**Examples:**
```bash
# Single plots for each power
python3 plot_rates_vs_bias_v2.py /path/to/SNSPD_analyzed_json/ --mode single

# Combined multi-power plot
python3 plot_rates_vs_bias_v2.py /path/to/SNSPD_analyzed_json/ --mode multi
```

#### Plot Count Rates vs Power

Plot signal rate, dark count rate, and efficiency vs optical power.

```bash
python3 plot_rates_vs_power.py <path_to_json_directory> [options]
```

**Options:**
- `--mode [single|multi]`: Plot mode
- `--output_dir <path>`: Output directory

**Examples:**
```bash
# Single plots for each bias
python3 plot_rates_vs_power.py /path/to/SNSPD_analyzed_json/ --mode single

# Combined multi-bias plot
python3 plot_rates_vs_power.py /path/to/SNSPD_analyzed_json/ --mode multi
```

#### Plot Event-by-Event Data

Plot histograms and scatter plots of individual pulse properties.

```bash
python3 plot_event_data.py <path_to_json_file> [options]
```

**Options:**
- `--output_dir <path>`: Output directory for plots

**Example:**
```bash
python3 plot_event_data.py /path/to/SNSPD_analyzed_json/sample.json
```

**Output:**
- Histograms of PTP, rise amplitude, rise/fall times
- Scatter plots showing correlations

#### Generate All Plots

Convenience script to generate all plot types at once.

```bash
python3 plot_all.py --input_dir <path_to_json_directory> [options]
```

**Options:**
- `--input_dir <path>`, `-i`: Input directory (or directories) with JSON files (default: ./plots/test)
- `--output_dir <path>`, `-d`: Output directory for all plots (default: current directory)
- `--pattern <pattern>`, `-p`: File pattern to match (default: *_analysis.json)
- `--mode {all|vs_bias|vs_power|pulse}`, `-m`: Which plots to generate (default: all)
- `--log_scale`: Use log scale for power plots
- `--recursive`, `-r`: Search recursively in subdirectories (default: True)

**Examples:**
```bash
# Generate all plots from a directory
python3 plot_all.py -i /path/to/SNSPD_analyzed_json/

# Generate only rate vs bias plots
python3 plot_all.py --input_dir /path/to/SNSPD_analyzed_json/ --mode vs_bias

# Multiple input directories with custom output
python3 plot_all.py -i dir1/ dir2/ -d output_plots/
```

This generates:
- All pulse characteristic plots
- All rate vs bias plots
- All rate vs power plots

## Workflow Example

Complete analysis workflow from raw data to plots:

```bash
# Step 1: Analyze TDMS files
python3 SelfTrigger.py /path/to/SNSPD_rawdata/

# Step 2: Generate all plots
python3 plot_all.py /path/to/SNSPD_analyzed_json/

# Step 3: Generate specific plots as needed
python3 plot_pulse_characteristics.py /path/to/SNSPD_analyzed_json/ --mode vs_bias
python3 plot_event_data.py /path/to/SNSPD_analyzed_json/specific_file.json
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
