# NIPXIe - TDMS Waveform Analysis

Event-by-event analysis of single-photon detection waveforms from NI PXIe digitizer TDMS files.

## Analysis Workflow

The analysis is organized into three sequential stages:

```
Stage 1: SelfTrigger.py       TDMS files → Event-by-event JSON
         ↓
Stage 2: analyze_events.py    Event JSON → Statistical analysis JSON
         ↓  
Stage 3: plot_all.py          Statistics JSON → Comparison plots
```

### Stage 1: Event Extraction (SelfTrigger.py)

Converts raw TDMS waveform data into condensed event-by-event physics variables:
- Extracts pulse characteristics (amplitude, rise/fall time, FWHM)
- Measures timing information (trigger delay, pulse interval)
- Calculates rates and efficiency
- **Output**: `*_analysis.json` with event data and summary statistics

### Stage 2: Statistical Analysis (analyze_events.py)

Performs statistical analysis on event-by-event data:
- Computes mean, standard deviation, standard error for all variables
- Fits Gaussian distributions to timing variables (e.g., trigger_check)
- Calculates uncertainties and error propagation
- **Output**: `statistics_*.json` with fitted parameters and errors
- **Optional**: Generate diagnostic plots with `--no-plots` to skip

### Stage 3: Comparison Plotting (plot_all.py)

Creates comparison plots across multiple measurements:
- Detection rate vs bias voltage (multiple powers)
- Detection rate vs optical power (multiple biases)
- Pulse characteristics vs operating conditions
- **Uses**: Statistical results from Stage 2 for error bars

## Quick Start

```bash
# Complete workflow example
# Stage 1: Extract events from TDMS
python SelfTrigger.py data.tdms -d output/

# Stage 2: Compute statistics (with plots)
python analyze_events.py output/*_analysis.json -d output/statistics/

# Stage 2: Compute statistics only (no plots, faster for batch)
python analyze_events.py output/*_analysis.json -d stats/ --no-plots

# Stage 3: Generate comparison plots
python plot_all.py -i stats/ -p 'statistics_*.json' -d final_plots/
```

## Features

- **Event Detection**: Automatic pulse detection with configurable thresholds
- **Pulse Characterization**: Rise time, fall time, amplitude, FWHM analysis
- **Timing Analysis**: Jitter measurement, trigger delay characterization with Gaussian fitting
- **Efficiency Calculation**: Detection efficiency vs bias voltage/optical power
- **Dark Count Analysis**: Background count rate characterization
- **Statistical Analysis**: Error propagation, histogram fitting, uncertainty quantification
- **Single Event Extraction**: Save individual events for detailed inspection
- **HEP-Style Plotting**: Professional publication-quality plots

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Detailed Script Documentation

### Stage 1: SelfTrigger.py - Event Extraction

Process TDMS files to extract pulse characteristics and convert scope data into physics variables.

Process TDMS files to extract pulse characteristics.

- `--display_report`, `-p`: Display matplotlib plots for each pulse (useful for debugging)

- `--debug_report`, `-b`: Enable debug output messages

- `--outputDir <path>`, `-d`: Output directory (default: ./Stats)

- `--report <N>`, `-r`: Report progress every N events (default: 1000)

- `--subset <N>`, `-s`: Process only first N events (default: -1 for all)

- `--checkSingleEvent <N>`, `-c`: Analyze only a specific event number (default: -1)

#### Examples

```bash
**Arguments:**
- `--display_report`, `-p`: Display matplotlib plots for each pulse (debugging)
- `--debug_report`, `-b`: Enable debug output messages
- `--outputDir <path>`, `-d`: Output directory (default: ./Stats)
- `--report <N>`, `-r`: Report progress every N events (default: 1000)
- `--subset <N>`, `-s`: Process only first N events (default: -1 for all)
- `--checkSingleEvent <N>`, `-c`: Analyze only a specific event number
- `--save_single_event <N>`: Extract and save a specific event's waveform
- `--recursive`: Process all TDMS files in directory recursively
- `--trigger_method`: Method for trigger detection ('spline' or 'simple')

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

---

### Stage 2: analyze_events.py - Statistical Analysis

Compute statistics, fit histograms, and quantify uncertainties from event-by-event data.

**Arguments:**
- `in_filenames`: One or more `*_analysis.json` files from Stage 1
- `--output_dir`, `-d`: Output directory for statistics and plots (default: .)
- `--bins`, `-b`: Number of histogram bins (default: 50, 100 for trigger_check)
- `--no-plots`: Skip plot generation, only compute statistics (faster for batch)

**Usage:**
```bash
# Full analysis with diagnostic plots
python analyze_events.py event0_analysis.json -d output/

# Batch processing without plots (faster)
python analyze_events.py output/*.json --no-plots -d statistics/

# Multiple files with custom binning
python analyze_events.py file1.json file2.json -b 100 -d stats/
```

**Output Files:**
- `statistics_*.json`: Comprehensive statistics with errors and fit parameters
- `*_histogram_*.png`: Histogram plots (if --no-plots not used)
- `*_vs_event_*.png`: Time series plots
- `correlation_*.png`: 2D correlation plots

**Statistics Computed:**
- Mean, median, standard deviation, standard error
- Min, max, quartiles (Q25, Q75)
- Gaussian fit parameters for `trigger_check` (μ, σ, amplitude with errors)

---

### Stage 3: plot_all.py - Comparison Plots

Generate comparison plots across multiple measurements using statistical results.

**Arguments:**
- `--input_dir`, `-i`: Directory/directories with `*_analysis.json` or `statistics_*.json`
- `--output_dir`, `-d`: Output directory for plots (default: .)
- `--pattern`, `-p`: File pattern (default: `*_analysis.json`)
- `--mode`, `-m`: Plot type: `all`, `vs_bias`, `vs_power`, `pulse`
- `--log_scale`: Use logarithmic scale for power plots
- `--recursive`, `-r`: Search subdirectories (default: True)

**Usage:**
```bash
# Using Stage 1 analysis files (original workflow)
python plot_all.py -i plots/test/ -p '*_analysis.json'

# Using Stage 2 statistics files (recommended - includes errors)
python plot_all.py -i statistics/ -p 'statistics_*.json'

# Multiple directories, specific plot type
python plot_all.py -i dir1/ dir2/ -m vs_bias -d comparison_plots/

# Only pulse characteristic plots
python plot_all.py -i statistics/ -m pulse
```

**Output Plots:**
- Detection rate vs bias voltage (multi-power comparison)
- Detection rate vs optical power (multi-bias comparison)
- Pulse characteristics vs operating conditions
- Efficiency curves with error bars

---

### Supporting Scripts

#### plot_multiple_events.py - Event Waveform Visualization

Visualize multiple single-event waveforms together, sorted by resistance.

**Usage:**
```bash
python plot_multiple_events.py /path/to/directory/with/event_jsons
```

#### plot_rates_vs_power.py - Power Dependence

Plot detection rates vs optical power for multiple bias voltages (can use directly, or via plot_all.py).

**Usage:**
```bash
python plot_rates_vs_power.py --input_dir /path/to/analyzed_json --output_dir ./plots
```

#### plot_rates_vs_bias_v2.py - Bias Dependence

Plot detection rates vs bias voltage for multiple power levels (can use directly, or via plot_all.py).

**Usage:**
```bash
python plot_rates_vs_bias_v2.py --input_dir /path/to/analyzed_json --output_dir ./plots
```

## Complete Workflow Example

```bash
# 1. Stage 1: Extract events from TDMS files
python SelfTrigger.py ~/SNSPD_data/*.tdms -d output/stage1/ --recursive

# 2. Stage 2: Compute statistics without plots (fast batch processing)
python analyze_events.py output/stage1/*_analysis.json --no-plots -d output/stage2/

# 3. Stage 3: Generate comparison plots
python plot_all.py -i output/stage2/ -p 'statistics_*.json' -d final_plots/

# Alternative: Stage 2 with diagnostic plots for select files
python analyze_events.py output/stage1/important_file_analysis.json -d diagnostics/
```

## Output JSON Structures

### Stage 1 Output: Analysis JSON (`*_analysis.json`)
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
      "pulse_max": 0.0125,
      "pulse_min": -0.0032,
      "rise_amplitude": 0.0091,
      "rise_time_10_90": 2.35e-9,
      "fall_time_90_10": 3.58e-9,
      "trigger_check": 197.3,
      "pulse_time_interval": 100.15
    }
  ]
}
```

### Stage 2 Output: Statistics JSON (`statistics_*.json`)
```json
{
  "input_file": "event0_analysis.json",
  "metadata": { ... },
  "summary_from_stage1": { ... },
  "variable_statistics": {
    "trigger_check": {
      "variable": "trigger_check",
      "count": 5000,
      "mean": 197.234,
      "std": 0.523,
      "sem": 0.0074,
      "median": 197.241,
      "min": 195.2,
      "max": 199.1,
      "q25": 196.89,
      "q75": 197.58,
      "gaussian_fit": {
        "amplitude": 1234.5,
        "amplitude_err": 15.2,
        "mu": 197.236,
        "mu_err": 0.0075,
        "sigma": 0.521,
        "sigma_err": 0.0053,
        "fit_success": true
      }
    },
    "rise_amplitude": {
      "variable": "rise_amplitude",
      "count": 5000,
      "mean": 0.00912,
      "std": 0.00034,
      "sem": 0.0000048,
      "median": 0.00910,
      "min": 0.0082,
      "max": 0.0105,
      "q25": 0.00895,
      "q75": 0.00928
    }
  },
  "analysis_timestamp": "2025-12-18T10:30:45.123456",
  "bins_used": 50
}
```

**Key differences:**
- Stage 1 provides raw event lists and basic summary statistics
- Stage 2 adds comprehensive statistical analysis with errors and fits
- Stage 3 uses either format but benefits from Stage 2's error estimates
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
