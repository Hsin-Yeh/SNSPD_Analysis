# NIPXIe SNSPD Analysis Pipeline

Comprehensive analysis pipeline for superconducting nanowire single-photon detector (SNSPD) measurements using NI PXIe data acquisition hardware. This pipeline processes raw TDMS data files into statistical analyses and publication-quality comparison plots.

## Overview

The NIPXIe analysis pipeline is organized as a three-stage workflow that processes SNSPD measurement data:

1. **Stage 1 (SelfTrigger.py)**: Raw TDMS data → Event-by-event JSON analysis
2. **Stage 2 (analyze_events.py)**: Event JSON → Statistical summaries with error estimates
3. **Stage 3 (plot_all.py)**: Statistical summaries → Comparison plots and figures

## Core Components

### Stage 1: `SelfTrigger.py`
Processes raw TDMS (TDM Streaming) files from NI PXIe measurements and extracts event-level information.

**Key Features:**
- Reads TDMS data files from SNSPD measurements
- Extracts pulse characteristics (timing, amplitude, pulse width)
- Classifies events as true signal or dark counts based on laser sync timing
- Calculates detection rates and quantum efficiency
- Outputs event-by-event analysis to JSON files
- Supports debug and display modes for troubleshooting

**Output:** `event0_analysis.json` (or numbered variants)

### Stage 2: `analyze_events.py`
Performs statistical analysis on event-by-event JSON data with error propagation.

**Key Features:**
- Reads event JSON files from Stage 1
- Computes statistical summaries (mean, median, standard deviation, SEM)
- Fits histograms with Gaussian distributions
- Propagates uncertainties through calculations
- Groups data by bias voltage and optical power
- Generates intermediate plots for validation
- Outputs detailed statistics to JSON with error estimates

**Output:** `statistics_*.json` files

**Usage:**
```bash
# Full analysis with diagnostic plots
python analyze_events.py event0_analysis.json -d output/

# Batch processing without plots
python analyze_events.py *.json --no-plots
```

### Stage 3: `plot_all.py`
Generates comparison plots across multiple measurements.

**Key Features:**
- Reads statistical analysis results from Stage 2 (preferred) or Stage 1
- Groups data by optical power and bias voltage
- Generates comparison plots for rates vs bias/power
- Plots pulse characteristics (peak-to-peak, timing distribution)
- Exports figures as high-resolution PNG files
- Supports both `*_analysis.json` and `statistics_*.json` file formats

**Usage:**
```bash
# Use Stage 2 statistics files (recommended - includes error bars)
python plot_all.py -i output/ -p 'statistics_*.json'

# Use Stage 1 analysis files
python plot_all.py -i plots/test/ -p '*_analysis.json'
```

### Supporting Modules: `plot_statistics_vs_power_bias.py`
Generates detailed statistical comparison plots organized by bias and power.

**Plot Types:**
- Signal count rate vs bias (for each power level)
- Dark count rate vs power (for each bias level)
- Efficiency vs bias/power
- Pulse amplitude vs bias/power
- Error bars included from Stage 2 analysis

### Utilities (`utils/`)

#### `Timing_Analyzer.py`
Timing analysis functions using cubic spline interpolation and root-finding algorithms.
- `Get_Xs()`: Find crossing times at specific signal levels
- `Get_Function_Arrival()`: Calculate arrival times for signal transitions

#### `tdmsUtils.py`
TDMS file I/O and metadata extraction utilities.
- Read and parse TDMS files from NI hardware
- Extract group and channel information

#### `plot_utils.py`
Common plotting and data organization functions.
- `read_analysis_files()`: Parse analysis JSON files
- `group_by_power()`: Organize data by optical power
- `group_by_bias()`: Organize data by bias voltage
- `calculate_errors_from_events()`: Compute rate uncertainties using time-binning method

#### `plotUtilscopy.py`
Additional plotting utilities (legacy naming).

#### `osUtils.py`
Operating system and file management utilities.

## Data Structure

### Event JSON Format (Stage 1 Output)
```json
{
  "measurement_info": {
    "bias_voltage": 70.5,
    "power": 1000.0,
    "timestamp": "2024-01-15 10:30:00"
  },
  "events": [
    {
      "pulse_time": 1234.56,
      "rise_amplitude": -0.245,
      "fall_amplitude": -0.180,
      "pulse_width": 2.34,
      "laser_sync_arrival": 198.5,
      "event_type": "signal"
    }
  ],
  "summary_statistics": {
    "signal_count_rate": 45000.0,
    "dark_count_rate": 150.0,
    "efficiency": 0.62
  }
}
```

### Statistics JSON Format (Stage 2 Output)
```json
{
  "measurement_info": {...},
  "pulse_statistics": {
    "signal_count_rate": {
      "mean": 45000.0,
      "median": 44950.0,
      "std": 1200.0,
      "sem": 120.0
    },
    "efficiency": {
      "mean": 0.62,
      "std": 0.02,
      "sem": 0.002
    }
  }
}
```

## Key Analysis Parameters

- **Laser Sync Window**: 194-203 ns (configurable for true vs dark event classification)
- **Signal Window**: 20 ns (default time window for signal detection)
- **Source Rate**: 10 MHz (laser pulse repetition rate)
- **Bias Voltages**: Typically 66-74 mV range
- **Optical Powers**: Variable (nW range)

## Workflow Example

```bash
# Step 1: Extract events from raw TDMS files
python SelfTrigger.py raw_measurement_001.tdms -o output/events/

# Step 2: Analyze extracted events with statistics
python analyze_events.py output/events/event0_analysis.json -d output/stats/ -p

# Step 3: Generate comparison plots across all measurements
python plot_all.py -i output/stats/ -p 'statistics_*.json' -o output/plots/
```

## Dependencies

- Python 3.7+
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `scipy` - Curve fitting and optimization
- `matplotlib` - Plotting
- `nptdms` - TDMS file I/O (National Instruments format)

## Output Directory Structure

```
output/
├── events/
│   ├── event0_analysis.json
│   ├── event1_analysis.json
│   └── ...
├── stats/
│   ├── statistics_event0.json
│   ├── statistics_event1.json
│   └── ...
└── plots/
    ├── bias/
    │   ├── signal_count_rate/
    │   ├── dark_count_rate/
    │   └── efficiency/
    ├── power/
    │   └── [similar structure]
    ├── pulse_ptp_vs_bias_*.png
    ├── pulse_ptp_vs_power_*.png
    └── comparison_*.png
```

## Features & Capabilities

- **Event Classification**: Automatic separation of signal (laser-sync) and dark count events
- **Error Propagation**: Full uncertainty tracking from event level to final statistics
- **Multi-parameter Comparison**: Analyze dependencies on bias voltage, optical power, and time
- **Pulse Characterization**: Peak-to-peak amplitude, timing distribution, rise/fall times
- **Efficiency Calculation**: Quantum efficiency determination with error estimates
- **Batch Processing**: Analyze multiple measurements in sequence
- **Debug Support**: Optional verbose output and diagnostic plotting

## Notes

- Stage 2 (analyze_events.py) recommended over Stage 1 output for final plots due to error estimates
- Ensure TDMS file format compatibility with `nptdms` library
- Output directories are created automatically if they don't exist
- All timestamps are in nanoseconds unless otherwise specified
