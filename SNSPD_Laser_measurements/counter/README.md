# Counter - Hardware Counter Analysis

Analysis tools for photon counting measurements using hardware counters (SR400, NI counter/timer, etc.).

## Features

- **Bias voltage scans**: Count rate vs bias for plateau characterization
- **Power dependence**: Detection efficiency vs optical power
- **Time-series analysis**: Monitor count rates over time
- **Dark count characterization**: Background rate analysis
- **Statistical analysis**: Proper error propagation with time binning
- **HEP-style plotting**: Publication-quality figures

## Quick Start

Run all analyses:
```bash
python run_all_counter_analysis.py
```

## Main Scripts

### plot_counter_generic.py - Unified Plotter

**Usage:**
```bash
# Basic usage
python plot_counter_generic.py /path/to/data_folder

# Specific bias voltages
python plot_counter_generic.py /path/to/data_folder --bias 68,70,72

# Specific powers (or "all")
python plot_counter_generic.py /path/to/data_folder --powers 369,446,534

# Custom y-axis scale
python plot_counter_generic.py /path/to/data_folder --yaxis-scale 0,10000

# Remove lowest power points
python plot_counter_generic.py /path/to/data_folder --remove-lowest 3
```

### Options:
- `--bias`: Comma-separated bias voltages in mV (default: 66,68,70,72,74)
- `--remove-lowest`: Number of lowest power points to remove (default: 0)
- `--tolerance`: Bias voltage tolerance in mV (default: 1.5)

### Examples:
```bash
# Analyze 1MHz data with specific biases
python plot_counter_generic.py /Users/ya/Documents/Projects/SNSPD/SNSPD_data/SMSPD_3/1MHz/2-7/6K --bias "68,70"

# Counter sweep with 3 lowest points removed
python plot_counter_generic.py /Users/ya/Documents/Projects/SNSPD/SNSPD_data/SMSPD_3/Counter_sweep_power_3/2-7/6K --bias "66,68,70,72,74" --remove-lowest 3
```

## Output

Each analysis generates a two-panel plot:

### Left Panel: Count Rate vs Bias Voltage
- **Dark-subtracted** count rates for each optical power
- Gray dotted line shows dark count level
- Title indicates "(Dark Subtracted)"

### Right Panel: Count Rate vs Power
- **Raw count rates** (no dark subtraction) at different bias voltages
- Linear fits with error metrics (R², RMSE, max residual)
- Gray dotted horizontal line shows dark count reference
- Equation format: `Rate = slope × P + intercept`

## Data Structure

Input files should be tab-separated text with:
- Column 1: Bias voltage
- Columns 7+: Individual 1s or 10s measurements

File naming: `SMSPD_3_2-7_{power}nW_{YYYYMMDD_HHMM}.txt`

Dark counts: Files with `0nW` in the name

## Scripts

- `plot_counter_generic.py`: Universal analysis script (recommended)
- `plot_counter_rates.py`: Counter_sweep_power_3 specific
- `plot_test_data.py`: Test folder specific
- `run_all_counter_analysis.py`: Batch process all folders

## Fit Quality Metrics

The analysis reports:
- **R²**: Coefficient of determination (>0.99 = excellent)
- **RMSE**: Root mean square error in counts/s
- **Relative RMSE**: RMSE as % of mean rate (<5% = good linear fit)
- **Max residual**: Largest deviation from fit line

### Typical Results:
- **Counter_sweep_power_3**: 0.3-0.8% RMSE (excellent linearity)
- **test data**: 0.4-0.8% RMSE (excellent linearity)
- **1MHz data**: 2.3% RMSE (very good linearity)
- **100kHz data**: 9.4% RMSE (moderate linearity, possible saturation effects)
