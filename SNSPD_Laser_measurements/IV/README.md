# IV - Current-Voltage Characterization

Analysis tools for SNSPD current-voltage (I-V) characteristic measurements.

## Features

- **I-V curve fitting**: Automated fitting to resistive transition models
- **Critical current determination**: Extract switching current and width
- **Resistance measurements**: Normal and superconducting state resistance
- **Multi-temperature analysis**: Compare I-V curves at different temperatures
- **Publication plots**: HEP-style figures with proper error bars

## Scripts

### plotIV_matplot.py - Main I-V Plotter

Plot I-V characteristics with model fitting.

**Usage:**
```bash
python plotIV_matplot.py /path/to/iv_data.txt
```

### plotIV.py - Basic I-V Plot

Simple I-V curve plotter without fitting.

**Usage:**
```bash
python plotIV.py /path/to/iv_data.txt
```

### fitbaseline.py - Baseline Subtraction

Remove baseline offsets from I-V measurements.

**Usage:**
```bash
python fitbaseline.py /path/to/iv_data.txt
```

## Data Format

I-V data files should be space/tab-separated with columns:
```
current(uA)    voltage(V)
```

Example:
```
0.0     0.000
10.5    0.001
21.0    0.002
...
```

## Analysis Workflow

1. **Collect Data**: Measure I-V characteristics at different temperatures
2. **Baseline Correction**: `python fitbaseline.py data.txt` (if needed)
3. **Plot and Fit**: `python plotIV_matplot.py data.txt`
4. **Extract Parameters**: Critical current, normal resistance from fit

## Output

- **I-V curves**: Current vs voltage plots with fitted models
- **Resistance**: Calculated from linear regions
- **Critical current**: From transition fitting
- **Fit parameters**: Saved to output files

## Tips

- Ensure sufficient points in transition region for accurate fitting
- Check for heating effects at high currents
- Multiple sweeps help verify repeatability
- Temperature stability critical for accurate measurements

## Dependencies

- numpy, scipy: Numerical analysis and fitting
- matplotlib: Plotting
