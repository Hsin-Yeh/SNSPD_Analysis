# Counter Analysis Configuration System

## Overview

The counter analysis scripts now use a centralized JSON configuration file (`counter_analysis_config.json`) to manage measurement parameters. This makes it easier to:
- See all measurement configurations in one place
- Add new measurements without modifying code
- Track bias voltages and power levels for each dataset
- Automatically discover new measurement folders
- **Use flexible bias voltage selection:** specific values, all available, or percentage-based sampling

## Files

### `counter_analysis_config.json`
Central configuration file containing:
- **base_path**: Root directory for all measurements
- **measurements**: Dictionary of measurement configurations
  - **folder**: Relative path from base_path
  - **bias_voltages**: Bias voltages to analyze - **flexible options:**
    - **Specific values**: `[66, 68, 70, 72, 74]` - exact bias voltages in mV
    - **All available**: `"all"` - use all bias voltages found in data
    - **Percentage sampling**: `"50%"` or `"20%"` - evenly spaced percentage of available bias voltages
  - **remove_lowest_points**: Number of lowest power points to exclude from linear fit
  - **description**: Human-readable description
  - **_comment_bias**: Reference showing all available bias voltages in the data (read-only)
  - **_comment_power**: Reference showing all available power levels in the data (read-only)
  - **_examples**: Quick reference for bias voltage options

**To change which bias voltages are analyzed**, edit the `bias_voltages` field. The `_comment_*` fields show all available options from the actual data files.

### `scan_measurements.py`
Utility script to manage measurements:

**View current configuration:**
```bash
python scan_measurements.py
```

**Scan for new measurements:**
```bash
python scan_measurements.py --scan
```

**Auto-add new measurements to config:**
```bash
python scan_measurements.py --scan --add-new
```

**Use custom config file:**
```bash
python scan_measurements.py --config my_config.json
```

### `run_all_counter_analysis.py`
Updated to read from config file automatically. Simply run:
```bash
python run_all_counter_analysis.py
```

## Current Measurements

Based on `counter_analysis_config.json`:

### Counter_sweep_power_3
- **Selected Bias:** 66, 68, 70, 72, 74 mV
- **Available Bias:** 38, 42, 46, 50, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78 mV
- **Powers:** 369, 446, 534, 629, 785, 969, 1193, 1443 nW
- **Remove lowest:** 3 points
- Main counter sweep with 8 power levels

### test
- **Selected Bias:** 66, 68, 70, 72, 74 mV
- **Available Bias:** 38, 42, 46, 50, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78 mV
- **Powers:** 629, 785, 969, 1193, 1443 nW
- **Remove lowest:** 0 points
- Test measurement round

### 1MHz
- **Selected Bias:** 66, 68, 70, 72, 74 mV
- **Available Bias:** 38, 42, 46, 50, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78 mV
- **Powers:** 629, 785, 969, 1193, 1443, 1823, 2162, 2690, 3310, 4080, 5060, 6180 nW
- **Remove lowest:** 0 points
- 1 MHz repetition rate measurements

### 100kHz
- **Selected Bias:** 66, 68, 70, 72, 74 mV
- **Available Bias:** 38, 42, 46, 50, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78 mV
- **Powers:** 302, 369, 446, 534, 629, 785, 969, 1193, 1443, 1823, 2162, 2690 nW
- **Remove lowest:** 0 points
- 100 kHz repetition rate measurements

### 10kHz
- **Selected Bias:** 66, 68, 70, 72, 74 mV
- **Available Bias:** 38, 42, 46, 50, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78 mV
- **Powers:** 302, 369, 446, 534, 629, 785, 969, 1193, 1443, 1823, 2162, 2690, 3310, 4080, 5060, 6180 nW
- **Remove lowest:** 0 points
- 10 kHz repetition rate measurements

## Adding New Measurements

### Changing Bias Voltages or Power Levels

The `bias_voltages` field supports multiple selection modes:

**1. Specific values (default):**
```json
"bias_voltages": [66, 68, 70, 72, 74]
```
Analyzes only the specified bias voltages.

**2. All available bias voltages:**
```json
"bias_voltages": "all"
```
Analyzes all bias voltages found in the data files (typically 17 voltages from 38-78 mV).

**3. Percentage-based sampling:**
```json
"bias_voltages": "50%"   // Use 50% of available bias voltages, evenly spaced
"bias_voltages": "20%"   // Use 20% of available bias voltages, evenly spaced
```
Automatically selects the specified percentage of bias voltages with even spacing.

**Examples from current config:**
- **Counter_sweep_power_3**: `[66, 68, 70, 72, 74]` - specific 5 values
- **1MHz**: `"50%"` - uses ~8-9 evenly spaced values
- **100kHz**: `"20%"` - uses ~3-4 evenly spaced values  
- **10kHz**: `"all"` - uses all 17 available bias voltages

After editing, run:
```bash
python run_all_counter_analysis.py
```

### Automatic Method
1. Run scan to detect new folders:
   ```bash
   python scan_measurements.py --scan
   ```

2. If new measurements are found, add them automatically:
   ```bash
   python scan_measurements.py --scan --add-new
   ```

3. Edit `counter_analysis_config.json` to adjust:
   - `bias_voltages` (default: [66, 68, 70, 72, 74])
   - `remove_lowest_points` (default: 0)
   - `description`

4. Run analysis:
   ```bash
   python run_all_counter_analysis.py
   ```

### Manual Method
1. Open `counter_analysis_config.json`
2. Add new entry under `measurements`:
   ```json
   "your_measurement_name": {
     "folder": "your_measurement_name/2-7/6K",
     "bias_voltages": [66, 68, 70, 72, 74],
     "remove_lowest_points": 0,
     "description": "Your description here"
   }
   ```
3. Save and run `python run_all_counter_analysis.py`

## Workflow Example

When you have a new measurement folder at `/Users/ya/Documents/Projects/SNSPD/SNSPD_data/SMSPD_3/new_measurement/2-7/6K`:

```bash
# 1. Discover it
python scan_measurements.py --scan

# 2. Add it to config with defaults
python scan_measurements.py --scan --add-new

# 3. Edit counter_analysis_config.json if needed
# (adjust bias voltages, remove_lowest_points, etc.)

# 4. Run analysis
python run_all_counter_analysis.py
```

## Output

Analysis plots are saved to:
- `output/Counter_sweep_power_3/Counter_sweep_power_3_analysis.png`
- `output/test/test_analysis.png`
- `output/1MHz/1MHz_analysis.png`
- `output/100kHz/100kHz_analysis.png`
- etc.
