# Attenuation - Optical Power Calibration

Tools for optical power measurements and calibration.

## Features

- **Rotation stage mapping**: Power vs rotation angle
- **Attenuation curves**: Fit optical density vs angle
- **Power calibration**: Convert angles to absolute power levels
- **HEP-style plots**: Publication-quality figures

## Scripts

### plot_rotation_power.py - Rotation Power Mapper

Plot optical power as a function of rotation stage angle.

**Usage:**
```bash
python plot_rotation_power.py Rotation_10MHz_5degrees_data.txt
```

## Data Format

Rotation data files should be space/tab-separated:
```
angle(degrees)    power(nW)    power_error(nW)
```

Example (`Rotation_10MHz_5degrees_data.txt`):
```
0      1000.0    10.0
5      794.3     8.0
10     630.9     6.3
15     501.2     5.0
...
```

## Analysis Workflow

1. **Collect Data**: Measure power at different rotation angles
2. **Plot Calibration**: `python plot_rotation_power.py data.txt`
3. **Fit Model**: Extract attenuation parameters
4. **Use Calibration**: Convert angles to powers in experiments

## Output

- **Power vs Angle**: Calibration curve with fitted model
- **Attenuation Factor**: Optical density per degree
- **Interpolation Table**: For converting angles to powers

## Tips

- Measure at least 10-15 angles for good calibration
- Include error estimates from power meter
- Verify linearity in log scale for exponential attenuation
- Re-calibrate periodically to account for alignment drift

## Dependencies

- numpy, scipy: Numerical analysis and fitting
- matplotlib: Plotting (uses HEP style from `../plot_style.py`)
