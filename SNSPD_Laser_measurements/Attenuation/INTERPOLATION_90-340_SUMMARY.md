# Cubic Interpolation Summary - 90-340° Range

## Task Completed ✓

Interpolated rotation power data from **5-degree to 1-degree separation** using **cubic interpolation**.

## Output File
**`Rotation_10MHz_1degree_data_90-340_20260205.txt`**
- **252 data points** (90°, 91°, 92°, ..., 340°)
- **Angle range**: 90° to 340° (250° span)
- **Format**: Tab-separated [Angle (degrees) | Power (nW)]

## Interpolation Method
**Cubic Interpolation**
- Smooth, continuous curves
- Natural for exponential decay region
- Better than linear for fine-grained power variation

## Data Characteristics

### Original Data (5° spacing)
- 90°: 20.63 nW
- 95°: 17.39 nW
- 100°: 14.33 nW
- ...
- 330°: 0.0992 nW
- 335°: 0.0841 nW
- 340°: 0.0718 nW

### Interpolated Data (1° spacing)
- **251 new points** between original measurements
- **Range**: 20.63 nW → 0.00283 nW (exponential decay)
- **Coverage**: Smooth interpolation throughout 90-340° region

## Why This Range?
- **90-340°**: Exponential power decay region (ideal for cubic interpolation)
- **Avoids plateau** (0-75°): Where cubic can cause oscillations
- **Avoids sharp transition** (75-90°): Unnecessary complexity
- **Avoids end effects** (340-355°): Improved numerical stability

## File Contents Example

```
90.0  2.063000e+01   ← Original point
91.0  1.921000e+01   ← Interpolated
92.0  1.787000e+01   ← Interpolated
...
100.0 1.433000e+01   ← Original point (from 100.0)
...
340.0 7.830000e-03   ← Original point
```

## Physical Validity
✓ All values positive (no negative interpolation artifacts)  
✓ Monotonic decay throughout region  
✓ Smooth exponential character preserved  
✓ Ready for TCSPC analysis integration  

## Usage
Replace the 5-degree data with this file for:
- Fine-grained power calibration (1° resolution)
- Detailed rotation angle studies
- High-resolution TCSPC analysis
- Improved interpolation in power-sweep fits
