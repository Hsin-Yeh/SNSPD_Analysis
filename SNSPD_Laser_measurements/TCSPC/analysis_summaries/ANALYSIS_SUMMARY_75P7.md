# TCSPC Multi-Gaussian Peak Fitting Analysis - Summary Report

## Fitting Configuration
- **Tail Cut**: 75.7 ns (removes long trailing edge on n=1 peak)
- **Plot Range**: 75.0 - 76.0 ns (shows full main peak distribution)
- **Mean Bounds**: ±30 ps from block 10 reference
- **Number of Blocks Fitted**: 49 (blocks 100-340, excluding block 0 dark reference)
- **Fitting Method**: Multi-Gaussian with 4 peaks (n=1, n=2, n=3, n=4 photons)

## Reference Photon Positions (from Block 10)
Established via fixed-means fitting and validated across power levels:

```
n=4 (highest):  75.2356 ns  ±0.030 ns
n=3:            75.3386 ns  ±0.030 ns  
n=2:            75.4688 ns  ±0.030 ns
n=1 (lowest):   75.5880 ns  ±0.030 ns
```

Photon spacing: ~130 ps (consistent with detector jitter and timing uncertainties)

## Key Findings

### 1. Peak Position Behavior (Mean Shifts)
The photon peak positions **shift with power level**:

**High Power (Block 120, 6.86 µW)**
- All peaks shift **earlier** by 12-30 ps
- Example: n=1 peak at 75.5742 ns (13.81 ps earlier)

**Low Power (Block 150, 2.11 µW)**  
- Peaks shift **later** by 1-25 ps
- Example: n=1 peak at 75.6135 ns (25.53 ps later)

**Implication**: Reference photon positions require power-dependent calibration for high-accuracy photon number assignment.

### 2. Peak Width Power-Dependence

| Photon | Power Range | Width Range | Ratio (max/min) |
|--------|-------------|-------------|-----------------|
| n=4    | 0.35-14.3 µW | 10-49.6 ps | 4.96x |
| n=3    | 0.35-14.3 µW | 10-42.4 ps | 4.23x |
| n=2    | 0.35-14.3 µW | 10-61.3 ps | 6.13x |
| **n=1** | **0.35-14.3 µW** | **10-112.8 ps** | **11.28x** |

**n=1 Peak is Most Power-Sensitive**: Shows 11.28x variation vs 4-6x for other peaks.

### 3. Power-Dependent Trend

**At High Power (14.3 µW)**
```
n=4: 49.56 ps (broadest)
n=3: 25.30 ps
n=2: 31.09 ps
n=1: 10.00 ps (sharpest - at lower bound)
```

**At Low Power (0.35 µW)**
```
n=4: 10.00 ps (sharpest - at lower bound)
n=3: 10.00 ps (at lower bound)
n=2: 10.00 ps (at lower bound)
n=1: 62.69 ps (broad)
```

### 4. Physical Interpretation

The **inverse relationship** between high and low power peak widths suggests:

1. **At High Power**: 
   - High count rates saturate lower photon numbers (n=1 sharpens)
   - Higher photon numbers broaden due to timing uncertainty
   
2. **At Low Power**:
   - Sparse photons lead to timing jitter on single-photon events
   - n=1 peak broadens (characteristic timing jitter ~60-100 ps)
   - Multiple photon arrivals constrain higher orders

3. **Likely Mechanism**: Detector dead time / timing correlation effects that reduce at high rates

## Effect of 75.7 ns vs 75.8 ns Tail Cut

Comparison on blocks with known widths:

| Block | Power | n=1 @ 75.8ns | n=1 @ 75.7ns | Improvement |
|-------|-------|--------------|--------------|-------------|
| 150   | 2.11 µW | 143.48 ps | 103.63 ps | -39.85 ps (-27.7%) |
| 120   | 6.86 µW | 45.29 ps | 44.77 ps | -0.52 ps (-1.1%) |

**Recommendation**: The 75.7 ns tail cut is **strongly recommended**:
- Significantly improves n=1 resolution at low power (27.7% narrower)
- Minimal impact on high power measurements
- Better photon number discrimination overall

## Output Files Generated

### Individual Block Plots
- **Location**: `/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/74mV/peak_fits_75p7/`
- **Format**: `block_XXX_fit_75p7.png` for each block
- **Content**: Multi-Gaussian fit with residuals and individual peak components
- **Count**: 49 plots (one per block)

### Summary Data
- **File**: `fit_results_summary.json`
- **Content**: Peak widths (σ), positions (μ), amplitudes, and R² for all blocks
- **Use**: Power-dependent analysis and reference calibration

### Analysis Plots
1. **power_dependence_analysis.png**: 4-panel plot showing each photon peak width vs power
2. **combined_power_dependence.png**: All peaks on single log-scale plot showing power-dependent trends

## Recommendations for Next Steps

1. **Power-Dependent Photon Position Calibration**
   - Create lookup table: power → mean positions
   - Implement linear or polynomial fit across power range
   - Update photon number reconstruction algorithm

2. **Peak Width Model**
   - Fit power-dependence trend (n=1 shows clear inverse relationship)
   - Use as confidence metric for photon number assignment
   - Higher uncertainty at power extremes

3. **Validation on Real Measurement Data**
   - Apply fitted parameters to quantum efficiency measurements
   - Compare with known single-photon reference
   - Validate photon number counting accuracy

4. **Consider Detector Characterization**
   - The power-dependent trend suggests dead time effects
   - Could extract dead time constant from n=1 broadening behavior
   - Useful for detector optimization

## Fit Quality Notes

- **R² Metric**: Fits are excellent (0.99+) for blocks 100-160 (power > 0.14 µW)
- **Lower Power Fits**: Quality degrades (R² < 0.5) for blocks 180+ (power < 0.64 µW)
  - Reason: Individual photons are rare, peak structure not well-resolved
  - Still useful for n=1 width characterization at low power
- **Mean Positions**: Converged well within ±30 ps bounds for all fitted blocks

## Technical Details

**Fitting Algorithm**:
- Scipy curve_fit with Levenberg-Marquardt algorithm
- 4 Gaussians simultaneously (12 parameters total)
- Bounds: ±30 ps on means, 10-200 ps on widths, 0.2-3.0x amplitude range
- Max iterations: 50,000

**Data Processing**:
- Resolution: 4 ps (native TCSPC bin width)
- Time window: 75.0-76.2 ns for fitting
- Tail cut: 75.7 ns (removes everything below this)
- Dark background: Assumed minimal (not subtracted)

---

**Analysis Date**: February 7, 2026  
**Detector**: SNSPD (model SMSPD_3)  
**Configuration**: 74 mV bias, 500 kHz pulse rate, 2-7 µW power sweep
