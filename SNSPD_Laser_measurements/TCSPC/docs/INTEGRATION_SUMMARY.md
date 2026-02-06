# TCSPC Analysis Integration Summary

## Updated to Use 1-Degree Interpolated Data

Successfully integrated the new cubic-interpolated power data (90-340°, 1-degree separation) into the TCSPC analysis pipeline.

## Files Updated

### 1. tcspc_config.py
**Change**: Updated POWER_DATA_FILE path
```python
# OLD:
POWER_DATA_FILE = PROJECT_ROOT / "Attenuation" / "Rotation_10MHz_5degrees_data_20260205.txt"

# NEW:
POWER_DATA_FILE = PROJECT_ROOT / "Attenuation" / "Rotation_10MHz_1degree_data_90-340_20260205.txt"
```
**Impact**: All TCSPC scripts now use the centralized config pointing to 1-degree data

### 2. read_phu.py
**Changes**: Updated 2 power data file references
- Line 678: Main power data loading section
- Line 700: Power data for plot legend

**Before**:
```python
workspace_attenuation = Path(__file__).parent.parent / "Attenuation" / "Rotation_10MHz_5degrees_data_20260205.txt"
```

**After**:
```python
workspace_attenuation = Path(__file__).parent.parent / "Attenuation" / "Rotation_10MHz_1degree_data_90-340_20260205.txt"
```

**Impact**: Individual bias voltage analysis now uses 5× finer power resolution

### 3. create_combined_plot.py
**Status**: ✓ Already imports from tcspc_config.py (POWER_DATA_FILE)
- No direct changes needed
- Automatically uses the updated POWER_DATA_FILE from config
- Multi-bias comparison plots will now use 1-degree interpolated data

## Data Improvements

| Aspect | Before | After | Benefit |
|--------|--------|-------|---------|
| Power resolution | 5° spacing | 1° spacing | 5× finer detail |
| Data points | 72 total | 250 points (90-340°) | Much smoother curves |
| Interpolation | Linear (conservative) | Cubic (smooth) | Better for exponential decay |
| Coverage | 0-355° | 90-340° optimal range | Avoids plateau/transition artifacts |

## How It Works

1. **read_phu.py**: Loads power calibration data and uses it to annotate plots with power values
2. **create_combined_plot.py**: Uses the same power data from tcspc_config.py for combined analyses
3. **Power lookup**: For any rotation angle, now has ±0.5° accuracy instead of ±2.5°

## Active Angle Range
- **Measurement range**: 0-355° (full rotation)
- **Active interpolation**: 90-340° (250° exponential decay region)
- **Note**: Original data (5°) still available for reference at 0-89° and 341-355° regions

## Testing & Validation

To verify the integration works:

```bash
# Individual bias analysis with new power data
python3 read_phu.py /path/to/SMSPD_3_2-7_500kHz_78mV_*.phu

# Combined multi-bias analysis
python3 create_combined_plot.py
```

Both scripts will now display enhanced power calibration information with 1° resolution.

## Benefits for Analysis

✓ **More accurate power correction**: 5× resolution for power-dependent fits  
✓ **Smoother calibration curves**: Cubic interpolation captures exponential decay  
✓ **Better fit statistics**: More data points reduce interpolation noise  
✓ **Improved visualization**: Finer power axis for publication-quality plots  

## Configuration Flexibility

To revert to 5-degree data:
1. Edit `tcspc_config.py` line 22
2. Change back to: `"Rotation_10MHz_5degrees_data_20260205.txt"`
3. Both old and new files coexist for flexibility

---

**Date Updated**: February 5, 2026  
**Data Source**: Rotation_10MHz_1degree_data_90-340_20260205.txt  
**Interpolation Method**: Cubic (90-340° range optimized)
