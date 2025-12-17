# SNSPD Analysis Workflow

## Three-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: Event Extraction                    │
│                         SelfTrigger.py                               │
├─────────────────────────────────────────────────────────────────────┤
│ Input:  TDMS files (raw oscilloscope waveforms)                     │
│ Action: - Detect pulses in waveform data                            │
│         - Extract timing, amplitude, rise/fall times                │
│         - Calculate rates and efficiency                            │
│ Output: *_analysis.json (event-by-event physics variables)          │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: Statistical Analysis                     │
│                       analyze_events.py                              │
├─────────────────────────────────────────────────────────────────────┤
│ Input:  *_analysis.json (from Stage 1)                              │
│ Action: - Compute mean, std, standard error for all variables       │
│         - Gaussian fit for trigger_check timing (μ ± σ)              │
│         - Calculate uncertainties and error propagation             │
│         - Optional: Generate diagnostic plots                       │
│ Output: statistics_*.json (fitted parameters with errors)           │
│         Optional: Histogram plots, correlation plots                │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     STAGE 3: Comparison Plots                        │
│                          plot_all.py                                 │
├─────────────────────────────────────────────────────────────────────┤
│ Input:  statistics_*.json (from Stage 2) or *_analysis.json         │
│ Action: - Group measurements by bias/power                          │
│         - Plot rate vs bias (multi-power comparison)                │
│         - Plot rate vs power (multi-bias comparison)                │
│         - Plot pulse characteristics                                │
│ Output: Comparison plots with error bars                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start Examples

### Full Pipeline
```bash
# Process all TDMS files in directory
python SelfTrigger.py ~/SNSPD_data/ --recursive -d stage1_output/

# Compute statistics (batch mode, no plots)
python analyze_events.py stage1_output/*_analysis.json --no-plots -d stage2_stats/

# Generate comparison plots
python plot_all.py -i stage2_stats/ -p 'statistics_*.json' -d final_plots/
```

### Single File Analysis with Diagnostics
```bash
# Stage 1: Extract events
python SelfTrigger.py data.tdms -d output/

# Stage 2: Full statistical analysis with plots
python analyze_events.py output/*_analysis.json -d diagnostics/

# Stage 3: Skip if only analyzing one file
```

### Debugging Workflow
```bash
# Extract specific event for inspection
python SelfTrigger.py data.tdms --save_single_event 42 -d debug/

# View waveform with plots
python SelfTrigger.py data.tdms --checkSingleEvent 42 --display_report

# Analyze with fine binning for better fits
python analyze_events.py output/event_analysis.json -b 200 -d high_res/
```

## Data Flow

```
Raw Data          Event Data              Statistics           Final Plots
--------          ----------              ----------           -----------
data.tdms    →    event0_analysis.json → statistics_event0.json → comparison.png
  (MB-GB)           (KB-MB)                  (KB)                  (publication)
                    
  Contains:         Contains:               Contains:             Shows:
  - Waveforms       - Pulse amplitudes      - Mean ± SEM          - Multi-condition
  - Timestamps      - Timing values         - Gaussian fits         comparison
  - Metadata        - Event counts          - Error estimates     - Error bars
                    - Summary stats         - Quartiles           - Trends
```

## Key Features by Stage

### Stage 1: SelfTrigger.py
- ✓ Handles large TDMS files efficiently
- ✓ Configurable trigger detection (spline/simple)
- ✓ Recursive directory processing
- ✓ Single event extraction for debugging
- ✓ Progress reporting for long runs

### Stage 2: analyze_events.py
- ✓ Comprehensive error propagation
- ✓ Gaussian fitting with uncertainties (trigger_check)
- ✓ Batch mode (--no-plots) for HPC workflows
- ✓ Percentile and quartile calculations
- ✓ Diagnostic plots for quality control

### Stage 3: plot_all.py
- ✓ Automatic grouping by bias/power
- ✓ Publication-quality HEP-style plots
- ✓ Error bar propagation from Stage 2
- ✓ Multiple plot modes (vs_bias, vs_power, pulse)
- ✓ Multi-directory support

## When to Use Each Stage

| Scenario | Run Stages | Notes |
|----------|------------|-------|
| Initial data processing | 1 only | Get event data quickly |
| Detailed single-file analysis | 1 → 2 (with plots) | Use for debugging, quality checks |
| Batch processing (many files) | 1 → 2 (--no-plots) | Fast statistics computation |
| Multi-condition comparison | 1 → 2 → 3 | Complete publication workflow |
| Re-analyze with different bins | 2 only | Use existing event data |
| Update comparison plots | 3 only | Use existing statistics |

## Tips and Best Practices

1. **Use `--no-plots` in Stage 2 for batch jobs** - 10x faster when processing many files
2. **Stage 2 statistics_*.json preferred for Stage 3** - Contains error estimates
3. **Organize by experiment**: Create subdirectories for stage1/, stage2/, final_plots/
4. **Check trigger_check Gaussian fit** - Should have μ ≈ 197 ns, verify fit quality
5. **Use `--recursive` cautiously** - May process more files than intended
6. **Save raw TDMS files separately** - They're large; don't mix with analysis outputs

## File Naming Convention

```
Original:     SMSPD_3_event0_Pulse_515_629nW_0degrees_32767uA_63mV_20251209.tdms
Stage 1:      SMSPD_3_event0_Pulse_515_629nW_0degrees_32767uA_63mV_20251209_analysis.json
Stage 2:      statistics_SMSPD_3_event0_Pulse_515_629nW_0degrees_32767uA_63mV_20251209_analysis.json
Stage 3:      output/SMSPD_3/rates_vs_bias_multi_power.png
```

## Output Directory Structure

Recommended organization:
```
project/
├── raw_data/
│   └── *.tdms                    # Original data
├── stage1_events/
│   └── *_analysis.json           # Event-by-event data
├── stage2_statistics/
│   ├── statistics_*.json         # Statistical results
│   └── diagnostics/              # Optional diagnostic plots
│       ├── *_histogram_*.png
│       └── correlation_*.png
└── final_plots/
    ├── output/
    │   └── SMSPD_3/
    │       ├── rates_vs_bias_*.png
    │       └── rates_vs_power_*.png
    └── publication_figures/
```
