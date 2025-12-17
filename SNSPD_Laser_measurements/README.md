# SNSPD Laser Measurements

Analysis tools for superconducting nanowire single-photon detector (SNSPD) characterization using pulsed laser measurements.

## Repository Structure

```
SNSPD_Laser_measurements/
├── NIPXIe/          # TDMS waveform analysis (NI PXIe digitizer data)
├── counter/         # Counter-based measurements (SR400, etc.)
├── IV/              # Current-voltage characteristic analysis
├── Attenuation/     # Optical power attenuation measurements
├── old/             # Archived legacy code
├── plot_style.py    # Centralized HEP-style plotting configuration
└── README.md        # This file
```

## Analysis Modules

### NIPXIe - Waveform Analysis
Analysis of single-photon detection events from NI PXIe digitizer TDMS files.
- Event-by-event pulse analysis
- Detection efficiency calculations
- Timing jitter analysis
- Dark count characterization

See [NIPXIe/README.md](NIPXIe/README.md) for detailed usage.

### Counter - Rate Measurements
Analysis of photon counting measurements using hardware counters.
- Count rate vs bias voltage
- Count rate vs optical power
- Dark count rate analysis
- Dead time corrections

See [counter/README.md](counter/README.md) for detailed usage.

### IV - Electrical Characterization
Current-voltage characteristic analysis for SNSPDs.
- I-V curve fitting
- Critical current determination
- Resistance measurements

See [IV/README.md](IV/README.md) for detailed usage.

### Attenuation - Optical Power Calibration
Optical power measurements and calibration.
- Rotation stage power mapping
- Attenuation curve fitting

See [Attenuation/README.md](Attenuation/README.md) for detailed usage.

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone git@github.com:Hsin-Yeh/SNSPD_Analysis.git
   cd SNSPD_Analysis/SNSPD_Laser_measurements
   ```

2. **Install dependencies:**
   ```bash
   pip install -r NIPXIe/requirements.txt
   ```

3. **Run analysis:**
   - For TDMS waveform analysis: See [NIPXIe/README.md](NIPXIe/README.md)
   - For counter data: See [counter/README.md](counter/README.md)

## Plotting Style

All plotting scripts use a centralized HEP (High Energy Physics) style defined in `plot_style.py`:

```python
from plot_style import setup_hep_style
setup_hep_style()
```

This provides consistent, publication-quality plots with:
- Serif fonts
- Inward-facing ticks on all sides
- Minor ticks visible
- Appropriate margins and spacing

## Data Organization

Raw data and analyzed results should be organized as:

```
SNSPD_rawdata/          # Raw measurement files
  └── DeviceName/
      └── Laser/
          └── Configuration/
              └── YYYYMMDD/
                  └── *.tdms or *.txt

SNSPD_analyzed_json/    # Analysis results (JSON format)
  └── [mirrors raw data structure]
```

## Contributing

When adding new analysis scripts:
1. Use the centralized `plot_style.py` for consistent plotting
2. Follow the existing directory structure
3. Add appropriate README documentation
4. Use meaningful commit messages

## Citation

If you use this code in your research, please cite the relevant publications.
