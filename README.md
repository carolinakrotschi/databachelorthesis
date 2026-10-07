# Interferometer Data Analysis

Data analysis and figures for my bachelor thesis at the Max Planck Institute of Quantum Optics:
an interferometric calibration of a motorized translation stage, including a quadrature homodyne
extension that detects the direction of motion.

The measurement software lives in a separate repository:
[translation-stage-calibration](https://github.com/carolinakrotschi/translation-stage-calibration).
This repository contains the raw data, the Python analysis scripts and the resulting plots.

## Analyses

| Folder | What is analyzed |
|---|---|
| `laser_wavelengths/` | Spectrometer data of the candidate lasers: Gaussian peak fits, FWHM and coherence-length estimates |
| `beam_quality_m2/` | M² beam-quality factor from beam-caustic measurements (fit with `laserbeamsize`) |
| `laser_intensity_comparison/` | Intensity stability and signal-to-noise ratio of each laser |
| `homodyne/` | Quadrature homodyne signals: comparison of camera, single photodiode and homodyne detection |
| `accuracy/` | Sine/cosine fits to the two homodyne channels and forward/backward direction classification |
| `oscilloscope/` | Optical power traces with and without the stage moving |
| `reproducibility/` | Reproducibility measurements (spreadsheet) |
| `thesis_figures/` | Final figures as used in the thesis |
| `datashare/` | Clean, self-contained versions of the main analyses (see below) |

### `datashare/`: reproducible figures

Each of the six numbered analyses in `datashare/` is self-contained, with the same layout:

```
datashare/04_quadrature_homodyne_sine_cosine_fit/
├── code/       # one script, paths resolved relative to the script
├── rawdata/    # the measurement data it needs
└── results/    # generated plots and tables
```

Run any of them directly and the figure is regenerated into `results/`:

```bash
python -m pip install -r requirements.txt
python datashare/04_quadrature_homodyne_sine_cosine_fit/code/plot_quadrature_homodyne_fit.py
```

## Selected results

- The HeNe lasers show an almost ideal Gaussian beam (M² ≈ 1.0–1.03), while the 787 nm diode
  laser is clearly multimode (M² ≈ 2–4.5, depending on the axis).
- The 787 nm diode laser, which is also the laser used for fringe counting, has the best
  intensity stability of all tested sources (SNR ≈ 58 dB vs. 34–47 dB for the HeNe lasers).

## Tech

Python · NumPy · matplotlib · pandas · least-squares fitting · `laserbeamsize`
