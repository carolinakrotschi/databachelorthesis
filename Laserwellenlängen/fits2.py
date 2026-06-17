from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import math
import os

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "rawdata"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_FILE = "uniphase1507p_2_USB4C031521__0__12-07-47-856"


def load_spectrum(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    wavelengths = []
    intensities = []
    reading_data = False

    with open(file_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "Begin Spectral Data" in line:
                reading_data = True
                continue
            if not reading_data or not line:
                continue

            parts = line.replace(",", ".").split()
            if len(parts) < 2:
                continue

            try:
                wavelengths.append(float(parts[0]))
                intensities.append(float(parts[1]))
            except ValueError:
                continue

    return np.array(wavelengths), np.array(intensities)


def resolve_file(name: str) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate

    if candidate.suffix:
        return DATA_DIR / candidate.name

    return DATA_DIR / f"{name}.txt"


def fit_gaussian(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    peak_index = int(np.argmax(y))
    mu0 = float(x[peak_index])
    offset0 = float(np.min(y))
    amp0 = float(np.max(y) - offset0)

    half_max = offset0 + 0.5 * amp0
    left = peak_index
    while left > 0 and y[left] > half_max:
        left -= 1

    right = peak_index
    while right < len(y) - 1 and y[right] > half_max:
        right += 1

    if left == 0 or right == len(y) - 1:
        sigma0 = 1.0
    else:
        fwhm0 = max(float(x[right] - x[left]), 0.1)
        sigma0 = fwhm0 / 2.354820045

    xfit = x[(x > mu0 - 3.0) & (x < mu0 + 3.0)]
    yfit = y[(x > mu0 - 3.0) & (x < mu0 + 3.0)]

    mu_vals = np.linspace(mu0 - 0.2, mu0 + 0.2, 81)
    sigma_vals = np.linspace(max(0.05, sigma0 * 0.7), max(0.1, sigma0 * 1.3), 81)

    best_error = None
    best_mu = mu0
    best_sigma = sigma0
    best_amp = amp0
    best_offset = offset0

    for mu in mu_vals:
        for sigma in sigma_vals:
            g = np.exp(-((xfit - mu) ** 2) / (2.0 * sigma**2))
            amp, offset = np.linalg.lstsq(
                np.column_stack([g, np.ones_like(g)]),
                yfit,
                rcond=None,
            )[0]
            fit = amp * g + offset
            error = float(np.mean((yfit - fit) ** 2))
            if best_error is None or error < best_error:
                best_error = error
                best_mu = mu
                best_sigma = sigma
                best_amp = float(amp)
                best_offset = float(offset)

    fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * best_sigma
    return best_mu, fwhm, best_amp, best_offset


def plot_fit(
    x: np.ndarray,
    y: np.ndarray,
    peak: float,
    fwhm: float,
    amp: float,
    offset: float,
    title: str,
    out_file: Path,
) -> None:
    sigma = fwhm / 2.354820045
    fit = amp * np.exp(-((x - peak) ** 2) / (2.0 * sigma**2)) + offset
    half = offset + 0.5 * amp

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color="black", linewidth=1, label="spectrum")
    plt.plot(x, fit, color="red", linewidth=2, label="gaussian fit")
    plt.axhline(half, color="blue", linestyle="--", linewidth=1, label="half max")
    plt.axvline(peak - 0.5 * fwhm, color="blue", linestyle=":", linewidth=1)
    plt.axvline(peak + 0.5 * fwhm, color="blue", linestyle=":", linewidth=1)
    plt.text(
        peak,
        half,
        f"FWHM = {fwhm:.3f} nm",
        color="blue",
        ha="center",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    plt.xlim(peak - 2.0, peak + 2.0)
    plt.xlabel("Wavelength / nm")
    plt.ylabel("Intensity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def main() -> None:
    parser = ArgumentParser(description="Fit one spectrum with a Gaussian and save a zoomed plot.")
    parser.add_argument(
        "filename",
        nargs="?",
        default=DEFAULT_FILE,
        help="File name with or without .txt, e.g. uniphase1023p_2_USB4C031521__0__12-07-47-856",
    )
    args = parser.parse_args()

    spectrum_file = resolve_file(args.filename)
    if not spectrum_file.exists():
        raise FileNotFoundError(f"File not found: {spectrum_file}")

    x, y = load_spectrum(spectrum_file)
    if len(x) == 0:
        raise ValueError(f"No spectral data found in {spectrum_file}")

    peak, fwhm, amp, offset = fit_gaussian(x, y)
    out_file = RESULTS_DIR / f"{spectrum_file.stem}_gauss_zoom.png"
    plot_fit(x, y, peak, fwhm, amp, offset, spectrum_file.stem, out_file)

    print(f"File: {spectrum_file.name}")
    print(f"Peak: {peak:.4f} nm")
    print(f"FWHM: {fwhm:.4f} nm")
    print(f"Plot saved to: {out_file}")


if __name__ == "__main__":
    main()
