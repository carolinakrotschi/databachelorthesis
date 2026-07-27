#!/usr/bin/env python3
"""
Fit Raw_S1_V with a sine and Raw_S2_V with a cosine.

Expected CSV columns:
Relative_Time_s,Raw_S1_V,Raw_S2_V
"""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "results" / "homodyneforward1_383_393.csv"


def estimate_frequency(t: np.ndarray, y: np.ndarray) -> float:
    """Estimate a dominant frequency from a detrended FFT."""
    if len(t) < 4:
        return 1.0

    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return 1.0

    trend = np.polyval(np.polyfit(t, y, 1), t)
    y0 = y - trend
    freqs = np.fft.rfftfreq(len(y0), d=dt)
    spectrum = np.abs(np.fft.rfft(y0))
    if len(freqs) < 2:
        return 1.0

    peak_index = int(np.argmax(spectrum[1:]) + 1)
    peak_freq = float(freqs[peak_index])
    return peak_freq if peak_freq > 0 else 1.0


def solve_channel(t: np.ndarray, y: np.ndarray, omega: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit y = c + m*t + a*sin(omega*t) + b*cos(omega*t)."""
    design = np.column_stack(
        [
            np.sin(omega * t),
            np.cos(omega * t),
            t,
            np.ones_like(t),
        ]
    )
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fit = design @ coeffs
    return coeffs, fit, float(np.sum((y - fit) ** 2))


def trig_description(coeffs: np.ndarray, omega: float, kind: str) -> str:
    a, b, m, c = coeffs
    amp = float(np.hypot(a, b))
    phase = float(np.arctan2(b, a))
    if kind == "sin":
        return (
            f"{c:.9g} + ({m:.9g})*t + "
            f"{amp:.9g}*sin({omega:.9g}*t + {phase:.9g})"
        )
    return (
        f"{c:.9g} + ({m:.9g})*t + "
        f"{amp:.9g}*cos({omega:.9g}*t - {phase:.9g})"
    )


def r_squared(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv",
        nargs="?",
        default=str(DEFAULT_CSV),
        help="Input CSV file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(BASE_DIR / "results" / "sin_cos_fit.png"),
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    t_abs_list = []
    y1_list = []
    y2_list = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = ["Relative_Time_s", "Raw_S1_V", "Raw_S2_V"]
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        for row in reader:
            try:
                t_abs_list.append(float(row["Relative_Time_s"]))
                y1_list.append(float(row["Raw_S1_V"]))
                y2_list.append(float(row["Raw_S2_V"]))
            except (TypeError, ValueError):
                continue

    if not t_abs_list:
        raise ValueError(f"No valid data rows found in {csv_path}")

    order = np.argsort(t_abs_list)
    t_abs = np.asarray(t_abs_list, dtype=float)[order]
    t = t_abs - t_abs[0]
    y1 = np.asarray(y1_list, dtype=float)[order]
    y2 = np.asarray(y2_list, dtype=float)[order]

    f0 = estimate_frequency(t, y1)
    f1 = estimate_frequency(t, y2)
    f_center = 0.5 * (f0 + f1)

    duration = float(t[-1] - t[0])
    if duration <= 0:
        raise ValueError("Time axis must span a positive range.")

    dt = float(np.median(np.diff(t)))
    f_low = max(0.05, 0.5 / duration)
    f_high = min(10.0, 0.45 / dt if dt > 0 else 10.0)

    if f_center <= 0:
        f_center = 1.0

    lower = max(f_low, 0.6 * f_center)
    upper = min(f_high, 1.4 * f_center)
    if lower >= upper:
        lower, upper = f_low, f_high

    freqs = np.linspace(lower, upper, 20000)
    best = None
    best_params = None
    best_fit1 = None
    best_fit2 = None

    for freq in freqs:
        omega = 2.0 * np.pi * freq
        coeffs1, fit1, rss1 = solve_channel(t, y1, omega)
        coeffs2, fit2, rss2 = solve_channel(t, y2, omega)
        rss = rss1 / max(np.var(y1), 1e-12) + rss2 / max(np.var(y2), 1e-12)
        if best is None or rss < best:
            best = rss
            best_params = (omega, coeffs1, coeffs2)
            best_fit1 = fit1
            best_fit2 = fit2

    if best_params is None or best_fit1 is None or best_fit2 is None:
        raise RuntimeError("No fit solution found.")

    w, coeffs1, coeffs2 = best_params
    f = w / (2 * np.pi)
    period = 1 / f
    fit1 = best_fit1
    fit2 = best_fit2

    print(f"Gemeinsame Frequenz: {f:.8f} Hz")
    print(f"Periode:             {period:.8f} s")
    print(f"S1: {trig_description(coeffs1, w, 'sin')}")
    print(f"S2: {trig_description(coeffs2, w, 'cos')}")
    print(f"R² S1: {r_squared(y1, fit1):.6f}")
    print(f"R² S2: {r_squared(y2, fit2):.6f}")

    # Dense curves for smooth plotting.
    td = np.linspace(t.min(), t.max(), 3000)
    design = np.column_stack([np.sin(w * td), np.cos(w * td), td, np.ones_like(td)])
    fit1d = design @ coeffs1
    fit2d = design @ coeffs2
    xd = td + t_abs[0]

    # Remove linear trend (m*t) from both raw data and fit lines
    m1 = coeffs1[2]
    m2 = coeffs2[2]

    y1_plot = y1 - m1 * t
    y2_plot = y2 - m2 * t

    fit1d_plot = fit1d - m1 * td
    fit2d_plot = fit2d - m2 * td

    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18
    })

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t_abs, y1_plot, "-", linewidth=1.0, alpha=0.45, label="Raw S1 line")
    axes[0].plot(t_abs, y1_plot, ".", ms=3, label="Raw S1 points")
    axes[0].plot(xd, fit1d_plot, linewidth=2.2, color="#d62728", label="Sinus-Fit")
    axes[0].set_ylabel("S1 [V]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title(
        f"Gemeinsamer Fit: f = {f:.5f} Hz, T = {period:.5f} s"
    )

    axes[1].plot(t_abs, y2_plot, "-", linewidth=1.0, alpha=0.45, label="Raw S2 line")
    axes[1].plot(t_abs, y2_plot, ".", ms=3, label="Raw S2 points")
    axes[1].plot(xd, fit2d_plot, linewidth=2.2, color="#d62728", label="Cosinus-Fit")
    axes[1].set_xlabel("Relative Time [s]")
    axes[1].set_ylabel("S2 [V]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for ax in axes:
        ax.axvline(388.0, color="black", linestyle="--", linewidth=1.2)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Plot gespeichert: {args.output}")


if __name__ == "__main__":
    main()
