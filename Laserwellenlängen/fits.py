from pathlib import Path
import csv
import math
import os

import numpy as np


folder = Path(__file__).resolve().parent / "rawdata"
out = Path(__file__).resolve().parent / "results"
out.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


pairs = [
    ("thorlabsCPS780S_2_NQ51B22081__0__11-36-36-140.txt", "backthorlabsCPS780S_2_NQ51B22081__0__11-35-54-129-1.txt"),
    ("uniphase1023p_USB4C031521__0__12-07-47-856.txt", "backuniphase1023p_USB4C031521__3__12-07-31-183.txt"),
    ("uniphase1103p11777_USB4C031521__0__12-13-38-733.txt", "backuniphase1103p11777_USB4C031521__0__12-12-07-080.txt"),
    ("uniphase1103p110838_USB4C031521__0__16-10-06-126.txt", "backuniphase1103p110838_USB4C031521__0__16-09-42-795.txt"),
    ("uniphase1122p_USB4C031521__0__16-13-14-685.txt", "backuniphase1122p_USB4C031521__0__16-12-54-022.txt"),
    ("uniphase1507p0_USB4C031521__0__16-18-35-607.txt", "backuniphase1507p0_USB4C031521__0__16-18-20-348.txt"),
]


def load(file_path):
    wl = []
    y = []
    read = False
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "Begin Spectral Data" in line:
                read = True
                continue
            if not read or not line:
                continue
            parts = line.replace(",", ".").split()
            if len(parts) >= 2:
                try:
                    wl.append(float(parts[0]))
                    y.append(float(parts[1]))
                except ValueError:
                    pass
    return np.array(wl), np.array(y)


def fit_gaussian(x, y):
    i = int(np.argmax(y))
    mu0 = float(x[i])
    base = float(np.min(y))
    half = base + 0.5 * (float(np.max(y)) - base)

    l = i
    while l > 0 and y[l] > half:
        l -= 1
    r = i
    while r < len(y) - 1 and y[r] > half:
        r += 1

    if l == 0 or r == len(y) - 1:
        sigma0 = 1.0
    else:
        fwhm0 = max(float(x[r] - x[l]), 0.1)
        sigma0 = fwhm0 / 2.354820045

    xfit = x[(x > mu0 - 3.0) & (x < mu0 + 3.0)]
    yfit = y[(x > mu0 - 3.0) & (x < mu0 + 3.0)]

    mu_vals = np.linspace(mu0 - 0.2, mu0 + 0.2, 81)
    sigma_vals = np.linspace(max(0.05, sigma0 * 0.7), max(0.1, sigma0 * 1.3), 81)

    best = None
    best_mu = mu0
    best_sigma = sigma0
    best_A = float(np.max(y) - np.min(y))
    best_offset = float(np.min(y))

    for mu in mu_vals:
        for sigma in sigma_vals:
            g = np.exp(-((xfit - mu) ** 2) / (2.0 * sigma**2))
            A, offset = np.linalg.lstsq(
                np.column_stack([g, np.ones_like(g)]),
                yfit,
                rcond=None,
            )[0]
            fit = A * g + offset
            err = float(np.mean((yfit - fit) ** 2))
            if best is None or err < best:
                best = err
                best_mu = mu
                best_sigma = sigma
                best_A = float(A)
                best_offset = float(offset)

    fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * best_sigma
    return best_mu, fwhm, best_A, best_offset


def plot_fit(x, y, peak, fwhm, A, offset, title, file_name):
    sigma = fwhm / 2.354820045
    fit = A * np.exp(-((x - peak) ** 2) / (2.0 * sigma**2)) + offset
    half = offset + 0.5 * A
    left = peak - 0.5 * fwhm
    right = peak + 0.5 * fwhm

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color="black", linewidth=1, label="spectrum")
    plt.plot(x, fit, color="red", linewidth=2, label="gaussian fit")
    plt.axhline(half, color="blue", linestyle="--", linewidth=1, label="half max")
    plt.axvline(left, color="blue", linestyle=":", linewidth=1)
    plt.axvline(right, color="blue", linestyle=":", linewidth=1)
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
    plt.savefig(out / file_name, dpi=300)
    plt.close()


rows = []

for signal_name, back_name in pairs:
    signal_file = folder / signal_name
    back_file = folder / back_name
    if not signal_file.exists() or not back_file.exists():
        continue

    wl, signal = load(signal_file)
    wl_back, back = load(back_file)
    if len(wl) != len(wl_back) or not np.allclose(wl, wl_back):
        continue

    corr = np.clip(signal - back, 0, None)
    peak, fwhm, A, offset = fit_gaussian(wl, corr)

    rows.append([signal_name.split("_")[0], peak, fwhm])
    plot_fit(wl, corr, peak, fwhm, A, offset, signal_name.split(".")[0], f"{signal_name.split('.')[0]}_fit.png")

    print(f"{signal_name}: peak={peak:.2f} nm, FWHM={fwhm:.4f} nm")


csv_file = out / "fwhm_table.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Laser", "Peak_nm", "FWHM_nm"])
    writer.writerows(rows)

print("\nTable saved:")
print(csv_file)
