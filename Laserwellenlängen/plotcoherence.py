from pathlib import Path
import csv
import math
import os

import numpy as np

out = Path(__file__).resolve().parent / "results"
out.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


folder = Path(__file__).resolve().parent / "rawdata"

pairs = [
    ("thorlabsCPS780S_2_NQ51B22081__0__11-36-36-140.txt", "backthorlabsCPS780S_2_NQ51B22081__0__11-35-54-129-1.txt"),
    ("uniphase1023p_USB4C031521__0__12-07-47-856.txt", "backuniphase1023p_USB4C031521__3__12-07-31-183.txt"),
    ("uniphase1103p11777_USB4C031521__0__12-13-38-733.txt", "backuniphase1103p11777_USB4C031521__0__12-12-07-080.txt"),
    ("uniphase1103p110838_USB4C031521__0__16-10-06-126.txt", "backuniphase1103p110838_USB4C031521__0__16-09-42-795.txt"),
    ("uniphase1122p_USB4C031521__0__16-13-14-685.txt", "backuniphase1122p_USB4C031521__0__16-12-54-022.txt"),
    ("uniphase1507p0_USB4C031521__0__16-18-35-607.txt", "backuniphase1507p0_USB4C031521__0__16-18-20-348.txt"),
]


def load(file_path):
    wl, y = [], []
    read = False
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "Begin Spectral Data" in line:
                read = True
                continue
            if not read or not line:
                continue
            a = line.replace(",", ".").split()
            if len(a) >= 2:
                try:
                    wl.append(float(a[0]))
                    y.append(float(a[1]))
                except ValueError:
                    pass
    return wl, y


def fit_gaussian(wl, y):
    x = np.asarray(wl, dtype=float)
    z = np.asarray(y, dtype=float)

    i = int(np.argmax(z))
    mu0 = x[i]

    base = float(np.min(z))
    half = base + 0.5 * (float(np.max(z)) - base)
    l = i
    while l > 0 and z[l] > half:
        l -= 1
    r = i
    while r < len(z) - 1 and z[r] > half:
        r += 1
    if l == 0 or r == len(z) - 1:
        sigma0 = 1.0
    else:
        fwhm0 = max(x[r] - x[l], 0.1)
        sigma0 = fwhm0 / 2.354820045

    mask = (x > mu0 - 3.0) & (x < mu0 + 3.0)
    x = x[mask]
    z = z[mask]
    if len(x) < 5:
        raise ValueError("not enough points")

    mu_vals = np.linspace(mu0 - 0.2, mu0 + 0.2, 81)
    sigma_vals = np.linspace(max(0.05, sigma0 * 0.7), max(0.1, sigma0 * 1.3), 81)

    best_err = None
    best_mu = mu0
    best_sigma = max(sigma0, 1.0)
    best_A = float(np.max(z) - np.min(z))
    best_offset = float(np.min(z))

    for mu in mu_vals:
        for sigma in sigma_vals:
            g = np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))
            A, offset = np.linalg.lstsq(
                np.column_stack([g, np.ones_like(g)]),
                z,
                rcond=None,
            )[0]
            fit = A * g + offset
            err = float(np.mean((z - fit) ** 2))
            if best_err is None or err < best_err:
                best_err = err
                best_mu = mu
                best_sigma = sigma
                best_A = float(A)
                best_offset = float(offset)

    fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * best_sigma
    return best_mu, fwhm, best_A, best_offset


def plot_spectrum_with_fit(wl, corr, peak, fwhm, A, offset, title, out_file):
    x = np.asarray(wl, dtype=float)
    y = np.asarray(corr, dtype=float)
    sigma = fwhm / 2.354820045
    fit = A * np.exp(-((x - peak) ** 2) / (2.0 * sigma**2)) + offset
    zoom = 2.0
    half = offset + 0.5 * A
    left = peak - 0.5 * fwhm
    right = peak + 0.5 * fwhm

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="corrected spectrum", color="black", linewidth=1)
    plt.plot(x, fit, label="gaussian fit", color="red", linewidth=2)
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
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    plt.xlim(peak - zoom, peak + zoom)
    plt.xlabel("Wavelength / nm")
    plt.ylabel("Intensity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


rows = []

for s_name, b_name in pairs:
    s_file = folder / s_name
    b_file = folder / b_name
    if not s_file.exists() or not b_file.exists():
        continue

    wl, signal = load(s_file)
    wl_b, back = load(b_file)
    if wl != wl_b or len(signal) != len(back):
        continue

    corr = [max(s - b, 0.0) for s, b in zip(signal, back)]
    peak, width, A, offset = fit_gaussian(wl, corr)

    
    # eigentliche Formel aus der Mail:
    # Lc = lambda^2 / FWHM
    lc_mm = (peak * peak / width) * 1e-6 if width > 0 else float("inf")

    rows.append([s_name.split("_")[0], peak, width, lc_mm])
    print(f"{s_name}: λ={peak:.2f} nm, FWHM={width:.4f} nm, Lc={lc_mm:.4f} mm")

    plot_spectrum_with_fit(
        wl,
        corr,
        peak,
        width,
        A,
        offset,
        s_name.split(".")[0],
        out / f"{s_name.split('.')[0]}_gaussian_fit.png",
    )


csv_file = out / "coherence_lengths.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Laser", "Wavelength_nm", "FWHM_nm", "CoherenceLength_mm"])
    writer.writerows(rows)

print("\nErgebnisse gespeichert:")
print(csv_file)
