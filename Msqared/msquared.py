from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# FOLDERS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "rawdata"
OUTPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# FILES TO INCLUDE
# file name, wavelength in nm
# ============================================================

DATA_FILES = [
    ("thorlabs.txt", 1576.293),
    # ("uniphase1023p.txt", 631.805),
    # ("uniphase1103p11777.txt", 631.805),
    # ("uniphase1103p110838.txt", 632.006),
    # ("uniphase1122p.txt", 632.208),
    # ("uniphase1507p0.txt", 634.822),
]

# ============================================================
# FUNCTIONS
# ============================================================

def load_knife_edge_file(file_path):
    z = []
    wx = []
    wy = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if "position" in line.lower() or "width" in line.lower():
                continue

            line = line.replace(";", ",")
            line = line.replace("\t", ",")
            line = line.replace(" mm", "")
            line = line.replace("mm", "")

            parts = line.split(",")

            if len(parts) < 3:
                continue

            try:
                z_val = float(parts[0].strip().replace(",", "."))
                wx_val = float(parts[1].strip().replace(",", "."))
                wy_val = float(parts[2].strip().replace(",", "."))

                z.append(z_val)
                wx.append(wx_val)
                wy.append(wy_val)

            except ValueError:
                continue

    z = np.array(z)
    wx = np.array(wx)
    wy = np.array(wy)

    sort_idx = np.argsort(z)

    return z[sort_idx], wx[sort_idx], wy[sort_idx]


def beam_radius_model(z, w0, z0, m2, wavelength_mm):
    return w0 * np.sqrt(
        1 + ((m2 * wavelength_mm * (z - z0)) / (np.pi * w0**2))**2
    )


def fit_m_squared(z_mm, width_mm, wavelength_nm):
    wavelength_mm = wavelength_nm * 1e-6

    # If your width is diameter, convert to radius
    w_mm = width_mm / 2

    min_idx = np.argmin(w_mm)

    p0 = [
        np.min(w_mm),       # w0
        z_mm[min_idx],      # z0
        1.5                 # M² start value
    ]

    bounds = (
        [0, np.min(z_mm) - 100, 0.1],
        [np.inf, np.max(z_mm) + 100, 100]
    )

    params, _ = curve_fit(
        lambda z, w0, z0, m2: beam_radius_model(
            z, w0, z0, m2, wavelength_mm
        ),
        z_mm,
        w_mm,
        p0=p0,
        bounds=bounds,
        maxfev=20000
    )

    w0, z0, m2 = params

    return w0, z0, m2, w_mm


# ============================================================
# PROCESS
# ============================================================

results = []

for filename, wavelength_nm in DATA_FILES:
    file_path = DATA_DIR / filename
    laser_name = Path(filename).stem

    print(f"\nProcessing: {laser_name}")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        continue

    z_mm, wx_mm, wy_mm = load_knife_edge_file(file_path)

    print(f"Loaded points: {len(z_mm)}")

    if len(z_mm) < 5:
        print("Not enough data points.")
        continue

    w0_x, z0_x, m2_x, radius_x = fit_m_squared(z_mm, wx_mm, wavelength_nm)
    w0_y, z0_y, m2_y, radius_y = fit_m_squared(z_mm, wy_mm, wavelength_nm)

    results.append((laser_name, wavelength_nm, w0_x, z0_x, m2_x, w0_y, z0_y, m2_y))

    print(f"M² x: {m2_x:.4f}")
    print(f"M² y: {m2_y:.4f}")

    z_fit = np.linspace(np.min(z_mm), np.max(z_mm), 1000)
    wavelength_mm = wavelength_nm * 1e-6

    fit_x = beam_radius_model(z_fit, w0_x, z0_x, m2_x, wavelength_mm)
    fit_y = beam_radius_model(z_fit, w0_y, z0_y, m2_y, wavelength_mm)

    plt.figure(figsize=(10, 6))

    plt.scatter(z_mm, radius_x, label="x data")
    plt.scatter(z_mm, radius_y, label="y data")

    plt.plot(z_fit, fit_x, label=f"x fit, M² = {m2_x:.2f}")
    plt.plot(z_fit, fit_y, label=f"y fit, M² = {m2_y:.2f}")

    plt.xlabel("Translation Stage Position / mm")
    plt.ylabel("Beam Radius / mm")
    plt.title(laser_name)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"{laser_name}_m_squared_fit.png", dpi=300)
    plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

with open(OUTPUT_DIR / "m_squared_results.txt", "w", encoding="utf-8") as file:
    file.write(
        "Laser\tWavelength_nm\t"
        "w0_x_mm\tz0_x_mm\tM2_x\t"
        "w0_y_mm\tz0_y_mm\tM2_y\n"
    )

    for r in results:
        file.write(
            f"{r[0]}\t{r[1]:.6f}\t"
            f"{r[2]:.6f}\t{r[3]:.6f}\t{r[4]:.6f}\t"
            f"{r[5]:.6f}\t{r[6]:.6f}\t{r[7]:.6f}\n"
        )

print("\nFinished.")
print(f"Results saved in: {OUTPUT_DIR.resolve()}")