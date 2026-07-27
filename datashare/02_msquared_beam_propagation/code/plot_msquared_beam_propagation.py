import os
from pathlib import Path
from tempfile import gettempdir

SCRIPT_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = Path(os.environ.get("MPLCONFIGDIR", Path(gettempdir()) / "msquared-matplotlib"))
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_DIR)

import numpy as np
import laserbeamsize as lbs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = SCRIPT_DIR.parent / "rawdata"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

THESIS_LABEL_SIZE = 18
THESIS_TICK_SIZE = 16
THESIS_LEGEND_SIZE = 16

# (raw data file, display name, laser wavelength in nm) for the M^2 fit
DATA_FILES = [
    ("thorlabs_cps780s.txt", "thorlabs", 787.324),
    ("uniphase_1023p.txt", "uniphase1023p", 631.805000),
    ("uniphase_1103p_1177761.txt", "uniphase1103p1177761", 631.805000),
    ("uniphase_1103p_1108380.txt", "uniphase1103p1108380", 632.006000),
    ("uniphase_1122p.txt", "uniphase1122p", 632.208000),
    ("uniphase_1507p.txt", "uniphase1507p0", 634.822000),
]


# Parse "z d1 d2" beam-width measurements, skipping header/unit lines
def read_txt_data(file_path):
    rows = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line = line.replace("mm", "").replace("≈", "").replace("~", "")
            lower = line.lower()
            if any(word in lower for word in ["position", "width", "radius", "z", "d1", "d2"]):
                continue

            parts = line.replace(";", "\t").replace(",", "\t").split()
            if len(parts) < 3:
                continue

            try:
                z = float(parts[0])
                d1 = float(parts[1])
                d2 = float(parts[2])
                rows.append([z, d1, d2])
            except ValueError:
                continue

    data = np.array(rows)
    data = data[np.argsort(data[:, 0])]

    z1_all = data[:, 0] * 1e-3
    d1_all = data[:, 1] * 1e-3
    d2_all = data[:, 2] * 1e-3

    return z1_all, d1_all, d2_all


# Fit the ISO 11146 M^2 beam-propagation model separately to both
# orthogonal beam-width axes (d1, d2), then build a dense curve for plotting.
def analyze_file(file_path, laser_name, wavelength_nm):
    wavelength_m = wavelength_nm * 1e-9
    z1_all, d1_all, d2_all = read_txt_data(file_path)

    t_valueM2d1 = lbs.M2_fit(z1_all, d1_all, wavelength_m, strict=True)
    t_valueM2d2 = lbs.M2_fit(z1_all, d2_all, wavelength_m, strict=True)

    a_fitd1 = t_valueM2d1[0]
    a_fitd2 = t_valueM2d2[0]

    # d0 (waist), z0 (waist position), theta (divergence), M2 (beam quality)
    v_dnewd1, v_znewd1, v_thetanewd1, v_M2newd1 = a_fitd1[0], a_fitd1[1], a_fitd1[2], a_fitd1[3]
    v_dnewd2, v_znewd2, v_thetanewd2, v_M2newd2 = a_fitd2[0], a_fitd2[1], a_fitd2[2], a_fitd2[3]

    # Hyperbolic beam-width model evaluated on the measured z positions
    a_d1new = np.sqrt(v_dnewd1**2 + v_thetanewd1**2 * (z1_all - v_znewd1)**2)
    a_d2new = np.sqrt(v_dnewd2**2 + v_thetanewd2**2 * (z1_all - v_znewd2)**2)

    # Finer z grid for a smooth fit curve
    step = (z1_all[1] - z1_all[0]) * 1e-2
    a_z1new = np.arange(np.min(z1_all), np.max(z1_all), step)

    a_d1newlong = np.interp(a_z1new, z1_all, a_d1new)
    a_d2newlong = np.interp(a_z1new, z1_all, a_d2new)

    return {
        "laser": laser_name,
        "wavelength_nm": wavelength_nm,
        "M2_d1": v_M2newd1,
        "M2_d2": v_M2newd2,
        "z0_d1_m": v_znewd1,
        "z0_d2_m": v_znewd2,
        "z_data": z1_all,
        "d1_data": d1_all,
        "d2_data": d2_all,
        "z_fit": a_z1new,
        "d1_fit": a_d1newlong,
        "d2_fit": a_d2newlong,
    }


# Run the M^2 fit for every laser that has raw data available
summary_results = []
for filename, laser_name, wavelength_nm in DATA_FILES:
    file_path = DATA_DIR / filename
    if file_path.exists():
        summary_results.append(analyze_file(file_path, laser_name, wavelength_nm))

# Combined figure: all lasers, both beam axes (solid/dashed), fit curves + data points
if summary_results:
    plt.figure(figsize=(14, 8))
    colors = plt.get_cmap("tab10")
    centered_z_sets = []

    for idx, r in enumerate(summary_results):
        color = colors(idx % 10)
        z_fit_d1 = r["z_fit"] - r["z0_d1_m"]
        z_fit_d2 = r["z_fit"] - r["z0_d2_m"]
        z_data_d1 = r["z_data"] - r["z0_d1_m"]
        z_data_d2 = r["z_data"] - r["z0_d2_m"]
        centered_z_sets.extend([z_fit_d1, z_fit_d2, z_data_d1, z_data_d2])

        plt.plot(
            z_fit_d1,
            r["d1_fit"],
            color=color,
            linewidth=2,
            linestyle="-",
            label=f"{r['laser']} d1, M2 = {r['M2_d1']:.2f}",
        )
        plt.plot(
            z_fit_d2,
            r["d2_fit"],
            color=color,
            linewidth=2,
            linestyle="--",
            label=f"{r['laser']} d2, M2 = {r['M2_d2']:.2f}",
        )

        plt.scatter(
            z_data_d1,
            r["d1_data"],
            color=color,
            marker="o",
            s=18,
            alpha=0.75,
            zorder=3,
        )
        plt.scatter(
            z_data_d2,
            r["d2_data"],
            color=color,
            marker="s",
            s=18,
            alpha=0.75,
            zorder=3,
        )

    plt.xlabel("z position [m]", fontsize=THESIS_LABEL_SIZE)
    plt.ylabel("beam radius [m]", fontsize=THESIS_LABEL_SIZE)
    plt.tick_params(axis="both", labelsize=THESIS_TICK_SIZE)
    all_z = np.concatenate(centered_z_sets)
    plt.xlim(float(np.min(all_z)), float(np.max(all_z)))
    plt.legend(
        fontsize=THESIS_LEGEND_SIZE,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    plt.tight_layout(rect=[0, 0, 0.77, 1])
    plt.savefig(RESULTS_DIR / "msquared_beam_propagation_all_lasers.png", dpi=300, bbox_inches="tight")
    plt.close()

print(f"Results saved in: {RESULTS_DIR}")
