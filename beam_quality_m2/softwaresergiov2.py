import os
from pathlib import Path
from tempfile import gettempdir

BASE_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = Path(os.environ.get("MPLCONFIGDIR", Path(gettempdir()) / "msquared-matplotlib"))
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_DIR)

import numpy as np
import laserbeamsize as lbs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt



DATA_DIR = BASE_DIR / "rawdata"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
THESIS_LABEL_SIZE = 18
THESIS_TICK_SIZE = 16
THESIS_LEGEND_SIZE = 16


DATA_FILES = [
    ("thorlabs_v2.txt", "thorlabs", 787.324),
    ("uniphase_1023p_v4.txt", "uniphase1023p", 631.805000),
    ("uniphase_1103p_1177761_v4.txt", "uniphase1103p1177761", 631.805000),
    ("uniphase_1103p_1180380_v4.txt", "uniphase1103p1108380", 632.006000),
    ("uniphase_1122p_v4.txt", "uniphase1122p", 632.208000),
    ("uniphase_1507p_v5.txt", "uniphase1507p0", 634.822000),
]


def read_txt_data(file_path):
    rows = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            line = line.replace("mm", "")
            line = line.replace("≈", "")
            line = line.replace("~", "")

            lower = line.lower()
            if any(word in lower for word in ["position", "width", "radius", "z", "d1", "d2"]):
                continue

            line = line.replace(";", "\t")
            line = line.replace(",", "\t")

            parts = line.split()

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

    if len(data) < 5:
        raise ValueError(f"Not enough valid data points in {file_path.name}")

    data = data[np.argsort(data[:, 0])]

    z1_all = data[:, 0] * 1e-3
    d1_all = data[:, 1] * 1e-3 
    d2_all = data[:, 2] * 1e-3

    return z1_all, d1_all, d2_all


def analyze_file(file_path, laser_name, wavelength_nm):
    wavelength_m = wavelength_nm * 1e-9

    print(f"\nProcessing: {laser_name}")
    print(f"File: {file_path.name}")
    print(f"Wavelength: {wavelength_nm:.6f} nm")

    z1_all, d1_all, d2_all = read_txt_data(file_path)

    print(f"Loaded points: {len(z1_all)}")

    lbs.M2_radius_plot(z1_all, d1_all, wavelength_m, strict=True)
    plt.savefig(RESULTS_DIR / f"{laser_name}_d1_M2_plot.png", dpi=300)
    plt.close()

    lbs.M2_radius_plot(z1_all, d2_all, wavelength_m, strict=True)
    plt.savefig(RESULTS_DIR / f"{laser_name}_d2_M2_plot.png", dpi=300)
    plt.close()

    t_valueM2d1 = lbs.M2_fit(z1_all, d1_all, wavelength_m, strict=True)
    t_valueM2d2 = lbs.M2_fit(z1_all, d2_all, wavelength_m, strict=True)

    a_fitd1 = t_valueM2d1[0]
    a_fitd2 = t_valueM2d2[0]

    v_dnewd1 = a_fitd1[0]
    v_znewd1 = a_fitd1[1]
    v_thetanewd1 = a_fitd1[2]
    v_M2newd1 = a_fitd1[3]
    v_zrnewd1 = a_fitd1[4]

    v_dnewd2 = a_fitd2[0]
    v_znewd2 = a_fitd2[1]
    v_thetanewd2 = a_fitd2[2]
    v_M2newd2 = a_fitd2[3]
    v_zrnewd2 = a_fitd2[4]

    a_d1ne = v_dnewd1**2 + v_thetanewd1**2 * (z1_all - v_znewd1)**2
    a_d2ne = v_dnewd2**2 + v_thetanewd2**2 * (z1_all - v_znewd2)**2

    a_d1new = np.sqrt(a_d1ne)
    a_d2new = np.sqrt(a_d2ne)

    step = (z1_all[1] - z1_all[0]) * 1e-2

    if step <= 0:
        raise ValueError(f"Invalid z step in {file_path.name}")

    a_z1new = np.arange(np.min(z1_all), np.max(z1_all), step)

    a_d1newlong = np.interp(a_z1new, z1_all, a_d1new)
    a_d2newlong = np.interp(a_z1new, z1_all, a_d2new)

    plt.figure(figsize=(10, 6))
    plt.plot(a_z1new, a_d1newlong, "b-", label=f"d1 fit, M2 = {v_M2newd1:.2f}")
    plt.plot(z1_all, d1_all, "b.", label="d1 data")
    plt.plot(a_z1new, a_d2newlong, "r-", label=f"d2 fit, M2 = {v_M2newd2:.2f}")
    plt.plot(z1_all, d2_all, "r.", label="d2 data")
    plt.xlabel("z position [m]")
    plt.ylabel("beam radius [m]")
    plt.title(f"{laser_name}, wavelength = {wavelength_nm:.3f} nm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{laser_name}_combined_fit.png", dpi=300)
    plt.close()

    with open(RESULTS_DIR / f"{laser_name}_d1_variables.txt", "w") as f:
        f.write("wavelength_nm\tv_dnewd1\tv_znewd1\tv_thetanewd1\tv_M2newd1\tv_zrnewd1\n")
        f.write(
            f"{wavelength_nm:.6f}\t{v_dnewd1:.6e}\t{v_znewd1:.6e}\t"
            f"{v_thetanewd1:.6e}\t{v_M2newd1:.6e}\t{v_zrnewd1:.6e}\n"
        )

    with open(RESULTS_DIR / f"{laser_name}_d2_variables.txt", "w") as f:
        f.write("wavelength_nm\tv_dnewd2\tv_znewd2\tv_thetanewd2\tv_M2newd2\tv_zrnewd2\n")
        f.write(
            f"{wavelength_nm:.6f}\t{v_dnewd2:.6e}\t{v_znewd2:.6e}\t"
            f"{v_thetanewd2:.6e}\t{v_M2newd2:.6e}\t{v_zrnewd2:.6e}\n"
        )

    with open(RESULTS_DIR / f"{laser_name}_d1_arrays.txt", "w") as f:
        f.write("z1_new_m\td1_new_long_m\n")
        for z, d1 in zip(a_z1new, a_d1newlong):
            f.write(f"{z:.6e}\t{d1:.6e}\n")

    with open(RESULTS_DIR / f"{laser_name}_d2_arrays.txt", "w") as f:
        f.write("z1_new_m\td2_new_long_m\n")
        for z, d2 in zip(a_z1new, a_d2newlong):
            f.write(f"{z:.6e}\t{d2:.6e}\n")

    print(f"M2 d1: {v_M2newd1:.4f}")
    print(f"M2 d2: {v_M2newd2:.4f}")

    return {
        "laser": laser_name,
        "wavelength_nm": wavelength_nm,
        "M2_d1": v_M2newd1,
        "M2_d2": v_M2newd2,
        "waist_d1_m": v_dnewd1,
        "waist_d2_m": v_dnewd2,
        "z0_d1_m": v_znewd1,
        "z0_d2_m": v_znewd2,
        "z_data": z1_all,
        "d1_data": d1_all,
        "d2_data": d2_all,
        "z_fit": a_z1new,
        "d1_fit": a_d1newlong,
        "d2_fit": a_d2newlong,
    }




summary_results = []

for filename, laser_name, wavelength_nm in DATA_FILES:
    file_path = DATA_DIR / filename

    if not file_path.exists():
        print(f"\nFile not found: {file_path}")
        continue

    try:
        result = analyze_file(file_path, laser_name, wavelength_nm)
        summary_results.append(result)
    except Exception as e:
        print(f"\nError processing {filename}: {e}")


def make_collage(results):
    if not results:
        return

    n = len(results)
    fig, axes = plt.subplots(
        2,
        n,
        figsize=(4.2 * n, 8.4),
        squeeze=False,
    )

    for col, r in enumerate(results):
        d1_path = RESULTS_DIR / f"{r['laser']}_d1_M2_plot.png"
        d2_path = RESULTS_DIR / f"{r['laser']}_d2_M2_plot.png"

        for row, (img_path, row_label) in enumerate(
            [(d1_path, "d1"), (d2_path, "d2")]
        ):
            ax = axes[row][col]
            if img_path.exists():
                ax.imshow(plt.imread(img_path))
            ax.axis("off")
            ax.set_title(
                f"{r['laser']} {row_label}",
                fontsize=THESIS_LEGEND_SIZE,
                pad=10,
            )

    fig.suptitle("M2 plots: d1 top row, d2 bottom row", fontsize=THESIS_LABEL_SIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(RESULTS_DIR / "m2_plots_collage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_simple_fit_plot(x, y, laser_name, row_label, xlim, ylim):
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(x, y, color="tab:blue", linewidth=2.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("z position [m]", fontsize=THESIS_LABEL_SIZE - 4)
    ax.set_ylabel("beam radius [m]", fontsize=THESIS_LABEL_SIZE - 4)
    ax.tick_params(axis="both", labelsize=THESIS_TICK_SIZE - 4)
    ax.set_title(f"{laser_name} {row_label}", fontsize=THESIS_LEGEND_SIZE)
    fig.tight_layout()
    out_path = RESULTS_DIR / f"{laser_name}_{row_label}_simple.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_simple_collage(results):
    if not results:
        return

    centered_x = []
    centered_y = []
    for r in results:
        centered_x.extend((r["z_fit"] - r["z0_d1_m"]).tolist())
        centered_x.extend((r["z_fit"] - r["z0_d2_m"]).tolist())
        centered_y.extend(r["d1_fit"].tolist())
        centered_y.extend(r["d2_fit"].tolist())

    x_vals = np.array(centered_x)
    y_vals = np.array(centered_y)
    x_span = float(np.max(x_vals) - np.min(x_vals))
    y_span = float(np.max(y_vals) - np.min(y_vals))
    x_pad = 0.05 * x_span if x_span > 0 else 1.0
    y_pad = 0.05 * y_span if y_span > 0 else 1.0
    xlim = (float(np.min(x_vals) - x_pad), float(np.max(x_vals) + x_pad))
    ylim = (float(np.min(y_vals) - y_pad), float(np.max(y_vals) + y_pad))

    n = len(results)
    fig, axes = plt.subplots(
        2,
        n,
        figsize=(3.6 * n, 6.4),
        squeeze=False,
    )

    for col, r in enumerate(results):
        d1_img = save_simple_fit_plot(
            r["z_fit"] - r["z0_d1_m"],
            r["d1_fit"],
            r["laser"],
            "d1",
            xlim,
            ylim,
        )
        d2_img = save_simple_fit_plot(
            r["z_fit"] - r["z0_d2_m"],
            r["d2_fit"],
            r["laser"],
            "d2",
            xlim,
            ylim,
        )

        for row, img_path in enumerate([d1_img, d2_img]):
            ax = axes[row][col]
            ax.imshow(plt.imread(img_path))
            ax.axis("off")

    fig.suptitle("Centered M2 fit curves", fontsize=THESIS_LABEL_SIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(RESULTS_DIR / "m2_simple_collage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


with open(RESULTS_DIR / "summary_M2_results.txt", "w") as f:
    f.write(
        "laser\twavelength_nm\tM2_d1\tM2_d2\t"
        "waist_d1_m\twaist_d2_m\tz0_d1_m\tz0_d2_m\n"
    )

    for r in summary_results:
        f.write(
            f"{r['laser']}\t{r['wavelength_nm']:.6f}\t"
            f"{r['M2_d1']:.6e}\t{r['M2_d2']:.6e}\t"
            f"{r['waist_d1_m']:.6e}\t{r['waist_d2_m']:.6e}\t"
            f"{r['z0_d1_m']:.6e}\t{r['z0_d2_m']:.6e}\n"
        )


# ============================================================
# COMBINED PLOT FOR ALL LASER SOURCES
# ============================================================

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
    plt.savefig(RESULTS_DIR / "all_lasers_combined_fit.png", dpi=300, bbox_inches="tight")
    plt.close()

    make_collage(summary_results)
    make_simple_collage(summary_results)

print("\nFinished.")
print(f"Results saved in: {RESULTS_DIR}")
