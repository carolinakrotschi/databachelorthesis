from pathlib import Path
import numpy as np
import laserbeamsize as lbs
import matplotlib.pyplot as plt



BASE_DIR = Path(
    "/Users/carolinakrotsch/Library/Mobile Documents/com~apple~CloudDocs/"
    "Studium/Bachelorarbeit/Lab/Messdaten/Msquared"
)

DATA_DIR = BASE_DIR / "rawdata"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


DATA_FILES = [
    ("thorlabs_v2.txt", "thorlabs", 1576.293000),
    ("uniphase_1023p_v4.txt", "uniphase1023p", 631.805000),
    ("uniphase_1103p_1177761_v4.txt", "uniphase1103p11777", 631.805000),
    ("uniphase_1103p_1180380_v4.txt", "uniphase1103p110838", 632.006000),
    ("uniphase_1122p_v4.txt", "uniphase1122p", 632.208000),
    ("uniphase_1507p_v4.txt", "uniphase1507p0", 634.822000),
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
    plt.xlabel("z position / m")
    plt.ylabel("beam radius / m")
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

print("\nFinished.")
print(f"Results saved in: {RESULTS_DIR}")