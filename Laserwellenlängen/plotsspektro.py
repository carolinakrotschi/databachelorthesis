from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FOLDERS
# ============================================================

DATA_DIR = Path("rawdata")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# FILE PAIRS
# signal file, background file
# both files are inside the folder "rawdata"
# ============================================================

FILE_PAIRS = [
    (
        "thorlabs_NQ51B22081__0__11-36-36-140.txt",
        "backthorlabs_NQ51B22081__0__11-35-54-129-1.txt"
    ),
    (
        "uniphase1023p_USB4C031521__0__12-07-47-856.txt",
        "backuniphase1023p_USB4C031521__3__12-07-31-183.txt"
    ),
    (
        "uniphase1103p11777_USB4C031521__0__12-13-38-733.txt",
        "backuniphase1103p11777_USB4C031521__0__12-12-07-080.txt"
    ),
    (
        "uniphase1103p110838_USB4C031521__0__16-10-06-126.txt",
        "backuniphase1103p110838_USB4C031521__0__16-09-42-795.txt"
    ),
    (
        "uniphase1122p_USB4C031521__0__16-13-14-685.txt",
        "backuniphase1122p_USB4C031521__0__16-12-54-022.txt"
    ),
    (
        "uniphase1507p0_USB4C031521__0__16-18-35-607.txt",
        "backuniphase1507p0_USB4C031521__0__16-18-20-348.txt"
    )
]

# ============================================================
# FUNCTIONS
# ============================================================

def read_spectrum(file_path):
    wavelengths = []
    intensities = []
    reading_data = False

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()

            if "Begin Spectral Data" in line:
                reading_data = True
                continue

            if not reading_data or not line:
                continue

            parts = line.replace(",", ".").split()

            if len(parts) >= 2:
                try:
                    wavelengths.append(float(parts[0]))
                    intensities.append(float(parts[1]))
                except ValueError:
                    continue

    return np.array(wavelengths), np.array(intensities)


def short_name(file_name):
    return Path(file_name).stem.split("_")[0]


# ============================================================
# PROCESS DATA
# ============================================================

peak_results = []
all_corrected_spectra = []

for signal_file, background_file in FILE_PAIRS:
    name = short_name(signal_file)

    signal_path = DATA_DIR / signal_file
    background_path = DATA_DIR / background_file

    print(f"\nProcessing: {name}")
    print(f"Signal file: {signal_path}")
    print(f"Background file: {background_path}")

    if not signal_path.exists():
        print(f"ERROR: Signal file not found: {signal_path}")
        continue

    if not background_path.exists():
        print(f"ERROR: Background file not found: {background_path}")
        continue

    wl_signal, intensity_signal = read_spectrum(signal_path)
    wl_background, intensity_background = read_spectrum(background_path)

    if len(wl_signal) == 0 or len(wl_background) == 0:
        print(f"ERROR: No spectral data found for {name}.")
        continue

    if len(wl_signal) != len(wl_background):
        print(f"ERROR: Spectrum lengths do not match for {name}.")
        continue

    if not np.allclose(wl_signal, wl_background, atol=1e-6):
        print(f"ERROR: Wavelength axes do not match for {name}.")
        continue

    # Background subtraction
    intensity = intensity_signal - intensity_background

    peak_index = np.argmax(intensity)
    peak_wavelength = wl_signal[peak_index]
    peak_intensity = intensity[peak_index]

    print(f"Min intensity: {np.min(intensity):.3f}")
    print(f"Max intensity: {np.max(intensity):.3f}")
    print(f"Peak wavelength: {peak_wavelength:.3f} nm")

    peak_results.append((name, peak_wavelength))
    all_corrected_spectra.append((name, wl_signal, intensity))

    # ========================================================
    # INDIVIDUAL PLOT
    # ========================================================

    plt.figure(figsize=(10, 6))
    plt.plot(wl_signal, intensity)

    text_y_position = np.min(intensity) + 0.55 * (np.max(intensity) - np.min(intensity))

    plt.text(
        peak_wavelength,
        text_y_position,
        f"{peak_wavelength:.3f} nm",
        horizontalalignment="center",
        verticalalignment="center"
    )

    plt.xlabel("Wavelength / nm")
    plt.ylabel("Intensity")
    plt.title(name)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=300)
    plt.close()


# ============================================================
# SAVE TXT FILE
# ============================================================

with open(OUTPUT_DIR / "laser_peak_wavelengths.txt", "w", encoding="utf-8") as file:
    file.write("Laser\tPeak_Wavelength_nm\n")

    for name, peak_wavelength in peak_results:
        file.write(f"{name}\t{peak_wavelength:.6f}\n")


# ============================================================
# COMBINED PLOT
# ============================================================

plt.figure(figsize=(12, 7))

for name, wavelengths, intensity in all_corrected_spectra:
    plt.plot(wavelengths, intensity, label=name)

plt.xlabel("Wavelength / nm")
plt.ylabel("Intensity")
plt.title("All Corrected Laser Spectra")
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "all_corrected_lasers.png", dpi=300)
plt.show()

print("\nFinished.")
print(f"Results saved in: {OUTPUT_DIR.resolve()}")