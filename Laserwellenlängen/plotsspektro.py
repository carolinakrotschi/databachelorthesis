from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

FILE_PAIRS = [
    ("thorlabs_NQ51B22081__0__11-36-36-140.txt",
     "backthorlabs_NQ51B22081__0__11-35-54-129-1.txt"),

    ("uniphase1023p_USB4C031521__0__12-07-47-856.txt",
     "backuniphase1023p_USB4C031521__3__12-07-31-183.txt"),

    ("uniphase1103p11777_USB4C031521__0__12-13-38-733.txt",
     "backuniphase1103p11777_USB4C031521__0__12-12-07-080.txt"),

    ("uniphase1103p110838_USB4C031521__0__16-10-06-126.txt",
     "backuniphase1103p110838_USB4C031521__0__16-09-42-795.txt"),

    ("uniphase1122p_USB4C031521__0__16-13-14-685.txt",
     "backuniphase1122p_USB4C031521__0__16-12-54-022.txt"),

    ("uniphase1507p0_USB4C031521__0__16-18-35-607.txt",
     "backuniphase1507p0_USB4C031521__0__16-18-20-348.txt")
]

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def read_spectrum(file_path):
    wavelengths = []
    intensities = []
    reading_data = False

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
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


results = []
all_corrected_spectra = []

for signal_file, background_file in FILE_PAIRS:
    name = short_name(signal_file)

    print(f"\nProcessing: {name}")

    wl_signal, intensity_signal = read_spectrum(signal_file)
    wl_background, intensity_background = read_spectrum(background_file)

    if len(wl_signal) != len(wl_background):
        print(f"ERROR: Spectrum lengths do not match for {name}.")
        continue

    if not np.allclose(wl_signal, wl_background, atol=1e-6):
        print(f"ERROR: Wavelength axes do not match for {name}.")
        continue

    intensity = intensity_signal - intensity_background
    intensity[intensity < 0] = 0

    peak_index = np.argmax(intensity)
    peak_wavelength = wl_signal[peak_index]
    peak_intensity = intensity[peak_index]

    print(f"Peak wavelength: {peak_wavelength:.3f} nm")

    results.append(peak_wavelength)
    all_corrected_spectra.append((name, wl_signal, intensity))

    plt.figure(figsize=(10, 6))
    plt.plot(wl_signal, intensity)

    plt.text(
        peak_wavelength,
        peak_intensity * 0.5,
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


with open(OUTPUT_DIR / "laser_peak_wavelengths.txt", "w", encoding="utf-8") as f:
    f.write("Laser\tPeak_Wavelength_nm\n")

    for name, wavelengths, intensity in all_corrected_spectra:

        peak_index = np.argmax(intensity)
        peak_wavelength = wavelengths[peak_index]

        f.write(f"{name}\t{peak_wavelength:.6f}\n")


plt.figure(figsize=(12, 7))

for name, wavelengths, intensity in all_corrected_spectra:
    plt.plot(wavelengths, intensity, label=name)

plt.xlabel("Wavelength / nm")
plt.ylabel("Intensity")
plt.title("All Laser Spectra")
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "all_laser_spectra.png", dpi=300)
plt.show()

print("\nFinished.")
print(f"Results saved in: {OUTPUT_DIR}")