from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# DIRECTORIES & PARAMETERS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "rawdata"
OUTPUT_DIR = SCRIPT_DIR.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

THESIS_LABEL_SIZE = 18
THESIS_TICK_SIZE = 18
THESIS_LEGEND_SIZE = 18

# Each laser has a signal spectrum and a dark/background spectrum recorded
# with the spectrometer; the background is subtracted below.
FILE_PAIRS = [
    ("thorlabsCPS780S", "thorlabs_cps780s_signal.txt", "thorlabs_cps780s_background.txt"),
    ("uniphase1023p", "uniphase_1023p_signal.txt", "uniphase_1023p_background.txt"),
    ("uniphase1103p1177761", "uniphase_1103p_1177761_signal.txt", "uniphase_1103p_1177761_background.txt"),
    ("uniphase1103p1108380", "uniphase_1103p_1108380_signal.txt", "uniphase_1103p_1108380_background.txt"),
    ("uniphase1122p", "uniphase_1122p_signal.txt", "uniphase_1122p_background.txt"),
    ("uniphase1507p0", "uniphase_1507p_signal.txt", "uniphase_1507p_background.txt"),
]


# Parse a spectrometer export: data rows only start after the
# "Begin Spectral Data" marker, columns are "wavelength intensity".
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


# Load signal + background for every laser and background-correct the spectrum
all_corrected_spectra = []

for name, signal_file, background_file in FILE_PAIRS:
    signal_path = DATA_DIR / signal_file
    background_path = DATA_DIR / background_file

    if not signal_path.exists() or not background_path.exists():
        print(f"File missing for {name}")
        continue

    wl_signal, intensity_signal = read_spectrum(signal_path)
    wl_background, intensity_background = read_spectrum(background_path)

    if len(wl_signal) == 0 or len(wl_background) == 0 or len(wl_signal) != len(wl_background):
        continue

    intensity = intensity_signal - intensity_background
    all_corrected_spectra.append((name, wl_signal, intensity))


# Broken-axis plot: left panel zooms on the ~630nm HeNe/uniphase lines,
# right panel zooms on the ~787nm thorlabs line. combine_uniphase merges the
# two near-identical uniphase1023p/uniphase1103p1177761 traces into one label.
def make_combined_plot(output_filename, combine_uniphase=True):
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(12, 7),
        sharey=True,
        gridspec_kw={"width_ratios": [15, 18], "wspace": 0.06},
    )

    for name, wavelengths, intensity in all_corrected_spectra:
        if combine_uniphase:
            if name == "uniphase1103p1177761":
                continue
            label = (
                "uniphase1023p & uniphase1103p1177761"
                if name == "uniphase1023p"
                else name
            )
        else:
            label = name

        ax_left.plot(wavelengths, intensity, label=label)
        ax_right.plot(wavelengths, intensity, label=label)

    ax_left.set_xlim(625, 640)
    ax_right.set_xlim(780, 798)

    ax_left.set_ylabel("Intensity", fontsize=THESIS_LABEL_SIZE)

    ax_left.tick_params(axis="both", labelsize=THESIS_TICK_SIZE)
    ax_right.tick_params(axis="both", labelsize=THESIS_TICK_SIZE)
    ax_right.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)

    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)

    d = 0.015
    break_kwargs = dict(color="black", clip_on=False, linewidth=1.5)
    ax_left.plot((1 - d, 1 + d), (-d, +d), transform=ax_left.transAxes, **break_kwargs)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_left.transAxes, **break_kwargs)
    ax_right.plot((-d, +d), (-d, +d), transform=ax_right.transAxes, **break_kwargs)
    ax_right.plot((-d, +d), (1 - d, 1 + d), transform=ax_right.transAxes, **break_kwargs)

    ax_left.set_xticks([625, 630, 635])
    ax_right.set_xticks([780, 785, 790, 795])

    ax_right.legend(
        fontsize=THESIS_LEGEND_SIZE,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    fig.supxlabel("Wavelength [nm]", fontsize=THESIS_LABEL_SIZE)
    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Two variants: one with overlapping uniphase traces merged, one with all lasers separate
make_combined_plot("wavelength_spectra_combined.png", combine_uniphase=True)
make_combined_plot("wavelength_spectra_separate.png", combine_uniphase=False)

print(f"Results saved in: {OUTPUT_DIR.resolve()}")
