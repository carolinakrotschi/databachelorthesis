#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = SCRIPT_DIR.parent / "rawdata"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
WAVELENGTH_NM = 787.324
# One interferometric fringe corresponds to a path-length change of lambda/2
DISTANCE_PER_FRINGE_MM = WAVELENGTH_NM * 1e-6 / 2.0


def load_csv(path: Path) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {
        "Relative_Time_s": [],
        "Raw_Voltage_V": [],
        "Intensity": [],
        "Fringe_Count": [],
    }

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["Relative_Time_s"].append(float(row["Relative_Time_s"]))
            data["Raw_Voltage_V"].append(float(row["Raw_Voltage_V"]))
            if "Intensity" in row and row["Intensity"] != "":
                data["Intensity"].append(float(row["Intensity"]))
            data["Fringe_Count"].append(float(row["Fringe_Count"]))

    return data


def load_camera_csv(path: Path) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {
        "Relative_Time_s": [],
        "Intensity": [],
        "Fringe_Count": [],
    }

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["Relative_Time_s"].append(float(row["Relative_Time_s"]))
            data["Intensity"].append(float(row["Intensity"]))
            data["Fringe_Count"].append(float(row["Fringe_Count"]))

    return data


# Stacked traces of all camera/photodiode runs side by side, time-normalized
# to a common axis so runs of different duration can be compared visually.
def plot_camera_diode_side_by_side(
    camera_series: list[tuple[str, dict[str, list[float]]]],
    diode_series: list[tuple[str, dict[str, list[float]]]],
) -> None:
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, wspace=0.1)

    camera_grid = outer[0].subgridspec(len(camera_series), 1, hspace=0.18)
    diode_grid = outer[1].subgridspec(len(diode_series), 1, hspace=0.18)

    thesis_label_size = 15
    thesis_tick_size = 11
    thesis_title_size = 13
    x_end = 1200.0
    line_color = "#1f77b4"

    camera_names = {
        "camera_measurement_01": "measurement 1",
        "camera_measurement_02": "measurement 2",
        "camera_measurement_03": "measurement 3",
    }
    diode_names = {
        "photodiode_measurement_01": "measurement 1",
        "photodiode_measurement_02": "measurement 2",
        "photodiode_measurement_03": "measurement 3",
    }

    fig.text(0.25, 1.015, "Camera", ha="center", va="top", fontsize=17)
    fig.text(0.75, 1.015, "Photodiode", ha="center", va="top", fontsize=17)

    camera_axes = []
    for idx, (label, data) in enumerate(camera_series):
        ax = fig.add_subplot(camera_grid[idx, 0])
        camera_axes.append(ax)
        time = data["Relative_Time_s"]
        values = data["Intensity"]
        if time and values:
            t0 = time[0]
            shifted_time = [t - t0 for t in time]
            max_time = max(shifted_time) if shifted_time else 0
            if max_time > 0:
                plot_time = [t / max_time * x_end for t in shifted_time]
                plot_values = values
                y_max = max(plot_values)
                ax.plot(plot_time, plot_values, linewidth=1.1, color=line_color)
                ax.set_xlim(0, x_end)
                ax.set_ylim(0, y_max + 6400)
        ax.set_ylabel("Intensity [arb. unit]", fontsize=thesis_label_size)
        ax.set_title(camera_names.get(label, label), loc="left", fontsize=thesis_title_size)
        ax.tick_params(axis="both", labelsize=thesis_tick_size)
        ax.grid(True, alpha=0.3)

    if camera_axes:
        camera_axes[-1].set_xlabel("Time [s]", fontsize=thesis_label_size)

    diode_axes = []
    for idx, (label, data) in enumerate(diode_series):
        ax = fig.add_subplot(diode_grid[idx, 0])
        diode_axes.append(ax)
        time = data["Relative_Time_s"]
        values = data["Raw_Voltage_V"]
        if time and values:
            t0 = time[0]
            shifted_time = [t - t0 for t in time]
            max_time = max(shifted_time) if shifted_time else 0
            if max_time > 0:
                plot_time = [t / max_time * x_end for t in shifted_time]
                plot_values = values
                y_min = min(plot_values)
                y_max = max(plot_values)
                y_span = y_max - y_min
                padding = max(y_span * 0.02, 0.02)
                ax.plot(plot_time, plot_values, linewidth=1.1, color=line_color)
                ax.set_xlim(0, x_end)
                ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_ylabel("Raw Voltage [V]", fontsize=thesis_label_size)
        ax.set_title(diode_names.get(label, label), loc="left", fontsize=thesis_title_size)
        ax.tick_params(axis="both", labelsize=thesis_tick_size)
        ax.grid(True, alpha=0.3)

    if diode_axes:
        diode_axes[-1].set_xlabel("Time [s]", fontsize=thesis_label_size)

    fig.savefig(RESULTS_DIR / "camera_vs_photodiode_side_by_side.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# Total fringe count per measurement run for camera vs. photodiode,
# with a secondary axis converting fringe count to travelled distance.
def plot_fringe_summary_combined(
    camera_series: list[tuple[str, dict[str, list[float]]]],
    diode_series: list[tuple[str, dict[str, list[float]]]],
) -> None:
    fig, ax = plt.subplots(figsize=(14, 9))
    ax_right = ax.twinx()

    thesis_label_size = 26
    thesis_tick_size = 22
    point_color = "#1f77b4"
    x_camera = 1.0
    x_diode = 2.0

    camera_values = []
    for _, data in camera_series:
        fringe_count = data["Fringe_Count"]
        if not fringe_count:
            continue
        delta = fringe_count[-1] - fringe_count[0]
        camera_values.append(delta)
        ax.scatter([x_camera], [delta], s=220, color=point_color, zorder=3)

    diode_values = [max(data["Fringe_Count"]) for _, data in diode_series]
    for fringe_count in diode_values:
        ax.scatter([x_diode], [fringe_count], s=220, color=point_color, zorder=3)

    ax.set_xticks([x_camera, x_diode], ["Camera", "Photodiode"])
    ax.set_xlim(0.6, 2.4)
    ax.set_ylabel("Fringe Count", fontsize=thesis_label_size)
    ax.tick_params(axis="both", labelsize=thesis_tick_size)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(1050, 1260)

    if camera_values:
        camera_delta = max(camera_values) - min(camera_values)
        ax.text(
            0.02,
            0.98,
            f"Camera Δ = {camera_delta:.0f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=thesis_label_size,
        )
    if diode_values:
        diode_delta = max(diode_values) - min(diode_values)
        ax.text(
            0.98,
            0.98,
            f"Photodiode Δ = {diode_delta:.0f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=thesis_label_size,
        )

    left_ylim = ax.get_ylim()
    right_ylim = tuple(value * DISTANCE_PER_FRINGE_MM for value in left_ylim)
    ax_right.set_ylim(right_ylim)
    ax_right.set_ylabel("Distance [mm]", fontsize=thesis_label_size)
    ax_right.tick_params(axis="y", labelsize=thesis_tick_size, right=True, labelright=True)
    ax_right.spines["right"].set_visible(True)
    ax_right.yaxis.set_ticks_position("right")

    left_ticks = ax.get_yticks()
    ax_right.set_yticks([tick * DISTANCE_PER_FRINGE_MM for tick in left_ticks])

    fig.subplots_adjust(left=0.13, right=0.87, top=0.95, bottom=0.12)
    fig.savefig(RESULTS_DIR / "fringe_summary_combined.png", dpi=200)
    plt.close(fig)


# Zoomed-in comparison of camera vs. photodiode for measurement 2 only,
# around a fixed time window where individual fringes are visible.
def plot_measurement_2_zoom(
    camera_data: dict[str, list[float]],
    diode_data: dict[str, list[float]],
) -> None:
    fig, (ax_camera, ax_diode) = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

    thesis_label_size = 18
    thesis_tick_size = 14
    zoom_center_s = 811.0
    zoom_half_width_s = 5.0

    camera_time = camera_data["Relative_Time_s"]
    camera_values = camera_data["Intensity"]
    if camera_time and camera_values:
        t0 = camera_time[0]
        shifted_time = [t - t0 for t in camera_time]
        camera_plot = [
            (t, v)
            for t, v in zip(shifted_time, camera_values, strict=True)
            if zoom_center_s - zoom_half_width_s <= t <= zoom_center_s + zoom_half_width_s
        ]
        if camera_plot:
            plot_time = [t for t, _ in camera_plot]
            plot_values = [v for _, v in camera_plot]
            ax_camera.plot(plot_time, plot_values, linewidth=1.3, color="#1f77b4")
            ax_camera.set_xlim(zoom_center_s - zoom_half_width_s, zoom_center_s + zoom_half_width_s)
            y_min = min(plot_values)
            y_max = max(plot_values)
            y_span = y_max - y_min
            padding = max(y_span * 0.15, 40.0)
            ax_camera.set_ylim(y_min - padding, y_max + padding)
    ax_camera.set_title("Camera - measurement 2", fontsize=thesis_label_size)
    ax_camera.set_xlabel("Time [s]", fontsize=thesis_label_size)
    ax_camera.set_ylabel("Intensity [arb. unit]", fontsize=thesis_label_size)
    ax_camera.tick_params(axis="both", labelsize=thesis_tick_size)
    ax_camera.grid(True, alpha=0.3)

    diode_time = diode_data["Relative_Time_s"]
    diode_values = diode_data["Raw_Voltage_V"]
    if diode_time and diode_values:
        t0 = diode_time[0]
        shifted_time = [t - t0 for t in diode_time]
        diode_plot = [
            (t, v)
            for t, v in zip(shifted_time, diode_values, strict=True)
            if zoom_center_s - zoom_half_width_s <= t <= zoom_center_s + zoom_half_width_s
        ]
        if diode_plot:
            plot_time = [t for t, _ in diode_plot]
            plot_values = [v for _, v in diode_plot]
            ax_diode.plot(plot_time, plot_values, linewidth=1.3, color="#1f77b4")
            ax_diode.set_xlim(zoom_center_s - zoom_half_width_s, zoom_center_s + zoom_half_width_s)
            ax_diode.set_ylim(2.67, 2.83)
    ax_diode.set_title("Photodiode - measurement 2", fontsize=thesis_label_size)
    ax_diode.set_xlabel("Time [s]", fontsize=thesis_label_size)
    ax_diode.set_ylabel("Raw Voltage [V]", fontsize=thesis_label_size)
    ax_diode.tick_params(axis="both", labelsize=thesis_tick_size)
    ax_diode.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "camera_vs_photodiode_measurement_02_zoom.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    camera_files = [
        ("camera_measurement_01", DATA_DIR / "camera_measurement_01.csv"),
        ("camera_measurement_02", DATA_DIR / "camera_measurement_02.csv"),
        ("camera_measurement_03", DATA_DIR / "camera_measurement_03.csv"),
    ]
    diode_files = [
        ("photodiode_measurement_01", DATA_DIR / "photodiode_measurement_01.csv"),
        ("photodiode_measurement_02", DATA_DIR / "photodiode_measurement_02.csv"),
        ("photodiode_measurement_03", DATA_DIR / "photodiode_measurement_03.csv"),
    ]

    camera_series = [(label, load_camera_csv(p)) for label, p in camera_files if p.exists()]
    diode_series = [(label, load_csv(p)) for label, p in diode_files if p.exists()]

    if camera_series and diode_series:
        plot_fringe_summary_combined(camera_series, diode_series)
        plot_camera_diode_side_by_side(camera_series, diode_series)

        # The zoom plot is only produced for measurement 2, used as the representative example
        cam2 = next((d for l, d in camera_series if l == "camera_measurement_02"), None)
        diode2 = next((d for l, d in diode_series if l == "photodiode_measurement_02"), None)
        if cam2 and diode2:
            plot_measurement_2_zoom(cam2, diode2)

    print(f"Results saved in: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
