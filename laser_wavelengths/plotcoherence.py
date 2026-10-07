from pathlib import Path
import csv


results = Path(__file__).resolve().parent / "results"
fwhm_file = results / "fwhm_table.csv"
out_file = results / "coherence_lengths.csv"

rows = []

with open(fwhm_file, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        laser = row["Laser"]
        wavelength = float(row["Peak_nm"])
        fwhm = float(row["FWHM_nm"])

        # formula from the mail:
        # Lc = lambda^2 / FWHM
        coherence_length_mm = (wavelength * wavelength / fwhm) * 1e-6
        rows.append([laser, wavelength, fwhm, coherence_length_mm])

with open(out_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Laser", "Wavelength_nm", "FWHM_nm", "CoherenceLength_mm"])
    writer.writerows(rows)

print("Table saved:")
print(out_file)
for row in rows:
    print(f"{row[0]}: FWHM={row[2]:.4f} nm, Lc={row[3]:.4f} mm")
