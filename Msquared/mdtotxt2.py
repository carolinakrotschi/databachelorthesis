from pathlib import Path

input_folder = Path("rawrawdata")
output_folder = Path("rawdata")

output_folder.mkdir(exist_ok=True)

for md_file in input_folder.iterdir():

    # Nur echte md-Dateien
    if md_file.suffix != ".md":
        continue

    print("Gefunden:", md_file)

    txt_file = output_folder / (md_file.stem + ".txt")

    content = md_file.read_text(errors="ignore")

    content = content.replace("|", "\t")

    txt_file.write_text(content)

    print("Erstellt:", txt_file)

print("Fertig")