from pathlib import Path
import re

# Ordnerpfade
input_folder = Path("rawrawdata")
output_folder = Path("rawdata")

# Output-Ordner anlegen
output_folder.mkdir(parents=True, exist_ok=True)

# Alle md-Dateien suchen
md_files = list(input_folder.glob("*.md"))

if not md_files:
    print("Keine .md Dateien gefunden!")
    print("Gesuchter Ordner:", input_folder.resolve())

for md_file in md_files:

    # Zielname
    txt_file = output_folder / f"{md_file.stem}.txt"

    # Datei lesen
    content = md_file.read_text(encoding="utf-8")

    lines = content.splitlines()
    cleaned_lines = []

    for line in lines:

        # Markdown-Tabelle entfernen
        line = line.replace("|", "\t")

        # Markdown-Trennlinien entfernen
        if re.match(r"^[\-\|\s:]+$", line):
            continue

        line = line.strip()

        if line:
            cleaned_lines.append(line)

    # Speichern
    txt_file.write_text("\n".join(cleaned_lines), encoding="utf-8")

    print(f"Konvertiert: {md_file.name} -> {txt_file.name}")

print("Fertig.")