import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path

# Import der eigenen Funktion zur Berechnung der Modeformen
from modal_utils import compute_mode_shapes

# Funktion zum Einlesen der FRF-Dateien
def read_frf_file(filepath):
    df = pd.read_csv(
        filepath,
        sep="\t",
        decimal=",",
        names=["f", "val"],
        header=0
    )
    return df # Rückgabe des DataFrames

# -----------------------------
# 1. DATEN EINLESEN
# -----------------------------

base_path = Path("final-assignment\markus-hinterkoerner\code\data") # Pfad zu den Daten
hochhaeuser = ["Hochhaus 1", "Hochhaus 2", "Hochhaus 3"] # Liste der Hochhäuser
knoten_liste = ["H11", "H12", "H13"] # Liste der Knoten

data = {} # dictionary zum Speichern der Daten
T = 10  # Messdauer in Sekunden
delta_f = 1 / T  # Frequenzauflösung

for haus in hochhaeuser: # Schleife über alle Hochhäuser
    data[haus] = {}
    for knoten in knoten_liste:
        im_file = base_path / haus / f"{knoten}_Im.txt" # Pfad zur Imaginärdatei
        re_file = base_path / haus / f"{knoten}_Re.txt" # Pfad zur Realdatei

        df_im = read_frf_file(im_file) # Imaginärteil einlesen
        df_re = read_frf_file(re_file) # Realteil einlesen

        # physikalische Frequenzachse erstellen
        N = len(df_im)
        f_phys = np.arange(N) * delta_f  # jeder Messpunkt bekommt eigene Frequenz

        data[haus][knoten] = {
            "f": f_phys,    # physikalische Frequenzen in Hz
            "Im": df_im["val"].values,
            "Re": df_re["val"].values
        } # Daten speichern

# -----------------------------
# 2. DATEN PLOTTEN (Imaginärteil)
# -----------------------------

# Zielordner für Plots
plot_path = Path(__file__).parent / "Plots" # Pfad zum Plot-Ordner
plot_path.mkdir(exist_ok=True) # Ordner erstellen, falls nicht vorhanden

# Plotten des Imaginärteils über der Frequenz für alle Hochhäuser und Knoten
for haus in hochhaeuser:
    plt.figure(figsize=(16, 9)) # Neuer Plot für jedes Hochhaus, großes Format
    for knoten in knoten_liste: # Schleife über alle Knoten
        plt.plot(
            data[haus][knoten]["f"],
            data[haus][knoten]["Im"],
            label=knoten
        ) # Imaginärteil über Frequenz plotten

    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("Imaginärteil")
    plt.title(f"Imaginärteil aller Knoten – {haus}")
    plt.legend()
    plt.grid(True)

    # Neue x-Ticks: von Minimum bis Maximum mit Schritt 1 Hz
    f_min = int(data[haus][knoten_liste[0]]["f"][0])
    f_max = int(data[haus][knoten_liste[0]]["f"][-1]) + 1
    plt.xticks(np.arange(f_min, f_max, 1))

    full_path = plot_path / f"{haus}_Imaginaerteil.png" # Pfad zum Speichern des Plots
    plt.savefig(full_path, dpi=100) # Plot speichern mit hoher Auflösung (dpi=300)
    # plt.show() # Plot anzeigen nach dem Speichern
    plt.close() # Plot schließen

# -----------------------------
# 3. EIGENFREQUENZEN BESTIMMEN
# -----------------------------

referenz_knoten = "H11" # Referenzknoten für Peaks (üblich: erster Knoten)

print("=== Gefundene Eigenfrequenzen für alle Hochhäuser ===\n")

for haus in hochhaeuser: # Schleife über alle Hochhäuser
    f_ref = data[haus][referenz_knoten]["f"] # Frequenzen des Referenzknotens
    im_ref = np.abs(data[haus][referenz_knoten]["Im"]) # Imaginärteil des Referenzknotens

    # Peaks finden
    # Relativer Threshold bezogen auf das Maximum
    rel_height = 0.25          # 25% des Maximums
    height_thr = rel_height * np.max(im_ref)

    # Mindestabstand zwischen Peaks (in Messpunkten)
    min_distance_hz = 1.5      # physikalisch sinnvoll
    min_distance_pts = int(min_distance_hz / delta_f)

    peaks, props = find_peaks(
        im_ref,
        height=height_thr,
        distance=min_distance_pts
    ) # Peaks finden

    eigenfrequenzen = f_ref[peaks] # Eigenfrequenzen aus Peaks extrahieren

    # Ausgabe
    print(f"{haus}:")
    if len(eigenfrequenzen) == 0:
        print("  Keine Peaks über Threshold gefunden")
    else:
        for i, freq in enumerate(eigenfrequenzen):
            print(f"  Mode {i+1}: {freq:.3f} Hz")
    print()

# -----------------------------
# 4. MODEFORMEN BERECHNEN UND DARSTELLEN
# -----------------------------

# Manuelle Frequenzfenster für die wichtigen Moden
mode_windows_all = {
    "Hochhaus 1": [(2, 4), (8, 10), (15,17), (30, 36)],
    "Hochhaus 2": [(2, 4), (8, 10), (15,17), (30, 36)],
    "Hochhaus 3": [(2, 4), (8, 10), (15,17), (30, 36)]
}

for haus in hochhaeuser:  # Schleife über alle Hochhäuser

    mode_windows = mode_windows_all[haus]
    mode_freqs = []

    # Peaks innerhalb der Fenster finden
    f_ref = data[haus][referenz_knoten]["f"]
    im_ref = np.abs(data[haus][referenz_knoten]["Im"])

    for f_min, f_max in mode_windows:
        mask = (f_ref >= f_min) & (f_ref <= f_max)
        rel_threshold = 0.15 # relativer Threshold innerhalb des Fensters (15%)
        peaks, props = find_peaks(
            im_ref[mask],
            prominence=rel_threshold * np.max(im_ref[mask])
        ) # Peaks im Fenster finden
        if len(peaks) == 0:
            mode_freqs.append(None)
        else:
            peak_idx = peaks[np.argmax(im_ref[mask][peaks])]
            mode_freq = f_ref[mask][peak_idx]
            mode_freqs.append(mode_freq)

    # Modeformen berechnen
    plt.figure(figsize=(7,8))  # Hochformat für Hochhaus

    # y-Positionen für Knoten + Fundament (Fundament = 0, H13 = 1, H12 = 2, H11 = 3)
    y_positions = np.arange(len(knoten_liste)+1)  # Fundamentpunkt + Knoten

    # Ursprungsform als gerade Linie am Fundament
    plt.plot(np.zeros(len(y_positions)), y_positions, color='red', linestyle='-', linewidth=2, label='Ursprüngliche Form')

    for i, f_mode in enumerate(mode_freqs):
        if f_mode is None:
            continue  # überspringe Fenster ohne Peaks
        mode_shape = []
        for knoten in knoten_liste:
            f = data[haus][knoten]["f"]
            im = data[haus][knoten]["Im"]
            idx = np.argmin(np.abs(f - f_mode))
            mode_shape.append(im[idx])
        mode_shape = np.array(mode_shape)

        # Fundamentpunkt für jede Mode bei 0 unterhalb von H13 hinzufügen
        mode_shape_with_fundament = np.append(0, mode_shape[::-1])  # H11 oben, H13 unten, Fundament = 0

        plt.plot(
            mode_shape_with_fundament,
            y_positions,
            marker="o",
            linewidth=2,
            label=f"Mode {i+1} ({f_mode:.2f} Hz)"
        )

    # y-Achse: Fundament unten, H13, H12, H11 oben
    y_labels = ["Fundament"] + knoten_liste[::-1]
    plt.yticks(y_positions, y_labels)
    plt.xlabel("Auslenkung")
    plt.ylabel("Knoten")
    plt.title(f"Modeformen – {haus}")
    plt.legend()
    plt.grid(True)
    full_path = plot_path / f"{haus}_Modeformen.png"
    plt.savefig(full_path, dpi=300)
    plt.show()