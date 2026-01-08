import sys # Für Systempfade
from pathlib import Path # Für Pfadoperationen
import pandas as pd # Für Datenmanipulation
import numpy as np # Für numerische Operationen
import pyvista as pv # Für 3D-Visualisierung
import matplotlib.pyplot as plt # Für Plotten
from scipy.signal import find_peaks # Für Peak-Erkennung

# Import der eigenen Funktion zur Berechnung der Modeformen
from modal_utils import compute_mode_shapes

# Funktion zum Einlesen der FRF-Dateien
def read_frf_file(filepath):
    df = pd.read_csv(
        filepath,
        sep="\t", # Trenner ist Tabulator
        decimal=",",
        names=["f", "val"], # Frequenz, Wert
        header=0
    )
    return df # Einlesen Textdatei mit Frequenz und Werten

# Funktion zum Einlesen der Zeit-Signal-Dateien
def read_time_signal(filepath):
    df = pd.read_csv(
        filepath,
        sep="\t",
        decimal=",",
        header=0
    )

    time = df.iloc[:, 0].values      # Zeit
    amplitude = df.iloc[:, 1].values # Signal

    return time, amplitude

# -----------------------------
# 1. DATEN EINLESEN
# -----------------------------

base_path = Path(r"final-assignment\markus-hinterkoerner\code\data") # Pfad zu den Daten (r für raw string wegen Backslashes)
hochhaeuser = ["Hochhaus 1", "Hochhaus 2", "Hochhaus 3"] # Liste der Hochhäuser
knoten_liste = ["H11", "H12", "H13"] # Liste der Knoten je Hochhaus

data = {} # dictionary zum Speichern der Daten
T = 10  # Messdauer in Sekunden
delta_f = 1 / T  # Frequenzauflösung

for haus in hochhaeuser: # Schleife über alle Hochhäuser
    data[haus] = {}
    for knoten in knoten_liste:
        im_file = base_path / haus / f"{knoten}_Im.txt" # Pfad zur Imaginärdatei
        re_file = base_path / haus / f"{knoten}_Re.txt" # Pfad zur Realdatei

        df_im = read_frf_file(im_file) # Daten des Imaginärteil einlesen
        df_re = read_frf_file(re_file) # Daten desRealteil einlesen

        # physikalische Frequenzachse erstellen
        N = len(df_im) # Anzahl der Messpunkte
        f_phys = np.arange(N) * delta_f  # jeder Messpunkt bekommt eigene Frequenz
        # [0, 1*delta_f, 2*delta_f, ..., (N-1)*delta_f]

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
    plt.figure(figsize=(16, 9)) # Neuer Plot für jedes Hochhaus, Hochformat
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

    # x-Ticks: von Minimum bis Maximum mit Schritt 1 Hz
    f_min = int(data[haus][knoten_liste[0]]["f"][0]) # knoten_liste[0] = H11, Frequenz, [0] = erster Wert (= 0 Hz)
    f_max = int(data[haus][knoten_liste[0]]["f"][-1]) + 1 # [-1] = letzter Wert, +1 für inklusives Maximum
    plt.xticks(np.arange(f_min, f_max, 1)) # x-Ticks mit 1 Hz Abstand setzen

    full_path = plot_path / f"{haus}_Imaginaerteil.png" # Pfad zum Speichern des Plots
    plt.savefig(full_path, dpi=100) # Plot speichern mit hoher Auflösung (dpi=100)
    # plt.show() # Plot anzeigen
    plt.close() # Plot schließen

# -----------------------------
# 2b. ZEITBEREICHSSIGNALE PLOTTEN
# -----------------------------

hochhaeuser_time = ["Hochhaus 2"] # Zeitdaten nur für Hochhaus 2 vorhanden
referenz_knoten = "H11"
t_zoom_min = 0.37  # untere Grenze des Zooms in Sekunden
t_zoom_max = 0.6  # obere Grenze des Zooms in Sekunden

for haus in hochhaeuser_time:

    acc_file = base_path / haus / f"{referenz_knoten}_beschleunigung.txt"
    force_file = base_path / haus / f"{referenz_knoten}_kraft.txt"

    # Zeitbereichsdaten einlesen
    t_acc, acc = read_time_signal(acc_file)
    t_force, force = read_time_signal(force_file)

    # =============================
    # Plot 1: Gesamte Zeitverläufe
    # =============================
    fig, axs = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    # Kraft
    axs[0].plot(t_force, force)
    axs[0].set_ylabel("Kraft [N]")
    axs[0].set_title(f"Zeitbereichssignale – {haus} ({referenz_knoten})")
    axs[0].grid(True)

    # Beschleunigung
    axs[1].plot(t_acc, acc)
    axs[1].set_xlabel("Zeit [s]")
    axs[1].set_ylabel("Beschleunigung [m/s²]")
    axs[1].grid(True)

    full_path = plot_path / f"{haus}_{referenz_knoten}_Zeitbereich_gesamt.png"
    plt.savefig(full_path, dpi=200)
    #plt.show()
    plt.close()

    # =============================
    # Plot 2: Gezoomter Zeitbereich
    # =============================
    mask_acc = (t_acc >= t_zoom_min) & (t_acc <= t_zoom_max)
    mask_force = (t_force >= t_zoom_min) & (t_force <= t_zoom_max)

    fig, axs = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    # Kraft (Zoom)
    axs[0].plot(t_force[mask_force], force[mask_force])
    axs[0].set_ylabel("Kraft [N]")
    axs[0].set_title(f"Zeitbereichssignale ({t_zoom_min}s–{t_zoom_max}s) – {haus} ({referenz_knoten})")
    axs[0].grid(True)

    # Beschleunigung (Zoom)
    axs[1].plot(t_acc[mask_acc], acc[mask_acc])
    axs[1].set_xlabel("Zeit [s]")
    axs[1].set_ylabel("Beschleunigung [m/s²]")
    axs[1].grid(True)

    full_path = plot_path / f"{haus}_{referenz_knoten}_Zeitbereich_Zoom.png"
    plt.savefig(full_path, dpi=200)
    #plt.show()
    plt.close()

# -----------------------------
# 3. EIGENFREQUENZEN BESTIMMEN UND AUSGEBEN
# -----------------------------

referenz_knoten = "H11" # Referenzknoten für Peaks

print("=== Gefundene Eigenfrequenzen für alle Hochhäuser ===\n")

for haus in hochhaeuser: # Schleife über alle Hochhäuser
    f_ref = data[haus][referenz_knoten]["f"] # Frequenzen des Referenzknotens
    im_ref = np.abs(data[haus][referenz_knoten]["Im"]) # Imaginärteil des Referenzknotens

    # Peaks finden
    # Relativer Threshold bezogen auf das Maximum
    rel_height = 0.25          # 25% des Maximums
    height_thr = rel_height * np.max(im_ref) # absoluter Threshold berechnen

    # Mindestabstand zwischen Peaks
    min_distance_hz = 1.5      # physikalisch sinnvoll
    min_distance_pts = int(min_distance_hz / delta_f) # in Messpunkten umrechnen

    peaks, props = find_peaks(
        im_ref,
        height=height_thr, # minimaler Peak-Höhe
        distance=min_distance_pts # Mindestabstand zwischen Peaks
    ) # Peaks finden

    eigenfrequenzen = f_ref[peaks] # Eigenfrequenzen aus Peaks extrahieren

    # Ausgabe
    print(f"{haus}:") # Name des Hochhauses
    if len(eigenfrequenzen) == 0: # keine Peaks gefunden
        print("  Keine Peaks über Threshold gefunden")
    else:
        for i, freq in enumerate(eigenfrequenzen): # Schleife über gefundene Eigenfrequenzen
            print(f"  Mode {i+1}: {freq:.3f} Hz") # Ausgabe der Eigenfrequenz mit 3 Nachkommastellen
    print()

# -----------------------------
# 4. MODEFORMEN BERECHNEN UND DARSTELLEN
# -----------------------------

# Manuelle Frequenzfenster für die Moden
mode_windows_all = {
    "Hochhaus 1": [(2, 4), (8, 10), (15,17), (30, 36)],
    "Hochhaus 2": [(2, 4), (8, 10), (15,17), (30, 36)],
    "Hochhaus 3": [(2, 4), (8, 10), (15,17), (30, 36)]
} # Frequenzfenster für jede Mode und jedes Hochhaus [(f_min, f_max)] in Hz

for haus in hochhaeuser:  # Schleife über alle Hochhäuser

    # Modeformen und deren Eigenfrequenzen mit eigener Funktion bestimmen
    mode_freqs, mode_shapes = compute_mode_shapes(
        data,
        haus,
        knoten_liste,
        mode_windows_all[haus],
        referenz_knoten,
        delta_f
    )

    plt.figure(figsize=(7,8))  # Hochformat für Hochhaus

    # y-Positionen für Fundament + Knoten
    # Fundament = 0, H13 = 1, H12 = 2, H11 = 3
    y_positions = np.arange(len(knoten_liste) + 1)

    # Ursprungsform als gerade Linie plotten
    plt.plot(
        np.zeros(len(y_positions)),
        y_positions,
        color='red',
        linestyle='-',
        linewidth=2,
        label='Ursprüngliche Form'
    )

    for i, f_mode in enumerate(mode_freqs): # Schleife über alle Moden
        if f_mode is None or np.isnan(f_mode): # Überspringen, falls keine Frequenz gefunden
         continue

        mode_shape = mode_shapes[i]  # Modeform (Reihenfolge: H11, H12, H13)

        # Fundament hinzufügen und Reihenfolge für Plot umdrehen (mit [::-1])
        mode_shape_with_fundament = np.append(0, mode_shape[::-1])
        # [Fundament, H13, H12, H11] 

        plt.plot(
            mode_shape_with_fundament,
            y_positions,
            marker="o",
            linewidth=2,
            label=f"Mode {i+1} ({f_mode:.2f} Hz)"
        )

    # y-Achse: Fundament unten, H13, H12, H11 oben
    y_labels = ["Fundament"] + knoten_liste[::-1]
    plt.yticks(y_positions, y_labels) # y-Ticks und Labels setzen

    plt.xlabel("Auslenkung")
    plt.ylabel("Knoten")
    plt.title(f"Modeformen – {haus}")
    plt.legend()
    plt.grid(True)

    full_path = plot_path / f"{haus}_Modeformen.png"
    plt.savefig(full_path, dpi=300) # Plot speichern mit hoher Auflösung (dpi=300)
    #plt.show() # Plot anzeigen
    plt.close() # Plot schließen

# -----------------------------
# 3D HOCHHAUS EINLESEN UND VON VORNE AUF DIE x-y EBENE ANSCHAUEN
# Hochrichtung = y, Breite = x, Auslenkung = z
# -----------------------------

# Pfad zur STL-Datei
stl_file = base_path / "Hochhaus.stl"

# Mesh laden
mesh = pv.read(stl_file)

# Plotter erstellen
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightgray", show_edges=True, opacity=1.0)

# Achsen hinzufügen
plotter.add_axes()   # X, Y, Z Achsen
plotter.show_grid()  # Raster anzeigen

# Kamera frontal auf x-y Ebene schauen
# Blick entlang +z (z nach hinten), Hochrichtung y
plotter.view_vector(vector=(0, 0, 1), viewup=(0, 1, 0))

plotter.show(title="Hochhaus 3D Ansicht (Frontansicht x-y, y=Höhe, z=Auslenkung)")