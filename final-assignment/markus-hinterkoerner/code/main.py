from pathlib import Path  # Für Pfadoperationen
import pandas as pd        # Für Datenmanipulation
import numpy as np         # Für numerische Berechnungen
import matplotlib.pyplot as plt  # Für 2D-Plots
import pyvista as pv       # Für 3D-Visualisierung
from scipy.signal import find_peaks          # Für Peak-Erkennung
from scipy.interpolate import PchipInterpolator  # Für glatte Interpolation der Modeformen

# Eigene Funktion zur Berechnung der Modeformen
from modal_utils import compute_mode_shapes

# =============================
# 1. HILFSFUNKTIONEN
# =============================

def read_frf_file(filepath):
    """
    Liest eine FRF-Datei ein (Re- oder Im-Teil)
    Erwartet Tab-getrennte Werte, Dezimalzeichen = ","
    """
    df = pd.read_csv(filepath, sep="\t", decimal=",", names=["f", "val"], header=0)
    return df

def read_time_signal(filepath):
    """
    Liest eine Zeitbereichs-Datei ein (z.B. Beschleunigung oder Kraft)
    Erwartet zwei Spalten: Zeit, Amplitude
    """
    df = pd.read_csv(filepath, sep="\t", decimal=",", header=0)
    time = df.iloc[:,0].values
    amplitude = df.iloc[:,1].values
    return time, amplitude

# =============================
# 2. DATEN EINLESEN
# =============================
base_path = Path(__file__).parent / "data"  # Basisverzeichnis für Daten
hochhaeuser = ["Hochhaus 1", "Hochhaus 2", "Hochhaus 3"]
knoten_liste = ["E1", "E2", "E3"]  # Messpunkte im Hochhaus

data = {}  # Dictionary zum Speichern der Messdaten
T = 10     # Messdauer in Sekunden
delta_f = 1 / T  # Frequenzauflösung

# Schleife über alle Hochhäuser und Knoten
for haus in hochhaeuser:
    data[haus] = {}
    for knoten in knoten_liste:
        # Pfade zu Real- und Imaginärteil
        im_file = base_path / haus / f"{knoten}_Im.txt"
        re_file = base_path / haus / f"{knoten}_Re.txt"
        df_im = read_frf_file(im_file)
        df_re = read_frf_file(re_file)

        # Physikalische Frequenzachse erstellen
        N = len(df_im)
        f_phys = np.arange(N) * delta_f

        # Daten abspeichern
        data[haus][knoten] = {
            "f": f_phys,
            "Im": df_im["val"].values,
            "Re": df_re["val"].values
        }

# =============================
# 3. 2D-PLOT: Imaginärteil
# =============================
plot_path = Path(__file__).parent / "Plots"
plot_path.mkdir(exist_ok=True)  # Ordner erstellen, falls nicht vorhanden

for haus in hochhaeuser:
    plt.figure(figsize=(16,9))
    for knoten in knoten_liste:
        plt.plot(data[haus][knoten]["f"], data[haus][knoten]["Im"], label=knoten)
    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("Imaginärteil")
    plt.title(f"Imaginärteil aller Knoten – {haus}")
    plt.legend()
    plt.grid(True)
    f_min = int(data[haus][knoten_liste[0]]["f"][0])
    f_max = int(data[haus][knoten_liste[0]]["f"][-1]) + 1
    plt.xticks(np.arange(f_min, f_max, 1))
    plt.savefig(plot_path / f"{haus}_Imaginaerteil.png", dpi=100)
    plt.close()

# =============================
# 4. EIGENFREQUENZEN BESTIMMEN
# =============================
referenz_knoten = "E1"

print("=== Gefundene Eigenfrequenzen ===\n")
for haus in hochhaeuser:
    f_ref = data[haus][referenz_knoten]["f"]
    im_ref = np.abs(data[haus][referenz_knoten]["Im"])

    # Peaks finden
    height_thr = 0.25 * np.max(im_ref)        # 25% des Maximums als Threshold
    min_distance_pts = int(1.5 / delta_f)     # Mindestabstand zwischen Peaks
    peaks, _ = find_peaks(im_ref, height=height_thr, distance=min_distance_pts)
    eigenfrequenzen = f_ref[peaks]

    # Ausgabe
    print(f"{haus}:")
    if len(eigenfrequenzen) == 0:
        print("  Keine Peaks über Threshold gefunden")
    else:
        for i, freq in enumerate(eigenfrequenzen):
            print(f"  Mode {i+1}: {freq:.3f} Hz")
    print()

# =============================
# 5. 2D-PLOT: Modeformen
# =============================
mode_windows_all = {
    "Hochhaus 1": [(2,4),(8,10),(15,17),(30,36)],
    "Hochhaus 2": [(2,4),(8,10),(15,17),(30,36)],
    "Hochhaus 3": [(2,4),(8,10),(15,17),(30,36)]
}

for haus in hochhaeuser:
    # Modeformen berechnen
    mode_freqs, mode_shapes = compute_mode_shapes(
        data, haus, knoten_liste, mode_windows_all[haus], referenz_knoten, delta_f
    )

    plt.figure(figsize=(7,8))
    y_positions = np.arange(len(knoten_liste) + 1)  # Fundament + Knoten

    # Ursprungsform: gerade Linie (Fundament = 0)
    plt.plot(np.zeros(len(y_positions)), y_positions, color='red', linewidth=2, label='Ursprüngliche Form')

    for i, f_mode in enumerate(mode_freqs):
        if f_mode is None or np.isnan(f_mode):
            continue
        mode_shape = mode_shapes[i]
        mode_shape_with_fundament = np.append(0, mode_shape[::-1])  # Fundament hinzufügen
        plt.plot(mode_shape_with_fundament, y_positions, marker='o', linewidth=2,
                 label=f"Mode {i+1} ({f_mode:.2f} Hz)")

    # y-Achsen Labels
    y_labels = ["Fundament"] + knoten_liste[::-1]
    plt.yticks(y_positions, y_labels)
    plt.xlabel("Auslenkung")
    plt.ylabel("Knoten")
    plt.title(f"Modeformen – {haus}")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path / f"{haus}_Modeformen.png", dpi=300)
    plt.close()

# -----------------------------
# 5. 3D AUSLENKUNG ALLER MODEN FÜR JEDES HOCHHAUS DARSTELLEN
# Hochrichtung = y, Breite = x, Auslenkung = z
# -----------------------------

from scipy.interpolate import PchipInterpolator  # Für glatte Interpolation der Modeformen

# STL-Datei laden (gemeinsames Mesh für alle Hochhäuser)
stl_file = base_path / "Hochhaus.stl"
mesh_orig = pv.read(stl_file)

# Knotenhöhen definieren (y-Werte der Ebenen/Hochhaus-Knoten)
y_min = mesh_orig.points[:, 1].min()
y_max = mesh_orig.points[:, 1].max()
knoten_hoehen = np.array([y_max, (y_min + y_max)/2, y_min])  # Fundament unten, H13, H12, H11

# Anzahl der Unterteilungen für feineres Mesh
subdivide_n = 2  # 0=Original, 1=1 Unterteilung, 2=2 Unterteilungen usw.

print("=== Warpingfaktoren für die 3D-Darstellung der Modeformen ===\n")

for haus in hochhaeuser:

    # Modeformen und Eigenfrequenzen bestimmen
    mode_freqs, mode_shapes = compute_mode_shapes(
        data,
        haus,
        knoten_liste,
        mode_windows_all[haus],
        referenz_knoten,
        delta_f
    )

    for mode_idx, mode_shape in enumerate(mode_shapes):

        # Kopie des Original-Meshes
        mesh = mesh_orig.copy()

        # Mesh fein unterteilen
        if subdivide_n > 0:
            mesh = mesh.subdivide(subdivide_n, "linear")

        # -----------------------------
        # Displacement berechnen
        # -----------------------------
        displacement = np.zeros_like(mesh.points)
        spline = PchipInterpolator(knoten_hoehen[::-1], mode_shape[::-1])
        displacement[:, 2] = spline(mesh.points[:, 1])
        mesh["U"] = displacement  # "U" = Standardattribut für Verschiebungen in PyVista

        # Skalierungsfaktor berechnen
        max_auslenkung = np.max(np.abs(mode_shape))
        hoehe_hh = y_max - y_min
        factor = hoehe_hh / max_auslenkung * 0.1
        print(f"{haus} – Mode {mode_idx+1}: Warping-Faktor = {factor:.2f}")

        # Mesh nach Auslenkung warpen
        warped_mesh = mesh.warp_by_vector("U", factor=factor)

        # -----------------------------
        # Interaktiver Plot (wird angezeigt)
        # -----------------------------
        plotter = pv.Plotter()
        plotter.add_mesh(warped_mesh, color="lightblue", show_edges=True, opacity=1.0)
        plotter.add_axes()
        plotter.show_grid()
        plotter.view_vector(vector=(-1, 0, 0), viewup=(0, 1, 0))
        plotter.show(title=f"{haus} – 3D Auslenkung Mode {mode_idx+1}")

        # -----------------------------
        # Off-screen Plot für Screenshot
        # -----------------------------
        plotter_off = pv.Plotter(off_screen=True)
        plotter_off.add_mesh(warped_mesh, color="lightblue", show_edges=True, opacity=1.0)
        plotter_off.add_axes()
        plotter_off.show_grid()
        plotter_off.view_vector(vector=(-1, 0, 0), viewup=(0, 1, 0))

        # Screenshot speichern
        full_path = plot_path / f"{haus}_Mode{mode_idx+1}_3D_Auslenkung.png"
        plotter_off.screenshot(full_path, window_size=(1600, 900))
        plotter_off.close()