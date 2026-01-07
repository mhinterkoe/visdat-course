def compute_mode_shapes(data, haus, knoten_liste, mode_windows, referenz_knoten, delta_f):
    # Bestimmt die Modeformen eines Hochhauses
    # aus dem Imaginärteil der FRFs.

    import numpy as np # Für numerische Operationen
    from scipy.signal import find_peaks # Für Peak-Erkennung

    mode_freqs = []  # Liste für Eigenfrequenzen der Moden
    mode_shapes = []  # Liste für Modeformen

    # Referenzdaten
    f_ref = data[haus][referenz_knoten]["f"]  # Frequenzen des Referenzknotens
    im_ref = np.abs(data[haus][referenz_knoten]["Im"])  # Imaginärteil des Referenzknotens

    for f_min, f_max in mode_windows: # Schleife über alle Frequenzfenster
        mask = (f_ref >= f_min) & (f_ref <= f_max) # Maske für das aktuelle Fenster
        rel_threshold = 0.15 # relativer Threshold innerhalb des Fensters (15%)

        peaks, props = find_peaks(
            im_ref[mask], # Peaks im Fenster finden
            prominence=rel_threshold * np.max(im_ref[mask]) # minimaler Prominenz (15% des Maximums im Fenster)
        )

        if len(peaks) == 0: # keine Peaks gefunden
            continue

        peak_idx = peaks[np.argmax(im_ref[mask][peaks])] # Index des höchsten Peaks im Fenster
        f_mode = f_ref[mask][peak_idx] # Eigenfrequenz des Modes
        mode_freqs.append(f_mode) # Eigenfrequenz zur Liste hinzufügen

        shape = [] # Liste für die Modeform erstellen
        for knoten in knoten_liste: # Schleife über alle Knoten
            f = data[haus][knoten]["f"] # Frequenzen des Knotens
            im = data[haus][knoten]["Im"] # Imaginärteil des Knotens
            idx = np.argmin(np.abs(f - f_mode)) # Index der Frequenz am nächsten zur Eigenfrequenz
            shape.append(im[idx]) # Imaginärteil an dieser Frequenz zur Modeform-Liste hinzufügen

        shape = np.array(shape) # Modeform-Liste in NumPy-Array umwandeln
        mode_shapes.append(shape) # Modeform zur Liste hinzufügen

    return mode_freqs, mode_shapes  # Rückgabe der Eigenfrequenzen und Modeformen

