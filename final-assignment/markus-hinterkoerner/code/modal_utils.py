def compute_mode_shapes(data, haus, knoten_liste, n_modes=4, peak_height=0.002):
    # Bestimmt die ersten n_modes Modeformen eines Hochhauses
    # aus dem Imaginärteil der FRFs.
    
    import numpy as np
    from scipy.signal import find_peaks

    ref = knoten_liste[0]  # Referenzknoten für Peaks
    f_ref = data[haus][ref]["f"] # Frequenzen des Referenzknotens
    im_ref = np.abs(data[haus][ref]["Im"]) # Imaginärteil des Referenzknotens

    peaks, _ = find_peaks(im_ref, height=peak_height) # Peaks finden

    mode_shapes = [] # Liste für Modeformen
    mode_freqs = [] # Liste für Eigenfrequenzen

    for peak in peaks[:n_modes]: # Nur die ersten n_modes Peaks verwenden
        f_mode = f_ref[peak] # Eigenfrequenz des Modes
        shape = [] # Modeform für diesen Mode

        for knoten in knoten_liste: # Schleife über alle Knoten
            f = data[haus][knoten]["f"] # Frequenzen des Knotens
            im = data[haus][knoten]["Im"] # Imaginärteil des Knotens

            idx = np.argmin(np.abs(f - f_mode)) # Index des Frequenzwerts nahe der Eigenfrequenz
            shape.append(im[idx]) # Imaginärteil an dieser Frequenz

        shape = np.array(shape) # In NumPy-Array umwandeln
        shape = shape / np.max(np.abs(shape))  # Normierung der Modeform

        mode_shapes.append(shape) # Modeform speichern
        mode_freqs.append(f_mode) # Eigenfrequenz speichern

    return mode_freqs, mode_shapes # Rückgabe der Eigenfrequenzen und Modeformen
