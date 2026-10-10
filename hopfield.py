import numpy as np


def print_image(pattern):
    symbols = {1: "#", -1: "."}
    img = pattern.reshape(10, 10)
    for row in img:
        print(" ".join(symbols[val] for val in row))
    print()


def create_ring_pattern(cx, cy):
    grid = np.full((10, 10), -1)
    # Escuadra de referencia fija inalterable en esquina inferior izquierda
    grid[8:10, 0:2] = 1
    # Geometria del aro
    r_start, r_end = cy - 2, cy + 2
    c_start, c_end = cx - 2, cx + 2
    grid[r_start:r_end, c_start:c_end] = 1
    grid[cy - 1 : cy + 1, cx - 1 : cx + 1] = -1
    return grid


def hebbian_learning(patterns):
    """Calcula la matriz de pesos sinapticos usando la regla de Hebb clasica."""
    n = patterns.shape[1]
    w = np.zeros((n, n))
    for p in patterns:
        w += np.outer(p, p)
    np.fill_diagonal(w, 0)
    return w / len(patterns)


def pseudoinverse_learning(patterns):
    """Calcula la matriz de pesos mediante proyeccion pseudoinversa (regla de proyeccion)."""
    X = patterns.T
    w = X @ np.linalg.pinv(X)
    np.fill_diagonal(w, 0)
    return w


def recall(w, pattern, steps=10):
    """Recupera el patron iterando sobre las neuronas de forma recurrente."""
    state = np.copy(pattern)
    for _ in range(steps):
        for i in range(len(state)):
            activation = np.dot(w[i], state)
            state[i] = 1 if activation >= 0 else -1
    return state


def extract_coordinates(pattern_flat):
    """Extrae las coordenadas del centro del aro descartando la escuadra fija."""
    grid = pattern_flat.reshape(10, 10).copy()
    grid[8:10, 0:2] = -1  # Excluir referencia fija
    y_coords, x_coords = np.where(grid == 1)
    if len(x_coords) == 0:
        return None, None
    cx = float(np.mean(x_coords)) + 0.5
    cy = float(np.mean(y_coords)) + 0.5
    return round(cx, 1), round(cy, 1)


def main():
    # 1. Definicion de multiples patrones de posicion del motor
    p1 = create_ring_pattern(cx=5, cy=4)  # Posicion nominal (A0)
    p2 = create_ring_pattern(cx=7, cy=4)  # Desplazamiento horizontal (+X)
    p3 = create_ring_pattern(cx=5, cy=2)  # Desplazamiento transversal (-Y)

    patterns = np.array([p1.flatten(), p2.flatten(), p3.flatten()])

    # 2. Entrenamiento con ambos metodos del Modulo 3
    w_hebb = hebbian_learning(patterns)
    w_pinv = pseudoinverse_learning(patterns)

    # 3. Prueba de recuperacion con 20% de ruido en el Patron 1
    np.random.seed(42)
    target = p1.flatten()
    noisy = target.copy()
    noise_idx = np.random.choice(100, 20, replace=False)
    noisy[noise_idx] *= -1

    print("=== PATRON OBJETIVO (POSICION NOMINAL) ===")
    print_image(target)

    print("=== ENTRADA CON 20% DE RUIDO ===")
    print_image(noisy)

    rec_hebb = recall(w_hebb, noisy)
    rec_pinv = recall(w_pinv, noisy)

    print("=== PATRON RECUPERADO (REGLA DE HEBB) ===")
    print_image(rec_hebb)
    coord_hebb = extract_coordinates(rec_hebb)
    print(f"Coordenadas del centro detectado: X={coord_hebb[0]}, Y={coord_hebb[1]}")
    print(f"Recuperacion exacta: {np.array_equal(rec_hebb, target)}\n")

    print("=== PATRON RECUPERADO (MATRIZ PSEUDOINVERSA) ===")
    print_image(rec_pinv)
    coord_pinv = extract_coordinates(rec_pinv)
    print(f"Coordenadas del centro detectado: X={coord_pinv[0]}, Y={coord_pinv[1]}")
    print(f"Recuperacion exacta: {np.array_equal(rec_pinv, target)}\n")

    # 4. Evaluacion sistematica de todos los patrones
    print("=== EVALUACION SISTEMATICA DE RECUPERACION (20% RUIDO) ===")
    nombres = ["Posicion Nominal", "Desplazamiento +X", "Desplazamiento -Y"]
    coords_teoricas = [(5.0, 4.0), (7.0, 4.0), (5.0, 2.0)]
    for idx, (p, nombre, coord_esp) in enumerate(zip([p1, p2, p3], nombres, coords_teoricas), 1):
        test_noisy = p.flatten().copy()
        test_noise_idx = np.random.choice(100, 20, replace=False)
        test_noisy[test_noise_idx] *= -1
        res_pinv = recall(w_pinv, test_noisy)
        c_calc = extract_coordinates(res_pinv)
        exito = np.array_equal(res_pinv, p.flatten())
        print(f"Patron {idx} ({nombre}): Esperado={coord_esp} | Detectado={c_calc} | Exacto={exito}")


if __name__ == "__main__":
    main()
