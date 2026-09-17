import numpy as np


def print_image(pattern):
    symbols = {1: "#", -1: "."}
    img = pattern.reshape(10, 10)
    for row in img:
        print(" ".join(symbols[val] for val in row))
    print()


# Calcula la matriz de pesos sinápticos usando la regla de Hebb (diagonal nula para evitar auto-conexiones)
def hebbian_learning(patterns):
    n = patterns.shape[1]
    w = np.zeros((n, n))
    for p in patterns:
        w += np.outer(p, p)
    np.fill_diagonal(w, 0)
    return w / len(patterns)


# Recupera el patrón iterando sobre las neuronas y actualizando sus estados según la función de activación
def recall(w, pattern, steps=5):
    state = np.copy(pattern)
    for _ in range(steps):
        for i in range(len(state)):
            activation = np.dot(w[i], state)
            state[i] = 1 if activation >= 0 else -1
    return state


def main():
    # Representación visual del patrón objetivo (10x10):
    # . . . . . . . . . .
    # . . . . . . . . . .
    # . . . . . . . . . .
    # . . . # # # # . . .  <- aro/figura central
    # . . . # . . # . . .
    # . . . # . . # . . .
    # . . . # # # # . . .
    # . . . . . . . . . .
    # # . . . . . . . .    <- referencia angular en esquina
    # # . . . . . . . .
    target = np.full((10, 10), -1)
    target[8:10, 0:2] = 1  # referencia fija en esquina
    target[3:7, 3:7] = 1   # figura central
    target[4:6, 4:6] = -1

    pattern = target.flatten()
    w = hebbian_learning(np.array([pattern]))

    # Introducir 20% de ruido invirtiendo pixeles
    np.random.seed(42)
    noisy = pattern.copy()
    noise_idx = np.random.choice(100, 20, replace=False)
    noisy[noise_idx] *= -1

    print("Patrón objetivo:")
    print_image(pattern)

    print("Entrada con ruido:")
    print_image(noisy)

    recovered = recall(w, noisy)
    print("Patrón recuperado:")
    print_image(recovered)


if __name__ == "__main__":
    main()
