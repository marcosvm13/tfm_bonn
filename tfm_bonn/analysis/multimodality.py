import cupy as np
import matplotlib.pyplot as plt


def analyze_kc_digit_multimodality(activations, y_test, C=10, output_path="kc_digit_effective_score.eps"):
    """
    Analiza cuántos dígitos activa cada KC en promedio (ponderado) y genera un histograma.

    Parámetros:
        activations (np.ndarray): Activaciones de las KCs, de tamaño (N, D).
        y_test (np.ndarray): Etiquetas verdaderas de los datos de entrada.
        C (int): Número de clases o categorías.
        output_path (str): Ruta donde guardar la figura generada.
    """
    activations_np = np.asnumpy(activations)
    y_np = np.asnumpy(y_test)
    N, D = activations_np.shape
    kc_digit_activity = np.zeros((D, C))

    for digit in range(C):
        indices = np.where(y_np == digit)[0]
        A = activations_np[indices, :]
        kc_digit_activity[:, digit] = A.mean(axis=0)

    max_per_kc = kc_digit_activity.max(axis=1, keepdims=True) + 1e-8
    kc_digit_score = kc_digit_activity.sum(axis=1) / max_per_kc[:, 0]

    plt.figure(figsize=(10, 6))
    plt.hist(kc_digit_score, bins=30, density=True,
             edgecolor='black', alpha=0.7)
    plt.xlabel("Nº efectivo de dígitos que activan cada KC (ponderado)")
    plt.ylabel("Frecuencia (normalizada)")
    plt.title("Multimodalidad en sensibilidad por KC (score continuo)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, format='eps')
    plt.show()

    return kc_digit_score
