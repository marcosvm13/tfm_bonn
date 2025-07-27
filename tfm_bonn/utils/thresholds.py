import cupy as np
import inspect
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu


def apply_threshold(matrix, threshold):
    """Aplica un umbral binario sobre la matriz."""
    return (matrix > threshold).astype(float)


def mean_threshold(matrix):
    """Umbral basado en la media."""
    threshold = np.mean(matrix)
    return apply_threshold(matrix, threshold)


def median_threshold(matrix):
    """Umbral basado en la mediana."""
    threshold = np.median(matrix)
    return apply_threshold(matrix, threshold)


def percentile_threshold(matrix, percentile=50):
    """Umbral basado en un percentil dado."""
    threshold = np.percentile(matrix, percentile)
    return apply_threshold(matrix, threshold)


def z_score_threshold(matrix):
    """Umbral basado en la puntuación z estándar."""
    z_scores = (matrix - np.mean(matrix)) / np.std(matrix)
    return apply_threshold(z_scores, 0)


def kmeans_threshold(matrix):
    """Umbral basado en clustering con K-Means (2 clusters)."""
    data = matrix.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42).fit(data.get())
    labels = kmeans.labels_
    return np.array(labels.reshape(matrix.shape))


def otsu_threshold(matrix):
    """Umbral óptimo calculado por el método de Otsu."""
    threshold = threshold_otsu(matrix.get())
    return apply_threshold(np.array(matrix), threshold)


def normalize_threshold(matrix):
    """Umbral fijo aplicado tras normalización entre 0 y 1."""
    normalized = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    return apply_threshold(normalized, 0.5)


def sign_based_threshold(matrix):
    """Umbral binario basado en el signo."""
    return apply_threshold(np.sign(matrix), 0)


def custom_threshold(matrix, threshold=0.8):
    """Umbral personalizado basado en un valor dado."""
    return apply_threshold(np.sign(matrix), threshold)


def binarization_factory(method_name, matrix, **kwargs):
    """
    Fábrica de binarización: aplica el método seleccionado al tensor dado.

    Args:
        method_name (str): Nombre del método ('mean', 'median', etc.).
        matrix (np.ndarray): Matriz a umbralizar.
        **kwargs: Parámetros opcionales específicos del método.

    Returns:
        np.ndarray: Matriz binarizada.
    """
    methods = {
        "mean": mean_threshold,
        "median": median_threshold,
        "percentile": percentile_threshold,
        "z_score": z_score_threshold,
        "kmeans": kmeans_threshold,
        "otsu": otsu_threshold,
        "normalize": normalize_threshold,
        "sign_based": sign_based_threshold,
        "custom_logic": custom_threshold,
    }

    if method_name not in methods:
        raise ValueError(
            f"Unknown method '{method_name}'. Available methods are: {list(methods.keys())}")

    method = methods[method_name]
    method_signature = inspect.signature(method)
    valid_kwargs = {k: v for k, v in kwargs.items(
    ) if k in method_signature.parameters}
    return method(matrix, **valid_kwargs)
