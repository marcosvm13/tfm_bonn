import cupy as np
from .activations import sigmoid
from .thresholds import median_threshold
from scipy.special import logit as scipy_logit


def logit(s):
    """Calcula el logit (inversa de la sigmoide)."""
    return scipy_logit(s) if s != 1. else 0


def median_output(weights, input=None):
    """Salida media ponderada por los pesos o vector constante si no hay input."""
    if input is not None:
        result = input.mean(axis=0, keepdims=True).dot(weights)
    else:
        result = np.full(weights.shape[1], 0.5).dot(weights)
    return result


def simple_logit(weights, s, input=None):
    """Inicialización basada en logit simple respecto al target s."""
    return median_output(weights, input) - logit(s)


def heuristic_gaussian_bias_init(weights, s, input=None, sigma=0.1):
    """Inicialización heurística gaussiana centrada en logit(s)."""
    mean_bias = logit(s)
    biases = np.random.normal(mean_bias, sigma, size=weights.shape[0])
    return median_output(weights, input) - biases


def heuristic_uniform_bias_init(weights, s, input=None, delta=0.1):
    """Inicialización heurística uniforme centrada en logit(s)."""
    mean_bias = logit(s)
    biases = np.random.uniform(
        mean_bias - delta, mean_bias + delta, size=weights.shape[0])
    return median_output(weights, input) - biases


def adaptive_scaling_bias_init(weights, s, input=None):
    """Inicialización adaptativa escalando según número de pesos activos."""
    mean_bias = logit(s)
    active_weights_count = np.sum(weights, axis=0)
    avg_active_weights = np.mean(active_weights_count)
    biases = mean_bias * (active_weights_count / avg_active_weights)
    return median_output(weights, input) - biases


def variance_scaling_bias_init(weights, s, input=None):
    """Inicialización de sesgos basada en varianza (tipo Glorot o He)."""
    n_out = np.sum(weights, axis=1)
    biases = np.random.normal(0, np.sqrt(2.0 / n_out), size=weights.shape[0])
    return biases


def adaptive_centering_bias_init(weights, s, input=None, eta=0.01, initial_forward_outputs=None):
    """Centrado adaptativo basado en outputs iniciales y un paso de gradiente suave."""
    mean_bias = logit(s)
    if initial_forward_outputs is None:
        biases = np.full(weights.shape[0], mean_bias)
    else:
        output_diff = s - initial_forward_outputs.mean(axis=0)
        biases = mean_bias + eta * output_diff
    return median_output(weights, input) - biases


def bias_factory(s, weights, input=None, tipo="simple", **args):
    """
    Fábrica de inicialización de sesgos.

    Args:
        s (float): Valor objetivo de activación.
        weights (np.ndarray): Matriz de pesos.
        input (np.ndarray): Entradas opcionales para estimar activación.
        tipo (str): Tipo de inicialización ('simple', 'heuristic_gaussian', ...).
        **args: Parámetros adicionales según el método.

    Returns:
        np.ndarray: Vector de sesgos inicializado.
    """
    if tipo == "simple":
        return simple_logit(weights, s, input)
    if tipo == "heuristic_gaussian":
        return heuristic_gaussian_bias_init(weights, s, input, **args)
    if tipo == "heuristic_uniform":
        return heuristic_uniform_bias_init(weights, s, input, **args)
    if tipo == "adaptive_scaling":
        return adaptive_scaling_bias_init(weights, s, input)
    if tipo == "variance_scaling":
        return variance_scaling_bias_init(weights, s, input)
    if tipo == "adaptive_centering":
        return adaptive_centering_bias_init(weights, s, input, **args)
    if tipo == "very_simple":
        return np.random.uniform(0, 1, size=(weights.shape[1])).astype(float)
    return simple_logit(weights, s)
