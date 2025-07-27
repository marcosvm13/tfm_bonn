import cupy as np
import inspect
from functools import wraps
from .constants import compute_constant


def regularization_decorator(func):
    """
    Decorador para funciones de regularización. Aplica la constante adaptativa y la normalización si es necesario.
    """
    @wraps(func)
    def wrapper(outputs, over_output_size=False, constant_type=None, **kwargs):
        func_signature = inspect.signature(func)
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k in func_signature.parameters}
        if constant_type is None:
            constant_type = compute_constant(1, mode="fix")
        gradient = func(outputs, **filtered_kwargs)
        gradient *= constant_type(outputs)
        if over_output_size:
            gradient /= outputs.size
        return gradient
    return wrapper


@regularization_decorator
def kl_gradient(outputs, s=0.1):
    """
    Calcula el gradiente de la divergencia KL entre la distribución de salida y una distribución objetivo.
    """
    target = np.random.normal(loc=s, scale=0.1, size=outputs.shape)
    epsilon = 1e-10
    output_distribution = np.clip(outputs, epsilon, None)
    output_distribution /= np.sum(output_distribution, axis=1, keepdims=True)
    target = np.clip(target, epsilon, None)
    target /= np.sum(target, axis=1, keepdims=True)
    return np.sum(output_distribution - target, axis=0)


@regularization_decorator
def l2_gradient(outputs, s=0.1):
    """
    Calcula el gradiente L2 entre las salidas y un valor objetivo.
    """
    return np.mean((outputs - s) ** 2)


@regularization_decorator
def mse_gradient(outputs, s=0.1):
    """
    Calcula el gradiente MSE entre las salidas y un valor objetivo.
    """
    return np.mean(outputs) - s


@regularization_decorator
def variance_gradient(outputs, s=0.1):
    """
    Calcula el gradiente de varianza entre las salidas y un valor objetivo.
    """
    return np.var(outputs) - s


@regularization_decorator
def activity(outputs, s=0.1):
    """
    Calcula el gradiente de actividad entre las salidas y un valor objetivo.
    """
    return np.mean(outputs, axis=0) - s


@regularization_decorator
def entropy(outputs, s=0.1):
    """
    Calcula la entropía de las salidas.
    """
    p = np.clip(outputs, 1e-8, None)
    p /= np.sum(p, axis=1, keepdims=True)
    entropy_vals = -np.sum(p * np.log(p), axis=1)
    return -np.mean(entropy_vals, axis=0)


@regularization_decorator
def lateral_inhibition(outputs):
    """
    Calcula la inhibición lateral entre las salidas.
    """
    similarity = outputs.T @ outputs / outputs.shape[1]
    np.fill_diagonal(similarity, 0)
    return np.sum(similarity ** 2, axis=1)


@regularization_decorator
def winner_take_most(outputs, k=5):
    """
    Aplica un umbral de "ganador se lleva la mayoría" a las salidas.
    Selecciona los k mayores valores y resta el umbral.
    """
    top_k = np.partition(outputs, -k, axis=1)[:, -k:]
    return outputs - top_k


@regularization_decorator
def diversity(outputs):
    """
    Calcula la diversidad de las salidas como la suma de las varianzas de cada neurona.
    """

    covariance = np.cov(outputs.T)
    return np.sum(np.abs(covariance))


@regularization_decorator
def synaptic_sacling(outputs, adaptive_constant=1, mode="fix", ord=2):
    """
    Escala sináptica adaptativa basada en la norma de las activaciones.
    Args:
        outputs (np.ndarray): Activaciones de la capa.
        adaptive_constant (float): Valor base de la constante adaptativa.
        mode (str): Modo de ajuste de la constante ('fix', 'mean', 'activity', 'variance', 'gradient_norm').
        ord (int): Orden de la norma para el modo 'gradient_norm'.
    Returns:
        np.ndarray: Escala adaptativa para las activaciones.
    """
    constant = compute_constant(adaptive_constant, mode=mode, ord=ord)(outputs)
    scale = np.linalg.norm(outputs, axis=0) + 1e-8
    return constant / scale


@regularization_decorator
def penalize_activity(outputs, s=0.1):
    """
    Penaliza la actividad de las neuronas que superan un umbral s.
    """
    return np.maximum(0, s - np.mean(outputs, axis=0))


@regularization_decorator
def decorrelation(outputs):
    """
    Calcula la decorrelación de las activaciones.
    """
    centered = outputs - np.mean(outputs, axis=1, keepdims=True)
    cov = centered.T @ centered / (outputs.shape[1] - 1)
    return np.sum(np.abs(cov), axis=0)


@regularization_decorator
def rotating_active_neurons(regs, s=0.1):
    """
    Aplica una rotación a las neuronas activas.
    """

    dead = np.random.binomial(1, 1 - s, size=regs.shape[0])
    regs[dead] = 0
    return regs


@regularization_decorator
def energy_efficiency(outputs):
    """
    Calcula la eficiencia energética como la suma de los cuadrados de las activaciones.
    """
    return np.sum(outputs ** 2, axis=0)


@regularization_decorator
def laplacian_loss(outputs, bias, sim_mtrx):
    """
    Calcula la pérdida de Laplaciano para regularizar la estructura de las activaciones.
    Args:
        outputs (np.ndarray): Activaciones de la capa.
        bias (np.ndarray): Sesgos de la capa.
        sim_mtrx (np.ndarray): Matriz de similitud entre las activaciones.
    Returns:
        np.ndarray: Pérdida de Laplaciano.
    """
    D = np.diag(sim_mtrx.sum(axis=1))
    D_inv = np.diag(1.0 / np.sqrt(D.diagonal()))
    L = D - sim_mtrx
    L_norm = D_inv @ L @ D_inv
    return bias @ (L_norm @ bias)


def gradient_factory(outputs, s=0.1, constant_type=None, over_output_size=False, tipo=["kl"], **kwargs):
    """
    Fábrica de regularización: aplica los métodos seleccionados sobre los outputs.

    Args:
        outputs (np.ndarray): Activaciones.
        s (float): Nivel deseado de activación.
        constant_type: Escalado adaptativo (puede ser lista).
        over_output_size (bool): Normaliza por el número de salidas.
        tipo (list|str): Métodos a aplicar.

    Returns:
        np.ndarray: Gradiente combinado.
    """
    funcs = {
        "kl": kl_gradient,
        "l2": l2_gradient,
        "mse": mse_gradient,
        "variance": variance_gradient,
        "activity": activity,
        "entropy": entropy,
        "decorrelation": decorrelation,
        "lateral_inhibition": lateral_inhibition,
        "synaptic_sacling": synaptic_sacling,
        "penalize_activity": penalize_activity,
        "energy_efficiency": energy_efficiency,
        "laplacian_loss": laplacian_loss,
        "winner_take_most": winner_take_most,
        "diversity": diversity
    }

    tipo = [tipo] if isinstance(tipo, str) else tipo
    if not isinstance(constant_type, list):
        constant_type = [constant_type] * len(tipo)

    reg = np.zeros(outputs.shape[1])
    for t, c in zip(tipo, constant_type):
        func = funcs.get(t)
        if func:
            term = func(outputs, constant_type=c,
                        over_output_size=over_output_size, **kwargs)
            if term.ndim == 0:
                reg += np.full(outputs.shape[1], term)
            else:
                reg += term

    if 'rotating_active_neurons' in tipo:
        reg = rotating_active_neurons(reg, s)

    return reg
