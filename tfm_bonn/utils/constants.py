import cupy as np


def compute_constant(constant, s=0.1, ord=2, mode="fix"):
    """
    Devuelve una función que ajusta dinámicamente una constante regularizadora.

    Args:
        constant (float): Valor base de la constante.
        s (float): Nivel objetivo de activación.
        ord (int): Orden de la norma (para modo 'gradient_norm').
        mode (str): Modo de ajuste ('fix', 'mean', 'activity', 'variance', 'gradient_norm').

    Returns:
        Callable: Función que recibe `outputs` y devuelve un escalar.
    """
    if mode == "fix":
        return lambda outputs: constant
    elif mode == "mean":
        return lambda outputs: constant * np.mean(np.abs(outputs))
    elif mode == "activity":
        return lambda outputs: constant * (1 + abs(np.mean(outputs) - s))
    elif mode == "variance":
        return lambda outputs: constant * np.var(outputs)
    elif mode == "gradient_norm":
        return lambda outputs: constant * np.linalg.norm(outputs, ord=ord)
    else:
        raise ValueError(f"Unknown mode for adaptive constant: {mode}")
