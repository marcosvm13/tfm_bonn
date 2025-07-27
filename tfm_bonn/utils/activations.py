import cupy as np


def sigmoid(x):
    """Función sigmoide."""
    return 1 / (1 + np.exp(-x))


def sigmoid_backward(Z):
    """Derivada de la función sigmoide."""
    return Z * (1 - Z)


def relu(x):
    """Función ReLU (Rectified Linear Unit)."""
    return np.greater(x, 0.).astype(np.float32)


def relu_backward(Z):
    """Derivada de la función ReLU."""
    return np.greater(Z, 0.).astype(np.float32)


def softmax(x):
    """Función softmax aplicada a cada columna."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def softmax_backward(Z):
    """Derivada simplificada del softmax (solo válida en casos especiales)."""
    return Z * (1 - Z)


def tanh(x):
    """Función tangente hiperbólica."""
    return np.tanh(x)


def tanh_backward(Z):
    """Derivada de la función tanh."""
    return 1 - Z ** 2


def f_factory(f):
    """
    Devuelve la función de activación y su derivada correspondiente.

    Args:
        f (str): Nombre de la función ('sigmoid', 'relu', 'tanh', 'softmax').

    Returns:
        Tuple[Callable, Callable]: Función de activación y su derivada.
    """
    if f == "sigmoid":
        return sigmoid, sigmoid_backward
    elif f == "relu":
        return relu, relu_backward
    elif f == "tanh":
        return tanh, tanh_backward
    elif f == "softmax":
        return softmax, softmax_backward
    else:
        return sigmoid, sigmoid_backward
