from abc import ABC, abstractmethod


class Layer(ABC):
    """
    Clase base abstracta para definir la interfaz de una capa en la red.
    """

    @abstractmethod
    def __init__(self):
        """Inicializa los parámetros de la capa."""
        pass

    @abstractmethod
    def forward(self, x):
        """
        Ejecuta la propagación hacia adelante (inferencia).

        Args:
            x (np.ndarray): Entrada a la capa.

        Returns:
            np.ndarray: Salida de la capa.
        """
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        """
        Calcula los gradientes para la retropropagación.

        Returns:
            Gradientes respecto a entradas y parámetros.
        """
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        """
        Aplica el paso de optimización a los parámetros.
        """
        pass
