from abc import ABC, abstractmethod


class Model(ABC):
    """
    Clase base abstracta para modelos.
    """

    @abstractmethod
    def __init__(self):
        """Inicializa los componentes del modelo."""
        pass

    @abstractmethod
    def forward(self, x):
        """
        Ejecuta la inferencia hacia adelante.

        Args:
            x (np.ndarray): Entrada al modelo.

        Returns:
            Salida del modelo.
        """
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        """
        Calcula los gradientes del modelo durante entrenamiento.
        """
        pass

    @abstractmethod
    def step(self):
        """
        Aplica el paso de optimización a todos los parámetros aprendibles.
        """
        pass
