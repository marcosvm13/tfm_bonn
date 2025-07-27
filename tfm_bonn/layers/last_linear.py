import cupy as np
import jax.numpy as jnp
import jax
from .base_layer import Layer
from ..utils.activations import sigmoid


class LastLinear(Layer):
    """
    Capa de salida con pesos y sesgos aprendibles.
    """

    def __init__(self, input_dim, output_dim):
        self.weight = np.random.uniform(-1, 1,
                                        size=(input_dim, output_dim)).astype(float)
        self.bias = np.random.uniform(0, 1, size=(output_dim)).astype(float)

        self.inputs = None
        self.outputs = 0
        self.gradient = 0

        self.db = 0
        self.dW = 0

    def forward(self, x):
        self.inputs = x.copy()
        self.outputs = sigmoid(np.dot(self.inputs, self.weight) - self.bias)
        return self.outputs.copy()

    def predict(self, x):
        return jax.nn.sigmoid(jnp.dot(x, self.weight.get()) - self.bias.get())

    def backward(self, targets, weight_train=False):
        self.gradient = self.outputs - targets
        self.dW = np.dot(
            self.inputs.T, self.gradient) if weight_train else np.zeros_like(self.weight)
        self.db = np.sum(self.gradient, axis=0)
        return self.gradient, self.db, None

    def step(self, lr=0.01):
        self.bias -= lr * self.db
        self.weight -= lr * self.dW
