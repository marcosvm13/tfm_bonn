import cupy as np
from src.training.validate import validate


class NeuronRankingFactory:
    """
    Fábrica de estrategias para ordenar neuronas en base a métricas como activación, pesos, bias, etc.
    """

    def __init__(self):
        self.methods = {
            "random": self.random_order,
            "activation": self.activation_based_order,
            "weight_magnitude": self.weight_magnitude_order,
            "bias_magnitude": self.bias_magnitude_order,
            "saliency": self.saliency_based_order,
            "random_2": self.random_order_2
        }

    def get_order(self, method_name, model, X, ascending=True):
        if method_name not in self.methods:
            raise ValueError(f"Method '{method_name}' is not supported.")
        return self.methods[method_name](model, X, ascending)

    @staticmethod
    def random_order(model, X, ascending=True):
        return np.random.permutation(model.first.bias.shape[1])

    @staticmethod
    def random_order_2(model, X, ascending=True):
        return np.random.permutation(model.first.bias.shape[0])

    @staticmethod
    def activation_based_order(model, X, ascending=True):
        activation = actividad(X, model)
        return np.argsort((1 if ascending else -1) * activation)

    @staticmethod
    def weight_magnitude_order(model, X, ascending=True):
        weight_magnitude = np.sum(model.first.weight, axis=0)
        return np.argsort((1 if ascending else -1) * weight_magnitude)

    @staticmethod
    def bias_magnitude_order(model, X, ascending=True):
        return np.argsort((1 if ascending else -1) * model.first.bias)

    @staticmethod
    def saliency_based_order(model, X, ascending=True):
        def sigmoid(z): return 1 / (1 + np.exp(-z))
        def sigmoid_derivative(z): return sigmoid(z) * (1 - sigmoid(z))

        z1 = np.dot(X, model.first.weight) + model.first.bias
        h1 = sigmoid(z1)
        z2 = np.dot(h1, model.output.weight) + model.output.bias
        y = sigmoid(z2)

        dL_dy = sigmoid_derivative(z2)
        dL_dh1 = np.dot(dL_dy, model.output.weight.T) * sigmoid_derivative(z1)

        contributions = np.abs(dL_dh1 * h1).sum(axis=0)
        return np.argsort(contributions if ascending else -contributions)


def model_number_of_neurons(model):
    """Devuelve el número de neuronas en la primera capa oculta."""
    return model.first.weight.shape[1]


def kill_progressive_neurons(model, layer, X, y, n=10, random_selector="random", ascending=False, number_of_neuron_to_prune=None):
    """
    Apaga progresivamente neuronas en una capa y mide la precisión del modelo tras cada apagado.
    """
    def ablate_neuron(weights, bias, neuron_idx):
        weights[:, neuron_idx] = 0

    accs = []
    index = []
    nrf = NeuronRankingFactory()

    weights = model.first.weight if layer == 0 else model.output.weight
    bias = model.first.bias if layer == 0 else model.output.bias
    original_weights = weights.copy()
    original_bias = bias.copy()

    total_neurons = model_number_of_neurons(model)
    order = nrf.get_order(random_selector, model, X, ascending)
    number_of_neuron_to_prune = total_neurons if number_of_neuron_to_prune == "ALL" else number_of_neuron_to_prune

    for i, idx in enumerate(order):
        ablate_neuron(weights, bias, idx)
        if i % n == 0:
            acc = validate(X, y, model)[3]
            accs.append(acc.get().view().item())
            index.append(i if number_of_neuron_to_prune else i / total_neurons)
        if number_of_neuron_to_prune is not None and i >= number_of_neuron_to_prune:
            break

    model.first.weight = original_weights
    model.first.bias = original_bias

    return accs, index
