from .validate import validate
import math
import cupy as np
import pandas as pd


def actividad(X, model):
    """Calcula la suma total de actividad por neurona del modelo dado un conjunto X."""
    return np.sum(model.forward(X)[1], axis=0)


def particion(X_valid, y_valid, model):
    """
    Divide el conjunto de validación por clase y calcula la activación significativa por neurona.

    Returns:
        pd.DataFrame: Activaciones por clase (normalizadas).
    """
    datasets_X, datasets_y = [], []
    for i in range(10):
        idx = np.argmax(y_valid, axis=1) == i
        datasets_X.append(X_valid[idx])
        datasets_y.append(y_valid[idx])

    resultados = {
        str(i): validate(dat_x, dat_y, model, val=0.5)[1].get()
        for i, (dat_x, dat_y) in enumerate(zip(datasets_X, datasets_y))
    }
    df = pd.DataFrame(resultados)
    return (df.T / df.T.sum()).T


def number_of_trainable_parameters(model, model_type="all"):
    """
    Devuelve el número de parámetros entrenables según el tipo de modelo.
    """
    first_w = model.first.weight.size
    first_b = model.first.bias.size
    output_w = model.output.weight.size
    output_b = model.output.bias.size

    model_type = model_type.lower()
    if model_type == "all":
        return first_b + output_b + output_w
    elif model_type == "ob":
        return first_b + output_b
    elif model_type == "ow":
        return output_w
    elif model_type == "perceptron":
        return first_b + first_w + output_b + output_w


def number_of_neuron_for_n_trainable_parameter(n, input_dim, output_dim, model_type="all"):
    """
    Estima cuántas neuronas ocultas se pueden usar para alcanzar n parámetros.
    """
    model_type = model_type.lower()
    if model_type == "all":
        return (n - output_dim) / (output_dim + 1)
    elif model_type == "ob":
        return n - output_dim
    elif model_type == "ow":
        return n / output_dim
    elif model_type == "perceptron":
        return (n - output_dim) / (input_dim + output_dim + 1)


def get_neurons_equivalence(model, model_input_type, model_output_type):
    n = number_of_trainable_parameters(model, model_input_type)
    return number_of_neuron_for_n_trainable_parameter(n, model.output.bias.shape[0], model.first.weight.shape[0], model_output_type)


def get_hidden_mult(model, model_input_type, model_output_type, tam_img=15):
    n = get_neurons_equivalence(model, model_input_type, model_output_type)
    return n / (tam_img * tam_img)
