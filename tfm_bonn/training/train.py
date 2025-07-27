import cupy as np
from ..utils.losses import log_loss

array_temp = []
array_final = []


def train(batch_size, X_train, y_train, model, epoch, verbose=True, weight_train=True, bias_learned=True):
    """
    Entrena el modelo una época sobre el conjunto de entrenamiento.

    Args:
        batch_size (int): Tamaño del batch.
        X_train (np.ndarray): Datos de entrada.
        y_train (np.ndarray): Etiquetas esperadas.
        model (Model): Instancia del modelo BONN.
        epoch (int): Número de época actual.
        verbose (bool): Si se imprime por pantalla el progreso.
        weight_train (bool): Si se actualizan los pesos.
        bias_learned (bool): Si se actualizan los sesgos.

    Returns:
        Tuple: Pérdida media y precisión sobre el conjunto de entrenamiento.
    """
    global array_temp, array_final
    perdida = 0
    accuracy = 0
    N = y_train.shape[0]
    num_batches = range(int(np.ceil(N / batch_size)))

    for n_batch in num_batches:
        batch_idx = np.arange(batch_size * n_batch,
                              min(N, batch_size * (n_batch + 1)))
        targets = y_train[batch_idx]
        inputs = X_train[batch_idx]

        outputs, _ = model.forward(inputs)
        model.backward(targets, weight_train=weight_train,
                       bias_learned=bias_learned)
        model.step()

        perdida += log_loss(targets, outputs)
        accuracy += np.sum(np.argmax(outputs, axis=1) ==
                           np.argmax(targets, axis=1))

    array_temp_ = np.array(array_temp)
    array_final.append(array_temp_)
    array_temp = []

    if verbose:
        print(
            f"(TRAIN) Época {epoch}: Loss: {perdida / len(num_batches):.4f}; Accuracy: {accuracy / N:.4f}")

    return perdida / len(num_batches), accuracy / N
