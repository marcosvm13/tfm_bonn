import cupy as np
from ..utils.losses import log_loss


def test(batch_size, X_test, y_test, model, epoch, verbose=True, hidden_layer_mult=1, mode="ALL"):
    """
    Evalúa el modelo sobre un conjunto de test.

    Args:
        batch_size (int): Tamaño del batch.
        X_test (np.ndarray): Datos de entrada.
        y_test (np.ndarray): Etiquetas verdaderas.
        model: Instancia del modelo BONN.
        epoch (int): Número de época.
        verbose (bool): Si imprimir métricas.
        hidden_layer_mult (int): Multiplicador del tamaño de la capa oculta.
        mode (str): Modo de entrenamiento ('ALL', 'perceptron', etc).

    Returns:
        Tuple: (pérdida media, accuracy, actividad media, accuracy final)
    """
    perdida = 0
    accuracy = 0
    N = X_test.shape[0]
    int_layer = np.zeros(int(2 * 15 * 15 * hidden_layer_mult))
    num_batches = range(int(np.ceil(N / batch_size)))

    for n_batch in num_batches:
        batch = np.arange(batch_size * n_batch,
                          min(N, batch_size * (n_batch + 1)))
        inputs = X_test[batch]
        targets = y_test[batch]

        outputs, y = model.forward(inputs)
        int_layer += np.sum(y, 0)

        perdida += log_loss(targets, outputs)
        accuracy += np.sum(np.argmax(outputs, axis=1) ==
                           np.argmax(targets, axis=1))

    if verbose:
        print(
            f"(TEST) Época {epoch}: Loss: {perdida / len(num_batches):.4f}; Accuracy: {accuracy / N:.4f}")

    return perdida / len(num_batches), accuracy / N, np.mean(int_layer / N), accuracy / N


def validate(X_test, y_test, model, val=0.7):
    """
    Realiza una validación sobre el conjunto de test.

    Args:
        X_test (np.ndarray): Datos de entrada.
        y_test (np.ndarray): Etiquetas esperadas.
        model: Modelo BONN ya entrenado.
        val (float): Umbral de activación.

    Returns:
        Tuple: Media de actividad, neuronas activas, fracción de activación > val, accuracy.
    """
    N = X_test.shape[0]
    outputs, activaciones = model.forward(X_test)
    accuracy = np.sum(np.argmax(outputs, axis=1) == np.argmax(y_test, axis=1))
    return (np.mean(activaciones, axis=0),
            np.sum(activaciones * (activaciones > val), axis=0),
            np.mean(activaciones > val),
            accuracy / N,
            outputs,
            activaciones)


def predict(X_test, model):
    """
    Ejecuta una predicción directa sobre el conjunto dado.

    Args:
        X_test (np.ndarray): Datos de entrada.
        model: Instancia del modelo.

    Returns:
        Tuple: Activaciones internas, salida final.
    """
    outputs, activaciones = model.forward(X_test)
    return activaciones, outputs
