import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist, fashion_mnist, cifar10
import tensorflow_datasets as tfds


def cargar_calles():
    """Carga el conjunto de datos SVHN (Street View House Numbers) desde TensorFlow Datasets."""
    (train_ds, test_ds) = tfds.load('svhn_cropped', split=[
        'train', 'test'], shuffle_files=False, as_supervised=True)
    train_X = [example.numpy() for example, _ in train_ds]
    train_y = [label.numpy() for _, label in train_ds]
    valid_X = [example.numpy() for example, _ in test_ds]
    valid_y = [label.numpy() for _, label in test_ds]
    return (np.array(train_X), np.array(train_y)), (np.array(valid_X), np.array(valid_y))


class Datos:
    """
    Clase que encapsula la carga y preprocesamiento de diferentes conjuntos de datos (MNIST, Fashion-MNIST, CIFAR-10, SVHN).

    Atributos:
        train_X (np.ndarray): Datos de entrenamiento preprocesados.
        valid_X (np.ndarray): Datos de validación preprocesados.
        train_y (np.ndarray): Etiquetas de entrenamiento codificadas one-hot.
        valid_y (np.ndarray): Etiquetas de validación codificadas one-hot.
        train_y_n (np.ndarray): Etiquetas originales de entrenamiento.
        valid_y_n (np.ndarray): Etiquetas originales de validación.
        clases (int): Número de clases presentes.
        is_ (list): Lista de clases activas.
        tam_img (int): Tamaño al que se redimensionan las imágenes.
        N (int): Número de muestras de entrenamiento.
    """

    def __init__(self, fashion=False, calles=False, cifar=False, half_mnist=False, perceptron=False, tam_img=15):
        """Inicializa la clase Datos cargando el dataset deseado y normalizando los datos."""
        if cifar:
            (train_X, train_y), (valid_X, valid_y) = cifar10.load_data()
            train_X = tf.image.rgb_to_grayscale(train_X).numpy()
            valid_X = tf.image.rgb_to_grayscale(valid_X).numpy()
        elif calles:
            (train_X, train_y), (valid_X, valid_y) = cargar_calles()
        elif fashion:
            (train_X, train_y), (valid_X, valid_y) = fashion_mnist.load_data()
        else:
            (train_X, train_y), (valid_X, valid_y) = mnist.load_data()

        self.is_ = list(range(10))
        if half_mnist:
            self.is_ = [8, 9]
            train_mask = np.isin(train_y, self.is_)
            test_mask = np.isin(valid_y, self.is_)
            train_X, train_y = train_X[train_mask], train_y[train_mask]
            valid_X, valid_y = valid_X[test_mask], valid_y[test_mask]
            train_y -= min(self.is_)
            valid_y -= min(self.is_)

        self.clases = len(self.is_)
        self.train_y_n = train_y
        self.valid_y_n = valid_y
        self.train_y = to_categorical(train_y)
        self.valid_y = to_categorical(valid_y)
        self.train_X = train_X
        self.valid_X = valid_X

        self.tam_img = tam_img
        self.normalizarDatos(expand=not perceptron)
        self.N = train_X.shape[0]

    def normalizarDatos(self, expand=True):
        """
        Normaliza y transforma los datos de entrada. Puede expandir el canal de entrada y aplicar centrado.

        Args:
            expand (bool): Si True, expande las dimensiones y aplica duplicado negativo.
        """
        def preprocess(X, expand):
            X = np.expand_dims(X, axis=-1)
            X = tf.image.resize(X, [self.tam_img, self.tam_img]).numpy()
            if expand:
                X = np.concatenate((X, np.abs(X - 255)), axis=2)
                X *= 1 / 255.
                X = X * 2 - 1
            else:
                X *= 1 / 255.
            X = X.reshape((X.shape[0], -1))
            return X

        self.train_X = preprocess(self.train_X, expand)
        self.valid_X = preprocess(self.valid_X, expand)


def cargar_datos(perceptron=False):
    """
    Carga y devuelve un objeto Datos con el conjunto seleccionado.

    Args:
        perceptron (bool): Si True, aplica el preprocesamiento sin expansión.

    Returns:
        Tuple: Datos y metadatos asociados para entrenamiento y validación.
    """
    data = Datos(perceptron=perceptron)
    return (data.N, data.train_X, data.train_y, data.valid_X, data.valid_y,
            data.train_y_n, data.valid_y_n, data.is_, data.clases)
