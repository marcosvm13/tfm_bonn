# EJEMPLO DE CÓDIGO PARA BONN

from tfm_bonn.data.load_data import cargar_datos
from tfm_bonn.models.bonn import BONN
from tfm_bonn.training.train import train
from tfm_bonn.training.validate import test, validate
from tfm_bonn.training.metrics import actividad, number_of_trainable_parameters

import cupy as np

if __name__ == '__main__':
    # Configuración
    batch_size = 64
    epochs = 5
    tam_img = 15
    hidden_layer_mult = 10

    # Cargar datos
    N, X_train, y_train, X_valid, y_valid, *_ = cargar_datos(perceptron=False)

    # Instanciar modelo BONN
    model = BONN(
        hidden_layer_mult=hidden_layer_mult,
        s1=0.1,
        pc=0.1,
        reg=True,
        constant1=1,
        lr=0.01,
        weight_learned=False,
        pc_method="constant",
        bias_method="simple",
        act="sigmoid",
        reg_method="kl",
        pretrained_weights=None,
        noise_type="Any",
        noise_level=0.1,
        mode="ALL",
        tam_img=tam_img
    )

    print("Parámetros entrenables:", number_of_trainable_parameters(model))

    # Entrenamiento
    for epoch in range(epochs):
        train(batch_size, X_train, y_train, model, epoch, verbose=True)
        test(batch_size, X_valid, y_valid, model, epoch, verbose=True)

    # Validación final
    _, _, actividad_ratio, acc, *_ = validate(X_valid, y_valid, model, val=0.7)
    print(f"\nActividad promedio > 0.7: {actividad_ratio:.4f}")
    print(f"Precisión final: {acc:.4f}")
