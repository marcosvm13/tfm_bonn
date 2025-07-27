# Uso del proyecto BONN

A continuación se detallan ejemplos de cómo ejecutar diferentes experimentos y modelos desde el repositorio.

## 1. Ejecutar BONN con entrenamiento superficial

```bash
python src/app.py
```

Este script entrena el modelo BONN con una sola capa entrenable tras proyecciones fijas y mide su precisión.

## 2. Ejecutar GateBONN (PyTorch)

```bash
python src/gatebonn/gatebonn_experiment.py
```

Este script entrena GateBONN sobre MNIST para distintos tamaños de entrenamiento y devuelve un DataFrame con la precisión media y desviación estándar.

## 3. Evaluar ensambles BONN

```python
from src.ensemble.ensemble_bonn import EnsembleBONN
```

Puedes construir un ensamble de modelos BONN con:

```python
ensemble = EnsembleBONN(fit_fn=my_fit_function)
accs = ensemble.evaluate_ensemble(X_test, y_test, ensemble_sizes=[1, 2, 5, 10])
```

## 4. Análisis de activación y escasez

```python
from src.analysis.multimodality import analyze_kc_digit_multimodality
from src.analysis.neuron_ranking import kill_progressive_neurons
```

Estos módulos permiten medir multimodalidad y simular pruning sobre las KCs.

## 5. Visualización de resultados

```python
from src.analysis.plot_utils import plot_surfaces_adjusted
from src.analysis.pruning_analysis import plot_pruning_progress_stylized
```

Las funciones de visualización permiten comparar arquitecturas y evaluar degradación de precisión ante eliminación de neuronas.
