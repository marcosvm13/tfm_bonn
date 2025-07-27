# Documentación Técnica del Proyecto BONN

Este repositorio implementa modelos bioinspirados en el sistema olfativo de los insectos, enfocados en tareas de clasificación con pocos datos. Las arquitecturas propuestas reproducen mecanismos funcionales clave del procesamiento sensorial en *Drosophila melanogaster*, como la codificación dispersa, la especialización funcional y la robustez estructural.

## 1. Arquitectura BONN

BONN (Bio-inspired Olfactory Neural Network) simula la vía sensorial AL → KC → MBON. Su estructura consta de:

* **Proyecciones fijas desde AL a KC**: Representadas mediante matrices de conexión binarias, dispersas y parcialmente estructuradas. Esta matriz se inicializa una única vez al comienzo y no se entrena.
* **Codificación dispersa**: Controlada mediante dos hiperparámetros:

  * `s`: tasa de activación objetivo para cada entrada.
  * `pc`: probabilidad de conexión de cada neurona de KC con entradas del AL.
* **Salida entrenable (MBON)**: Una única capa lineal entrenada con descenso por gradiente.
* **Función de activación**: Sigmoide o variantes suaves, usada para mantener la interpretabilidad de las tasas de activación.

La implementación principal está escrita en CuPy para acelerar la ejecución mediante operaciones vectorizadas en GPU. El uso de máscaras binarias permite simular la conectividad estructural AL→KC de forma eficiente.

## 2. GateBONN (versión bayesiana con compuertas)

GateBONN extiende BONN incorporando mecanismos bayesianos de control de activación:

* **Compuertas estocásticas diferenciables**: Se utiliza la relajación de Gumbel-Sigmoid para permitir el paso de gradientes en compuertas binarias.
* **Distribuciones beta**: Cada compuerta tiene una distribución Beta con parámetros aprendidos. Esto permite modelar explícitamente la probabilidad de activación de cada neurona.
* **Regularización por divergencia KL**: Se añade un término en la función de pérdida que penaliza desviaciones respecto a un prior Beta, promoviendo escasez y control de entropía.
* **Cálculo de información mutua**: Se estima la dependencia entre las activaciones internas y las clases objetivo usando un término tipo InfoNCE o una medida basada en codificación cruzada.

Este modelo está implementado completamente en PyTorch, aprovechando su compatibilidad con autograd y su ecosistema de entrenamiento distribuido.

## 3. EnsembleBONN

* **Agregación de modelos BONN independientes**: Se generan múltiples instancias de BONN con distintas inicializaciones estructurales (máscaras de conexión).
* **Votación suave (soft voting)**: Las salidas de cada modelo se combinan promediando las probabilidades clase a clase.
* **Evaluación progresiva**: Se mide cómo mejora la precisión del conjunto a medida que se agregan modelos, desde 1 hasta 100.
* **Motivación biológica**: Refleja la diversidad funcional de subpoblaciones KCs en el MB (α, β, γ).

## 4. Experimentos y Evaluación

Se han llevado a cabo varios experimentos para validar el rendimiento y las propiedades internas de BONN y sus variantes:

* **Comparación con modelos clásicos**: Se comparan BONN, GateBONN y EnsembleBONN con MLP, ELM y RFM bajo entrenamiento superficial (una época).
* **Fast learning**: Evaluación de precisión en función del número de muestras de entrenamiento, desde 1 hasta 10k.
* **Pruning sin reentrenamiento**: Evaluación de robustez al eliminar un porcentaje creciente de neuronas ocultas sin volver a entrenar.
* **Especialización funcional**: Análisis de la proporción de KCs especialistas y generalistas a través de métricas basadas en la sensibilidad por clase.
* **Multimodalidad de activaciones**: Se estudia la distribución del número efectivo de clases que activan cada KC, buscando indicios de subpoblaciones funcionales diferenciadas.

## 5. Entradas y Salidas

* **Datos**: MNIST (grises, 28×28 píxeles, 10 clases).
* **Preprocesamiento**: Flatten + normalización a \[0, 1].
* **Entrada del modelo**: vectores de 784 dimensiones.
* **Salida del modelo**: logits (10 dimensiones), convertidos a probabilidades vía softmax.
* **Métricas**: accuracy, sparsity media, especialización funcional, robustez al pruning.

## 6. Implementación técnica

| Componente    | Librería     | Detalles técnicos                                                  |
| ------------- | ------------ | ------------------------------------------------------------------ |
| BONN          | CuPy         | Máscaras binarias, operaciones vectorizadas en GPU                 |
| GateBONN      | PyTorch      | Uso de `torch.distributions.Beta`, `RelaxedBernoulli`, `nn.Module` |
| Pruning       | NumPy / CuPy | Eliminación de neuronas según índice                               |
| Visualización | Matplotlib   | Histogramas de especialización, evolución de precisión             |

### Ficheros clave

* `bonn.py`: clase BONN (CuPy)
* `gatebonn.py`: clase GateBONN (PyTorch)
* `ensemble.py`: funciones para crear y evaluar ensambles
* `experiments/`: scripts de evaluación rápida, pruning y análisis interno
* `utils/`: funciones de activación, inicialización y análisis de activaciones

## 7. Requisitos

```bash
cupy>=11.0.0
torch>=2.0.0
matplotlib
scikit-learn
scipy
```

---

Para un análisis más detallado del marco conceptual, referirse al Capítulo de Desarrollo del TFM. Allí se explican las motivaciones biológicas, las propiedades funcionales y los resultados experimentales con referencias cruzadas entre secciones.
