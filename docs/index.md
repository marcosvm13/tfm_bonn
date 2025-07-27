

# BONN: Redes Neuronales Eficientes para Aprendizaje Rápido

**BONN (Bio-inspired Olfactory Neural Network)** es una arquitectura diseñada para entornos donde la eficiencia, la rapidez de entrenamiento y la robustez son prioritarias. Frente a los modelos tradicionales, BONN ofrece una alternativa ligera y altamente competitiva en escenarios de clasificación con pocos datos y entrenamiento superficial.

Este repositorio implementa y analiza una familia de modelos que combinan conectividad estructurada fija con mecanismos de activación controlada, obteniendo representaciones internas especializadas sin necesidad de múltiples capas entrenables.

---

## Características destacadas

* Arquitectura con una sola capa entrenable y conexiones fijas.
* Entrenamiento eficiente: convergencia en una única época.
* Codificación dispersa y representaciones especializadas emergentes.
* Alta tolerancia al pruning sin necesidad de reentrenamiento.
* Extensiones con activación estocástica y agregación estructurada.

---

## Modelos incluidos

* `BONN`: Modelo base con proyecciones aleatorias fijas y salida entrenable.
* `GateBONN`: Variante con compuertas estocásticas diferenciables y regularización bayesiana.
* `EnsembleBONN`: Combinación de múltiples instancias independientes para mejorar estabilidad y precisión.
* Comparativas con modelos de referencia: `MLP`, `ELM`, `RFM`.

---

## Estructura del proyecto

```bash
src/
├── bonn/              # Arquitectura principal BONN
├── gatebonn/          # GateBONN: versión con activación estocástica
├── ensemble/          # Lógica de ensamble y agregación
├── training/          # Entrenamiento, validación y evaluación
├── analysis/          # Visualización de activaciones, escasez y especialización
├── experiments/       # Scripts para ejecución de experimentos controlados
└── app.py             # Punto de entrada para lanzar entrenamientos
```

---

## Documentación

- [Installation](installation.md): Cómo instalar y configurar el entorno
- [Usage](usage.md): Ejecución de los experimentos
- [Technical Details](technical.md): Diseño interno de los modelos y su justificación biológica
