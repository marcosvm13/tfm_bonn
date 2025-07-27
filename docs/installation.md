# Installation

## Stable release

To install tfm_bonn, run this command in your terminal:

```sh
pip install tfm_bonn
```

This is the preferred method to install tfm_bonn, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.

## From sources

The sources for tfm_bonn can be downloaded from the [Github repo](https://github.com/marcosvm13/tfm_bonn).

You can either clone the public repository:

```sh
git clone git://github.com/marcosvm13/tfm_bonn
```

Or download the [tarball](https://github.com/marcosvm13/tfm_bonn/tarball/master):

```sh
curl -OJL https://github.com/marcosvm13/tfm_bonn/tarball/master
```

Once you have a copy of the source, you can install it with:

```sh
python setup.py install
```


# Instalación del proyecto BONN

Este proyecto puede ejecutarse en entornos con GPU o CPU, aunque se recomienda el uso de GPU con soporte CUDA para experimentar con CuPy y PyTorch eficientemente.

## Requisitos

- Python 3.9 o superior
- CUDA (si se usa GPU + CuPy)

## Instalación con entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## CuPy según tu versión de CUDA

Instala `cupy` correspondiente a tu versión CUDA (verifica con `nvcc --version`):

```bash
pip install cupy-cuda12x   # Para CUDA 12.x
# o
pip install cupy-cuda11x   # Para CUDA 11.x
```

Si no dispones de GPU, puedes eliminar `cupy` del proyecto y reemplazarlo por `numpy` en el código.

## Descarga de datasets

Los experimentos utilizan MNIST. PyTorch lo descarga automáticamente en la primera ejecución.

```bash
./data/  # Se crea al correr los scripts por primera vez
```

## Prueba rápida

```bash
python src/app.py
```

Esto ejecuta un pipeline de ejemplo con la arquitectura BONN.