# Práctica 4 - Inteligencia Artificial  
## Clasificadores basados en métricas

Este repositorio contiene la **Práctica 4 de Inteligencia Artificial**, en la cual se implementan clasificadores basados en métricas de distancia utilizando el dataset **Wine** del repositorio UCI.

El objetivo principal de la práctica es implementar desde cero algoritmos de clasificación, aplicar diferentes funciones de distancia, evaluar los modelos con distintos métodos de validación y comparar los resultados obtenidos.

---

## Contenido del proyecto

El proyecto incluye:

- Análisis exploratorio de datos.
- Implementación de funciones de distancia.
- Clasificador de Distancia Mínima al Centroide.
- Clasificador 1NN.
- Clasificador KNN con distintos valores de `k`.
- Validación Hold-Out 70/30.
- Validación 10-Fold Cross-Validation.
- Validación Leave-One-Out.
- Tabla comparativa de resultados.
- Pruebas unitarias con `pytest`.

---

## Dataset utilizado

Se utiliza el dataset **Wine** del repositorio UCI Machine Learning Repository.

Este dataset contiene información química de vinos italianos clasificados en tres clases diferentes.

Características principales del dataset:

- 178 muestras.
- 13 atributos químicos.
- 3 clases.
- Dataset obtenido desde UCI.

Algunos atributos incluidos son:

- Alcohol.
- Ácido málico.
- Cenizas.
- Magnesio.
- Fenoles totales.
- Flavonoides.
- Intensidad de color.
- Tono.
- Prolina.

---

## Estructura del proyecto

```text
Practica4_IA/
│
├── eda.py
├── classifiers.py
├── main.py
├── test_classifiers.py
├── requirements.txt
├── README.md
├── scatter_plots.png
├── boxplots.png
└── .github/
    └── workflows/
        └── python-app.yml
