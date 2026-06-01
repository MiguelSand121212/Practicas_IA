"""
Práctica 5 - Inteligencia Artificial
Programa principal: Clasificador Naive Bayes Gaussiano.

Este programa realiza:
    1. Carga de los datasets Wine e Iris.
    2. Probabilidades a priori.
    3. Media y desviación estándar por característica y clase.
    4. Gráficas KDE con curvas Gaussianas.
    5. Matrices de correlación por clase.
    6. Evaluación de independencia.
    7. Implementación de Naive Bayes Gaussiano desde cero.
    8. Hold-Out 80/20.
    9. 10-Fold Cross-Validation.
   10. Leave-One-Out.
   11. Comparación contra GaussianNB de scikit-learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .comparison import compare_models
from .config import (
    OUTPUTS_DIR,
    RANDOM_SEED,
    TARGET_COL,
    create_project_directories,
)
from .datasets import load_all_datasets
from .eda import run_eda


# ---------------------------------------------------------------------------
# Preparación de datos para clasificación
# ---------------------------------------------------------------------------

def dataframe_to_xy(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide un DataFrame en características X y clases y.

    Returns:
        X: Matriz numérica de atributos.
        y: Vector de etiquetas de clase.
    """

    X = (
        df
        .drop(columns=[TARGET_COL])
        .to_numpy(dtype=float)
    )

    y = df[TARGET_COL].to_numpy()

    return X, y


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main() -> None:
    """Ejecuta automáticamente todos los puntos de la práctica."""

    create_project_directories()

    print("\n" + "=" * 80)
    print(" PRÁCTICA 5 — CLASIFICADOR NAIVE BAYES GAUSSIANO")
    print("=" * 80)

    print(
        " Datasets seleccionados: Wine e Iris "
        "(UCI Machine Learning Repository)."
    )

    datasets = load_all_datasets()

    global_results: list[pd.DataFrame] = []

    for dataset_name, dataframe in datasets.items():
        # --------------------------------------------------------------
        # Puntos 2 a 7: EDA
        # --------------------------------------------------------------

        run_eda(
            dataset_name,
            dataframe,
        )

        # --------------------------------------------------------------
        # Preparar X e y
        # --------------------------------------------------------------

        X, y = dataframe_to_xy(
            dataframe
        )

        results_dir = (
            OUTPUTS_DIR
            / dataset_name
            / "resultados"
        )

        # --------------------------------------------------------------
        # Puntos 8, 9 y 10: clasificación, validación y comparación
        # --------------------------------------------------------------

        comparison = compare_models(
            dataset_name=dataset_name,
            X=X,
            y=y,
            output_dir=results_dir,
            seed=RANDOM_SEED,
        )

        global_results.append(
            comparison
        )

        print("\n Resultados de validación y comparación:")

        print(
            comparison[
                [
                    "validacion",
                    "modelo",
                    "exactitud_porcentaje",
                    "tiempo_segundos",
                ]
            ].to_string(
                index=False,
                formatters={
                    "exactitud_porcentaje": "{:.2f}%".format,
                    "tiempo_segundos": "{:.6f}".format,
                },
            )
        )

        print(
            f"\n Resultados guardados en outputs/{dataset_name}/resultados/"
        )

    # ------------------------------------------------------------------
    # Guardar comparación global
    # ------------------------------------------------------------------

    global_comparison = pd.concat(
        global_results,
        ignore_index=True,
    )

    global_comparison.to_csv(
        OUTPUTS_DIR / "comparacion_global.csv",
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print(" EJECUCIÓN COMPLETADA CORRECTAMENTE")
    print("=" * 80)

    print(
        " Todos los CSV, gráficas KDE, matrices de correlación "
        "y comparaciones se encuentran en la carpeta outputs/."
    )

    print(
        " Para el punto 5, revisa las imágenes de cada carpeta kde/ "
        "y redacta tu análisis visual del ajuste Gaussiano."
    )


if __name__ == "__main__":
    main()