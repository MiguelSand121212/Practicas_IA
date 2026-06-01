"""
Práctica 5 - Inteligencia Artificial
Comparación del clasificador propio contra scikit-learn.

El algoritmo principal de la práctica es GaussianNaiveBayesScratch.
GaussianNB de scikit-learn únicamente se usa para comparar resultados,
como solicita el punto 10 de la práctica.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from .naive_bayes import GaussianNaiveBayesScratch
from .validation import (
    evaluate_splits,
    get_validation_splits,
    save_confusion_matrix_csv,
)


# ---------------------------------------------------------------------------
# Métodos de validación solicitados
# ---------------------------------------------------------------------------

VALIDATION_METHODS = [
    "Hold-Out 80/20",
    "10-Fold Cross-Validation",
    "Leave-One-Out",
]


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def safe_method_name(
    validation_method: str,
) -> str:
    """Convierte un método de validación en nombre seguro de archivo."""

    return (
        validation_method
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


# ---------------------------------------------------------------------------
# Comparación de modelos
# ---------------------------------------------------------------------------

def compare_models(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compara Naive Bayes propio contra GaussianNB de scikit-learn.

    Ambos modelos reciben exactamente los mismos conjuntos de
    entrenamiento y prueba para que la comparación sea justa.

    Args:
        dataset_name: Nombre del dataset evaluado.
        X: Matriz de características.
        y: Vector de clases.
        output_dir: Carpeta donde se guardarán los resultados.
        seed: Semilla para particiones reproducibles.

    Returns:
        DataFrame con los resultados obtenidos.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, object]] = []

    models = {
        "Naive Bayes desde cero": (
            lambda: GaussianNaiveBayesScratch(
                var_smoothing=1e-9
            )
        ),
        "Scikit-learn GaussianNB": (
            lambda: GaussianNB(
                var_smoothing=1e-9
            )
        ),
    }

    for validation_method in VALIDATION_METHODS:
        splits = get_validation_splits(
            y,
            validation_method,
            seed=seed,
        )

        for model_name, model_factory in models.items():
            result = evaluate_splits(
                model_factory=model_factory,
                X=X,
                y=y,
                splits=splits,
            )

            rows.append(
                {
                    "dataset": dataset_name,
                    "validacion": validation_method,
                    "modelo": model_name,
                    "exactitud": result["accuracy"],
                    "exactitud_porcentaje": round(
                        float(result["accuracy"]) * 100,
                        4,
                    ),
                    "exactitud_media_folds": (
                        result["accuracy_media_folds"]
                    ),
                    "predicciones_evaluadas": (
                        result["n_predicciones"]
                    ),
                    "tiempo_segundos": (
                        result["tiempo_segundos"]
                    ),
                }
            )

            model_file_name = (
                "desde_cero"
                if model_name == "Naive Bayes desde cero"
                else "sklearn"
            )

            confusion_path = output_dir / (
                f"matriz_confusion_"
                f"{safe_method_name(validation_method)}_"
                f"{model_file_name}.csv"
            )

            save_confusion_matrix_csv(
                result,
                confusion_path,
            )

    comparison_dataframe = pd.DataFrame(rows)

    comparison_dataframe.to_csv(
        output_dir / "comparacion_sklearn.csv",
        index=False,
        encoding="utf-8",
    )

    own_model_results = comparison_dataframe[
        comparison_dataframe["modelo"]
        == "Naive Bayes desde cero"
    ]

    own_model_results.to_csv(
        output_dir / "resultados_validacion_desde_cero.csv",
        index=False,
        encoding="utf-8",
    )

    return comparison_dataframe