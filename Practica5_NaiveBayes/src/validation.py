"""
Práctica 5 - Inteligencia Artificial
Métodos de validación y métricas implementados desde cero.

Métodos requeridos:
    - Hold-Out 80/20 estratificado.
    - 10-Fold Cross-Validation estratificado.
    - Leave-One-Out.

Métricas:
    - Accuracy.
    - Matriz de confusión.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Alias de tipos
# ---------------------------------------------------------------------------

SplitPair = tuple[np.ndarray, np.ndarray]
ModelFactory = Callable[[], Any]


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calcula la exactitud del clasificador.

    Fórmula:
        accuracy = predicciones correctas / total de predicciones
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        raise ValueError(
            "No se puede calcular accuracy sin muestras."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true e y_pred deben tener la misma longitud."
        )

    return float(
        np.mean(
            y_true == y_pred
        )
    )


def confusion_matrix_scratch(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye una matriz de confusión desde cero.

    Filas:
        Clases reales.

    Columnas:
        Clases predichas.

    Returns:
        Tupla:
            matriz_de_confusion, etiquetas
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = np.unique(
            np.concatenate(
                [
                    y_true,
                    y_pred,
                ]
            )
        )

    label_to_index = {
        label: index
        for index, label in enumerate(labels)
    }

    matrix = np.zeros(
        (len(labels), len(labels)),
        dtype=int,
    )

    for real_class, predicted_class in zip(
        y_true,
        y_pred,
    ):
        row = label_to_index[real_class]
        column = label_to_index[predicted_class]

        matrix[row, column] += 1

    return matrix, labels


# ---------------------------------------------------------------------------
# Hold-Out 80/20 estratificado
# ---------------------------------------------------------------------------

def stratified_holdout_indices(
    y: np.ndarray,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> SplitPair:
    """
    Genera índices para Hold-Out estratificado.

    Por defecto:
        - 80% entrenamiento.
        - 20% prueba.

    La estratificación conserva muestras de cada clase en ambos grupos.
    """

    y = np.asarray(y)

    if not 0.0 < test_ratio < 1.0:
        raise ValueError(
            "test_ratio debe encontrarse entre 0 y 1."
        )

    rng = np.random.default_rng(seed)

    train_indices: list[int] = []
    test_indices: list[int] = []

    for class_name in np.unique(y):
        class_indices = np.where(
            y == class_name
        )[0]

        shuffled_indices = rng.permutation(
            class_indices
        )

        number_of_test_samples = max(
            1,
            int(
                round(
                    len(shuffled_indices)
                    * test_ratio
                )
            ),
        )

        if number_of_test_samples >= len(shuffled_indices):
            number_of_test_samples = (
                len(shuffled_indices) - 1
            )

        test_indices.extend(
            shuffled_indices[
                :number_of_test_samples
            ].tolist()
        )

        train_indices.extend(
            shuffled_indices[
                number_of_test_samples:
            ].tolist()
        )

    train_array = rng.permutation(
        np.asarray(
            train_indices,
            dtype=int,
        )
    )

    test_array = rng.permutation(
        np.asarray(
            test_indices,
            dtype=int,
        )
    )

    return train_array, test_array


# ---------------------------------------------------------------------------
# K-Fold estratificado
# ---------------------------------------------------------------------------

def stratified_kfold_indices(
    y: np.ndarray,
    k: int = 10,
    seed: int = 42,
) -> list[SplitPair]:
    """
    Genera índices para K-Fold Cross-Validation estratificado.

    Cada muestra aparece exactamente una vez como elemento de prueba.
    """

    y = np.asarray(y)

    if k < 2:
        raise ValueError(
            "k debe ser por lo menos 2."
        )

    smallest_class_size = min(
        np.sum(y == class_name)
        for class_name in np.unique(y)
    )

    if k > smallest_class_size:
        raise ValueError(
            "k no puede ser mayor que la cantidad de muestras "
            "de la clase más pequeña."
        )

    rng = np.random.default_rng(seed)

    classes = np.unique(y)

    class_folds: dict[object, list[np.ndarray]] = {}

    for class_name in classes:
        class_indices = np.where(
            y == class_name
        )[0]

        shuffled_indices = rng.permutation(
            class_indices
        )

        class_folds[class_name] = list(
            np.array_split(
                shuffled_indices,
                k,
            )
        )

    all_indices = np.arange(len(y))

    splits: list[SplitPair] = []

    for fold_index in range(k):
        test_indices = np.concatenate(
            [
                class_folds[class_name][fold_index]
                for class_name in classes
            ]
        )

        test_indices = rng.permutation(
            test_indices
        )

        train_mask = np.ones(
            len(y),
            dtype=bool,
        )

        train_mask[test_indices] = False

        train_indices = all_indices[
            train_mask
        ]

        splits.append(
            (
                train_indices,
                test_indices,
            )
        )

    return splits


# ---------------------------------------------------------------------------
# Leave-One-Out
# ---------------------------------------------------------------------------

def loo_indices(
    n_samples: int,
) -> list[SplitPair]:
    """
    Genera particiones Leave-One-Out.

    En cada iteración:
        - Una sola muestra se utiliza como prueba.
        - Las restantes se utilizan para entrenamiento.
    """

    if n_samples < 2:
        raise ValueError(
            "Leave-One-Out requiere al menos dos muestras."
        )

    all_indices = np.arange(
        n_samples
    )

    splits: list[SplitPair] = []

    for sample_index in range(n_samples):
        test_indices = all_indices[
            sample_index: sample_index + 1
        ]

        train_indices = np.concatenate(
            [
                all_indices[:sample_index],
                all_indices[sample_index + 1:],
            ]
        )

        splits.append(
            (
                train_indices,
                test_indices,
            )
        )

    return splits


# ---------------------------------------------------------------------------
# Selector de validaciones
# ---------------------------------------------------------------------------

def get_validation_splits(
    y: np.ndarray,
    method: str,
    seed: int = 42,
) -> list[SplitPair]:
    """
    Obtiene las particiones correspondientes al método solicitado.
    """

    if method == "Hold-Out 80/20":
        return [
            stratified_holdout_indices(
                y,
                test_ratio=0.20,
                seed=seed,
            )
        ]

    if method == "10-Fold Cross-Validation":
        return stratified_kfold_indices(
            y,
            k=10,
            seed=seed,
        )

    if method == "Leave-One-Out":
        return loo_indices(
            len(y)
        )

    raise ValueError(
        f"Método de validación desconocido: '{method}'."
    )


# ---------------------------------------------------------------------------
# Evaluación general
# ---------------------------------------------------------------------------

def evaluate_splits(
    model_factory: ModelFactory,
    X: np.ndarray,
    y: np.ndarray,
    splits: Iterable[SplitPair],
) -> dict[str, object]:
    """
    Evalúa cualquier clasificador compatible con fit() y predict().

    Esta función será utilizada tanto por:
        - GaussianNaiveBayesScratch.
        - sklearn.naive_bayes.GaussianNB.
    """

    X = np.asarray(
        X,
        dtype=float,
    )

    y = np.asarray(y)

    true_values: list[object] = []
    predicted_values: list[object] = []
    fold_accuracies: list[float] = []

    start_time = time.perf_counter()

    for train_indices, test_indices in splits:
        model = model_factory()

        model.fit(
            X[train_indices],
            y[train_indices],
        )

        predictions = model.predict(
            X[test_indices]
        )

        expected = y[test_indices]

        true_values.extend(
            expected.tolist()
        )

        predicted_values.extend(
            predictions.tolist()
        )

        fold_accuracies.append(
            accuracy(
                expected,
                predictions,
            )
        )

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    y_true = np.asarray(true_values)
    y_pred = np.asarray(predicted_values)

    matrix, labels = confusion_matrix_scratch(
        y_true,
        y_pred,
    )

    return {
        "accuracy": accuracy(
            y_true,
            y_pred,
        ),
        "accuracy_media_folds": float(
            np.mean(fold_accuracies)
        ),
        "n_predicciones": len(y_true),
        "tiempo_segundos": elapsed_time,
        "confusion_matrix": matrix,
        "labels": labels,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Guardado de matriz de confusión
# ---------------------------------------------------------------------------

def save_confusion_matrix_csv(
    result: dict[str, object],
    destination: Path,
) -> None:
    """
    Guarda una matriz de confusión como archivo CSV.
    """

    labels = result["labels"]
    matrix = result["confusion_matrix"]

    dataframe = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    dataframe.index.name = "Clase real"

    dataframe.to_csv(
        destination,
        encoding="utf-8",
    )