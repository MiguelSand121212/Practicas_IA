"""
Pruebas unitarias de los métodos de validación y métricas.
"""

import numpy as np
import pytest

from src.naive_bayes import GaussianNaiveBayesScratch
from src.validation import (
    accuracy,
    confusion_matrix_scratch,
    evaluate_splits,
    get_validation_splits,
    loo_indices,
    stratified_holdout_indices,
    stratified_kfold_indices,
)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def test_accuracy_perfect_prediction() -> None:
    """
    Una predicción completamente correcta debe producir accuracy=1.
    """

    y = np.array(
        ["A", "B", "C"]
    )

    assert accuracy(
        y,
        y,
    ) == pytest.approx(1.0)


def test_accuracy_partial_prediction() -> None:
    """
    Dos aciertos de cuatro deben producir accuracy=0.5.
    """

    y_true = np.array(
        ["A", "B", "C", "D"]
    )

    y_pred = np.array(
        ["A", "X", "C", "Y"]
    )

    assert accuracy(
        y_true,
        y_pred,
    ) == pytest.approx(0.5)


def test_accuracy_with_empty_array_raises_error() -> None:
    """
    No se debe calcular accuracy sin muestras.
    """

    with pytest.raises(
        ValueError,
        match="sin muestras",
    ):
        accuracy(
            np.array([]),
            np.array([]),
        )


# ---------------------------------------------------------------------------
# Matriz de confusión
# ---------------------------------------------------------------------------

def test_confusion_matrix_correct_values() -> None:
    """
    Verifica el cálculo básico de una matriz de confusión.
    """

    y_true = np.array(
        ["A", "A", "B", "B"]
    )

    y_pred = np.array(
        ["A", "B", "B", "B"]
    )

    matrix, labels = confusion_matrix_scratch(
        y_true,
        y_pred,
    )

    assert list(labels) == ["A", "B"]

    assert np.array_equal(
        matrix,
        np.array(
            [
                [1, 1],
                [0, 2],
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Hold-Out 80/20
# ---------------------------------------------------------------------------

def test_holdout_is_80_20_and_preserves_classes() -> None:
    """
    Con 150 muestras balanceadas deben quedar:
        - 120 para entrenamiento.
        - 30 para prueba.
    """

    y = np.array(
        ["A"] * 50
        + ["B"] * 50
        + ["C"] * 50
    )

    train_indices, test_indices = stratified_holdout_indices(
        y,
        test_ratio=0.20,
        seed=42,
    )

    assert len(train_indices) == 120
    assert len(test_indices) == 30

    assert set(
        y[train_indices]
    ) == {"A", "B", "C"}

    assert set(
        y[test_indices]
    ) == {"A", "B", "C"}

    assert set(
        train_indices.tolist()
    ).isdisjoint(
        set(test_indices.tolist())
    )


def test_holdout_is_reproducible_with_same_seed() -> None:
    """
    Dos ejecuciones con la misma semilla deben generar el mismo split.
    """

    y = np.array(
        ["A"] * 40
        + ["B"] * 40
    )

    first_train, first_test = stratified_holdout_indices(
        y,
        seed=8,
    )

    second_train, second_test = stratified_holdout_indices(
        y,
        seed=8,
    )

    assert np.array_equal(
        first_train,
        second_train,
    )

    assert np.array_equal(
        first_test,
        second_test,
    )


# ---------------------------------------------------------------------------
# 10-Fold Cross-Validation
# ---------------------------------------------------------------------------

def test_ten_fold_returns_ten_splits() -> None:
    """
    10-Fold debe generar exactamente diez particiones.
    """

    y = np.array(
        ["A"] * 50
        + ["B"] * 50
        + ["C"] * 50
    )

    splits = stratified_kfold_indices(
        y,
        k=10,
        seed=42,
    )

    assert len(splits) == 10


def test_ten_fold_covers_every_sample_once_as_test() -> None:
    """
    Cada muestra debe aparecer exactamente una vez como dato de prueba.
    """

    y = np.array(
        ["A"] * 50
        + ["B"] * 50
        + ["C"] * 50
    )

    splits = stratified_kfold_indices(
        y,
        k=10,
        seed=42,
    )

    all_test_indices = np.concatenate(
        [
            test_indices
            for _, test_indices in splits
        ]
    )

    assert sorted(
        all_test_indices.tolist()
    ) == list(
        range(len(y))
    )


def test_ten_fold_has_no_overlap_inside_each_fold() -> None:
    """
    En ningún fold deben repetirse datos entre entrenamiento y prueba.
    """

    y = np.array(
        ["A"] * 30
        + ["B"] * 30
        + ["C"] * 30
    )

    splits = stratified_kfold_indices(
        y,
        k=10,
        seed=42,
    )

    for train_indices, test_indices in splits:
        assert set(
            train_indices.tolist()
        ).isdisjoint(
            set(test_indices.tolist())
        )


# ---------------------------------------------------------------------------
# Leave-One-Out
# ---------------------------------------------------------------------------

def test_loo_generates_one_split_per_sample() -> None:
    """
    Leave-One-Out debe generar n particiones.
    """

    splits = loo_indices(13)

    assert len(splits) == 13


def test_loo_uses_one_test_sample_each_time() -> None:
    """
    Cada partición debe contener una sola muestra de prueba.
    """

    splits = loo_indices(13)

    assert all(
        len(test_indices) == 1
        and len(train_indices) == 12
        for train_indices, test_indices in splits
    )


def test_loo_covers_all_samples() -> None:
    """
    Todas las muestras deben aparecer una vez como elemento de prueba.
    """

    splits = loo_indices(10)

    test_indices = [
        int(test_index[0])
        for _, test_index in splits
    ]

    assert sorted(test_indices) == list(
        range(10)
    )


# ---------------------------------------------------------------------------
# Selector de métodos
# ---------------------------------------------------------------------------

def test_get_validation_splits_holdout_returns_one_split() -> None:
    """
    Hold-Out debe retornar una sola partición.
    """

    y = np.array(
        ["A"] * 20
        + ["B"] * 20
    )

    splits = get_validation_splits(
        y,
        "Hold-Out 80/20",
    )

    assert len(splits) == 1


def test_get_validation_splits_unknown_method_raises_error() -> None:
    """
    Un nombre de validación desconocido debe producir error.
    """

    y = np.array(
        ["A", "A", "B", "B"]
    )

    with pytest.raises(
        ValueError,
        match="desconocido",
    ):
        get_validation_splits(
            y,
            "Metodo inexistente",
        )


# ---------------------------------------------------------------------------
# Evaluación completa del modelo
# ---------------------------------------------------------------------------

def test_evaluate_splits_runs_with_naive_bayes() -> None:
    """
    Comprueba que evaluate_splits puede entrenar y evaluar el
    clasificador Naive Bayes implementado desde cero.
    """

    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, 0.1],
            [5.0, 5.0],
            [5.1, 5.2],
            [4.9, 5.1],
            [0.0, 5.0],
            [0.1, 5.1],
            [0.2, 4.9],
        ]
    )

    y = np.array(
        [
            "A", "A", "A",
            "B", "B", "B",
            "C", "C", "C",
        ]
    )

    splits = [
        (
            np.array(
                [0, 1, 3, 4, 6, 7]
            ),
            np.array(
                [2, 5, 8]
            ),
        )
    ]

    result = evaluate_splits(
        model_factory=lambda: GaussianNaiveBayesScratch(),
        X=X,
        y=y,
        splits=splits,
    )

    assert result["accuracy"] == pytest.approx(1.0)

    assert result["n_predicciones"] == 3

    assert result["confusion_matrix"].shape == (3, 3)