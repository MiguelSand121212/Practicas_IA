"""
Pruebas unitarias del clasificador Gaussian Naive Bayes implementado
desde cero.
"""

import numpy as np
import pytest

from src.naive_bayes import GaussianNaiveBayesScratch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def separated_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Dataset sintético con tres clases claramente separadas.
    """

    rng = np.random.default_rng(7)

    X = np.vstack(
        [
            rng.normal(
                loc=[0.0, 0.0],
                scale=0.25,
                size=(25, 2),
            ),
            rng.normal(
                loc=[5.0, 5.0],
                scale=0.25,
                size=(25, 2),
            ),
            rng.normal(
                loc=[0.0, 5.0],
                scale=0.25,
                size=(25, 2),
            ),
        ]
    )

    y = np.array(
        ["A"] * 25
        + ["B"] * 25
        + ["C"] * 25
    )

    return X, y


# ---------------------------------------------------------------------------
# Validaciones de inicialización y entrenamiento
# ---------------------------------------------------------------------------

def test_invalid_var_smoothing_raises_error() -> None:
    """
    var_smoothing negativo no debe aceptarse.
    """

    with pytest.raises(
        ValueError,
        match="var_smoothing",
    ):
        GaussianNaiveBayesScratch(
            var_smoothing=-1.0
        )


def test_predict_before_fit_raises_error() -> None:
    """
    No se debe poder predecir antes de entrenar el modelo.
    """

    model = GaussianNaiveBayesScratch()

    with pytest.raises(
        RuntimeError,
        match="no ha sido entrenado",
    ):
        model.predict(
            np.array([[1.0, 2.0]])
        )


def test_fit_returns_self() -> None:
    """
    fit() debe retornar el mismo objeto.
    """

    X = np.array(
        [
            [0.0],
            [0.2],
            [5.0],
            [5.2],
        ]
    )

    y = np.array(
        ["A", "A", "B", "B"]
    )

    model = GaussianNaiveBayesScratch()

    assert model.fit(X, y) is model


def test_fit_with_empty_dataset_raises_error() -> None:
    """
    No se debe aceptar un conjunto de entrenamiento vacío.
    """

    model = GaussianNaiveBayesScratch()

    with pytest.raises(
        ValueError,
        match="vacío",
    ):
        model.fit(
            np.empty((0, 2)),
            np.array([]),
        )


# ---------------------------------------------------------------------------
# Probabilidades a priori, medias y varianzas
# ---------------------------------------------------------------------------

def test_fit_calculates_priors_correctly() -> None:
    """
    Verifica P(A)=3/4 y P(B)=1/4.
    """

    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [10.0],
        ]
    )

    y = np.array(
        ["A", "A", "A", "B"]
    )

    model = GaussianNaiveBayesScratch()
    model.fit(X, y)

    assert list(model.classes_) == ["A", "B"]

    assert np.allclose(
        model.class_prior_,
        [0.75, 0.25],
    )


def test_fit_calculates_class_means_correctly() -> None:
    """
    Verifica las medias calculadas para cada clase.
    """

    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [10.0, 12.0],
            [14.0, 16.0],
        ]
    )

    y = np.array(
        ["A", "A", "B", "B"]
    )

    model = GaussianNaiveBayesScratch()
    model.fit(X, y)

    assert np.allclose(
        model.theta_[0],
        [2.0, 3.0],
    )

    assert np.allclose(
        model.theta_[1],
        [12.0, 14.0],
    )


def test_class_priors_sum_to_one(
    separated_dataset,
) -> None:
    """
    Las probabilidades a priori deben sumar 1.
    """

    X, y = separated_dataset

    model = GaussianNaiveBayesScratch()
    model.fit(X, y)

    assert np.sum(
        model.class_prior_
    ) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Predicciones
# ---------------------------------------------------------------------------

def test_predict_has_high_accuracy_for_separated_classes(
    separated_dataset,
) -> None:
    """
    El modelo debe lograr alta exactitud en clases bien separadas.
    """

    X, y = separated_dataset

    model = GaussianNaiveBayesScratch()
    model.fit(X, y)

    predictions = model.predict(X)

    obtained_accuracy = np.mean(
        predictions == y
    )

    assert obtained_accuracy >= 0.95


def test_predict_output_shape(
    separated_dataset,
) -> None:
    """
    El vector de predicciones debe tener la cantidad correcta de elementos.
    """

    X, y = separated_dataset

    model = GaussianNaiveBayesScratch()
    model.fit(X, y)

    predictions = model.predict(
        X[:10]
    )

    assert predictions.shape == (10,)


def test_predict_proba_rows_sum_to_one(
    separated_dataset,
) -> None:
    """
    Las probabilidades posteriores de cada muestra deben sumar 1.
    """

    X, y = separated_dataset

    model = GaussianNaiveBayesScratch()
    model.fit(X, y)

    probabilities = model.predict_proba(
        X[:10]
    )

    assert np.allclose(
        np.sum(
            probabilities,
            axis=1,
        ),
        1.0,
    )


def test_predict_log_proba_matches_predict_proba(
    separated_dataset,
) -> None:
    """
    exp(predict_log_proba) debe coincidir con predict_proba.
    """

    X, y = separated_dataset

    model = GaussianNaiveBayesScratch()
    model.fit(X, y)

    log_probabilities = model.predict_log_proba(
        X[:5]
    )

    probabilities = model.predict_proba(
        X[:5]
    )

    assert np.allclose(
        np.exp(log_probabilities),
        probabilities,
    )


def test_constant_feature_does_not_generate_nan() -> None:
    """
    Una variable constante no debe generar divisiones inválidas.
    """

    X = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.2],
            [1.0, 5.0],
            [1.0, 5.2],
        ]
    )

    y = np.array(
        ["A", "A", "B", "B"]
    )

    model = GaussianNaiveBayesScratch()
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert not np.any(
        np.isnan(probabilities)
    )

    assert not np.any(
        np.isinf(probabilities)
    )


def test_different_feature_count_raises_error(
    separated_dataset,
) -> None:
    """
    No se deben aceptar muestras con distinta cantidad de características.
    """

    X, y = separated_dataset

    model = GaussianNaiveBayesScratch()
    model.fit(X, y)

    with pytest.raises(
        ValueError,
        match="cantidad de características",
    ):
        model.predict(
            np.array(
                [
                    [1.0, 2.0, 3.0]
                ]
            )
        )