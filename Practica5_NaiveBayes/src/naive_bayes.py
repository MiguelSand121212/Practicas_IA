"""
Práctica 5 - Inteligencia Artificial
Clasificador Naive Bayes Gaussiano implementado desde cero.

El núcleo del algoritmo utiliza únicamente NumPy.
Scikit-learn solamente será utilizado posteriormente para comparar resultados.
"""

from __future__ import annotations

import numpy as np


class GaussianNaiveBayesScratch:
    """
    Clasificador Gaussian Naive Bayes implementado manualmente.

    Suposiciones:
        - Cada característica se distribuye normalmente dentro de cada clase.
        - Las características son condicionalmente independientes dada la clase.

    Durante fit() se calculan:
        - Probabilidades a priori P(C).
        - Medias μ por clase y característica.
        - Varianzas σ² por clase y característica.

    Durante predict() se calcula:
        log P(C | X) ∝ log P(C) + Σ log P(x_i | C)

    Se utilizan logaritmos para evitar problemas de precisión numérica
    ocasionados por multiplicar probabilidades muy pequeñas.
    """

    def __init__(
        self,
        var_smoothing: float = 1e-9,
    ) -> None:
        """
        Inicializa el clasificador.

        Args:
            var_smoothing:
                Factor utilizado para agregar una pequeña cantidad
                a la varianza y evitar división entre cero.
        """

        if var_smoothing < 0:
            raise ValueError(
                "var_smoothing debe ser mayor o igual a cero."
            )

        self.var_smoothing = var_smoothing

        self.classes_: np.ndarray = np.array([])
        self.class_count_: np.ndarray = np.array([])
        self.class_prior_: np.ndarray = np.array([])

        self.theta_: np.ndarray = np.empty((0, 0))
        self.var_: np.ndarray = np.empty((0, 0))

        self.epsilon_: float = 0.0

        self.n_features_in_: int = 0

        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "GaussianNaiveBayesScratch":
        """
        Entrena el modelo mediante el cálculo de parámetros Gaussianos.

        Args:
            X: Matriz de características, forma (n_muestras, n_atributos).
            y: Vector de clases, forma (n_muestras,).

        Returns:
            self
        """

        X = np.asarray(
            X,
            dtype=float,
        )

        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(
                "X debe ser una matriz bidimensional."
            )

        if y.ndim != 1:
            raise ValueError(
                "y debe ser un vector unidimensional."
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X e y deben contener la misma cantidad de muestras."
            )

        if X.shape[0] == 0:
            raise ValueError(
                "No se puede entrenar con un dataset vacío."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contiene valores no finitos."
            )

        self.n_features_in_ = X.shape[1]

        self.classes_, self.class_count_ = np.unique(
            y,
            return_counts=True,
        )

        # --------------------------------------------------------------
        # Probabilidades a priori:
        # P(C) = muestras de C / muestras totales
        # --------------------------------------------------------------

        self.class_prior_ = (
            self.class_count_.astype(float)
            / X.shape[0]
        )

        # --------------------------------------------------------------
        # Media por clase y característica
        # --------------------------------------------------------------

        self.theta_ = np.vstack(
            [
                np.mean(
                    X[y == class_name],
                    axis=0,
                )
                for class_name in self.classes_
            ]
        )

        # --------------------------------------------------------------
        # Varianza por clase y característica
        #
        # Se utiliza ddof=0 para estimación por máxima verosimilitud,
        # comportamiento habitual de Gaussian Naive Bayes.
        # --------------------------------------------------------------

        raw_variances = np.vstack(
            [
                np.var(
                    X[y == class_name],
                    axis=0,
                    ddof=0,
                )
                for class_name in self.classes_
            ]
        )

        global_variance = np.var(
            X,
            axis=0,
            ddof=0,
        )

        largest_global_variance = float(
            np.max(global_variance)
        )

        self.epsilon_ = (
            self.var_smoothing
            * largest_global_variance
        )

        if self.epsilon_ <= 0.0:
            self.epsilon_ = np.finfo(float).eps

        self.var_ = (
            raw_variances
            + self.epsilon_
        )

        self._fitted = True

        return self

    # ------------------------------------------------------------------
    # Verificación de estado
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        """Verifica que el modelo ya haya sido entrenado."""

        if not self._fitted:
            raise RuntimeError(
                "El modelo no ha sido entrenado. "
                "Ejecuta fit(X, y) antes de predict(X)."
            )

    def _validate_prediction_input(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Valida las muestras que serán clasificadas.
        """

        X = np.asarray(
            X,
            dtype=float,
        )

        if X.ndim != 2:
            raise ValueError(
                "X debe ser una matriz bidimensional."
            )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "La cantidad de características de X no coincide "
                "con los datos utilizados en entrenamiento."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contiene valores no finitos."
            )

        return X

    # ------------------------------------------------------------------
    # Log-verosimilitud conjunta
    # ------------------------------------------------------------------

    def _joint_log_likelihood(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Calcula la log-verosimilitud conjunta para cada muestra y clase.

        Fórmula:
            log P(C | X) ∝ log P(C) + Σ log P(x_i | C)

        Returns:
            Matriz de forma (n_muestras, n_clases).
        """

        self._check_is_fitted()

        X = self._validate_prediction_input(X)

        scores: list[np.ndarray] = []

        for class_index in range(len(self.classes_)):
            class_mean = self.theta_[class_index]
            class_variance = self.var_[class_index]

            log_prior = np.log(
                self.class_prior_[class_index]
            )

            log_normalization = -0.5 * np.sum(
                np.log(
                    2.0
                    * np.pi
                    * class_variance
                )
            )

            log_exponent = -0.5 * np.sum(
                (
                    (X - class_mean) ** 2
                )
                / class_variance,
                axis=1,
            )

            scores.append(
                log_prior
                + log_normalization
                + log_exponent
            )

        return np.column_stack(scores)

    # ------------------------------------------------------------------
    # Predicciones
    # ------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predice la clase de cada muestra.

        Returns:
            Vector con las etiquetas predichas.
        """

        log_likelihoods = self._joint_log_likelihood(X)

        best_indices = np.argmax(
            log_likelihoods,
            axis=1,
        )

        return self.classes_[best_indices]

    def predict_log_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Retorna log-probabilidades posteriores normalizadas.

        Se utiliza log-sum-exp para estabilidad numérica.
        """

        joint_log_likelihood = self._joint_log_likelihood(X)

        maximum_per_row = np.max(
            joint_log_likelihood,
            axis=1,
            keepdims=True,
        )

        normalization = (
            maximum_per_row
            + np.log(
                np.sum(
                    np.exp(
                        joint_log_likelihood
                        - maximum_per_row
                    ),
                    axis=1,
                    keepdims=True,
                )
            )
        )

        return (
            joint_log_likelihood
            - normalization
        )

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Retorna probabilidades posteriores normalizadas.

        Returns:
            Matriz de forma (n_muestras, n_clases).
        """

        return np.exp(
            self.predict_log_proba(X)
        )

    # ------------------------------------------------------------------
    # Presentación del modelo
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "GaussianNaiveBayesScratch("
            f"var_smoothing={self.var_smoothing}, "
            f"fitted={self._fitted})"
        )


# ---------------------------------------------------------------------------
# Prueba rápida del clasificador
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    X_train = np.vstack(
        [
            rng.normal(
                loc=[0.0, 0.0],
                scale=0.30,
                size=(30, 2),
            ),
            rng.normal(
                loc=[5.0, 5.0],
                scale=0.30,
                size=(30, 2),
            ),
            rng.normal(
                loc=[0.0, 5.0],
                scale=0.30,
                size=(30, 2),
            ),
        ]
    )

    y_train = np.array(
        ["Clase A"] * 30
        + ["Clase B"] * 30
        + ["Clase C"] * 30
    )

    X_test = np.array(
        [
            [0.1, 0.1],
            [5.1, 4.9],
            [0.1, 5.1],
        ]
    )

    model = GaussianNaiveBayesScratch()
    model.fit(
        X_train,
        y_train,
    )

    print("Modelo:")
    print(model)

    print("\nClases aprendidas:")
    print(model.classes_)

    print("\nProbabilidades a priori:")
    print(model.class_prior_)

    print("\nMedias aprendidas:")
    print(model.theta_)

    print("\nPredicciones:")
    print(model.predict(X_test))

    print("\nProbabilidades posteriores:")
    print(model.predict_proba(X_test))