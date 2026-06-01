"""
Práctica 5 - Inteligencia Artificial
Análisis Exploratorio de Datos para Gaussian Naive Bayes.

Este archivo realiza:
    2. Probabilidad a priori de cada clase.
    3. Media y desviación estándar de cada característica por clase.
    4. Gráficas KDE por característica con curvas por clase.
    5. Superposición de curvas Gaussianas para verificación visual.
    6. Matrices de correlación por clase.
    7. Análisis de independencia mediante correlaciones fuertes.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import OUTPUTS_DIR, TARGET_COL


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def feature_columns(df: pd.DataFrame) -> list[str]:
    """Obtiene las columnas de características, excluyendo la clase."""

    return [
        column
        for column in df.columns
        if column != TARGET_COL
    ]


def safe_filename(text: str) -> str:
    """
    Convierte texto en un nombre seguro para archivos.

    Ejemplos:
        Clase 1 -> Clase_1
        Iris-setosa -> Iris-setosa
    """

    return re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(text),
    ).strip("_")


# ---------------------------------------------------------------------------
# Información general
# ---------------------------------------------------------------------------

def print_basic_info(
    dataset_name: str,
    df: pd.DataFrame,
) -> None:
    """Imprime información general del dataset."""

    print("\n" + "=" * 80)
    print(f" DATASET: {dataset_name.upper()}")
    print("=" * 80)

    print(
        f" Muestras: {df.shape[0]} | "
        f"Características: {len(feature_columns(df))} | "
        f"Clases: {df[TARGET_COL].nunique()}"
    )

    print("\n Distribución de clases:")

    counts = df[TARGET_COL].value_counts().sort_index()

    for class_name, count in counts.items():
        percentage = count / len(df) * 100

        print(
            f"   {class_name:<20} "
            f"{count:>4} muestras "
            f"({percentage:>6.2f}%)"
        )

    print("\n Primeras cinco filas:")
    print(df.head().to_string(index=False))


# ---------------------------------------------------------------------------
# Punto 2: Probabilidades a priori
# ---------------------------------------------------------------------------

def calculate_priors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la probabilidad a priori de cada clase.

    Fórmula:
        P(C) = número de muestras de la clase C / total de muestras
    """

    counts = df[TARGET_COL].value_counts().sort_index()

    result = pd.DataFrame(
        {
            "clase": counts.index,
            "muestras": counts.values,
        }
    )

    result["probabilidad_a_priori"] = (
        result["muestras"] / len(df)
    )

    result["porcentaje"] = (
        result["probabilidad_a_priori"] * 100.0
    )

    return result


# ---------------------------------------------------------------------------
# Punto 3: Media y desviación estándar por clase
# ---------------------------------------------------------------------------

def calculate_class_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula media y desviación estándar muestral por clase y característica.

    En este reporte se utiliza desviación estándar muestral con ddof=1,
    debido a que se están describiendo las muestras observadas del dataset.
    """

    rows: list[dict[str, object]] = []

    for class_name, class_df in df.groupby(
        TARGET_COL,
        sort=True,
    ):
        for feature in feature_columns(df):
            values = class_df[feature].to_numpy(dtype=float)

            mean_value = float(
                np.mean(values)
            )

            standard_deviation = float(
                np.std(values, ddof=1)
            )

            rows.append(
                {
                    "clase": class_name,
                    "caracteristica": feature,
                    "media_mu": mean_value,
                    "desviacion_estandar_sigma": standard_deviation,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Distribución Gaussiana
# ---------------------------------------------------------------------------

def gaussian_pdf(
    x_values: np.ndarray,
    mean: float,
    standard_deviation: float,
) -> np.ndarray:
    """
    Calcula la función de densidad de probabilidad Gaussiana.

    Fórmula:
        f(x) = 1 / (σ sqrt(2π)) * exp(-(x - μ)^2 / (2σ²))
    """

    safe_std = max(
        float(standard_deviation),
        np.finfo(float).eps,
    )

    coefficient = 1.0 / (
        safe_std * np.sqrt(2.0 * np.pi)
    )

    exponent = np.exp(
        -0.5
        * (
            (x_values - mean)
            / safe_std
        ) ** 2
    )

    return coefficient * exponent


# ---------------------------------------------------------------------------
# Puntos 4 y 5: KDE con curva Gaussiana
# ---------------------------------------------------------------------------

def plot_kde_with_gaussian(
    dataset_name: str,
    df: pd.DataFrame,
) -> None:
    """
    Genera una gráfica KDE por cada característica.

    Cada imagen presenta:
        - Una KDE por clase.
        - Una curva Gaussiana estimada por clase.

    La comparación visual entre la línea continua KDE y la línea punteada
    Gaussiana permite analizar si la distribución normal se ajusta de forma
    razonable a los datos.
    """

    output_dir = (
        OUTPUTS_DIR
        / dataset_name
        / "kde"
    )

    class_names = sorted(
        df[TARGET_COL].unique()
    )

    colors = sns.color_palette(
        "tab10",
        n_colors=len(class_names),
    )

    palette = dict(
        zip(
            class_names,
            colors,
        )
    )

    for feature in feature_columns(df):
        fig, ax = plt.subplots(
            figsize=(11, 6)
        )

        for class_name in class_names:
            values = df.loc[
                df[TARGET_COL] == class_name,
                feature,
            ].to_numpy(dtype=float)

            color = palette[class_name]

            sns.kdeplot(
                x=values,
                ax=ax,
                color=color,
                linewidth=2.4,
                fill=False,
                warn_singular=False,
                label=f"{class_name} - KDE",
            )

            margin = (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else 1.0
            )

            x_values = np.linspace(
                float(values.min()) - margin,
                float(values.max()) + margin,
                400,
            )

            y_values = gaussian_pdf(
                x_values,
                mean=float(np.mean(values)),
                standard_deviation=float(
                    np.std(values, ddof=1)
                ),
            )

            ax.plot(
                x_values,
                y_values,
                color=color,
                linestyle="--",
                linewidth=1.8,
                label=f"{class_name} - Gaussiana",
            )

        ax.set_title(
            f"{dataset_name.upper()} — KDE y curva Gaussiana: {feature}",
            fontsize=13,
            fontweight="bold",
        )

        ax.set_xlabel(feature)
        ax.set_ylabel("Densidad")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        fig.tight_layout()

        image_path = output_dir / (
            f"kde_{safe_filename(feature)}.png"
        )

        fig.savefig(
            image_path,
            dpi=170,
            bbox_inches="tight",
        )

        plt.close(fig)


# ---------------------------------------------------------------------------
# Punto 6: Matrices de correlación por clase
# ---------------------------------------------------------------------------

def plot_correlation_matrices(
    dataset_name: str,
    df: pd.DataFrame,
) -> None:
    """
    Genera una matriz de correlación para cada clase del dataset.
    """

    output_dir = (
        OUTPUTS_DIR
        / dataset_name
        / "correlacion"
    )

    features = feature_columns(df)

    for class_name, class_df in df.groupby(
        TARGET_COL,
        sort=True,
    ):
        correlation_matrix = class_df[features].corr()

        figure_size = max(
            7,
            min(16, len(features) * 0.90),
        )

        fig, ax = plt.subplots(
            figsize=(figure_size, figure_size)
        )

        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            annot_kws={"fontsize": 7},
        )

        ax.set_title(
            (
                f"{dataset_name.upper()} — Matriz de correlación\n"
                f"Clase: {class_name}"
            ),
            fontsize=13,
            fontweight="bold",
        )

        fig.tight_layout()

        image_path = output_dir / (
            f"correlacion_{safe_filename(class_name)}.png"
        )

        fig.savefig(
            image_path,
            dpi=170,
            bbox_inches="tight",
        )

        plt.close(fig)


# ---------------------------------------------------------------------------
# Punto 7: Evaluación de independencia
# ---------------------------------------------------------------------------

def analyze_independence(
    df: pd.DataFrame,
    threshold: float = 0.70,
) -> pd.DataFrame:
    """
    Detecta pares de características con correlación fuerte por clase.

    Naive Bayes supone independencia condicional entre características.
    Una correlación alta dentro de una clase constituye evidencia de que
    esta suposición puede no cumplirse completamente.

    Args:
        df: Dataset completo.
        threshold: Valor absoluto mínimo para considerar correlación fuerte.

    Returns:
        DataFrame con correlaciones fuertes encontradas.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "El umbral de correlación debe encontrarse entre 0 y 1."
        )

    rows: list[dict[str, object]] = []

    features = feature_columns(df)

    for class_name, class_df in df.groupby(
        TARGET_COL,
        sort=True,
    ):
        correlation_matrix = class_df[features].corr()

        for first_index, first_feature in enumerate(features):
            for second_feature in features[first_index + 1:]:
                correlation_value = float(
                    correlation_matrix.loc[
                        first_feature,
                        second_feature,
                    ]
                )

                absolute_value = abs(correlation_value)

                if absolute_value >= threshold:
                    rows.append(
                        {
                            "clase": class_name,
                            "caracteristica_1": first_feature,
                            "caracteristica_2": second_feature,
                            "correlacion": correlation_value,
                            "correlacion_absoluta": absolute_value,
                            "interpretacion": (
                                "Correlación fuerte: posible incumplimiento "
                                "de la independencia condicional."
                            ),
                        }
                    )

    columns = [
        "clase",
        "caracteristica_1",
        "caracteristica_2",
        "correlacion",
        "correlacion_absoluta",
        "interpretacion",
    ]

    if not rows:
        return pd.DataFrame(
            columns=columns
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by="correlacion_absoluta",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def save_independence_conclusion(
    dataset_name: str,
    strong_correlations: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Guarda una conclusión textual para apoyar el reporte de independencia.
    """

    if strong_correlations.empty:
        conclusion = (
            f"ANÁLISIS DE INDEPENDENCIA — DATASET {dataset_name.upper()}\n\n"
            "Con el umbral |r| >= 0.70 no se identificaron correlaciones "
            "fuertes entre características dentro de las clases. "
            "Por lo tanto, el análisis de correlación no muestra evidencia "
            "fuerte contra la suposición de independencia condicional "
            "de Naive Bayes.\n"
        )

    else:
        conclusion = (
            f"ANÁLISIS DE INDEPENDENCIA — DATASET {dataset_name.upper()}\n\n"
            f"Se identificaron {len(strong_correlations)} pares de "
            "características con correlación absoluta mayor o igual a 0.70 "
            "dentro de alguna clase. Esto indica que la suposición de "
            "independencia condicional de Naive Bayes no se cumple "
            "completamente para todos los atributos.\n\n"
            "Aun cuando existan correlaciones, el clasificador Naive Bayes "
            "puede conservar un buen desempeño predictivo, lo cual debe "
            "verificarse mediante Hold-Out, 10-Fold Cross-Validation y "
            "Leave-One-Out.\n"
        )

    output_path.write_text(
        conclusion,
        encoding="utf-8",
    )


def save_gaussian_visual_review_guide(
    dataset_name: str,
    output_path: Path,
) -> None:
    """
    Guarda una guía para redactar la verificación visual del ajuste Gaussiano.
    """

    content = (
        f"GUÍA DE REVISIÓN VISUAL DEL AJUSTE GAUSSIANO — "
        f"{dataset_name.upper()}\n\n"
        "Revisa cada imagen de la carpeta kde/.\n\n"
        "Interpretación sugerida:\n"
        "- Si la curva KDE continua se parece a la curva Gaussiana punteada, "
        "la suposición normal es razonable para esa característica y clase.\n"
        "- Si la KDE presenta fuerte asimetría, varios picos o una forma muy "
        "distinta a la curva punteada, la suposición Gaussiana es menos "
        "adecuada.\n\n"
        "Texto base para el reporte:\n"
        "Se compararon las curvas KDE observadas contra distribuciones "
        "Gaussianas estimadas mediante la media y desviación estándar de "
        "cada característica dentro de cada clase. Las variables cuyas curvas "
        "presentan formas semejantes muestran un ajuste normal razonable, "
        "mientras que las diferencias marcadas evidencian limitaciones de "
        "la suposición Gaussiana.\n"
    )

    output_path.write_text(
        content,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Ejecución completa del EDA
# ---------------------------------------------------------------------------

def run_eda(
    dataset_name: str,
    df: pd.DataFrame,
) -> None:
    """
    Ejecuta todos los análisis requeridos para un dataset.
    """

    statistics_dir = (
        OUTPUTS_DIR
        / dataset_name
        / "estadisticas"
    )

    print_basic_info(
        dataset_name,
        df,
    )

    priors = calculate_priors(df)

    statistics = calculate_class_statistics(df)

    strong_correlations = analyze_independence(
        df,
        threshold=0.70,
    )

    print("\n Probabilidades a priori:")

    print(
        priors.to_string(
            index=False,
            formatters={
                "probabilidad_a_priori": "{:.6f}".format,
                "porcentaje": "{:.2f}%".format,
            },
        )
    )

    print(
        "\n Media (μ) y desviación estándar (σ) "
        "por característica y clase:"
    )

    print(
        statistics.to_string(
            index=False,
            formatters={
                "media_mu": "{:.5f}".format,
                "desviacion_estandar_sigma": "{:.5f}".format,
            },
        )
    )

    priors.to_csv(
        statistics_dir / "probabilidades_a_priori.csv",
        index=False,
        encoding="utf-8",
    )

    statistics.to_csv(
        statistics_dir / "medias_desviaciones_por_clase.csv",
        index=False,
        encoding="utf-8",
    )

    strong_correlations.to_csv(
        statistics_dir / "correlaciones_fuertes.csv",
        index=False,
        encoding="utf-8",
    )

    save_independence_conclusion(
        dataset_name,
        strong_correlations,
        statistics_dir / "conclusion_independencia.txt",
    )

    save_gaussian_visual_review_guide(
        dataset_name,
        statistics_dir / "guia_revision_gaussiana.txt",
    )

    print("\n Generando gráficas KDE con curvas Gaussianas...")

    plot_kde_with_gaussian(
        dataset_name,
        df,
    )

    print(" Generando matrices de correlación por clase...")

    plot_correlation_matrices(
        dataset_name,
        df,
    )

    if strong_correlations.empty:
        print(
            " No se detectaron correlaciones fuertes "
            "con |r| >= 0.70."
        )

    else:
        print(
            f" Se detectaron {len(strong_correlations)} "
            "correlaciones fuertes con |r| >= 0.70."
        )

    print(
        " Resultados del EDA guardados en: "
        f"outputs/{dataset_name}/"
    )