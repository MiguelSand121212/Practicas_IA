"""
Práctica 5 - Inteligencia Artificial
Descarga, carga y limpieza de los datasets Wine e Iris.

Los datasets principales se descargan desde el repositorio UCI.
Si ocurre un problema de conexión, se utiliza una copia de respaldo
incluida en scikit-learn únicamente para obtener los datos.
"""

from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path

import pandas as pd

from .config import (
    DATASETS,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    TARGET_COL,
    DatasetConfig,
    create_project_directories,
)


# ---------------------------------------------------------------------------
# Descarga de archivos originales
# ---------------------------------------------------------------------------

def download_dataset(
    config: DatasetConfig,
    force_download: bool = False,
) -> Path:
    """
    Descarga un dataset desde UCI hacia la carpeta data/raw/.

    Args:
        config: Configuración del dataset.
        force_download: Si es True, reemplaza el archivo existente.

    Returns:
        Ruta local del archivo descargado.
    """

    create_project_directories()

    destination = DATA_RAW_DIR / config.raw_filename

    if destination.exists() and not force_download:
        return destination

    print(f" Descargando dataset {config.name.upper()} desde UCI...")

    try:
        ssl_context = ssl.create_default_context()

        with urllib.request.urlopen(
            config.url,
            context=ssl_context,
            timeout=30,
        ) as response:
            destination.write_bytes(response.read())

    except Exception as error:
        print(
            f" Aviso: no fue posible descargar {config.name.upper()} "
            f"desde UCI: {error}"
        )
        print(
            " Se utilizará una copia de respaldo incluida en "
            "scikit-learn para continuar la ejecución."
        )

        _write_sklearn_fallback(
            dataset_name=config.name,
            destination=destination,
        )

    return destination


# ---------------------------------------------------------------------------
# Respaldo de datos en caso de fallo de descarga
# ---------------------------------------------------------------------------

def _write_sklearn_fallback(
    dataset_name: str,
    destination: Path,
) -> None:
    """
    Genera un archivo equivalente usando datasets incluidos en scikit-learn.

    Este método sólo obtiene los datos si UCI no está disponible.
    El clasificador principal continúa siendo implementado desde cero.
    """

    from sklearn.datasets import load_iris, load_wine

    if dataset_name == "wine":
        dataset = load_wine()

        fallback_df = pd.DataFrame(dataset.data)

        # UCI utiliza etiquetas 1, 2 y 3.
        fallback_df.insert(
            0,
            "class",
            dataset.target + 1,
        )

    elif dataset_name == "iris":
        dataset = load_iris()

        fallback_df = pd.DataFrame(dataset.data)

        labels = [
            f"Iris-{dataset.target_names[target]}"
            for target in dataset.target
        ]

        fallback_df[4] = labels

    else:
        raise ValueError(
            f"No existe respaldo para el dataset '{dataset_name}'."
        )

    fallback_df.to_csv(
        destination,
        index=False,
        header=False,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Carga y limpieza
# ---------------------------------------------------------------------------

def load_dataset(
    dataset_name: str,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Carga y limpia uno de los datasets de la práctica.

    Args:
        dataset_name: Nombre del dataset: 'wine' o 'iris'.
        force_download: Si es True, descarga nuevamente el archivo.

    Returns:
        DataFrame limpio con características numéricas y clase categórica.
    """

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset desconocido: '{dataset_name}'. "
            f"Opciones disponibles: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]

    raw_path = download_dataset(
        config,
        force_download=force_download,
    )

    df = pd.read_csv(
        raw_path,
        header=None,
        names=list(config.columns),
    )

    # El archivo original de Iris contiene una línea vacía al final.
    df = df.dropna().reset_index(drop=True)

    if dataset_name == "wine":
        df[TARGET_COL] = (
            df[TARGET_COL]
            .astype(int)
            .map(lambda value: f"Clase {value}")
        )

    elif dataset_name == "iris":
        df[TARGET_COL] = (
            df[TARGET_COL]
            .astype(str)
            .str.strip()
        )

    feature_cols = [
        column
        for column in df.columns
        if column != TARGET_COL
    ]

    df[feature_cols] = df[feature_cols].apply(
        pd.to_numeric,
        errors="raise",
    )

    processed_path = (
        DATA_PROCESSED_DIR
        / config.processed_filename
    )

    df.to_csv(
        processed_path,
        index=False,
        encoding="utf-8",
    )

    return df


def load_all_datasets(
    force_download: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Carga los dos datasets utilizados en la práctica.

    Returns:
        Diccionario:
            {
                "wine": DataFrame,
                "iris": DataFrame
            }
    """

    return {
        dataset_name: load_dataset(
            dataset_name,
            force_download=force_download,
        )
        for dataset_name in DATASETS
    }


# ---------------------------------------------------------------------------
# Prueba rápida del archivo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    datasets = load_all_datasets()

    for name, dataframe in datasets.items():
        print("\n" + "=" * 70)
        print(f"DATASET: {name.upper()}")
        print("=" * 70)
        print(
            f"Muestras: {dataframe.shape[0]} | "
            f"Características: {dataframe.shape[1] - 1} | "
            f"Clases: {dataframe[TARGET_COL].nunique()}"
        )
        print(dataframe.head().to_string(index=False))