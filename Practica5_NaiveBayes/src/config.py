"""
Práctica 5 - Inteligencia Artificial
Configuración general del proyecto.

Datasets seleccionados:
    - Wine, UCI Machine Learning Repository.
    - Iris, UCI Machine Learning Repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Rutas generales
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TARGET_COL = "class"
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Configuración de datasets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuración necesaria para descargar y preparar un dataset.

    Attributes:
        name: Nombre interno del dataset.
        url: URL directa del archivo original de UCI.
        raw_filename: Nombre del archivo descargado.
        processed_filename: Nombre del archivo limpio generado.
        columns: Nombres de las columnas.
    """

    name: str
    url: str
    raw_filename: str
    processed_filename: str
    columns: tuple[str, ...]


DATASETS: dict[str, DatasetConfig] = {
    "wine": DatasetConfig(
        name="wine",
        url=(
            "https://archive.ics.uci.edu/ml/"
            "machine-learning-databases/wine/wine.data"
        ),
        raw_filename="wine.data",
        processed_filename="wine_clean.csv",
        columns=(
            "class",
            "alcohol",
            "malic_acid",
            "ash",
            "alcalinity_of_ash",
            "magnesium",
            "total_phenols",
            "flavanoids",
            "nonflavanoid_phenols",
            "proanthocyanins",
            "color_intensity",
            "hue",
            "od280_od315",
            "proline",
        ),
    ),
    "iris": DatasetConfig(
        name="iris",
        url=(
            "https://archive.ics.uci.edu/ml/"
            "machine-learning-databases/iris/iris.data"
        ),
        raw_filename="iris.data",
        processed_filename="iris_clean.csv",
        columns=(
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "class",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Creación automática de carpetas
# ---------------------------------------------------------------------------

def create_project_directories() -> None:
    """
    Crea automáticamente las carpetas necesarias del proyecto.

    Carpetas generadas:
        data/raw/
        data/processed/
        outputs/wine/estadisticas/
        outputs/wine/kde/
        outputs/wine/correlacion/
        outputs/wine/resultados/
        outputs/iris/estadisticas/
        outputs/iris/kde/
        outputs/iris/correlacion/
        outputs/iris/resultados/
    """

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASETS:
        base_output = OUTPUTS_DIR / dataset_name

        for subdirectory in (
            "estadisticas",
            "kde",
            "correlacion",
            "resultados",
        ):
            (base_output / subdirectory).mkdir(
                parents=True,
                exist_ok=True,
            )