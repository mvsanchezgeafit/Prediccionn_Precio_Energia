"""
Configuración central — layout GitHub (src/ + data/ en raíz del repo).
"""
from __future__ import annotations

from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _PROJECT_DIR.parent
PROJECT_ROOT = _PROJECT_DIR

DATA_DIR = REPO_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR
FIGURES_DIR = DATA_DIR / "figures"

METRIC_PRECIOS_BOLSA = ("PrecBolsNaci", "Sistema")
METRIC_DEMANDA_COMERCIAL = ("DemaCome", "Sistema")
METRIC_PORC_APORTE = ("PorcApor", "Sistema")
METRIC_VOLUMEN_UTIL_ENERGIA_SISTEMA = ("VoluUtilDiarEner", "Sistema")
METRIC_PORC_VOLUMEN_UTIL = ("PorcVoluUtilDiar", "Sistema")
METRIC_GENERACION = ("Gene", "Recurso")
METRIC_LISTADO_RECURSOS = ("ListadoRecursos", "Sistema")
METRIC_LISTADO_EMBALSES = ("ListadoEmbalses", "Sistema")

DEFAULT_PANEL_HOURLY_PARQUET = "xm_panel_hourly.parquet"
TARGET_COL_DAILY_MAX_PRICE = "precio_max_cop_kwh"
TARGET_COL_DAILY_WEIGHTED_PRICE = "precio_prom_ponderado_cop_kwh"
DEFAULT_FORECAST_MONTHS = 6
DEFAULT_DAILY_MAX_DATASET = "xm_daily_max_price_dataset.parquet"
ENSO_CSV_DEFAULT = DATA_DIR / "ENSO_2010_2025_manual.csv"
ENSO_MISSING_SENTINELS = (-99.9, -999.0)
