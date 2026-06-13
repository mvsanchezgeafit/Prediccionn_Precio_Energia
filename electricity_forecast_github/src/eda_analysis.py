"""
Utilidades para EDA del dataset diario (precio diario objetivo + covariables).

En notebooks, el objetivo suele fijarse con ``resolve_daily_price_target`` (ponderado si existe).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from xm_config import (
    DATA_PROCESSED_DIR,
    DEFAULT_DAILY_MAX_DATASET,
    TARGET_COL_DAILY_MAX_PRICE,
    TARGET_COL_DAILY_WEIGHTED_PRICE,
)
from xm_daily_target import build_daily_max_price_dataset, save_daily_max_dataset
from xm_download import download_core_series

logger = logging.getLogger(__name__)


def resolve_daily_price_target(
    df: pd.DataFrame,
    *,
    prefer_weighted: bool = True,
) -> str:
    """
    Elige columna objetivo para modelado diario.

    Por defecto usa **precio promedio ponderado** por demanda horaria si existe y tiene datos;
    si no, **precio máximo** diario.
    """
    if prefer_weighted and TARGET_COL_DAILY_WEIGHTED_PRICE in df.columns:
        s = df[TARGET_COL_DAILY_WEIGHTED_PRICE]
        if s.notna().any():
            return TARGET_COL_DAILY_WEIGHTED_PRICE
    return TARGET_COL_DAILY_MAX_PRICE


def load_cache_only() -> pd.DataFrame:
    """Carga únicamente el Parquet/CSV en ``data/processed`` (sin llamar a la API)."""
    p = DATA_PROCESSED_DIR / DEFAULT_DAILY_MAX_DATASET
    if p.is_file():
        return pd.read_parquet(p)
    alt = p.with_suffix(".csv")
    if alt.is_file():
        return pd.read_csv(alt, parse_dates=["date"])
    raise FileNotFoundError(f"No hay dataset en {p} ni {alt}")


def load_daily_auto(
    start: date | None = None,
    end: date | None = None,
    *,
    persist: bool = True,
) -> pd.DataFrame:
    """
    Lee ``xm_daily_max_price_dataset`` en ``data/processed`` si existe; si no,
    descarga desde la API XM entre ``start`` y ``end`` (por defecto 2015–2025).

    Tras la primera descarga, con ``persist=True`` (por defecto) guarda Parquet/CSV
    para que las siguientes ejecuciones no vuelvan a llamar a la API.

    Uso recomendado en notebooks cuando aún no has ejecutado ``download_xm_data.py``.
    """
    s = start or date(2015, 1, 1)
    e = end or date(2025, 12, 31)
    return load_or_build_daily(s, e, use_cache=True, persist=persist)


def load_or_build_daily(
    start,
    end,
    *,
    cache_path: Path | str | None = None,
    use_cache: bool = True,
    persist: bool = True,
) -> pd.DataFrame:
    """
    Carga Parquet/CSV en ``cache_path`` si existe y ``use_cache`` es True;
    si no, descarga XM y construye el dataset diario (con ENSO por defecto).

    Si hubo descarga/construcción y ``persist`` es True, guarda el resultado en
    ``cache_path`` (Parquet o CSV según ``save_daily_max_dataset``) para reutilizarlo.
    Use ``persist=False`` solo si no quiere escribir disco (p. ej. ``use_cache=False``
    y varias pruebas sin sobrescribir la caché local).
    """
    p = Path(cache_path) if cache_path else DATA_PROCESSED_DIR / DEFAULT_DAILY_MAX_DATASET
    if use_cache:
        if p.is_file():
            if p.suffix.lower() == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p, parse_dates=["date"])
        alt = p.with_suffix(".csv")
        if alt.is_file():
            return pd.read_csv(alt, parse_dates=["date"])

    bundle = download_core_series(start, end)
    df = build_daily_max_price_dataset(bundle, include_enso=True)
    if persist:
        out = save_daily_max_dataset(df, path=p)
        logger.info("Dataset diario guardado para caché local: %s", out)
    return df


def numeric_columns_for_corr(df: pd.DataFrame, target: str = TARGET_COL_DAILY_MAX_PRICE) -> list[str]:
    """Columnas numéricas para correlación: objetivo primero, luego el resto ordenado."""
    skip = {"date", "enso_regime"}
    num = [
        c
        for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c])
    ]
    if target in num:
        return [target] + sorted([c for c in num if c != target])
    return sorted(num)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    m = df.isna().mean().sort_values(ascending=False) * 100
    return pd.DataFrame({"pct_missing": m.round(2)})
