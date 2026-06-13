"""
Índices ENSO (El Niño / La Niña) y unión al dataset diario de precios.

Los índices suelen ser **mensuales** (p. ej. ONI, anomalía Niño 3.4). Cada día del mes
recibe el mismo valor del índice (estándar en modelos de energía e hidrología).

El CSV esperado tiene columnas: `Año`, `Mes`, `Value` (como `ENSO_2010_2025_manual.csv`).
Valores sentinela (p. ej. -99.9 para meses sin dato) se convierten en NaN.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from xm_config import ENSO_CSV_DEFAULT, ENSO_MISSING_SENTINELS

logger = logging.getLogger(__name__)

# Umbrales típicos ONI (°C) para clasificación (NOAA); ajusta si tu índice usa otra escala.
ENSO_EL_NINO_THRESHOLD = 0.5
ENSO_LA_NINA_THRESHOLD = -0.5


def _clean_enso_values(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    for bad in ENSO_MISSING_SENTINELS:
        v = v.mask(v == bad, np.nan)
    v = v.mask(v < -90, np.nan)
    return v


def load_enso_monthly(path: Path | str | None = None) -> pd.DataFrame:
    """
    Lee el CSV mensual y devuelve Year, Month, enso_index.

    Parameters
    ----------
    path :
        Ruta al CSV. Por defecto `xm_config.ENSO_CSV_DEFAULT`.
    """
    p = Path(path) if path else ENSO_CSV_DEFAULT
    if not p.is_file():
        raise FileNotFoundError(f"No existe archivo ENSO: {p}")

    df = pd.read_csv(p, encoding="utf-8-sig")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    # Aceptar nombres en español o inglés
    col_year = "Año" if "Año" in df.columns else "Year"
    col_month = "Mes" if "Mes" in df.columns else "Month"
    col_val = "Value" if "Value" in df.columns else "value"

    if col_year not in df.columns or col_month not in df.columns or col_val not in df.columns:
        raise ValueError(
            f"CSV ENSO debe tener columnas Año/Mes/Value (o Year/Month/Value). Encontradas: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "Year": pd.to_numeric(df[col_year], errors="coerce").astype("Int64"),
            "Month": pd.to_numeric(df[col_month], errors="coerce").astype("Int64"),
            "enso_index": _clean_enso_values(df[col_val]),
        }
    )
    out = out.dropna(subset=["Year", "Month"])
    out = out.sort_values(["Year", "Month"]).reset_index(drop=True)

    # Rezago 1 mes (útil como regresor: condición del mes anterior)
    out["enso_index_lag1"] = out["enso_index"].shift(1)

    return out


def enso_regime_label(
    enso_index: pd.Series,
    el_nino: float = ENSO_EL_NINO_THRESHOLD,
    la_nina: float = ENSO_LA_NINA_THRESHOLD,
) -> pd.Series:
    """Clasificación: el_nino / neutral / la_nina (para modelos categóricos o one-hot)."""
    x = pd.to_numeric(enso_index, errors="coerce")
    return pd.cut(
        x,
        bins=[-np.inf, la_nina, el_nino, np.inf],
        labels=["la_nina", "neutral", "el_nino"],
    ).astype(str).replace("nan", np.nan)


def merge_enso_onto_daily(
    daily: pd.DataFrame,
    path: Path | str | None = None,
    *,
    include_regime: bool = True,
    include_lag1: bool = True,
) -> pd.DataFrame:
    """
    Añade columnas ENSO a un DataFrame con columna `date` (una fila por día).

    - ``enso_index``: valor mensual del CSV para el (año, mes) del día.
    - ``enso_index_lag1``: índice del **mes calendario anterior** (si existe en CSV).
    - ``enso_regime`` (opcional): ``el_nino`` / ``neutral`` / ``la_nina`` según umbrales ONI.
    """
    if daily.empty or "date" not in daily.columns:
        return daily

    try:
        em = load_enso_monthly(path)
    except FileNotFoundError as e:
        logger.warning("%s — dataset sin columnas ENSO.", e)
        return daily
    except Exception as e:
        logger.warning("No se pudo cargar ENSO: %s", e)
        return daily

    d = daily.copy()
    d["Year"] = pd.to_datetime(d["date"]).dt.year
    d["Month"] = pd.to_datetime(d["date"]).dt.month

    cols = ["Year", "Month", "enso_index"]
    if include_lag1 and "enso_index_lag1" in em.columns:
        cols.append("enso_index_lag1")

    merged = d.merge(em[cols], on=["Year", "Month"], how="left")
    merged = merged.drop(columns=["Year", "Month"])

    if include_regime and "enso_index" in merged.columns:
        merged["enso_regime"] = enso_regime_label(merged["enso_index"])

    return merged


def resolve_enso_path(path: Optional[Path | str]) -> Path | None:
    """Resuelve ruta al CSV ENSO: argumento explícito, si no existe prueba ``ENSO_CSV_DEFAULT``."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    candidates.append(Path(ENSO_CSV_DEFAULT))
    for p in candidates:
        if p.is_file():
            return p
    return None
