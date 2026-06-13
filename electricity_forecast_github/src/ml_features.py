"""
Features para modelos de machine learning (rezagos, calendario, hidrología, ENSO).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xm_config import TARGET_COL_DAILY_MAX_PRICE


def _enso_regime_numeric(s: pd.Series) -> pd.Series:
    m = {"la_nina": -1.0, "neutral": 0.0, "el_nino": 1.0}
    return s.map(m)


def _colombia_fixed_holiday_mask(dates: pd.Series) -> pd.Series:
    """
    Festivos fijos aproximados (no mueve Semana Santa). Solo bandera 0/1 para el árbol.
    """
    dt = pd.to_datetime(dates).dt
    md = list(zip(dt.month.astype(int), dt.day.astype(int)))
    fixed = {
        (1, 1),
        (1, 6),
        (3, 24),
        (3, 25),
        (5, 1),
        (7, 20),
        (8, 7),
        (12, 8),
        (12, 25),
    }
    return pd.Series([1.0 if m in fixed else 0.0 for m in md], index=dates.index)


# Covariables para las que el LGBM usa rezagos / media móvil corta (además del nivel en la fila).
_EXOG_FOR_EXTRA_FEATURES: tuple[str, ...] = (
    "demanda_max_kwh",
    "porc_vol_util_diario",
    "enso_index",
)
_EXOG_LAG_DAYS: tuple[int, ...] = (1, 7)


def build_tree_features(
    df: pd.DataFrame,
    target: str = TARGET_COL_DAILY_MAX_PRICE,
    lags: tuple[int, ...] = (1, 7, 14, 30),
    *,
    include_exog_lags_and_rolls: bool = True,
    include_extended_calendar: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Añade rezagos del objetivo, medias móviles, armónicos de mes y covariables numéricas.

    Si ``include_exog_lags_and_rolls`` (por defecto), también crea **rezagos 1 y 7 días** y
    **media móvil 7 días** (con desfase 1) sobre demanda, % vol. útil y ENSO, para que modelos
    de árbol (p. ej. LGBM cuantílico recursivo) no dependan solo del nivel contemporáneo.

    Si ``include_extended_calendar``, añade día del año y día de la semana (seno/coseno) y un
    indicador aproximado de festivo fijo en Colombia.
    Elimina ``vol_util_energia_sistema`` si existe ``porc_vol_util_diario`` (multicolinealidad).
    """
    d = df.sort_values("date").reset_index(drop=True).copy()
    if target not in d.columns:
        raise ValueError(f"Falta columna objetivo {target!r}")

    y = pd.to_numeric(d[target], errors="coerce")

    for lag in lags:
        d[f"lag_{lag}"] = y.shift(lag)

    d["roll_mean_7"] = y.shift(1).rolling(7, min_periods=1).mean()
    d["roll_mean_30"] = y.shift(1).rolling(30, min_periods=2).mean()

    d["month"] = pd.to_datetime(d["date"]).dt.month.astype(float)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12.0)

    calendar_extra: list[str] = []
    if include_extended_calendar:
        dt = pd.to_datetime(d["date"])
        doy = dt.dt.dayofyear.astype(float)
        d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        dow = dt.dt.dayofweek.astype(float)
        d["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        d["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        d["is_holiday_co_fix"] = _colombia_fixed_holiday_mask(d["date"])
        calendar_extra.extend(
            ["doy_sin", "doy_cos", "dow_sin", "dow_cos", "is_holiday_co_fix"]
        )

    if "enso_regime" in d.columns:
        d["enso_regime_ord"] = _enso_regime_numeric(d["enso_regime"])

    for c in (
        "demanda_max_kwh",
        "porc_vol_util_diario",
        "vol_util_energia_sistema",
        "enso_index",
        "enso_index_lag1",
        "porc_aporte_value",
        "porc_aporte_max",
    ):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    exog_derived: list[str] = []
    if include_exog_lags_and_rolls:
        for col in _EXOG_FOR_EXTRA_FEATURES:
            if col not in d.columns:
                continue
            s = pd.to_numeric(d[col], errors="coerce")
            for k in _EXOG_LAG_DAYS:
                name = f"{col}_lag_{k}"
                d[name] = s.shift(k)
                exog_derived.append(name)
            rname = f"{col}_roll7"
            d[rname] = s.shift(1).rolling(7, min_periods=1).mean()
            exog_derived.append(rname)

    # Lista explícita de features para el árbol
    candidates = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        "roll_mean_7",
        "roll_mean_30",
        "month_sin",
        "month_cos",
        *calendar_extra,
        "demanda_max_kwh",
        "porc_vol_util_diario",
        "enso_index",
        "enso_regime_ord",
        "porc_aporte_value",
        "porc_aporte_max",
    ]
    if "porc_vol_util_diario" not in d.columns and "vol_util_energia_sistema" in d.columns:
        candidates.append("vol_util_energia_sistema")

    candidates.extend(exog_derived)

    feature_names = [c for c in candidates if c in d.columns]
    # Evitar duplicar: si hay % vol útil, no usar volumen energía (muy correlacionados)
    if "porc_vol_util_diario" in feature_names and "vol_util_energia_sistema" in feature_names:
        feature_names = [c for c in feature_names if c != "vol_util_energia_sistema"]

    return d, feature_names


def dropna_xy(
    d: pd.DataFrame,
    feature_names: list[str],
    target: str = TARGET_COL_DAILY_MAX_PRICE,
) -> pd.DataFrame:
    """Quita filas con NaN en objetivo o features."""
    cols = [target] + [c for c in feature_names if c in d.columns]
    return d.dropna(subset=cols).reset_index(drop=True)


def sarimax_exog_columns(df: pd.DataFrame) -> list[str]:
    """Regresores para SARIMAX: niveles + estacionalidad en seno/coseno."""
    base = [
        "demanda_max_kwh",
        "porc_vol_util_diario",
        "enso_index",
        "porc_aporte_value",
        "porc_aporte_max",
        "month_sin",
        "month_cos",
        "week_sin",
        "week_cos",
    ]
    # Evitar dos columnas de aportes redundantes
    out = [c for c in base if c in df.columns]
    if "porc_aporte_value" in out and "porc_aporte_max" in out:
        out = [c for c in out if c != "porc_aporte_max"]
    return out


def ensure_sarimax_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Fourier mensual y semanal desde ``date`` (sin fuga; determinístico en train y exog_future)."""
    d = df.copy()
    if "date" not in d.columns:
        return d
    dt = pd.to_datetime(d["date"])
    m = dt.dt.month.astype(float)
    d["month_sin"] = np.sin(2 * np.pi * m / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * m / 12.0)
    dow = dt.dt.dayofweek.astype(float)
    d["week_sin"] = np.sin(2 * np.pi * dow / 7.0)
    d["week_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return d
