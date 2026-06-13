"""
Construcción del objetivo diario (COP/kWh) y dataset para modelado.

- **precio_max_cop_kwh:** máximo horario del precio de bolsa del día.
- **precio_prom_ponderado_cop_kwh:** media diaria ponderada por demanda comercial horaria
  ``sum_h P_h D_h / sum_h D_h`` (véase :func:`daily_weighted_price_by_demand_hourly`).

Horizonte de predicción: por defecto **6 meses calendario** desde el último día observado.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from enso_data import merge_enso_onto_daily, resolve_enso_path
from xm_config import (
    DATA_PROCESSED_DIR,
    DEFAULT_DAILY_MAX_DATASET,
    DEFAULT_FORECAST_MONTHS,
    TARGET_COL_DAILY_MAX_PRICE,
    TARGET_COL_DAILY_WEIGHTED_PRICE,
)

logger = logging.getLogger(__name__)


def _as_float_series(s: pd.Series | pd.DataFrame) -> pd.Series:
    """Una sola columna numérica float64 (evita object y columnas duplicadas `Value`)."""
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    v = pd.to_numeric(s, errors="coerce")
    return v.astype(np.float64)


def _to_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """La API XM a veces entrega números como object (strings); evita fallos en max/mean."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def hourly_price_columns(df: pd.DataFrame) -> list[str]:
    """Columnas horarias tipo Values_Hour01..Values_Hour24."""
    return sorted(
        [c for c in df.columns if c.startswith("Values_Hour")],
        key=lambda x: int(x.replace("Values_Hour", "").lstrip("0") or "0"),
    )


def daily_agg_from_hourly_wide(
    df: pd.DataFrame,
    hour_cols: list[str],
    out_name: str,
    how: str = "max",
) -> pd.DataFrame:
    """
    Una fila por día natural con agregación horizontal (max/mean/min) sobre las 24 horas.
    """
    if df.empty or not hour_cols:
        return pd.DataFrame(columns=["date", out_name])

    x = _to_numeric_cols(df, hour_cols)
    x["date"] = pd.to_datetime(x["Date"]).dt.normalize()
    if how == "max":
        agg = x[hour_cols].max(axis=1)
    elif how == "mean":
        agg = x[hour_cols].mean(axis=1)
    elif how == "min":
        agg = x[hour_cols].min(axis=1)
    else:
        raise ValueError("how debe ser max, mean o min")

    out = pd.DataFrame({"date": x["date"], out_name: agg})
    return out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


def daily_weighted_price_by_demand_hourly(
    price: pd.DataFrame,
    demand: pd.DataFrame,
    out_name: str = TARGET_COL_DAILY_WEIGHTED_PRICE,
) -> pd.DataFrame:
    """
    Precio promedio ponderado por día usando **demanda comercial horaria** como peso:

    ``sum_h P_h * D_h / sum_h D_h`` por fecha natural (COP/kWh).

    Requiere las mismas columnas ``Values_Hour*`` en precio y demanda.
    """
    if price.empty or demand.empty:
        return pd.DataFrame(columns=["date", out_name])

    h_p = hourly_price_columns(price)
    h_d = hourly_price_columns(demand)
    common = sorted(
        set(h_p) & set(h_d),
        key=lambda x: int(x.replace("Values_Hour", "").lstrip("0") or "0"),
    )
    if not common:
        logger.warning("No hay columnas horarias comunes entre precio y demanda; sin ponderado.")
        return pd.DataFrame(columns=["date", out_name])

    p = price[["Date"] + common].copy()
    d = demand[["Date"] + common].copy()
    p = _to_numeric_cols(p, common)
    d = _to_numeric_cols(d, common)
    p["date"] = pd.to_datetime(p["Date"], errors="coerce").dt.normalize()
    d["date"] = pd.to_datetime(d["Date"], errors="coerce").dt.normalize()
    p = p.drop(columns=["Date"]).groupby("date", as_index=True)[common].mean()
    d = d.drop(columns=["Date"]).groupby("date", as_index=True)[common].mean()

    idx = p.index.intersection(d.index)
    if len(idx) == 0:
        return pd.DataFrame(columns=["date", out_name])

    num = np.zeros(len(idx), dtype=np.float64)
    den = np.zeros(len(idx), dtype=np.float64)
    for c in common:
        pv = p.loc[idx, c].to_numpy(dtype=np.float64)
        dv = d.loc[idx, c].to_numpy(dtype=np.float64)
        num += np.nan_to_num(pv, nan=0.0) * np.nan_to_num(dv, nan=0.0)
        den += np.nan_to_num(dv, nan=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(den > 1e-12, num / den, np.nan)
    out = pd.DataFrame({"date": idx.values, out_name: w})
    return out.sort_values("date").reset_index(drop=True)


def _daily_one_value_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Tabla diaria API con columna `Value` (una observación por día)."""
    if df.empty or "Value" not in df.columns or "Date" not in df.columns:
        return pd.DataFrame(columns=["date", col_name])

    x = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], errors="coerce").dt.normalize(),
            "_val": _as_float_series(df["Value"]),
        }
    )
    # np.nanmean evita fallos de groupby.mean() cuando pandas infiere dtype object (p. ej. 3.12)
    out = (
        x.groupby("date", sort=True)["_val"]
        .apply(lambda s: float(np.nanmean(np.asarray(s, dtype=np.float64))))
        .reset_index()
    )
    return out.rename(columns={"_val": col_name})


def build_daily_max_price_dataset(
    bundle: dict[str, pd.DataFrame],
    include_demanda_max: bool = True,
    *,
    include_weighted_average_price: bool = True,
    include_enso: bool = True,
    enso_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Dataset diario con:
    - **precio_max_cop_kwh**: máximo horario del precio de bolsa del día.
    - **precio_prom_ponderado_cop_kwh** (opcional): media diaria ponderada por demanda horaria
      ``sum_h P_h D_h / sum_h D_h`` si hay demanda comercial horaria y
      ``include_weighted_average_price`` es True.
    - **demanda_max_kwh** (opcional): máximo horario de demanda comercial.
    - **vol_util_energia_sistema**, **porc_vol_util_diario**: variables hídricas diarias (si vienen con `Value`).
    - **porc_aporte_***: si `porc_aporte` es horario, se agrega `porc_aporte_max` (máximo diario);
      si es una sola columna `Value`, se une como `porc_aporte_value`.
    - **ENSO** (opcional): ``enso_index``, ``enso_index_lag1``, ``enso_regime`` vía ``merge_enso_onto_daily``
      si existe CSV (por defecto `ENSO_CSV_DEFAULT` en ``xm_config``).
    """
    price = bundle.get("precio_bolsa")
    if price is None or price.empty:
        raise ValueError("Se requiere bundle['precio_bolsa'] no vacío.")

    h_price = hourly_price_columns(price)
    if not h_price:
        raise ValueError("precio_bolsa no tiene columnas Values_Hour*.")

    daily = daily_agg_from_hourly_wide(price, h_price, TARGET_COL_DAILY_MAX_PRICE, how="max")

    dem = bundle.get("demanda_comercial")
    if dem is not None and not dem.empty and include_weighted_average_price:
        wavg = daily_weighted_price_by_demand_hourly(
            price, dem, TARGET_COL_DAILY_WEIGHTED_PRICE
        )
        if not wavg.empty:
            daily = daily.merge(wavg, on="date", how="left")
            miss = float(daily[TARGET_COL_DAILY_WEIGHTED_PRICE].isna().mean())
            if miss > 0.05:
                logger.warning(
                    "precio ponderado: %.1f%% fechas sin dato tras merge; revise alineación API.",
                    100.0 * miss,
                )

    if include_demanda_max and dem is not None and not dem.empty:
        h_dem = hourly_price_columns(dem)
        if h_dem:
            dmax = daily_agg_from_hourly_wide(dem, h_dem, "demanda_max_kwh", how="max")
            daily = daily.merge(dmax, on="date", how="left")

    vol = bundle.get("vol_util_energia")
    if vol is not None and not vol.empty:
        v = _daily_one_value_column(vol, "vol_util_energia_sistema")
        if not v.empty:
            daily = daily.merge(v, on="date", how="left")

    pvol = bundle.get("porc_vol_util")
    if pvol is not None and not pvol.empty:
        pv = _daily_one_value_column(pvol, "porc_vol_util_diario")
        if not pv.empty:
            daily = daily.merge(pv, on="date", how="left")

    pa = bundle.get("porc_aporte")
    if pa is not None and not pa.empty:
        if hourly_price_columns(pa):
            h_pa = hourly_price_columns(pa)
            pam = daily_agg_from_hourly_wide(pa, h_pa, "porc_aporte_max", how="max")
            daily = daily.merge(pam, on="date", how="left")
        elif "Value" in pa.columns:
            pv2 = _daily_one_value_column(pa, "porc_aporte_value")
            if not pv2.empty:
                daily = daily.merge(pv2, on="date", how="left")

    if include_enso:
        p_enso = resolve_enso_path(enso_path)
        if p_enso is not None:
            daily = merge_enso_onto_daily(daily, path=p_enso)
        else:
            logger.warning(
                "ENSO: no se encontró CSV (use enso_path= o coloque el archivo en %s).",
                "xm_config.ENSO_CSV_DEFAULT",
            )

    return daily.sort_values("date").reset_index(drop=True)


def forecast_calendar_days_index(
    last_observed_date: date | datetime | pd.Timestamp,
    months: int = DEFAULT_FORECAST_MONTHS,
) -> pd.DatetimeIndex:
    """
    Fechas futuras **diarias** desde el día siguiente al último observado hasta
    `last_observed + months` (desplazamiento de meses calendario de pandas).

    Ejemplo: último dato 2025-03-20 → predicción de 2025-03-21 hasta ~2025-09-20 (6 meses después).
    """
    last = pd.Timestamp(last_observed_date).normalize()
    end_fc = last + pd.DateOffset(months=months)
    start_fc = last + pd.Timedelta(days=1)
    if start_fc > end_fc:
        return pd.DatetimeIndex([])
    return pd.date_range(start_fc, end_fc, freq="D")


def forecast_next_n_days(
    last_observed_date: date | datetime | pd.Timestamp,
    n: int = 30,
) -> pd.DatetimeIndex:
    """
    ``n`` días naturales consecutivos desde el día siguiente al último observado
    (p. ej. último dato 2025-12-31 → 2026-01-01 … +30 días).
    """
    last = pd.Timestamp(last_observed_date).normalize()
    start_fc = last + pd.Timedelta(days=1)
    return pd.date_range(start_fc, periods=int(n), freq="D")


def save_daily_max_dataset(
    df: pd.DataFrame,
    path: Path | str | None = None,
) -> Path:
    """Guarda Parquet o CSV según disponibilidad."""
    out = Path(path) if path else DATA_PROCESSED_DIR / DEFAULT_DAILY_MAX_DATASET
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out, index=False)
    except Exception as e:
        logger.warning("Parquet no disponible (%s); guardando CSV.", e)
        out = out.with_suffix(".csv")
        df.to_csv(out, index=False)
    return out


def summarize_target_and_horizon(
    daily_df: pd.DataFrame,
    months: int = DEFAULT_FORECAST_MONTHS,
) -> dict:
    """Resumen para reportes: última fecha, nº de días a pronosticar, rango futuro."""
    if daily_df.empty or "date" not in daily_df.columns:
        return {}
    last = pd.Timestamp(daily_df["date"].max()).normalize()
    idx = forecast_calendar_days_index(last, months=months)
    return {
        "ultima_fecha_observada": last.date(),
        "horizonte_meses": months,
        "dias_a_predecir": len(idx),
        "primera_fecha_prediccion": idx.min().date() if len(idx) else None,
        "ultima_fecha_prediccion": idx.max().date() if len(idx) else None,
    }
