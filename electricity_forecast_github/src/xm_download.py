"""
Descarga orquestada de series XM para el modelo de precios (panel multivariado).

Incluye:
- Precio bolsa, demanda, aportes (horario)
- Variables hídricas agregadas Sistema (diario)
- Listados (recursos, embalses) para cruces posteriores
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **_kwargs):
        return iterable

from xm_config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    METRIC_DEMANDA_COMERCIAL,
    METRIC_LISTADO_EMBALSES,
    METRIC_LISTADO_RECURSOS,
    METRIC_PORC_APORTE,
    METRIC_PORC_VOLUMEN_UTIL,
    METRIC_PRECIOS_BOLSA,
    METRIC_GENERACION,
    METRIC_VOLUMEN_UTIL_ENERGIA_SISTEMA,
)
from xm_connection import XMAPIClient
from xm_daily_target import build_daily_max_price_dataset, save_daily_max_dataset

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def merge_hourly_with_daily(
    df_hourly: pd.DataFrame,
    df_daily: pd.DataFrame,
    daily_value_cols: list[str],
    prefix: str = "daily_",
) -> pd.DataFrame:
    """
    Une columnas diarias al índice horario usando la fecha calendario (sin alterar hora).
    Las columnas diarias se repiten para las 24 horas del mismo día.
    """
    if df_hourly.empty or df_daily.empty:
        return df_hourly

    h = df_hourly.copy()
    h["_merge_date"] = pd.to_datetime(h["Date"]).dt.normalize()
    d = df_daily.copy()
    d["_merge_date"] = pd.to_datetime(d["Date"]).dt.normalize()
    use_cols = ["_merge_date"] + [c for c in daily_value_cols if c in d.columns]
    d = d[use_cols].drop_duplicates(subset=["_merge_date"])
    merged = h.merge(d, on="_merge_date", how="left")
    rename = {c: f"{prefix}{c}" for c in daily_value_cols if c in merged.columns}
    merged = merged.rename(columns=rename)
    merged = merged.drop(columns=["_merge_date"])
    return merged


def download_core_series(
    start: date | datetime | str,
    end: date | datetime | str,
    client: Optional[XMAPIClient] = None,
    include_generation: bool = False,
    parallel: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Descarga las series base para modelado.

    Parameters
    ----------
    include_generation :
        Si True, descarga `Gene` por Recurso (volumen alto; usar con cuidado).
    parallel :
        Descarga mensual en paralelo (más rápido; más carga al servidor).
    """
    c = client or XMAPIClient()
    out: dict[str, pd.DataFrame] = {}

    jobs: list[tuple[str, Callable[[], pd.DataFrame]]] = [
        ("precio_bolsa", lambda: c.request_data(*METRIC_PRECIOS_BOLSA, start, end, parallel=parallel)),
        ("demanda_comercial", lambda: c.request_data(*METRIC_DEMANDA_COMERCIAL, start, end, parallel=parallel)),
        ("porc_aporte", lambda: c.request_data(*METRIC_PORC_APORTE, start, end, parallel=parallel)),
        (
            "vol_util_energia",
            lambda: c.request_data(*METRIC_VOLUMEN_UTIL_ENERGIA_SISTEMA, start, end, parallel=parallel),
        ),
        ("porc_vol_util", lambda: c.request_data(*METRIC_PORC_VOLUMEN_UTIL, start, end, parallel=parallel)),
    ]
    if include_generation:
        jobs.append(
            ("generacion_recurso", lambda: c.request_data(*METRIC_GENERACION, start, end, parallel=parallel)),
        )

    print(
        f"[xm_download] Descargando {len(jobs)} series (cada una muestra progreso por mes en la consola)...",
        flush=True,
    )
    for name, fetch in tqdm(jobs, desc="XM series", unit="serie"):
        logger.info("Descargando %s...", name)
        print(f"[xm_download]   → {name}", flush=True)
        out[name] = fetch()

    return out


def download_list_catalogs(client: Optional[XMAPIClient] = None) -> dict[str, pd.DataFrame]:
    """Catálogos list (sin fechas): recursos y embalses."""
    c = client or XMAPIClient()
    return {
        "listado_recursos": c.request_list_data(*METRIC_LISTADO_RECURSOS),
        "listado_embalses": c.request_list_data(*METRIC_LISTADO_EMBALSES),
    }


def build_hourly_panel(
    bundle: dict[str, pd.DataFrame],
    daily_value_cols_vol: Optional[list[str]] = None,
    daily_value_cols_porc: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Construye un panel horario: parte de precio y demanda y añade variables diarias hídricas.
    Las columnas exactas dependen de la API; por defecto se intenta `Value` en diarios.
    """
    price = bundle.get("precio_bolsa", pd.DataFrame()).copy()
    if price.empty:
        raise ValueError("Falta precio_bolsa en el bundle.")

    panel = price.copy()
    dem = bundle.get("demanda_comercial")
    if dem is not None and not dem.empty:
        panel = panel.merge(dem, on="Date", how="outer", suffixes=("", "_dema"))

    vol = bundle.get("vol_util_energia", pd.DataFrame())
    pvol = bundle.get("porc_vol_util", pd.DataFrame())

    dvc_vol = daily_value_cols_vol or (["Value"] if "Value" in vol.columns else [])
    dvc_porc = daily_value_cols_porc or (["Value"] if "Value" in pvol.columns else [])

    if not vol.empty and dvc_vol:
        panel = merge_hourly_with_daily(panel, vol, dvc_vol, prefix="hidro_")
    if not pvol.empty and dvc_porc:
        panel = merge_hourly_with_daily(panel, pvol, dvc_porc, prefix="hidro_")

    return panel.sort_values("Date").reset_index(drop=True)


def save_bundle(
    bundle: dict[str, pd.DataFrame],
    directory: Path | str | None = None,
    prefix: str = "xm_",
) -> list[Path]:
    """Guarda cada DataFrame en Parquet (o CSV si no hay engine pyarrow/fastparquet)."""
    base = Path(directory) if directory else DATA_RAW_DIR
    _ensure_dir(base)
    written: list[Path] = []
    for name, df in bundle.items():
        if df.empty:
            logger.warning("Serie vacía, no se guarda: %s", name)
            continue
        p_parquet = base / f"{prefix}{name}.parquet"
        p_csv = base / f"{prefix}{name}.csv"
        try:
            df.to_parquet(p_parquet, index=False)
            written.append(p_parquet)
        except Exception as e:
            logger.warning("Parquet no disponible (%s), usando CSV.", e)
            df.to_csv(p_csv, index=False)
            written.append(p_csv)
    return written


def run_default_download(
    start: date | datetime | str,
    end: date | datetime | str,
    output_dir: Path | str | None = None,
    include_generation: bool = False,
    save_daily_max_price: bool = True,
    include_enso: bool = True,
    enso_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Flujo listo para script: descarga core + catálogos + panel horario y guarda en `data/raw`.

    Si ``save_daily_max_price`` es True, genera además el dataset diario con **precio máximo por día**
    (objetivo para horizonte multi-paso de 6 meses en la fase de modelado), incluyendo **ENSO** si hay CSV.
    """
    client = XMAPIClient()
    print("[xm_download] Series temporales (API)...", flush=True)
    bundle = download_core_series(
        start, end, client=client, include_generation=include_generation
    )
    print("[xm_download] Catálogos (listas)...", flush=True)
    catalogs = download_list_catalogs(client)

    full = {**bundle, **{f"cat_{k}": v for k, v in catalogs.items()}}
    raw_dir = (Path(output_dir) / "raw") if output_dir else DATA_RAW_DIR
    _ensure_dir(raw_dir)
    print(f"[xm_download] Guardando Parquet/CSV en {raw_dir}...", flush=True)
    paths = save_bundle(full, directory=raw_dir)

    print("[xm_download] Panel horario y escritura en processed...", flush=True)
    panel = build_hourly_panel(bundle)
    if output_dir:
        panel_dir = Path(output_dir) / "processed"
    else:
        panel_dir = DATA_PROCESSED_DIR
    _ensure_dir(panel_dir)
    try:
        panel_path = panel_dir / "xm_panel_hourly.parquet"
        panel.to_parquet(panel_path, index=False)
        paths.append(panel_path)
    except Exception as e:
        logger.warning("No se pudo guardar panel parquet (%s); CSV.", e)
        panel_path = panel_dir / "xm_panel_hourly.csv"
        panel.to_csv(panel_path, index=False)
        paths.append(panel_path)

    daily_max = None
    if save_daily_max_price:
        print("[xm_download] Dataset precio máximo diario (+ ENSO si aplica)...", flush=True)
        try:
            daily_max = build_daily_max_price_dataset(
                bundle, include_enso=include_enso, enso_path=enso_path
            )
            dpath = save_daily_max_dataset(daily_max, path=panel_dir / "xm_daily_max_price_dataset.parquet")
            paths.append(dpath)
        except Exception as e:
            logger.warning("No se pudo construir/guardar dataset precio máximo diario: %s", e)

    out: dict[str, Any] = {"bundle": full, "panel": panel, "saved_paths": paths}
    if daily_max is not None:
        out["daily_max_price"] = daily_max
    print("[xm_download] Listo.", flush=True)
    return out
