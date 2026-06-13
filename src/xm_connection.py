"""
Conexión y consultas a la API pública de XM (servapibi.xm.com.co).

No requiere autenticación. Implementación nativa con `requests` para evitar
dependencia estricta de versiones de `pydataxm` y permitir reintentos y logging.

Referencia conceptual: Equipo Analítica XM — pydataxm / API_XM.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
import requests

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover

    def _tqdm(iterable, **_kwargs):
        return iterable


logger = logging.getLogger(__name__)

BASE = "https://servapibi.xm.com.co"
ENDPOINT_LISTS = f"{BASE}/Lists"

PERIOD_PATH = {
    "HourlyEntities": "hourly",
    "DailyEntities": "daily",
    "MonthlyEntities": "monthly",
    "AnnualEntities": "annual",
}

NEST_KEY = {
    "HourlyEntities": "HourlyEntities",
    "DailyEntities": "DailyEntities",
    "MonthlyEntities": "MonthlyEntities",
    "AnnualEntities": "AnnualEntities",
}


@dataclass
class MetricSpec:
    """Par (MetricId, Entity) validado contra el inventario."""

    metric_id: str
    entity: str
    entity_type: str
    metric_name: str = ""


def _to_date(d: date | datetime | str) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return pd.Timestamp(d).date()


def _month_ranges(start: date, end: date) -> Iterator[Tuple[str, str]]:
    """Genera pares (inicio, fin) por mes calendario para respetar límites de la API."""
    cur = pd.Timestamp(start).to_period("M").to_timestamp().date()
    end_ts = pd.Timestamp(end)
    while pd.Timestamp(cur) <= end_ts:
        month_start = max(pd.Timestamp(cur).date(), start)
        month_end = min((pd.Timestamp(cur) + pd.offsets.MonthEnd(0)).date(), end)
        yield str(month_start), str(month_end)
        cur = (pd.Timestamp(cur) + pd.offsets.MonthBegin(1)).date()


class XMAPIClient:
    """
    Cliente HTTP para la API XM.

    Uso:
        client = XMAPIClient()
        inv = client.fetch_inventory()
        df = client.request_data("PrecBolsNaci", "Sistema", "2024-01-01", "2024-03-31")
    """

    def __init__(
        self,
        timeout: int = 120,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.session = session or requests.Session()
        self._inventory: Optional[pd.DataFrame] = None

    def _post_json(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.post(
                    url,
                    json=body,
                    headers={"Content-Type": "application/json", "Connection": "close"},
                    timeout=self.timeout,
                )
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                logger.warning("POST %s intento %s/%s: %s", url, attempt + 1, self.max_retries, e)
                time.sleep(self.retry_sleep * (attempt + 1))
        msg = f"Fallo la petición a {url}: {last_err}"
        err_l = str(last_err).lower()
        if (
            "failed to resolve" in err_l
            or "getaddrinfo" in err_l
            or "name or service not known" in err_l
            or "name resolution" in err_l
        ):
            msg += (
                "\n\nNo hay resolución DNS o no hay ruta al servidor (sin internet, firewall, DNS bloqueado, VPN, etc.). "
                "La URL oficial sigue siendo https://servapibi.xm.com.co . "
                "Para trabajar sin red: genera o copia el archivo "
                "`electricity_forecast/data/processed/xm_daily_max_price_dataset.parquet` "
                "(o .csv) desde un equipo con conexión (`python download_xm_data.py --start 2015-01-01 --end 2025-12-31`) "
                "y en el notebook usa `load_cache_only()` en lugar de `load_daily_auto(...)`."
            )
        raise RuntimeError(msg) from last_err

    def fetch_inventory(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Descarga el catálogo completo de métricas (equivalente a ListadoMetricas).
        """
        if self._inventory is not None and not force_refresh:
            return self._inventory

        data = self._post_json(ENDPOINT_LISTS, {"MetricId": "ListadoMetricas"})
        rows: list[dict[str, Any]] = []
        for item in data.get("Items", []):
            d = item.get("Date")
            for le in item.get("ListEntities", []):
                vals = le.get("Values") or {}
                row = {"Date": d, **vals}
                rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        self._inventory = df
        return df

    def resolve_metric(self, metric_id: str, entity: str) -> MetricSpec:
        inv = self.fetch_inventory()
        m = inv[(inv["MetricId"] == metric_id) & (inv["Entity"] == entity)]
        if m.empty:
            raise ValueError(f"No existe la combinación MetricId={metric_id!r}, Entity={entity!r} en la API.")
        row = m.iloc[0]
        return MetricSpec(
            metric_id=metric_id,
            entity=entity,
            entity_type=str(row["Type"]),
            metric_name=str(row.get("MetricName", "")),
        )

    def search_metrics(self, text: str, columns: Iterable[str] = ("MetricId", "MetricName")) -> pd.DataFrame:
        """Filtra el inventario por texto (sin distinguir mayúsculas)."""
        inv = self.fetch_inventory()
        t = text.lower()
        mask = pd.Series(False, index=inv.index)
        for c in columns:
            if c in inv.columns:
                mask = mask | inv[c].astype(str).str.lower().str.contains(t, na=False)
        return inv.loc[mask].drop_duplicates()

    def request_list_data(self, metric_id: str, entity: str) -> pd.DataFrame:
        """
        Métricas tipo ListsEntities (catálogos: recursos, embalses, etc.).
        """
        spec = self.resolve_metric(metric_id, entity)
        if spec.entity_type != "ListsEntities":
            raise ValueError(f"{metric_id}/{entity} no es ListsEntities (es {spec.entity_type}).")

        data = self._post_json(f"{BASE}/lists", {"MetricId": metric_id, "Entity": entity})
        items = data.get("Items", [])
        if not items:
            return pd.DataFrame()

        # Misma idea que pydataxm.json_normalize(..., 'ListEntities', 'Date')
        df = pd.json_normalize(items, "ListEntities", "Date", sep="_")
        return df

    def _request_period_chunk(
        self,
        metric_id: str,
        entity: str,
        entity_type: str,
        start_str: str,
        end_str: str,
        filtros: Optional[list[Any]],
    ) -> pd.DataFrame:
        path = PERIOD_PATH.get(entity_type)
        nest = NEST_KEY.get(entity_type)
        if not path or not nest:
            raise ValueError(f"Tipo no soportado: {entity_type}")

        url = f"{BASE}/{path}"
        body = {
            "MetricId": metric_id,
            "StartDate": start_str,
            "EndDate": end_str,
            "Entity": entity,
            "Filter": filtros or [],
        }
        data = self._post_json(url, body)
        items = data.get("Items", [])
        if not items:
            return pd.DataFrame()

        df = pd.json_normalize(items, nest, "Date", sep="_")
        if not df.empty and "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df

    def request_data(
        self,
        metric_id: str,
        entity: str,
        start: date | datetime | str,
        end: date | datetime | str,
        filtros: Optional[list[Any]] = None,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> pd.DataFrame:
        """
        Descarga serie temporal (hourly/daily/monthly/annual).

        Particiona por mes para alinearse con MaxDays de la API y reduce fallos por timeout.
        """
        spec = self.resolve_metric(metric_id, entity)
        if spec.entity_type == "ListsEntities":
            return self.request_list_data(metric_id, entity)

        d0, d1 = _to_date(start), _to_date(end)
        if d0 > d1:
            raise ValueError("start debe ser <= end")

        ranges = list(_month_ranges(d0, d1))
        if not parallel or len(ranges) == 1:
            frames: list[pd.DataFrame] = []
            month_iter = (
                ranges
                if len(ranges) <= 1
                else _tqdm(
                    ranges,
                    desc=f"XM {entity} · {metric_id}",
                    unit="mes",
                    leave=False,
                    mininterval=0.3,
                )
            )
            for a, b in month_iter:
                frames.append(
                    self._request_period_chunk(
                        metric_id, entity, spec.entity_type, a, b, filtros
                    )
                )
            out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            out = self._request_parallel(
                metric_id,
                entity,
                spec.entity_type,
                ranges,
                filtros,
                max_workers=max_workers,
            )

        if out.empty:
            return out

        out = out.sort_values("Date").reset_index(drop=True)
        return out

    def _request_parallel(
        self,
        metric_id: str,
        entity: str,
        entity_type: str,
        ranges: List[Tuple[str, str]],
        filtros: Optional[list[Any]],
        max_workers: int,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(
                    self._request_period_chunk,
                    metric_id,
                    entity,
                    entity_type,
                    a,
                    b,
                    filtros,
                ): (a, b)
                for a, b in ranges
            }
            done_iter = (
                as_completed(futs)
                if len(futs) <= 1
                else _tqdm(
                    as_completed(futs),
                    total=len(futs),
                    desc=f"XM {entity} · {metric_id} (paralelo)",
                    unit="mes",
                    leave=False,
                    mininterval=0.3,
                )
            )
            for fut in done_iter:
                frames.append(fut.result())
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def test_connection() -> bool:
    """Comprueba conectividad con el endpoint de inventario."""
    try:
        c = XMAPIClient()
        df = c.fetch_inventory()
        return not df.empty
    except Exception as e:
        logger.error("test_connection falló: %s", e)
        return False
