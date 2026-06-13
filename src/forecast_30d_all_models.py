"""
Pronóstico **N días hacia adelante** (por defecto 30) desde el último día del panel, con varios modelos alineados por fecha.

Incluye (misma lógica base que el notebook 10, sin TVT):
  - **SARIMAX** + exógenas proyectadas (bloque 1 STL/ETS),
  - **SARIMAX + GARCH (sim)** punto en COP (media de trayectorias, σ̂ multi-paso),
  - **VAR** en nivel (precio + covariables en diferencias),
  - **Híbrido LGBM + SARIMAX** (`final_pred` de ``run_full_pipeline`` / ``block7``),
  - **Ensemble + spike** (``train_hybrid_ensemble_spike`` + ``ensemble_forecast_df``),
  - **TFT p50** (opcional, ``forecast_30d_tft``): Lightning + pytorch-forecasting; ``sigma_feat=0``;
    covariables futuras = ``exog_future`` del mismo pipeline.

Ejemplos::

    python forecast_30d_all_models.py --data-cache
    python forecast_30d_all_models.py --data data/processed/xm_daily_max_price_dataset.parquet --days 30
    python forecast_30d_all_models.py --data-cache --skip-tft

Salida por defecto: ``data/processed/forecast_30d_all_models_<última_fecha_hist>.csv`` + JSON meta.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from eda_analysis import load_cache_only, load_or_build_daily, resolve_daily_price_target
from garch_intervals import (
    fit_garch,
    forecast_garch_volatility,
    simulate_sarimax_garch_point_cop,
    var_forecast_and_fit,
)
from hybrid_direct_30d import normalize_column_names, run_full_pipeline
from hybrid_ensemble_spike import ensemble_forecast_df, train_hybrid_ensemble_spike
from ml_features import ensure_sarimax_calendar
from xm_config import DATA_PROCESSED_DIR

logger = logging.getLogger(__name__)


def _load_df(*, data: Path | None, data_cache: bool, start: str | None, end: str | None) -> pd.DataFrame:
    if data is not None:
        path = Path(data)
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, parse_dates=["date"])
    elif start and end:
        df = load_or_build_daily(date.fromisoformat(start), date.fromisoformat(end), use_cache=False)
    elif data_cache:
        df = load_cache_only()
    else:
        df = load_or_build_daily(date(2015, 1, 1), date(2025, 12, 31), use_cache=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def run_forecast_30d_all(
    df: pd.DataFrame,
    *,
    target: str,
    days: int = 30,
    garch_n_paths: int = 2000,
    skip_ensemble: bool = False,
    include_tft: bool = True,
    tft_max_epochs: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    d = normalize_column_names(df.sort_values("date").copy())
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    last_hist = pd.Timestamp(d["date"].max()).normalize()

    pipe = run_full_pipeline(
        d,
        target=target,
        days=int(days),
        exog_method="ets",
        univariate_structure="stl_trend_season_error",
        volatility_mode="residual_bootstrap",
        volatility_strength=1.0,
        exog_random_state=42,
        exog_scenario_replicates=3,
        exog_enso_flat_max_horizon=max(30, int(days)),
        origin_stride=14,
        min_train_size=500,
    )
    fc = pipe["forecast_table"].copy()
    b2 = pipe["sarimax_block"]
    ex_f = pipe["exog_future"]

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(fc["date"]).dt.normalize(),
            "sarimax_cop": fc["sarimax_pred"].astype(float).values,
            "hibrido_lgbm_sarimax_cop": fc["final_pred"].astype(float).values,
            "hibrido_q10_cop": fc["q10"].astype(float).values,
            "hibrido_q50_cop": fc["q50"].astype(float).values,
            "hibrido_q90_cop": fc["q90"].astype(float).values,
        }
    )

    # --- SARIMAX + GARCH (sim), misma idea que notebook 10 (innovaciones independientes por h) ---
    meta_extra: dict = {"sarimax_garch_sim": None}
    r_log = np.asarray(b2["residuals_in_sample_log"], dtype=float).ravel()
    g_fit, g_err = fit_garch(r_log, p=1, q=1, min_obs=200)
    if g_fit is not None:
        try:
            sig_h = forecast_garch_volatility(g_fit, int(days))
            y_log = np.asarray(b2["sarimax_pred_future_log"], dtype=float).ravel()[: int(days)]
            if y_log.size == sig_h.size:
                sim_cop = simulate_sarimax_garch_point_cop(
                    y_log,
                    sig_h,
                    n_paths=int(garch_n_paths),
                    aggregate="mean_cop",
                    rng=np.random.default_rng(42),
                )
                out["sarimax_garch_sim_cop"] = sim_cop
                meta_extra["sarimax_garch_sim"] = {"ok": True, "garch_fit_error": None}
            else:
                out["sarimax_garch_sim_cop"] = np.nan
                meta_extra["sarimax_garch_sim"] = {
                    "ok": False,
                    "reason": f"shape y_log={y_log.size} vs sig={sig_h.size}",
                }
        except Exception as e:  # pragma: no cover
            out["sarimax_garch_sim_cop"] = np.nan
            meta_extra["sarimax_garch_sim"] = {"ok": False, "reason": repr(e)}
    else:
        out["sarimax_garch_sim_cop"] = np.nan
        meta_extra["sarimax_garch_sim"] = {"ok": False, "garch_fit_error": g_err}

    # --- VAR (nivel; exógenas futuras Bloque 1, alineado con notebook 10 §3) ---
    pv, _vres, verr = var_forecast_and_fit(
        d,
        int(days),
        target,
        ensure_sarimax_calendar=ensure_sarimax_calendar,
        normalize_column_names=normalize_column_names,
        projected_exog_future=ex_f,
        exog_futr_source="projected",
    )
    if pv is not None and len(pv) >= int(days):
        out["var_cop"] = np.asarray(pv[: int(days)], dtype=float)
        meta_extra["var"] = {"ok": True, "error": None}
    elif pv is not None:
        v = np.full(int(days), np.nan, dtype=float)
        v[: len(pv)] = np.asarray(pv, dtype=float)
        out["var_cop"] = v
        meta_extra["var"] = {"ok": True, "error": f"VAR devolvió h={len(pv)} (se rellenó con NaN)"}
    else:
        out["var_cop"] = np.nan
        meta_extra["var"] = {"ok": False, "error": verr}

    # --- Ensemble + spike (sin matriz de volatilidad futura; igual que muchas corridas del notebook) ---
    meta_extra["ensemble_spike"] = None
    if not skip_ensemble:
        try:
            ens_b = train_hybrid_ensemble_spike(
                d,
                target=target,
                max_horizon=int(days),
                min_train_size=500,
                origin_stride=14,
            )
            ens_df = ensemble_forecast_df(d, ex_f, ens_b, b2, target=target)
            out["ensemble_spike_cop"] = ens_df["ensemble_pred"].astype(float).values[: int(days)]
            meta_extra["ensemble_spike"] = {"ok": True}
        except Exception as e:  # pragma: no cover
            out["ensemble_spike_cop"] = np.nan
            meta_extra["ensemble_spike"] = {"ok": False, "error": repr(e)}
            logger.warning("Ensemble+spike omitido: %s", e)
    else:
        out["ensemble_spike_cop"] = np.nan
        meta_extra["ensemble_spike"] = {"ok": False, "skipped": True}

    meta_extra["tft"] = None
    if include_tft:
        try:
            from forecast_30d_tft import tft_forecast_p50_future

            tft_pred, tft_m = tft_forecast_p50_future(
                d,
                ex_f,
                target,
                int(days),
                max_epochs=tft_max_epochs,
                use_disk_cache=True,
            )
            v = np.asarray(tft_pred, dtype=float).ravel()
            if v.size < int(days):
                v2 = np.full(int(days), np.nan, dtype=float)
                v2[: v.size] = v
                v = v2
            out["tft_cop"] = v[: int(days)]
            meta_extra["tft"] = tft_m
        except ImportError as e:  # pragma: no cover
            out["tft_cop"] = np.nan
            meta_extra["tft"] = {"ok": False, "error": f"ImportError: {e}"}
            logger.warning("TFT omitido (dependencias): %s", e)
        except Exception as e:  # pragma: no cover
            out["tft_cop"] = np.nan
            meta_extra["tft"] = {"ok": False, "error": repr(e)}
            logger.warning("TFT omitido: %s", e)
    else:
        out["tft_cop"] = np.nan
        meta_extra["tft"] = {"ok": False, "skipped": True}

    meta = {
        "ultima_fecha_historica": str(last_hist.date()),
        "target": target,
        "n_dias": int(days),
        **meta_extra,
    }
    return out, meta


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Pronóstico N días — varios modelos (+ TFT opcional)")
    p.add_argument("--data", type=str, default=None, help="Parquet/CSV diario")
    p.add_argument("--data-cache", action="store_true", help="Usar caché EDA (load_cache_only)")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--garch-paths", type=int, default=2000, dest="garch_n_paths")
    p.add_argument("--skip-ensemble", action="store_true", help="No entrena ensemble+spike (más rápido)")
    p.add_argument("--skip-tft", action="store_true", help="No entrena TFT (evita torch/Lightning)")
    p.add_argument("--out", type=str, default=None, help="CSV de salida (default: data/processed/...)")
    args = p.parse_args()

    data_path = Path(args.data) if args.data else None
    try:
        df = _load_df(data=data_path, data_cache=bool(args.data_cache), start=args.start, end=args.end)
    except FileNotFoundError as e:
        logging.error("%s", e)
        return 2

    target = resolve_daily_price_target(df, prefer_weighted=True)
    logging.info("Target: %s | filas: %d", target, len(df))

    try:
        out, meta = run_forecast_30d_all(
            df,
            target=target,
            days=int(args.days),
            garch_n_paths=int(args.garch_n_paths),
            skip_ensemble=bool(args.skip_ensemble),
            include_tft=not bool(args.skip_tft),
        )
    except ImportError as e:
        logging.error("Falta dependencia (p. ej. lightgbm). Instale el entorno del proyecto. %s", e)
        return 1
    except Exception as e:
        logging.exception("Fallo en pipeline: %s", e)
        return 1

    out_dir = DATA_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"forecast_fwd_{meta['n_dias']}d_{meta['ultima_fecha_historica']}"
    csv_path = Path(args.out) if args.out else out_dir / f"{stem}.csv"
    json_path = csv_path.with_suffix(".meta.json")

    out.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    print("Última fecha histórica:", meta["ultima_fecha_historica"])
    print("Días:", meta["n_dias"], "| Target:", meta["target"])
    print("CSV:", csv_path.resolve())
    print("Meta:", json_path.resolve())
    print(out.head(3).to_string(index=False))
    print("...")
    print(out.tail(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
