"""
División temporal **train / validación / test** para paneles diarios (sin mezclar el futuro).

Orden cronológico: ``train`` | ``val`` | ``test`` (el test es la cola de la serie).

Protocolo recomendado:
- **Validación:** ajuste y comparación de modelos / hiperparámetros (p. ej. backtest con corte al final del train).
- **Test:** una sola evaluación al final, entrenando con **train + val** y cortando al final de val.

``train_val_test_compare_three_models`` añade **SARIMAX** y **ensemble+spike** (mismo corte y horizonte
que el híbrido) para comparar en val y test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from xm_config import TARGET_COL_DAILY_MAX_PRICE


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas del panel diario (misma lógica que ``hybrid_direct_30d``).

    Se define aquí para que ``split_train_val_test`` no importe ``hybrid_direct_30d`` (evita
    cargar LightGBM y el resto del híbrido solo por usar el TVT).
    """
    if "porc_vol_util" in df.columns and "porc_vol_util_diario" not in df.columns:
        return df.rename(columns={"porc_vol_util": "porc_vol_util_diario"})
    return df.copy()


@dataclass
class TemporalTVTSplit:
    """Límites y subconjuntos disjuntos por fecha."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    date_train_end: pd.Timestamp
    date_val_end: pd.Timestamp
    date_test_end: pd.Timestamp


def split_train_val_test(
    df: pd.DataFrame,
    *,
    val_days: int,
    test_days: int,
    min_train_days: int = 500,
    date_col: str = "date",
) -> TemporalTVTSplit:
    """
    Parte el panel en tres tramos contiguos por **calendario**:

    - **test:** últimos ``test_days`` (fechas naturales).
    - **val:** ``val_days`` inmediatamente anteriores al test.
    - **train:** todo lo anterior (debe tener al menos ``min_train_days`` filas).
    """
    if val_days < 1 or test_days < 1:
        raise ValueError("val_days y test_days deben ser >= 1.")
    d = normalize_column_names(df).sort_values(date_col).copy()
    d[date_col] = pd.to_datetime(d[date_col]).dt.normalize()
    last = d[date_col].max()

    test_start = last - pd.Timedelta(days=test_days - 1)
    test_df = d[d[date_col] >= test_start].copy()

    val_end = test_start - pd.Timedelta(days=1)
    val_start = val_end - pd.Timedelta(days=val_days - 1)
    val_df = d[(d[date_col] >= val_start) & (d[date_col] <= val_end)].copy()

    train_df = d[d[date_col] < val_start].copy()

    if len(train_df) < min_train_days:
        raise ValueError(
            f"Train tiene {len(train_df)} filas < min_train_days={min_train_days}. "
            "Reduzca val_days/test_days o use más historia."
        )
    if val_df.empty or test_df.empty:
        raise ValueError("Val o test vacío; revise val_days/test_days.")

    return TemporalTVTSplit(
        train=train_df,
        val=val_df,
        test=test_df,
        date_train_end=train_df[date_col].max(),
        date_val_end=val_df[date_col].max(),
        date_test_end=test_df[date_col].max(),
    )


def hybrid_train_val_test_backtests(
    df: pd.DataFrame,
    *,
    target: str = TARGET_COL_DAILY_MAX_PRICE,
    val_days: int = 120,
    test_days: int = 60,
    forecast_days: int = 30,
    min_train_days: int = 500,
    exog_block1_kw: dict[str, Any] | None = None,
    **train_kw: Any,
) -> dict[str, Any]:
    """
    Ejecuta el híbrido directo (``hybrid_direct_30d``) en dos ventanas:

    1. **Val:** entrena solo con ``train``, corte = último día del train, pronostica los
       primeros ``min(forecast_days, len(val))`` días de val y compara con realizados.
    2. **Test:** entrena con ``train + val``, corte = último día de val, pronostica los
       primeros ``min(forecast_days, len(test))`` días de test y compara con realizados.

    Use **val** para elegir hiperparámetros; **test** solo para cifras finales (una pasada).
    """
    spl = split_train_val_test(
        df,
        val_days=val_days,
        test_days=test_days,
        min_train_days=min_train_days,
    )

    h_val = min(forecast_days, len(spl.val))
    h_test = min(forecast_days, len(spl.test))

    from hybrid_direct_30d import temporal_backtest_at_cutoff

    bt_val = temporal_backtest_at_cutoff(
        df,
        spl.date_train_end,
        target=target,
        forecast_days=h_val,
        train_df=spl.train,
        exog_block1_kw=exog_block1_kw,
        **train_kw,
    )

    train_val = (
        pd.concat([spl.train, spl.val], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    bt_test = temporal_backtest_at_cutoff(
        df,
        spl.date_val_end,
        target=target,
        forecast_days=h_test,
        train_df=train_val,
        exog_block1_kw=exog_block1_kw,
        **train_kw,
    )

    return {
        "split": spl,
        "forecast_days_val": h_val,
        "forecast_days_test": h_test,
        "val_backtest": bt_val,
        "test_backtest": bt_test,
    }


def summarize_tvt_backtest(name: str, bt: dict[str, Any]) -> dict[str, float]:
    """Extrae MAE/RMSE y n del dict devuelto por ``temporal_backtest_at_cutoff``."""
    cmp = bt.get("comparison_df")
    if cmp is None or cmp.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0.0}
    y = cmp["y_true"].astype(float)
    p = cmp["final_pred"].astype(float)
    mae = float((y - p).abs().mean())
    rmse = float(((y - p) ** 2).mean() ** 0.5)
    return {"mae": mae, "rmse": rmse, "n": float(len(cmp))}


def three_models_metrics_summary(
    m_val: dict[str, float],
    m_test: dict[str, float],
) -> pd.DataFrame:
    """
    Tabla lista para informe: MAE/RMSE en val y test para híbrido, SARIMAX y ensemble+spike
    (claves como las devuelve ``extend_hybrid_backtest_with_ensemble_spike``).
    """
    specs: list[tuple[str, str]] = [
        ("Híbrido (LGBM + SARIMAX, final_pred)", "hybrid_final"),
        ("SARIMAX solo", "hybrid_sarimax"),
        ("Ensemble + spike", "ensemble_spike"),
    ]
    rows: list[dict[str, Any]] = []
    for label, key in specs:
        rows.append(
            {
                "Modelo": label,
                "MAE_val": float(m_val.get(f"mae_{key}", float("nan"))),
                "RMSE_val": float(m_val.get(f"rmse_{key}", float("nan"))),
                "MAE_test": float(m_test.get(f"mae_{key}", float("nan"))),
                "RMSE_test": float(m_test.get(f"rmse_{key}", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def train_val_test_compare_three_models(
    df: pd.DataFrame,
    *,
    target: str = TARGET_COL_DAILY_MAX_PRICE,
    val_days: int = 120,
    test_days: int = 60,
    forecast_days: int = 90,
    min_train_days: int = 500,
    exog_method: str = "ets",
    univariate_structure: str = "stl_trend_season_error",
    exog_block1_kw: dict[str, Any] | None = None,
    hybrid_train_kw: dict[str, Any] | None = None,
    ensemble_extra_kw: dict[str, Any] | None = None,
    run_ensemble_spike: bool = True,
    exog_future_val: pd.DataFrame | None = None,
    exog_future_test: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Mismo protocolo que ``hybrid_train_val_test_backtests``, pero:

    - Horizonte de evaluación ``forecast_days`` (p. ej. **90 ≈ 3 meses**), acotado por la
      longitud de val/test.
    - Tras cada backtest híbrido, aplica ``extend_hybrid_backtest_with_ensemble_spike`` para
      alinear **SARIMAX**, **híbrido** y **ensemble+spike** sobre las mismas fechas.

    ``exog_future_val`` / ``exog_future_test``: si se proveen, saltan B1 completamente
    (reutilizan exógenas pre-calculadas de §1). Ahorra ~50% del tiempo de cómputo.

    El ensemble se reentrena con el mismo ``train`` implícito que el híbrido
    (``date <= cutoff``), coherente con el TVT.
    """
    hkw: dict[str, Any] = dict(hybrid_train_kw or {})
    ekw: dict[str, Any] = dict(ensemble_extra_kw or {})

    spl = split_train_val_test(
        df,
        val_days=val_days,
        test_days=test_days,
        min_train_days=min_train_days,
    )

    h_val = min(int(forecast_days), len(spl.val))
    h_test = min(int(forecast_days), len(spl.test))

    merged_b1: dict[str, Any] = {
        "method": exog_method,
        "univariate_structure": univariate_structure,
        **(exog_block1_kw or {}),
    }

    from hybrid_direct_30d import temporal_backtest_at_cutoff

    bt_val = temporal_backtest_at_cutoff(
        df,
        spl.date_train_end,
        target=target,
        forecast_days=h_val,
        train_df=spl.train,
        exog_block1_kw=merged_b1,
        exog_future_precomputed=exog_future_val,
        **hkw,
    )
    train_val = (
        pd.concat([spl.train, spl.val], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    bt_test = temporal_backtest_at_cutoff(
        df,
        spl.date_val_end,
        target=target,
        forecast_days=h_test,
        train_df=train_val,
        exog_block1_kw=merged_b1,
        exog_future_precomputed=exog_future_test,
        **hkw,
    )

    ext_kw: dict[str, Any] = {
        "exog_block1_kw": merged_b1,
        "origin_stride": int(hkw.get("origin_stride", 14)),
        "min_train_size": int(hkw.get("min_train_size", 500)),
    }
    for _sar_k in (
        "sarimax_order",
        "sarimax_seasonal_order",
        "sarimax_exog_cols",
        "sarimax_standardize_exog",
    ):
        if _sar_k in hkw:
            ext_kw[_sar_k] = hkw[_sar_k]
    ext_kw.update(ekw)
    # ``ensemble_extra_kw`` puede incluir p. ej. ``use_garch_vol_spike=True`` y ``spike_vol_prob_boost=0.08``.

    if run_ensemble_spike:
        from hybrid_ensemble_spike import extend_hybrid_backtest_with_ensemble_spike

        cmp_val, m_val = extend_hybrid_backtest_with_ensemble_spike(
            df, bt_val, target=target, **ext_kw
        )
        cmp_test, m_test = extend_hybrid_backtest_with_ensemble_spike(
            df, bt_test, target=target, **ext_kw
        )
    else:
        cmp_val = bt_val["comparison_df"].copy()
        cmp_test = bt_test["comparison_df"].copy()
        cmp_val["ensemble_pred"] = float("nan")
        cmp_test["ensemble_pred"] = float("nan")
        m_val = {
            "mae_hybrid_final": float(bt_val.get("mae_final", float("nan"))),
            "rmse_hybrid_final": float(bt_val.get("rmse_final", float("nan"))),
            "mae_hybrid_sarimax": float("nan"),
            "rmse_hybrid_sarimax": float("nan"),
            "mae_ensemble_spike": float("nan"),
            "rmse_ensemble_spike": float("nan"),
        }
        m_test = {
            "mae_hybrid_final": float(bt_test.get("mae_final", float("nan"))),
            "rmse_hybrid_final": float(bt_test.get("rmse_final", float("nan"))),
            "mae_hybrid_sarimax": float("nan"),
            "rmse_hybrid_sarimax": float("nan"),
            "mae_ensemble_spike": float("nan"),
            "rmse_ensemble_spike": float("nan"),
        }

    return {
        "split": spl,
        "forecast_days_val": h_val,
        "forecast_days_test": h_test,
        "val_backtest": bt_val,
        "test_backtest": bt_test,
        "val_comparison_three": cmp_val,
        "test_comparison_three": cmp_test,
        "metrics_val_three": m_val,
        "metrics_test_three": m_test,
    }
