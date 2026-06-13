"""
Diagnóstico y experimentos **estructurales** para SARIMAX + exógenas proyectadas (Bloque 1).

Objetivo metodológico
----------------------
Separar tres fuentes de error en TVT multi-paso:

1. **Especificación / ajuste SARIMAX** (orden, estacionalidad, escala log).
2. **Proyección de exógenas** (STL + bootstrap vs trayectoria realizable en oráculo).
3. **Horizonte y régimen** (60 d de error acumulado vs ventanas cortas).

Sin fugas temporales
---------------------
- **Val:** se entrena solo con ``train``; las exógenas futuras parten del último día del train.
  El oráculo usa las **exógenas realizadas en val** (solo para acotar error de B1; no entran al fit).
- **Test:** se entrena con ``train ∪ val``; exógenas futuras desde el último día de val.
  Oráculo = **exógenas realizadas en test**. Los precios de test **nunca** entren al entrenamiento.

Uso rápido
----------
::

    python -m sarimax_exog_robustness

Genera CSV + figuras bajo ``data/processed/sarimax_robustness/`` (crear directorio si no existe).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from hybrid_direct_30d import (
    block1_build_exog_future,
    block2_sarimax_fit_forecast,
    from_log_price,
    normalize_column_names,
)
from ml_features import ensure_sarimax_calendar, sarimax_exog_columns
from ml_metrics import metrics_standard
from ts_splits import TemporalTVTSplit, split_train_val_test
from xm_config import DATA_PROCESSED_DIR, TARGET_COL_DAILY_MAX_PRICE

logger = logging.getLogger(__name__)

ExogVol = Literal["none", "residual_bootstrap"]
OracleMode = Literal["projected", "oracle"]


@dataclass
class SlicePack:
    """Realizados y horizonte efectivo para una ventana (val o test)."""

    h: int
    y_cop: np.ndarray
    dates: np.ndarray


def _assert_no_price_leakage(spl: TemporalTVTSplit, target: str) -> None:
    """Comprueba que train/val/test no se solapen en fechas ni mezclen precios."""
    tr_max = pd.Timestamp(spl.train["date"].max()).normalize()
    va_min = pd.Timestamp(spl.val["date"].min()).normalize()
    va_max = pd.Timestamp(spl.val["date"].max()).normalize()
    te_min = pd.Timestamp(spl.test["date"].min()).normalize()
    assert tr_max < va_min, (tr_max, va_min)
    assert va_max < te_min, (va_max, te_min)
    for name, a, b in (
        ("train", spl.train, spl.val),
        ("val", spl.val, spl.test),
    ):
        if a.empty or b.empty:
            continue
        assert pd.Timestamp(a["date"].max()).normalize() < pd.Timestamp(b["date"].min()).normalize()


def slice_future_actuals(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    h: int,
    target: str,
) -> SlicePack:
    """Primeros ``h`` días estrictamente después de ``cutoff``: precio real COP."""
    cutoff = pd.Timestamp(cutoff).normalize()
    d = normalize_column_names(df).sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    sub = d[(d["date"] > cutoff) & (d["date"] <= cutoff + pd.Timedelta(days=int(h)))]
    h_eff = len(sub)
    return SlicePack(
        h=h_eff,
        y_cop=sub[target].values.astype(float),
        dates=sub["date"].values,
    )


def naive_persistence_mae(y_cop: np.ndarray, last_price: float) -> dict[str, float]:
    """Baseline: repetir el último precio observado en todo el horizonte."""
    pred = np.full(len(y_cop), float(last_price), dtype=float)
    return metrics_standard(y_cop, pred)


def build_oracle_exog_future(
    train_hist: pd.DataFrame,
    future_panel: pd.DataFrame,
    *,
    h: int,
    exog_cols: list[str],
) -> pd.DataFrame:
    """
    Matriz de exógenas **realizadas** en la ventana futura (val o test), alineada al orden
    cronológico tras ``cutoff = último día de train_hist``.

    Solo para diagnóstico tipo **oráculo** (cota superior de calidad de información exógena).
    """
    tr = normalize_column_names(train_hist).sort_values("date").copy()
    tr["date"] = pd.to_datetime(tr["date"]).dt.normalize()
    last = tr["date"].max()
    fut = normalize_column_names(future_panel).sort_values("date").copy()
    fut["date"] = pd.to_datetime(fut["date"]).dt.normalize()
    sub = fut[fut["date"] > last].head(int(h)).copy()
    sub = ensure_sarimax_calendar(sub)
    cols = [c for c in exog_cols if c in sub.columns]
    if not cols:
        raise ValueError("Oráculo: ninguna columna exógena disponible en future_panel.")
    return sub[["date"] + cols].reset_index(drop=True)


def exog_subset_presets(full_cols: list[str]) -> dict[str, list[str]]:
    """
    Conjuntos de regresores para ablación (nombres deben existir en el panel).

    - **full**: lista canónica (puede haber caído ``porc_aporte_max`` por redundancia).
    - **no_enso**: quita ``enso_index`` (ENSO mal proyectado suele dañar test).
    - **no_aportes**: quita columnas de aportes hídricos.
    - **demand_cal**: demanda + Fourier mensual/semanal (mínimo físico + calendario).
    - **demand_cal_enso**: ``demand_cal`` + ``enso_index`` (ENSO estructural sin hidrología diaria).
    """
    full = list(full_cols)
    no_enso = [c for c in full if c != "enso_index"]
    no_aportes = [c for c in full if not str(c).startswith("porc_aporte")]
    cal_fourier = ("month_sin", "month_cos", "week_sin", "week_cos")
    demand_cal = [c for c in full if c in ("demanda_max_kwh", *cal_fourier)]
    demand_cal_enso = [c for c in full if c in ("demanda_max_kwh", "enso_index", *cal_fourier)]
    return {
        "full": full,
        "no_enso": no_enso,
        "no_aportes": no_aportes,
        "demand_cal": demand_cal,
        "demand_cal_enso": demand_cal_enso,
    }


def sarimax_coef_table(b2: dict[str, Any]) -> pd.DataFrame:
    """Parámetros del SARIMAX con error estándar y |t| (diagnóstico de explosividad)."""
    res = b2["sarimax_result"]
    p = pd.Series(res.params, name="coef")
    se = pd.Series(getattr(res, "bse", pd.Series(index=p.index, dtype=float)), name="bse")
    out = pd.concat([p, se], axis=1)
    out["t_abs"] = (out["coef"].abs() / out["bse"].replace(0, np.nan)).fillna(np.nan)
    out = out.loc[[i for i in out.index if i in b2["exog_cols"]]].copy()
    return out.reset_index().rename(columns={"index": "param"})


def exog_impact_proxy(b2: dict[str, Any], train_hist: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Proxy de “importancia” en COP aproximada: |β| × std(regresor en train), en escala log1p
    los β ya actúan sobre ``y_log``; esto solo ordena magnitudes relativas entre exógenas.
    """
    tr = ensure_sarimax_calendar(normalize_column_names(train_hist).sort_values("date").copy())
    cols = b2["exog_cols"]
    coef = b2["sarimax_result"].params
    rows: list[dict[str, Any]] = []
    for c in cols:
        if c not in coef.index:
            continue
        std = float(pd.to_numeric(tr[c], errors="coerce").std(ddof=0) or 0.0)
        rows.append({"exog": c, "abs_coef": float(abs(float(coef[c]))), "std_train": std, "proxy": abs(float(coef[c])) * std})
    return pd.DataFrame(rows).sort_values("proxy", ascending=False)


def cumulative_abs_error_curve(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MAE acumulado día a día: sum_{i<=h} |e_i| / h (vector longitud h)."""
    e = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    return np.cumsum(e) / np.arange(1, len(e) + 1, dtype=float)


def run_one_case(
    *,
    df: pd.DataFrame,
    spl: TemporalTVTSplit,
    target: str,
    h_req: int,
    exog_preset: str,
    exog_cols: list[str],
    vol: ExogVol,
    oracle: OracleMode,
    standardize_exog: bool,
    b1_kw: dict[str, Any],
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> dict[str, Any]:
    """Un caso: val + test, métricas + coef + curva MAE acumulada (usa h efectivo mínimo)."""
    train_val = pd.concat([spl.train, spl.val], ignore_index=True).sort_values("date").reset_index(drop=True)

    hv = int(min(h_req, len(spl.val), len(spl.test)))
    if hv < 1:
        raise ValueError("h_req demasiado pequeño vs partición.")

    last_px_val = float(pd.to_numeric(spl.train.sort_values("date").iloc[-1][target], errors="coerce"))
    last_px_test = float(pd.to_numeric(train_val.sort_values("date").iloc[-1][target], errors="coerce"))

    # --- Val ---
    pack_v = slice_future_actuals(df, spl.date_train_end, hv, target)
    if oracle == "oracle":
        ex_v = build_oracle_exog_future(spl.train, spl.val, h=hv, exog_cols=exog_cols)
    else:
        kw = {**b1_kw, "volatility_mode": vol}
        ex_v = block1_build_exog_future(spl.train, days=hv, **kw)
    b2_v = block2_sarimax_fit_forecast(
        spl.train,
        ex_v,
        target=target,
        order=order,
        seasonal_order=seasonal_order,
        exog_cols=exog_cols,
        standardize_exog=standardize_exog,
    )
    pred_v = from_log_price(np.asarray(b2_v["sarimax_pred_future_log"], dtype=float))[: pack_v.h]
    yv = pack_v.y_cop[: len(pred_v)]
    m_v = metrics_standard(yv, pred_v)
    n_v = naive_persistence_mae(yv, last_px_val)
    cum_v = cumulative_abs_error_curve(yv, pred_v)

    # --- Test ---
    pack_t = slice_future_actuals(df, spl.date_val_end, hv, target)
    if oracle == "oracle":
        ex_t = build_oracle_exog_future(train_val, spl.test, h=hv, exog_cols=exog_cols)
    else:
        kw = {**b1_kw, "volatility_mode": vol}
        ex_t = block1_build_exog_future(train_val, days=hv, **kw)
    b2_t = block2_sarimax_fit_forecast(
        train_val,
        ex_t,
        target=target,
        order=order,
        seasonal_order=seasonal_order,
        exog_cols=exog_cols,
        standardize_exog=standardize_exog,
    )
    pred_t = from_log_price(np.asarray(b2_t["sarimax_pred_future_log"], dtype=float))[: pack_t.h]
    yt = pack_t.y_cop[: len(pred_t)]
    m_t = metrics_standard(yt, pred_t)
    n_t = naive_persistence_mae(yt, last_px_test)
    cum_t = cumulative_abs_error_curve(yt, pred_t)

    coef_v = sarimax_coef_table(b2_v)

    return {
        "h_req": h_req,
        "h_eff": hv,
        "exog_preset": exog_preset,
        "volatility_mode": vol,
        "oracle": oracle,
        "standardize_exog": standardize_exog,
        "mae_val": m_v["mae"],
        "rmse_val": m_v["rmse"],
        "mape_val": m_v["mape_pct"],
        "mae_naive_val": n_v["mae"],
        "ratio_val": m_v["mae"] / max(n_v["mae"], 1e-12),
        "mae_test": m_t["mae"],
        "rmse_test": m_t["rmse"],
        "mape_test": m_t["mape_pct"],
        "mae_naive_test": n_t["mae"],
        "ratio_test": m_t["mae"] / max(n_t["mae"], 1e-12),
        "coef_table_val": coef_v,
        "cum_mae_val": cum_v,
        "cum_mae_test": cum_t,
        "b2_val": b2_v,
        "b2_test": b2_t,
        "pred_val": pred_v,
        "pred_test": pred_t,
        "y_val": yv,
        "y_test": yt,
    }


def run_experiment_grid(
    df: pd.DataFrame,
    *,
    spl: TemporalTVTSplit | None = None,
    target: str = TARGET_COL_DAILY_MAX_PRICE,
    val_days: int = 120,
    test_days: int = 60,
    min_train_days: int = 500,
    horizons: tuple[int, ...] = (7, 14, 30, 60),
    b1_kw: dict[str, Any] | None = None,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 7),
    standardize_grid: tuple[bool, ...] = (False,),
) -> pd.DataFrame:
    """
    Recorre horizontes × volatilidad × presets × oráculo × (opcional) estandarización de exógenas.

    Retorna tabla resumen (una fila por experimento).
    """
    if spl is None:
        spl = split_train_val_test(df, val_days=val_days, test_days=test_days, min_train_days=min_train_days)
    _assert_no_price_leakage(spl, target)

    tr0 = ensure_sarimax_calendar(normalize_column_names(spl.train).copy())
    full_cols = sarimax_exog_columns(tr0)
    presets = exog_subset_presets(full_cols)

    b1_base: dict[str, Any] = dict(b1_kw or {})
    b1_base.setdefault("method", "ets")
    b1_base.setdefault("univariate_structure", "stl_trend_season_error")
    b1_base.setdefault("volatility_strength", 0.55)
    b1_base.setdefault("random_state", 42)
    b1_base.setdefault("exog_scenario_replicates", 1)
    b1_base.setdefault("enso_flat_max_horizon", 0)
    b1_base.setdefault("stl_trend_decay", 0.022)

    rows: list[dict[str, Any]] = []

    for std_ex in standardize_grid:
        for h in horizons:
            for vol in ("none", "residual_bootstrap"):
                for preset_name, cols in presets.items():
                    if preset_name == "demand_cal" and len(cols) < 2:
                        continue
                    for oracle in ("projected", "oracle"):
                        tag = f"h{h}_{vol}_{preset_name}_{oracle}_std{int(std_ex)}"
                        logger.info("Caso %s", tag)
                        r = run_one_case(
                            df=df,
                            spl=spl,
                            target=target,
                            h_req=h,
                            exog_preset=preset_name,
                            exog_cols=cols,
                            vol=vol,
                            oracle=oracle,
                            standardize_exog=std_ex,
                            b1_kw=b1_base,
                            order=order,
                            seasonal_order=seasonal_order,
                        )
                        rows.append(
                            {
                                "tag": tag,
                                "h_req": r["h_req"],
                                "h_eff": r["h_eff"],
                                "volatility_mode": r["volatility_mode"],
                                "exog_preset": r["exog_preset"],
                                "oracle": r["oracle"],
                                "standardize_exog": r["standardize_exog"],
                                "mae_val": r["mae_val"],
                                "ratio_val": r["ratio_val"],
                                "mae_test": r["mae_test"],
                                "ratio_test": r["ratio_test"],
                                "mae_naive_val": r["mae_naive_val"],
                                "mae_naive_test": r["mae_naive_test"],
                                "rmse_val": r["rmse_val"],
                                "rmse_test": r["rmse_test"],
                                "mape_val": r["mape_val"],
                                "mape_test": r["mape_test"],
                            }
                        )
    return pd.DataFrame(rows)


def save_figures(
    summary: pd.DataFrame,
    out_dir: Path,
    df: pd.DataFrame,
    spl: TemporalTVTSplit,
    *,
    target: str,
    b1_kw: dict[str, Any],
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> None:
    """Guarda heatmaps, brecha oráculo vs proyectado y un ejemplo de MAE acumulado (h=30)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib no disponible; se omiten figuras.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    sub2 = summary[(summary["oracle"] == "projected") & (summary["standardize_exog"] == False)].copy()
    if not sub2.empty:
        sub2["key"] = sub2["exog_preset"] + "|" + sub2["volatility_mode"]
        pv = sub2.pivot_table(index="h_req", columns="key", values="mae_test", aggfunc="first")
        pv = pv.reindex(sorted(pv.index))
        fig, ax = plt.subplots(figsize=(10, 4))
        vals = np.ma.masked_invalid(pv.values.astype(float))
        im = ax.imshow(vals, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pv.columns)))
        ax.set_xticklabels(list(pv.columns), rotation=35, ha="right")
        ax.set_yticks(range(len(pv.index)))
        ax.set_yticklabels([str(x) for x in pv.index])
        ax.set_ylabel("Horizonte solicitado (d)")
        ax.set_title("MAE test (COP/kWh) — exógenas proyectadas")
        fig.colorbar(im, ax=ax, label="MAE")
        fig.tight_layout()
        fig.savefig(out_dir / "heatmap_mae_test_projected.png", dpi=140)
        plt.close(fig)

    merged: list[tuple[Any, ...]] = []
    for h in sorted(summary["h_req"].unique()):
        for preset in summary["exog_preset"].unique():
            for vol in summary["volatility_mode"].unique():
                for std in summary["standardize_exog"].unique():
                    a = summary[
                        (summary["h_req"] == h)
                        & (summary["exog_preset"] == preset)
                        & (summary["volatility_mode"] == vol)
                        & (summary["standardize_exog"] == std)
                    ]
                    o = a[a["oracle"] == "oracle"]["mae_test"]
                    p = a[a["oracle"] == "projected"]["mae_test"]
                    if len(o) and len(p):
                        merged.append((h, preset, vol, std, float(o.values[0] - p.values[0])))
    if merged:
        gap = pd.DataFrame(merged, columns=["h", "preset", "vol", "std", "mae_test_gap"])
        gap = gap[(gap["std"] == False) & (gap["vol"] == "residual_bootstrap")].copy()
        if not gap.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            labs = [f"h={int(row['h'])} {row['preset']}" for _, row in gap.iterrows()]
            ax.barh(labs, gap["mae_test_gap"].values)
            ax.axvline(0.0, color="k", lw=0.8)
            ax.set_xlabel("MAE_test(oracle) − MAE_test(projected)  [COP/kWh]")
            ax.set_title(
                "Gap test: oráculo vs proyectado (≥0 implica que la proyección B1 añade error OOS)"
            )
            fig.tight_layout()
            fig.savefig(out_dir / "oracle_gap_mae_test.png", dpi=140)
            plt.close(fig)

    tr0 = ensure_sarimax_calendar(normalize_column_names(spl.train).copy())
    cols_full = exog_subset_presets(sarimax_exog_columns(tr0))["full"]
    pick = run_one_case(
        df=df,
        spl=spl,
        target=target,
        h_req=30,
        exog_preset="full",
        exog_cols=cols_full,
        vol="residual_bootstrap",
        oracle="projected",
        standardize_exog=False,
        b1_kw=b1_kw,
        order=order,
        seasonal_order=seasonal_order,
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(pick["cum_mae_test"]) + 1), pick["cum_mae_test"], label="test cum MAE")
    ax.plot(np.arange(1, len(pick["cum_mae_val"]) + 1), pick["cum_mae_val"], label="val cum MAE")
    ax.set_xlabel("Día del horizonte")
    ax.set_ylabel("MAE acumulado medio")
    ax.legend()
    ax.set_title("MAE acumulado (full exog, bootstrap, h=30, proyectado)")
    fig.tight_layout()
    fig.savefig(out_dir / "cum_mae_example.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(pick["y_test"], "o-", ms=3, label="Real (test)")
    ax.plot(pick["pred_test"], "-", lw=1.5, label="SARIMAX (proyectado)")
    ax.set_title("Test: precio vs pronóstico (ancla h=30, full exog, bootstrap)")
    ax.set_xlabel("Índice día en horizonte")
    ax.set_ylabel("COP/kWh")
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "forecast_vs_real_test_anchor.png", dpi=140)
    plt.close(fig)


def run_all(
    df: pd.DataFrame | None = None,
    *,
    out_dir: Path | None = None,
) -> Path:
    """Carga datos cache, ejecuta rejilla, escribe CSV + figuras."""
    from eda_analysis import load_cache_only, load_daily_auto
    from datetime import date

    out_dir = out_dir or (DATA_PROCESSED_DIR / "sarimax_robustness")
    out_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        try:
            df = load_cache_only()
        except FileNotFoundError:
            df = load_daily_auto(date(2015, 1, 1), date(2025, 12, 31))
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    spl = split_train_val_test(df, val_days=120, test_days=60, min_train_days=500)
    summary = run_experiment_grid(df, spl=spl, target=TARGET_COL_DAILY_MAX_PRICE)
    summary.to_csv(out_dir / "experiments_summary.csv", index=False)

    # Coeficientes (caso ancla)
    tr0 = ensure_sarimax_calendar(normalize_column_names(spl.train).copy())
    cols = exog_subset_presets(sarimax_exog_columns(tr0))["full"]
    kw = {
        "method": "ets",
        "univariate_structure": "stl_trend_season_error",
        "volatility_mode": "residual_bootstrap",
        "volatility_strength": 0.55,
        "random_state": 42,
        "exog_scenario_replicates": 1,
        "enso_flat_max_horizon": 0,
        "stl_trend_decay": 0.022,
    }
    hv = int(min(30, len(spl.val), len(spl.test)))
    ex_v = block1_build_exog_future(spl.train, days=hv, **kw)
    b2_v = block2_sarimax_fit_forecast(
        spl.train,
        ex_v,
        target=TARGET_COL_DAILY_MAX_PRICE,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        exog_cols=cols,
        standardize_exog=False,
    )
    sarimax_coef_table(b2_v).to_csv(out_dir / "coef_val_anchor_h30_full_bootstrap.csv", index=False)

    kw_plot = {
        "method": "ets",
        "univariate_structure": "stl_trend_season_error",
        "volatility_mode": "residual_bootstrap",
        "volatility_strength": 0.55,
        "random_state": 42,
        "exog_scenario_replicates": 1,
        "enso_flat_max_horizon": 0,
        "stl_trend_decay": 0.022,
    }
    save_figures(
        summary,
        out_dir,
        df,
        spl,
        target=TARGET_COL_DAILY_MAX_PRICE,
        b1_kw=kw_plot,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
    )

    # Impacto proxy (train)
    imp = exog_impact_proxy(b2_v, spl.train, TARGET_COL_DAILY_MAX_PRICE)
    imp.to_csv(out_dir / "exog_impact_proxy_train.csv", index=False)

    logger.info("Resultados en %s", out_dir)
    return out_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = run_all()
    print("Listo:", p)
