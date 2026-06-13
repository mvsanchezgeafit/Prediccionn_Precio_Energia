"""
Pipeline híbrido **direct multi-step** 30 días: SARIMAX + LightGBM en residuos (log), sin LSTM ni recursividad.

- **Escala log:** ``y_log = log(precio + 1)`` para SARIMAX, features de rezagos y objetivos LGBM; salida en COP/kWh vía ``expm1``.
- **Residuos (log):** ``resid = y_log - sarimax_pred_log``; se **centran** por horizonte (media muestral) antes de LGBM; al predecir se suma la media.
- **LGBM:** regularización fuerte (``num_leaves=20``, ``max_depth=5``, ``lr=0.03``).
- **Punto final (log):** ``final_log = sarimax + w * resid`` con ``w=1`` por defecto (antes ``w=0.5``).
- **Cuantiles (log):** ``q*_log = sarimax + resid_q`` (residuo cuantílico recentrado); ordenación monótona; ``expm1`` al final.

**Exógenas futuras (bloque 1):** por defecto **STL** + ``volatility_mode=residual_bootstrap``,
réplicas con agregación configurable (``scenario_aggregation``: mean, percentile_75, etc.),

Notebook: ``notebooks/07_hybrid_direct_lgbm_30d.ipynb``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

ScenarioAggregation = Literal["median", "mean", "percentile_75", "single_path_seeded"]

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from exog_univariate_forecast import (
    DEFAULT_VOL_STRENGTH_SCALE_BY_COL,
    ExogMethod,
    ExogUnivariateStructure,
    ExogVolatilityMode,
    build_exog_forecast_univariate,
)
from ml_features import ensure_sarimax_calendar, sarimax_exog_columns
from xm_config import TARGET_COL_DAILY_MAX_PRICE
from xm_daily_target import forecast_next_n_days

from ml_metrics import lgbm_quantile_oos_metrics

logger = logging.getLogger(__name__)

SARIMAX_FC_LOG_COL = "sarimax_fc_log"
# ``expm1(10)`` ≈ 22k COP/kWh; evita overflow si LGBM extrapola en log.
LOG_PRICE_CLIP = 10.0
# Corrección híbrida no debe alejarse más de ~2.7× del SARIMAX en multiplicativo.
MAX_HYBRID_LOG_OFFSET = 1.0


def clip_log_price(y_log: np.ndarray | pd.Series | float) -> np.ndarray:
    return np.clip(np.asarray(y_log, dtype=float), -LOG_PRICE_CLIP, LOG_PRICE_CLIP)


def bound_hybrid_final_log(s_h: float, final_log: float, *, direct: bool) -> float:
    """Acota ``final_log`` respecto al SARIMAX antes de ``expm1``."""
    del direct  # mismo anclaje en residual y direct_log
    s_c = float(clip_log_price(s_h))
    fl = float(np.clip(final_log, s_c - MAX_HYBRID_LOG_OFFSET, s_c + MAX_HYBRID_LOG_OFFSET))
    return float(clip_log_price(fl))

def pop_sarimax_block2_kwargs(train_kw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Extrae kwargs para ``block2_sarimax_fit_forecast`` / ``precompute_sarimax_h_step_from_origins``
    desde ``hybrid_train_kw`` (p. ej. ``sarimax_exog_cols`` alineado con el SARIMAX del §1).

    Devuelve ``(block2_kwargs, train_kw_restante)`` con claves
    ``order``, ``seasonal_order``, ``exog_cols``, ``standardize_exog``.
    """
    rest = dict(train_kw)
    out: dict[str, Any] = {
        "order": (1, 1, 1),
        "seasonal_order": (1, 0, 1, 7),
        "exog_cols": None,
        "standardize_exog": False,
    }
    if "sarimax_order" in rest:
        out["order"] = rest.pop("sarimax_order")
    if "sarimax_seasonal_order" in rest:
        out["seasonal_order"] = rest.pop("sarimax_seasonal_order")
    if "sarimax_exog_cols" in rest:
        out["exog_cols"] = rest.pop("sarimax_exog_cols")
    if "sarimax_standardize_exog" in rest:
        out["standardize_exog"] = bool(rest.pop("sarimax_standardize_exog"))
    return out, rest


def to_log_price(price: np.ndarray | pd.Series | float) -> np.ndarray:
    """``log(precio + 1)`` (estable para precios ≥ 0)."""
    x = np.asarray(price, dtype=float)
    return np.log(np.maximum(x, 0.0) + 1.0)


def from_log_price(y_log: np.ndarray | pd.Series | float) -> np.ndarray:
    """Vuelve a COP/kWh."""
    return np.expm1(clip_log_price(y_log))


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    if "porc_vol_util" in df.columns and "porc_vol_util_diario" not in df.columns:
        return df.rename(columns={"porc_vol_util": "porc_vol_util_diario"})
    return df.copy()


# --- BLOQUE 1 ---


def _aggregate_exog_scenarios(
    parts: list[pd.DataFrame],
    *,
    aggregation: ScenarioAggregation,
) -> pd.DataFrame:
    """Combina réplicas bootstrap día a día (sin mediana por defecto en experimentos B1)."""
    ref = parts[0]
    if aggregation == "single_path_seeded" or len(parts) == 1:
        out = ref.copy()
        out.attrs = ref.attrs.copy()
        out.attrs["exog_scenario_aggregation"] = "single_path_seeded"
        return out
    scenario = ref.copy()
    num_cols = [c for c in ref.columns if c != "date"]
    for c in num_cols:
        if not all(c in p.columns for p in parts):
            continue
        stacked = np.stack([pd.to_numeric(p[c], errors="coerce").to_numpy(float) for p in parts], axis=0)
        if aggregation == "mean":
            scenario[c] = np.nanmean(stacked, axis=0)
        elif aggregation == "percentile_75":
            scenario[c] = np.nanpercentile(stacked, 75.0, axis=0)
        elif aggregation == "median":
            scenario[c] = np.nanmedian(stacked, axis=0)
        else:
            raise ValueError(f"scenario_aggregation desconocido: {aggregation}")
    scenario.attrs = ref.attrs.copy()
    scenario.attrs["exog_scenario_aggregation"] = aggregation
    return scenario


def block1_build_exog_future(
    df_hist: pd.DataFrame,
    *,
    days: int = 30,
    method: ExogMethod = "ets",
    univariate_structure: ExogUnivariateStructure = "stl_trend_season_error",
    volatility_mode: ExogVolatilityMode = "residual_bootstrap",
    volatility_strength: float = 1.0,
    volatility_strength_scale_by_col: dict[str, float] | None = None,
    volatility_pool_days: int = 252,
    random_state: int | None = 42,
    exog_scenario_replicates: int = 1,
    scenario_aggregation: ScenarioAggregation = "mean",
    enso_flat_max_horizon: int = 30,
    stl_seasonal_period: dict[str, int] | None = None,
    stl_residual_in_forecast: dict[str, bool] | None = None,
    stl_trend_decay: float = 0.0,
    stl_trend_decay_by_col: dict[str, float] | None = None,
    stl_trend_revert_mean_by_col: dict[str, bool] | None = None,
    stl_trend_fit_window: int = 120,
    stl_trend_fit_window_by_col: dict[str, int] | None = None,
    log_exog_diagnostics: bool = False,
    use_realistic_exog_train: bool = False,
) -> pd.DataFrame:
    """
    Construye ``exog_future`` con STL + variabilidad opcional y réplicas agregadas.

    ``exog_scenario_replicates`` > 1: varias trayectorias (semillas distintas) agregadas con
    ``scenario_aggregation`` (``mean``, ``percentile_75``, ``single_path_seeded``, o ``median`` legacy).

    ``use_realistic_exog_train``: reservado para alinear entrenamiento LGBM con producción
    (sin implementación aún; solo registro en log).
    """
    if use_realistic_exog_train:
        logger.info(
            "use_realistic_exog_train=True: hook reservado (SARIMAX+LGBM con exógenas proyectadas "
            "en orígenes históricos); sin efecto en esta versión."
        )

    df_hist = df_hist.sort_values("date").copy()
    df_hist["date"] = pd.to_datetime(df_hist["date"]).dt.normalize()
    last = df_hist["date"].max()
    future_dates = forecast_next_n_days(last, n=days)

    n_rep = max(1, int(exog_scenario_replicates))
    agg: ScenarioAggregation = str(scenario_aggregation)  # type: ignore[assignment]
    if agg == "single_path_seeded":
        n_rep = 1
    base_seed = int(random_state) if random_state is not None else 0
    vol_scale = volatility_strength_scale_by_col
    if vol_scale is None:
        vol_scale = DEFAULT_VOL_STRENGTH_SCALE_BY_COL

    def _one_scenario(seed: int | None) -> pd.DataFrame:
        return build_exog_forecast_univariate(
            df_hist,
            future_dates,
            method=method,
            univariate_structure=univariate_structure,
            stl_seasonal_period=stl_seasonal_period,
            stl_residual_in_forecast=stl_residual_in_forecast,
            stl_trend_decay=float(stl_trend_decay),
            stl_trend_decay_by_col=stl_trend_decay_by_col,
            stl_trend_revert_mean_by_col=stl_trend_revert_mean_by_col,
            stl_trend_fit_window=int(stl_trend_fit_window),
            stl_trend_fit_window_by_col=stl_trend_fit_window_by_col,
            volatility_mode=volatility_mode,
            volatility_strength=float(volatility_strength),
            volatility_strength_scale_by_col=vol_scale,
            volatility_pool_days=int(volatility_pool_days),
            random_state=seed,
            enso_flat_max_horizon=int(enso_flat_max_horizon),
        )

    if n_rep == 1:
        scenario = _one_scenario(base_seed if random_state is not None else None)
        scenario.attrs["exog_scenario_aggregation"] = agg
    else:
        parts: list[pd.DataFrame] = []
        for k in range(n_rep):
            sk = None if random_state is None else base_seed + k * 100_003
            parts.append(_one_scenario(sk))
        scenario = _aggregate_exog_scenarios(parts, aggregation=agg)
        scenario.attrs["exog_scenario_replicates"] = n_rep
        scenario.attrs["exog_scenario_seeds"] = [
            None if random_state is None else base_seed + k * 100_003 for k in range(n_rep)
        ]

    out = ensure_sarimax_calendar(scenario)
    if log_exog_diagnostics:
        from exog_univariate_forecast import summarize_exog_forecast_df

        diag = summarize_exog_forecast_df(out)
        logger.info("Exógenas futuras: NaN totales=%s", diag["nan_total"])
        for c, st in diag["per_column"].items():
            logger.info("  %s: std=%.6g n_nan=%s min=%.4g max=%.4g", c, st["std"], st["n_nan"], st.get("min", float("nan")), st.get("max", float("nan")))
        if diag["flat_like_columns"]:
            logger.warning("Columnas con std casi nula (trayectoria plana): %s", diag["flat_like_columns"])
    return out


def _fit_sarimax_one(
    y: pd.Series,
    exog: pd.DataFrame,
    *,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 7),
):
    last_res = None
    for seas in (seasonal_order, (0, 0, 0, 0)):
        try:
            model = SARIMAX(
                y,
                exog=exog,
                order=order,
                seasonal_order=seas,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            last_res = model.fit(disp=False, maxiter=300)
            break
        except Exception as e:
            logger.debug("SARIMAX seasonal=%s: %s", seas, e)
            last_res = None
    if last_res is None:
        raise RuntimeError("SARIMAX no convergió.")
    return last_res


# --- BLOQUE 2: SARIMAX en log(precio+1) ---


def block2_sarimax_fit_forecast(
    df_hist: pd.DataFrame,
    exog_future: pd.DataFrame,
    *,
    target: str = TARGET_COL_DAILY_MAX_PRICE,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 7),
    exog_cols: list[str] | None = None,
    standardize_exog: bool = False,
) -> dict[str, Any]:
    """
    SARIMAX sobre ``log(precio+1)`` con exógenas.

    ``exog_cols``: si se pasa, usa solo ese subconjunto (debe existir en ``df_hist`` / futuro).
    Útil para ablaciones (p. ej. sin ENSO, solo demanda+calendario).

    ``standardize_exog``: z-score de las exógenas **con media y desvío del train**; la misma
    transformación se aplica al bloque futuro. Mejora estabilidad numérica cuando una covariable
    domina la escala (p. ej. demanda en kWh vs proporciones en [0,1]); los coeficientes pasan a
    interpretarse en “σ de train” por regresor.
    """
    tr = ensure_sarimax_calendar(df_hist.sort_values("date").copy())
    tr["date"] = pd.to_datetime(tr["date"]).dt.normalize()
    if exog_cols is None:
        exog_cols_use = sarimax_exog_columns(tr)
    else:
        exog_cols_use = [c for c in exog_cols if c in tr.columns]
    if not exog_cols_use:
        raise ValueError("No hay exógenas para SARIMAX (lista vacía o columnas ausentes).")

    y_price = pd.to_numeric(tr[target], errors="coerce").astype(float)
    y_log = pd.Series(to_log_price(y_price.values), index=y_price.index)
    X = tr[exog_cols_use].astype(float)
    med = X.median()
    X = X.fillna(med)

    exog_mu: pd.Series | None = None
    exog_sd: pd.Series | None = None
    if standardize_exog:
        exog_mu = X.mean(axis=0)
        exog_sd = X.std(axis=0, ddof=0).replace(0.0, 1.0)
        X = (X - exog_mu) / exog_sd

    res = _fit_sarimax_one(y_log, X, order=order, seasonal_order=seasonal_order)
    fv = np.asarray(res.fittedvalues, dtype=float)
    yv = y_log.values.astype(float)
    n = min(len(yv), len(fv))
    r = yv[:n] - fv[:n]

    Xf = ensure_sarimax_calendar(exog_future.copy())
    Xf = Xf[exog_cols_use].astype(float).fillna(med)
    if standardize_exog and exog_mu is not None and exog_sd is not None:
        Xf = (Xf - exog_mu) / exog_sd
    fc = res.get_forecast(steps=len(exog_future), exog=Xf)
    pred_log = np.asarray(fc.predicted_mean, dtype=float)

    out: dict[str, Any] = {
        "sarimax_result": res,
        "exog_cols": exog_cols_use,
        "exog_median_train": med,
        "sarimax_pred_future_log": pred_log,
        "fittedvalues_log": fv,
        "residuals_in_sample_log": r,
        "y_train_log": yv,
        "target_scale": "log1p",
        "standardize_exog": bool(standardize_exog),
    }
    if standardize_exog and exog_mu is not None:
        out["exog_train_mean"] = exog_mu
        out["exog_train_std"] = exog_sd
    return out


# --- BLOQUE 3: features desde log-precio (rezagos coherentes con el objetivo) ---


def block3_build_direct_features(
    df: pd.DataFrame,
    *,
    target: str = TARGET_COL_DAILY_MAX_PRICE,
    use_log_price_lags: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    d = df.sort_values("date").reset_index(drop=True).copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    price = pd.to_numeric(d[target], errors="coerce")
    if use_log_price_lags:
        y_feat = pd.Series(to_log_price(price.values), index=d.index)
    else:
        y_feat = price

    for lag in (1, 7, 14, 30):
        d[f"lag_{lag}"] = y_feat.shift(lag)
    d["roll_mean_7"] = y_feat.shift(1).rolling(7, min_periods=1).mean()
    d["roll_mean_14"] = y_feat.shift(1).rolling(14, min_periods=2).mean()

    dt = d["date"]
    d["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12.0)
    d["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
    d["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)

    exog_numeric = [
        c
        for c in (
            "demanda_max_kwh",
            "porc_vol_util_diario",
            "vol_util_energia_sistema",
            "enso_index",
            "porc_aporte_value",
            "porc_aporte_max",
        )
        if c in d.columns
    ]
    for c in exog_numeric:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    exog_derived: list[str] = []
    for col in exog_numeric:
        s = d[col]
        for k in (1, 7):
            name = f"{col}_lag_{k}"
            d[name] = s.shift(k)
            exog_derived.append(name)

    feature_names = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        "roll_mean_7",
        "roll_mean_14",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
    ]
    feature_names.extend([c for c in exog_numeric if c in d.columns])
    feature_names.extend([c for c in exog_derived if c in d.columns])
    if "porc_vol_util_diario" in feature_names and "vol_util_energia_sistema" in feature_names:
        feature_names = [c for c in feature_names if c != "vol_util_energia_sistema"]

    return d, feature_names


def precompute_sarimax_h_step_from_origins(
    df: pd.DataFrame,
    *,
    target: str,
    exog_cols: list[str],
    max_horizon: int = 30,
    min_train_size: int = 500,
    origin_stride: int = 14,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 7),
    standardize_exog: bool = False,
) -> dict[int, np.ndarray]:
    tr = ensure_sarimax_calendar(df.sort_values("date").copy()).reset_index(drop=True)
    tr["date"] = pd.to_datetime(tr["date"]).dt.normalize()
    n = len(tr)
    y_price = pd.to_numeric(tr[target], errors="coerce").astype(float)
    y_all = pd.Series(to_log_price(y_price.values), index=y_price.index)
    X_all = tr[exog_cols].astype(float)
    med_global = X_all.median()
    X_all = X_all.fillna(med_global)

    out: dict[int, np.ndarray] = {}
    for t in range(min_train_size - 1, n - max_horizon, origin_stride):
        y_tr = y_all.iloc[: t + 1]
        X_tr = X_all.iloc[: t + 1].copy()
        X_fut = X_all.iloc[t + 1 : t + 1 + max_horizon].copy()
        if len(X_fut) < max_horizon:
            break
        if standardize_exog:
            mu = X_tr.mean(axis=0)
            sig = X_tr.std(axis=0).replace(0.0, 1.0)
            X_tr = (X_tr - mu) / sig
            X_fut = (X_fut - mu) / sig
        try:
            res = _fit_sarimax_one(y_tr, X_tr, order=order, seasonal_order=seasonal_order)
            fc = res.get_forecast(steps=max_horizon, exog=X_fut)
            out[t] = np.asarray(fc.predicted_mean, dtype=float).ravel()
        except Exception as e:
            logger.warning("Origen t=%d omitido: %s", t, e)
    return out


def _train_lgbm_point(X: np.ndarray, y: np.ndarray) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=500,
        num_leaves=20,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.85,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=0.3,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def _train_lgbm_quantile(X: np.ndarray, y: np.ndarray, alpha: float) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=float(alpha),
        n_estimators=500,
        num_leaves=20,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.85,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=0.3,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def _order_quantiles_monotone(q10: float, q50: float, q90: float) -> tuple[float, float, float]:
    a = sorted([q10, q50, q90])
    return float(a[0]), float(a[1]), float(a[2])


def _split_conformal_cqr_delta(scores: np.ndarray, *, miscoverage: float) -> float:
    """
    Ajuste CQR (Romano et al.): ensancha [q_lo, q_hi] en ±δ con δ = cuantil finito
    de E_i = max(q_lo(X_i) - Y_i, Y_i - q_hi(X_i)) sobre el conjunto de calibración.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = int(s.size)
    if n == 0:
        return 0.0
    alpha = float(miscoverage)
    if not (0.0 < alpha < 1.0):
        alpha = 0.2
    idx = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    idx = max(0, min(idx, n - 1))
    return float(s[idx])


def _calibration_origin_split(
    origins: list[int],
    *,
    cal_fraction: float,
) -> tuple[set[int], set[int]]:
    """Últimos orígenes (tiempo) = calibración CQR; el resto = entrenamiento."""
    u = sorted(set(origins))
    n_o = len(u)
    if n_o < 2:
        return set(u), set()
    n_cal = max(1, int(round(float(cal_fraction) * n_o)))
    n_cal = min(n_cal, n_o - 1)
    cal = set(u[-n_cal:])
    train = set(u[:-n_cal])
    return train, cal


@dataclass
class DirectHybridModels30d:
    feature_names: list[str]
    horizon: int = 30
    models_residual_mse: dict[int, lgb.LGBMRegressor] = field(default_factory=dict)
    models_residual_q: dict[tuple[int, float], lgb.LGBMRegressor] = field(default_factory=dict)
    residual_mean_by_h: dict[int, float] = field(default_factory=dict)
    origins_used: dict[int, np.ndarray] = field(default_factory=dict)
    cqr_delta_log_by_h: dict[int, float] = field(default_factory=dict)
    residual_blend_weight: float = 1.0
    target_mode: str = "residual"


def block456_train_direct_hybrid(
    df: pd.DataFrame,
    *,
    target: str = TARGET_COL_DAILY_MAX_PRICE,
    max_horizon: int = 30,
    min_train_size: int = 500,
    origin_stride: int = 14,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 7),
    sarimax_exog_cols: list[str] | None = None,
    sarimax_standardize_exog: bool = False,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    residual_blend_weight: float = 1.0,
    use_cqr: bool = True,
    cqr_cal_origin_fraction: float = 0.15,
    cqr_miscoverage: float = 0.2,
    target_mode: str = "residual",
    min_train_rows_after_split: int = 80,
    min_cal_rows_for_cqr: int = 8,
) -> DirectHybridModels30d:
    """
    ``target_mode``:
    - ``residual`` (default): LGBM sobre residuo ``y_log - sarimax_fc`` (centrado por h).
    - ``direct_log``: LGBM predice ``y_log`` con feature extra ``sarimax_fc_log`` (misma escala).

    ``residual_blend_weight`` w: punto final ``final_log = sarimax + w * resid`` (antes w=0.5).

    CQR: últimos orígenes reservados para calibrar δ_h en log y ensanchar P10/P90.
    """
    if target_mode not in ("residual", "direct_log"):
        raise ValueError("target_mode debe ser 'residual' o 'direct_log'.")
    tr = normalize_column_names(df)
    tr = ensure_sarimax_calendar(tr.sort_values("date").copy()).reset_index(drop=True)
    exog_cols = (
        [c for c in (sarimax_exog_cols or []) if c in tr.columns]
        if sarimax_exog_cols is not None
        else sarimax_exog_columns(tr)
    )
    if not exog_cols:
        raise ValueError("sarimax_exog_columns vacío.")

    feat_df, feat_names = block3_build_direct_features(tr, target=target, use_log_price_lags=True)
    y_log = to_log_price(pd.to_numeric(feat_df[target], errors="coerce").values)

    sarimax_by_t = precompute_sarimax_h_step_from_origins(
        tr,
        target=target,
        exog_cols=exog_cols,
        max_horizon=max_horizon,
        min_train_size=min_train_size,
        origin_stride=origin_stride,
        order=order,
        seasonal_order=seasonal_order,
        standardize_exog=bool(sarimax_standardize_exog),
    )
    if not sarimax_by_t:
        raise RuntimeError(
            "No se pudo precomputar ningún origen SARIMAX; baje min_train_size u origin_stride."
        )

    out_feat_names = (
        list(feat_names) + [SARIMAX_FC_LOG_COL] if target_mode == "direct_log" else list(feat_names)
    )
    bundle = DirectHybridModels30d(
        feature_names=out_feat_names,
        horizon=max_horizon,
        origins_used=sarimax_by_t,
        residual_blend_weight=float(residual_blend_weight),
        target_mode=target_mode,
    )

    for h in range(1, max_horizon + 1):
        origin_ts: list[int] = []
        X_base: list[np.ndarray] = []
        s_logs: list[float] = []
        y_log_true: list[float] = []
        y_res_raw: list[float] = []

        for t, fc_vec in sarimax_by_t.items():
            if t + h >= len(y_log):
                continue
            if np.isnan(y_log[t + h]):
                continue
            row = feat_df.iloc[t]
            if row[feat_names].isna().any():
                continue
            s_h = float(fc_vec[h - 1])
            y_lt = float(y_log[t + h])
            origin_ts.append(int(t))
            X_base.append(row[feat_names].to_numpy(dtype=float))
            s_logs.append(s_h)
            y_log_true.append(y_lt)
            y_res_raw.append(float(y_lt - s_h))

        n_all = len(origin_ts)
        if n_all < min_train_rows_after_split:
            logger.warning("Horizonte h=%d: pocas muestras (%d).", h, n_all)
        if not origin_ts:
            raise RuntimeError(f"No hay filas de entrenamiento para h={h}")

        train_origins, cal_origins = _calibration_origin_split(
            origin_ts, cal_fraction=cqr_cal_origin_fraction
        )
        idx_train = [i for i, t in enumerate(origin_ts) if t in train_origins]
        idx_cal = [i for i, t in enumerate(origin_ts) if t in cal_origins]

        if len(idx_train) < min_train_rows_after_split or not idx_cal:
            idx_train = list(range(n_all))
            idx_cal = []
            if use_cqr:
                logger.debug("h=%d: CQR desactivado (split insuficiente).", h)

        Xb = np.vstack(X_base)
        if target_mode == "direct_log":
            s_arr = np.asarray(s_logs, dtype=float).reshape(-1, 1)
            X_full = np.hstack([Xb, s_arr])
        else:
            X_full = Xb

        if target_mode == "residual":
            y_raw = np.asarray(y_res_raw, dtype=float)
        else:
            y_raw = np.asarray(y_log_true, dtype=float)

        mu_h = float(np.mean(y_raw[np.array(idx_train, dtype=int)]))
        bundle.residual_mean_by_h[h] = mu_h
        y_centered = y_raw - mu_h

        X_tr = X_full[np.array(idx_train, dtype=int)]
        y_tr = y_centered[np.array(idx_train, dtype=int)]
        bundle.models_residual_mse[h] = _train_lgbm_point(X_tr, y_tr)
        for alpha in quantiles:
            bundle.models_residual_q[(h, alpha)] = _train_lgbm_quantile(X_tr, y_tr, float(alpha))

        delta_h = 0.0
        if use_cqr and len(idx_cal) >= min_cal_rows_for_cqr:
            scores: list[float] = []
            for i in idx_cal:
                xi = X_full[i : i + 1]
                s_hi = s_logs[i]
                y_i = y_log_true[i]
                if target_mode == "residual":
                    r_c10 = float(bundle.models_residual_q[(h, 0.1)].predict(xi)[0])
                    r_c50 = float(bundle.models_residual_q[(h, 0.5)].predict(xi)[0])
                    r_c90 = float(bundle.models_residual_q[(h, 0.9)].predict(xi)[0])
                    q10l = s_hi + r_c10 + mu_h
                    q50l = s_hi + r_c50 + mu_h
                    q90l = s_hi + r_c90 + mu_h
                else:
                    q10l = float(mu_h + bundle.models_residual_q[(h, 0.1)].predict(xi)[0])
                    q50l = float(mu_h + bundle.models_residual_q[(h, 0.5)].predict(xi)[0])
                    q90l = float(mu_h + bundle.models_residual_q[(h, 0.9)].predict(xi)[0])
                q10l, q50l, q90l = _order_quantiles_monotone(q10l, q50l, q90l)
                e_i = max(q10l - y_i, y_i - q90l)
                scores.append(float(e_i))
            delta_h = _split_conformal_cqr_delta(np.asarray(scores, dtype=float), miscoverage=cqr_miscoverage)

        bundle.cqr_delta_log_by_h[h] = delta_h

    return bundle


def block7_forecast_output_df(
    df: pd.DataFrame,
    exog_future: pd.DataFrame,
    bundle: DirectHybridModels30d,
    block2_result: dict[str, Any],
    *,
    target: str = TARGET_COL_DAILY_MAX_PRICE,
) -> pd.DataFrame:
    tr = normalize_column_names(df)
    tr = ensure_sarimax_calendar(tr.sort_values("date").copy()).reset_index(drop=True)
    feat_df, feat_names = block3_build_direct_features(tr, target=target, use_log_price_lags=True)
    x_row = feat_df.iloc[-1][feat_names].astype(float)
    if x_row.isna().any():
        x_row = x_row.fillna(x_row.median())
    X_one = x_row.to_numpy(dtype=float).reshape(1, -1)

    s_log = np.asarray(block2_result["sarimax_pred_future_log"], dtype=float).ravel()
    h_max = min(len(s_log), bundle.horizon)
    dates = pd.to_datetime(exog_future["date"]).dt.normalize().iloc[:h_max]

    sarimax_price: list[float] = []
    final_price: list[float] = []
    q10_p: list[float] = []
    q50_p: list[float] = []
    q90_p: list[float] = []

    direct = bundle.target_mode == "direct_log"

    for h in range(1, h_max + 1):
        s_h = float(s_log[h - 1])
        mu_h = bundle.residual_mean_by_h[h]

        if direct:
            X_h = np.concatenate([X_one.ravel(), np.array([s_h], dtype=float)]).reshape(1, -1)
        else:
            X_h = X_one

        rq10_c = float(bundle.models_residual_q[(h, 0.1)].predict(X_h)[0])
        rq50_c = float(bundle.models_residual_q[(h, 0.5)].predict(X_h)[0])
        rq90_c = float(bundle.models_residual_q[(h, 0.9)].predict(X_h)[0])
        if direct:
            q10l = rq10_c + mu_h
            q50l = rq50_c + mu_h
            q90l = rq90_c + mu_h
        else:
            q10l = s_h + rq10_c + mu_h
            q50l = s_h + rq50_c + mu_h
            q90l = s_h + rq90_c + mu_h
        q10l, q50l, q90l = _order_quantiles_monotone(q10l, q50l, q90l)

        d_cqr = float(bundle.cqr_delta_log_by_h.get(h, 0.0))
        if d_cqr > 0.0:
            q10l -= d_cqr
            q90l += d_cqr
            q10l, q50l, q90l = _order_quantiles_monotone(q10l, q50l, q90l)

        s_h = float(clip_log_price(s_h))
        # Punto final = P50 (más estable OOS que el head MSE; alineado con ensemble+spike).
        final_log = bound_hybrid_final_log(s_h, q50l, direct=direct)
        q10l, q50l, q90l = _order_quantiles_monotone(
            float(clip_log_price(q10l)),
            float(clip_log_price(q50l)),
            float(clip_log_price(q90l)),
        )
        sarimax_price.append(float(np.expm1(s_h)))
        final_price.append(float(np.expm1(final_log)))
        q10_p.append(float(np.expm1(q10l)))
        q50_p.append(float(np.expm1(q50l)))
        q90_p.append(float(np.expm1(q90l)))

    return pd.DataFrame(
        {
            "date": dates.values,
            "sarimax_pred": sarimax_price,
            "final_pred": final_price,
            "q10": q10_p,
            "q50": q50_p,
            "q90": q90_p,
        }
    )


def compute_validation_metrics(
    comparison_df: pd.DataFrame,
    *,
    pred_col: str = "final_pred",
    actual_col: str = "y_true",
    q10_col: str = "q10",
    q50_col: str = "q50",
    q90_col: str = "q90",
) -> dict[str, float]:
    """
    Cobertura [q10, q90], sesgo del punto, pinball P10/P50/P90, interval score (80 %).
    """
    d = comparison_df.dropna(subset=[actual_col, pred_col]).copy()
    if d.empty:
        return {
            "coverage_p10_p90": float("nan"),
            "mean_bias_pred_minus_actual": float("nan"),
            "n": 0,
        }
    y = d[actual_col].values.astype(float)
    p = d[pred_col].values.astype(float)
    bias = float(np.mean(p - y))
    if q10_col in d.columns and q90_col in d.columns:
        lo = d[q10_col].values.astype(float)
        hi = d[q90_col].values.astype(float)
        inside = float(np.mean((y >= lo) & (y <= hi)))
    else:
        inside = float("nan")
    result: dict[str, float] = {
        "mean_bias_pred_minus_actual": bias,
        "n": int(len(d)),
        "coverage_p10_p90": inside,
    }
    if (
        q10_col in d.columns
        and q50_col in d.columns
        and q90_col in d.columns
    ):
        m = lgbm_quantile_oos_metrics(
            y,
            d[q10_col].values.astype(float),
            d[q50_col].values.astype(float),
            d[q90_col].values.astype(float),
        )
        for k, v in m.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                result[str(k)] = float(v)
        result["coverage_p10_p90"] = result["coverage_p10_p90_empirical"]
    return result


def feature_importance_for_horizon(
    bundle: DirectHybridModels30d,
    horizon: int = 15,
) -> pd.DataFrame:
    m = bundle.models_residual_mse.get(horizon)
    if m is None:
        raise KeyError(f"No hay modelo MSE para h={horizon}")
    imp = np.asarray(m.feature_importances_, dtype=float)
    return (
        pd.DataFrame({"feature": bundle.feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# Nombres que solo aplican a ``block1_build_exog_future``; no deben llegar a ``block456_train_direct_hybrid``.
_BLOCK1_KW_KEYS = frozenset(
    {
        "exog_block1_kw",
        "method",
        "exog_method",
        "univariate_structure",
        "volatility_mode",
        "volatility_strength",
        "volatility_strength_scale_by_col",
        "volatility_pool_days",
        "random_state",
        "exog_random_state",
        "exog_scenario_replicates",
        "scenario_aggregation",
        "enso_flat_max_horizon",
        "exog_enso_flat_max_horizon",
        "stl_seasonal_period",
        "exog_stl_seasonal_period",
        "stl_residual_in_forecast",
        "exog_stl_residual_in_forecast",
        "stl_trend_decay",
        "stl_trend_decay_by_col",
        "stl_trend_revert_mean_by_col",
        "stl_trend_fit_window",
        "stl_trend_fit_window_by_col",
        "log_exog_diagnostics",
        "use_realistic_exog_train",
    }
)


def _partition_block1_vs_lgbm_train_kw(train_kw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Separa kwargs del bloque 1 (exógenas) de los del LGBM. Evita ``TypeError`` si el usuario
    mezcla claves en ``HYBRID_KW`` o si un kernel tenía una firma antigua de ``temporal_backtest_at_cutoff``.
    """
    rest = dict(train_kw)
    b1: dict[str, Any] = {}
    for k in list(rest.keys()):
        if k not in _BLOCK1_KW_KEYS:
            continue
        v = rest.pop(k)
        if k == "exog_block1_kw" and isinstance(v, dict):
            b1 = {**v, **b1}
        elif k == "exog_method":
            b1["method"] = v
        elif k == "exog_random_state":
            b1["random_state"] = v
        elif k == "exog_enso_flat_max_horizon":
            b1["enso_flat_max_horizon"] = v
        elif k == "exog_stl_seasonal_period":
            b1["stl_seasonal_period"] = v
        elif k == "exog_stl_residual_in_forecast":
            b1["stl_residual_in_forecast"] = v
        elif k != "exog_block1_kw":
            b1[k] = v
    return b1, rest


def temporal_backtest_at_cutoff(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp | str,
    *,
    target: str = TARGET_COL_DAILY_MAX_PRICE,
    forecast_days: int = 30,
    train_df: pd.DataFrame | None = None,
    exog_block1_kw: dict[str, Any] | None = None,
    exog_future_precomputed: pd.DataFrame | None = None,
    **train_kw: Any,
) -> dict[str, Any]:
    """
    Backtest con **corte explícito**: entrena con ``train_df`` si se pasa; si no, con todo
    ``date <= cutoff``. Evalúa pronóstico en ``(cutoff, cutoff + forecast_days]`` contra ``df``.

    ``exog_block1_kw``: argumentos extra para :func:`block1_build_exog_future` (p. ej.
    ``univariate_structure``, ``volatility_mode``, ``exog_scenario_replicates``).

    ``exog_future_precomputed``: si se provee, se usa directamente como exógenas futuras
    saltando completamente B1 (STL + bootstrap). Debe tener columna ``date`` y las exógenas.

    Cualquier clave de bloque 1 que venga por error en ``**train_kw`` se redirige a ``block1``.
    """
    b1_from_train, lgbm_kw = _partition_block1_vs_lgbm_train_kw(train_kw)
    b2_fit_kw, lgbm_kw = pop_sarimax_block2_kwargs(lgbm_kw)
    merged_exog_kw = {**b1_from_train, **(exog_block1_kw or {})}

    d = normalize_column_names(df).sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    cutoff = pd.Timestamp(cutoff_date).normalize()

    if train_df is None:
        train = d[d["date"] <= cutoff].copy()
    else:
        train = normalize_column_names(train_df).sort_values("date").copy()
        train["date"] = pd.to_datetime(train["date"]).dt.normalize()
        tmax = train["date"].max()
        if tmax != cutoff:
            logger.warning(
                "Última fecha de train_df (%s) != cutoff (%s); se usa el train_df proporcionado.",
                tmax,
                cutoff,
            )

    actual = d[
        (d["date"] > cutoff)
        & (d["date"] <= cutoff + pd.Timedelta(days=forecast_days))
    ]

    if exog_future_precomputed is not None:
        ex_f = exog_future_precomputed.copy()
        logger.info("B1 skipped: using pre-computed exog_future (%d rows)", len(ex_f))
    else:
        ex_f = block1_build_exog_future(train, days=forecast_days, **merged_exog_kw)
    b2 = block2_sarimax_fit_forecast(
        train,
        ex_f,
        target=target,
        order=b2_fit_kw["order"],
        seasonal_order=b2_fit_kw["seasonal_order"],
        exog_cols=b2_fit_kw["exog_cols"],
        standardize_exog=b2_fit_kw["standardize_exog"],
    )
    bundle = block456_train_direct_hybrid(
        train,
        target=target,
        max_horizon=forecast_days,
        order=b2_fit_kw["order"],
        seasonal_order=b2_fit_kw["seasonal_order"],
        sarimax_exog_cols=b2_fit_kw["exog_cols"],
        sarimax_standardize_exog=b2_fit_kw["standardize_exog"],
        **lgbm_kw,
    )
    fc = block7_forecast_output_df(train, ex_f, bundle, b2, target=target)

    cmp = fc.merge(actual[["date", target]], on="date", how="inner")
    cmp = cmp.rename(columns={target: "y_true"})
    mae = float(np.mean(np.abs(cmp["y_true"] - cmp["final_pred"]))) if len(cmp) else float("nan")
    rmse = float(np.sqrt(np.mean((cmp["y_true"] - cmp["final_pred"]) ** 2))) if len(cmp) else float("nan")
    val_metrics = compute_validation_metrics(cmp)

    return {
        "cutoff_date": cutoff,
        "forecast_df": fc,
        "comparison_df": cmp,
        "mae_final": mae,
        "rmse_final": rmse,
        "validation_metrics": val_metrics,
    }


def temporal_backtest_last_window(
    df: pd.DataFrame,
    *,
    target: str = TARGET_COL_DAILY_MAX_PRICE,
    holdout_days: int = 60,
    forecast_days: int = 30,
    exog_block1_kw: dict[str, Any] | None = None,
    **train_kw: Any,
) -> dict[str, Any]:
    d = normalize_column_names(df).sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    last = d["date"].max()
    cutoff = last - pd.Timedelta(days=holdout_days)
    return temporal_backtest_at_cutoff(
        df,
        cutoff,
        target=target,
        forecast_days=forecast_days,
        train_df=None,
        exog_block1_kw=exog_block1_kw,
        **train_kw,
    )


def run_full_pipeline(
    df: pd.DataFrame,
    *,
    target: str = TARGET_COL_DAILY_MAX_PRICE,
    days: int = 30,
    exog_method: ExogMethod = "ets",
    univariate_structure: ExogUnivariateStructure = "stl_trend_season_error",
    volatility_mode: ExogVolatilityMode = "residual_bootstrap",
    volatility_strength: float = 1.0,
    volatility_pool_days: int = 252,
    exog_random_state: int | None = 42,
    exog_scenario_replicates: int = 1,
    exog_enso_flat_max_horizon: int = 30,
    exog_stl_seasonal_period: dict[str, int] | None = None,
    exog_stl_residual_in_forecast: dict[str, bool] | None = None,
    log_exog_diagnostics: bool = False,
    use_realistic_exog_train: bool = False,
    origin_stride: int = 14,
    min_train_size: int = 500,
    **train_kw: Any,
) -> dict[str, Any]:
    d = normalize_column_names(df).sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()

    ex_f = block1_build_exog_future(
        d,
        days=days,
        method=exog_method,
        univariate_structure=univariate_structure,
        volatility_mode=volatility_mode,
        volatility_strength=volatility_strength,
        volatility_pool_days=volatility_pool_days,
        random_state=exog_random_state,
        exog_scenario_replicates=exog_scenario_replicates,
        enso_flat_max_horizon=exog_enso_flat_max_horizon,
        stl_seasonal_period=exog_stl_seasonal_period,
        stl_residual_in_forecast=exog_stl_residual_in_forecast,
        log_exog_diagnostics=log_exog_diagnostics,
        use_realistic_exog_train=use_realistic_exog_train,
    )
    b2 = block2_sarimax_fit_forecast(d, ex_f, target=target)
    bundle = block456_train_direct_hybrid(
        d,
        target=target,
        max_horizon=days,
        min_train_size=min_train_size,
        origin_stride=origin_stride,
        **train_kw,
    )
    out_df = block7_forecast_output_df(d, ex_f, bundle, b2, target=target)

    return {
        "forecast_table": out_df,
        "exog_future": ex_f,
        "sarimax_block": b2,
        "direct_models": bundle,
        "feature_importance_h15": feature_importance_for_horizon(bundle, 15),
    }
