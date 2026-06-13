"""
Ensemble híbrido mejorado (opción tipo paper aplicado):

1. **SARIMAX solo como covariable:** en cada horizonte ``h`` se añade el pronóstico
   SARIMAX en log (``sarimax_fc_log``) a las mismas features tipo ``block3`` — el LightGBM
   aprende el objetivo **directamente en log1p** (no stack 0.5·SARIMAX + residuo).
2. **LGBM cuantílico:** tres modelos (P10, P50, P90) por día de horizonte.
3. **Modelo de picos (clasificación):** ``LGBMClassifier`` predice si el precio real
   superará un umbral alto (percentil histórico del train, p. ej. 90 %).
4. **Combinación:** si la probabilidad de pico es alta, el punto final se mezcla hacia
   arriba (acercamiento controlado al cuantil alto) para no subestimar colas.

Reutiliza precomputo SARIMAX por orígenes de ``hybrid_direct_30d``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from hybrid_direct_30d import (
    block1_build_exog_future,
    block2_sarimax_fit_forecast,
    block3_build_direct_features,
    normalize_column_names,
    pop_sarimax_block2_kwargs,
    precompute_sarimax_h_step_from_origins,
    to_log_price,
    _order_quantiles_monotone,
    _train_lgbm_quantile,
)
from ml_features import ensure_sarimax_calendar, sarimax_exog_columns
from hybrid_direct_30d import clip_log_price
from ml_metrics import lgbm_quantile_oos_metrics
from xm_config import TARGET_COL_DAILY_MAX_PRICE

logger = logging.getLogger(__name__)

SARIMAX_FC_LOG_COL = "sarimax_fc_log"


def _train_lgbm_classifier(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=400,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.85,
        min_child_samples=25,
        reg_alpha=0.05,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ).fit(X, y)


@dataclass
class HybridSpikeEnsembleBundle:
    feature_names: list[str]
    horizon: int
    models_q: dict[tuple[int, float], lgb.LGBMRegressor] = field(default_factory=dict)
    models_spike: dict[int, lgb.LGBMClassifier] = field(default_factory=dict)
    mean_by_h: dict[int, float] = field(default_factory=dict)
    origins_used: dict[int, np.ndarray] = field(default_factory=dict)
    spike_threshold_price: float = 0.0
    spike_train_quantile: float = 0.9
    vol_dim: int = 0
    vol_feature_names: list[str] = field(default_factory=list)


def train_hybrid_ensemble_spike(
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
    spike_train_quantile: float = 0.9,
    min_samples_per_h: int = 80,
    vol_feature_array: np.ndarray | None = None,
    vol_feature_names: list[str] | None = None,
) -> HybridSpikeEnsembleBundle:
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
    y_price = pd.to_numeric(feat_df[target], errors="coerce").astype(float)
    y_log = to_log_price(y_price.values)

    vol_dim = 0
    v_names: list[str] = []
    V: np.ndarray | None = None
    if vol_feature_array is not None:
        V = np.asarray(vol_feature_array, dtype=float)
        if V.ndim != 2:
            raise ValueError("vol_feature_array debe ser 2D (n_rows, n_vol).")
        if V.shape[0] != len(feat_df):
            raise ValueError(f"vol_feature_array.shape[0]={V.shape[0]} != len(feat_df)={len(feat_df)}")
        vol_dim = int(V.shape[1])
        v_names = list(vol_feature_names or [f"vol_f{i}" for i in range(vol_dim)])
        if len(v_names) != vol_dim:
            raise ValueError("vol_feature_names debe tener la misma longitud que columnas de vol.")

    thr = float(np.nanquantile(y_price.values, spike_train_quantile))

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
        raise RuntimeError("No hay orígenes SARIMAX; ajuste min_train_size u origin_stride.")

    out_names = list(feat_names) + [SARIMAX_FC_LOG_COL] + v_names
    bundle = HybridSpikeEnsembleBundle(
        feature_names=out_names,
        horizon=max_horizon,
        origins_used=sarimax_by_t,
        spike_threshold_price=thr,
        spike_train_quantile=spike_train_quantile,
        vol_dim=vol_dim,
        vol_feature_names=v_names,
    )

    for h in range(1, max_horizon + 1):
        X_rows: list[np.ndarray] = []
        y_log_tgt: list[float] = []
        y_spike: list[int] = []
        for t, fc_vec in sarimax_by_t.items():
            if t + h >= len(y_log):
                continue
            if np.isnan(y_log[t + h]) or np.isnan(y_price.iloc[t + h]):
                continue
            row = feat_df.iloc[t]
            if row[feat_names].isna().any():
                continue
            s_h = float(fc_vec[h - 1])
            base = row[feat_names].to_numpy(dtype=float)
            parts = [base, np.array([s_h], dtype=float)]
            if vol_dim > 0 and V is not None:
                parts.append(np.asarray(V[t, :vol_dim], dtype=float))
            X_rows.append(np.concatenate(parts))
            y_log_tgt.append(float(y_log[t + h]))
            y_spike.append(1 if float(y_price.iloc[t + h]) >= thr else 0)

        if len(y_log_tgt) < min_samples_per_h:
            logger.warning("h=%d: pocas muestras (%d)", h, len(y_log_tgt))
        if not X_rows:
            raise RuntimeError(f"Sin datos para h={h}")

        X_mat = np.vstack(X_rows)
        y_raw = np.asarray(y_log_tgt, dtype=float)
        mu_h = float(np.mean(y_raw))
        bundle.mean_by_h[h] = mu_h
        y_c = y_raw - mu_h
        y_bin = np.asarray(y_spike, dtype=int)

        for alpha in quantiles:
            bundle.models_q[(h, float(alpha))] = _train_lgbm_quantile(X_mat, y_c, float(alpha))

        pos = int(np.sum(y_bin))
        if pos < 15 or (len(y_bin) - pos) < 15:
            logger.warning(
                "h=%d: clases desbalanceadas (positivos=%d); clasificador puede ser débil.",
                h,
                pos,
            )
        bundle.models_spike[h] = _train_lgbm_classifier(X_mat, y_bin)

    return bundle


def ensemble_forecast_df(
    df: pd.DataFrame,
    exog_future: pd.DataFrame,
    bundle: HybridSpikeEnsembleBundle,
    block2_result: dict[str, Any],
    *,
    target: str = TARGET_COL_DAILY_MAX_PRICE,
    spike_prob_threshold: float = 0.30,
    spike_upward_blend: float = 0.35,
    vol_forecast_matrix: np.ndarray | None = None,
    spike_vol_prob_boost: float = 0.0,
) -> pd.DataFrame:
    """
    ``spike_upward_blend``: si ``P(pico) >= spike_prob_threshold``, el punto en **log**
    se acerca al P90: ``(1-w)*P50_log + w*((1-a)*P50_log + a*P90_log)`` con ``a = spike_upward_blend``.
    """
    tr = normalize_column_names(df)
    tr = ensure_sarimax_calendar(tr.sort_values("date").copy()).reset_index(drop=True)
    feat_df, feat_names = block3_build_direct_features(tr, target=target, use_log_price_lags=True)
    x_row = feat_df.iloc[-1][feat_names].astype(float)
    if x_row.isna().any():
        x_row = x_row.fillna(x_row.median())
    X_base = x_row.to_numpy(dtype=float)

    s_log = np.asarray(block2_result["sarimax_pred_future_log"], dtype=float).ravel()
    h_max = min(len(s_log), bundle.horizon)
    dates = pd.to_datetime(exog_future["date"]).dt.normalize().iloc[:h_max]

    Vf: np.ndarray | None = None
    if vol_forecast_matrix is not None:
        Vf = np.asarray(vol_forecast_matrix, dtype=float)
        if Vf.ndim != 2 or Vf.shape[0] < h_max or Vf.shape[1] != bundle.vol_dim:
            raise ValueError(
                f"vol_forecast_matrix incompatible: shape={getattr(Vf, 'shape', None)} "
                f"vs h_max={h_max}, vol_dim={bundle.vol_dim}"
            )

    sarimax_price: list[float] = []
    final_price: list[float] = []
    q10_p: list[float] = []
    q50_p: list[float] = []
    q90_p: list[float] = []
    sigma_hat_out: list[float] = []
    vol_future_z_out: list[float] = []
    hi_pct_out: list[float] = []
    hi_z_out: list[float] = []
    risk_num_out: list[float] = []
    risk_flag_out: list[float] = []

    for h in range(1, h_max + 1):
        s_h = float(s_log[h - 1])
        mu_h = bundle.mean_by_h[h]
        parts = [X_base, np.array([s_h], dtype=float)]
        if bundle.vol_dim > 0 and Vf is not None:
            parts.append(np.asarray(Vf[h - 1, : bundle.vol_dim], dtype=float))
        X_h = np.concatenate(parts).reshape(1, -1)

        rq10 = float(bundle.models_q[(h, 0.1)].predict(X_h)[0])
        rq50 = float(bundle.models_q[(h, 0.5)].predict(X_h)[0])
        rq90 = float(bundle.models_q[(h, 0.9)].predict(X_h)[0])
        q10l = rq10 + mu_h
        q50l = rq50 + mu_h
        q90l = rq90 + mu_h
        q10l, q50l, q90l = _order_quantiles_monotone(q10l, q50l, q90l)

        p_spike = float(bundle.models_spike[h].predict_proba(X_h)[0, 1])
        if Vf is not None and spike_vol_prob_boost > 0:
            rowv = np.asarray(Vf[h - 1], dtype=float)
            hi_any = float(max(rowv[4] if len(rowv) > 4 else 0.0, rowv[5] if len(rowv) > 5 else 0.0))
            if hi_any >= 0.5:
                p_spike = min(1.0, p_spike + float(spike_vol_prob_boost))
        if Vf is not None:
            rowv = np.asarray(Vf[h - 1], dtype=float)
            sigma_hat_out.append(float(rowv[0]) if len(rowv) > 0 else float("nan"))
            vol_future_z_out.append(float(rowv[1]) if len(rowv) > 1 else float("nan"))
            hi_pct_out.append(float(rowv[4]) if len(rowv) > 4 else float("nan"))
            hi_z_out.append(float(rowv[5]) if len(rowv) > 5 else float("nan"))
            rn = float(rowv[6]) if len(rowv) > 6 else float("nan")
            risk_num_out.append(rn)
            risk_flag_out.append(
                1.0
                if (len(rowv) > 4 and rowv[4] >= 0.5)
                or (len(rowv) > 5 and rowv[5] >= 0.5)
                or (len(rowv) > 6 and rn >= 2.0 - 1e-9)
                else 0.0
            )
        else:
            sigma_hat_out.append(float("nan"))
            vol_future_z_out.append(float("nan"))
            hi_pct_out.append(float("nan"))
            hi_z_out.append(float("nan"))
            risk_num_out.append(float("nan"))
            risk_flag_out.append(float("nan"))

        med_log = q50l
        if p_spike >= spike_prob_threshold:
            hi_log = (1.0 - spike_upward_blend) * q50l + spike_upward_blend * q90l
            w = min(1.0, (p_spike - spike_prob_threshold) / max(1e-6, 1.0 - spike_prob_threshold))
            final_log = (1.0 - w) * med_log + w * hi_log
        else:
            final_log = med_log

        s_h = float(clip_log_price(s_h))
        final_log = float(clip_log_price(final_log))
        q10l, q50l, q90l = (
            float(clip_log_price(q10l)),
            float(clip_log_price(q50l)),
            float(clip_log_price(q90l)),
        )
        sarimax_price.append(float(np.expm1(s_h)))
        final_price.append(float(np.expm1(final_log)))
        q10_p.append(float(np.expm1(q10l)))
        q50_p.append(float(np.expm1(q50l)))
        q90_p.append(float(np.expm1(q90l)))

    out = {
        "date": dates.values,
        "sarimax_pred": sarimax_price,
        "ensemble_pred": final_price,
        "q10": q10_p,
        "q50": q50_p,
        "q90": q90_p,
        "sigma_hat_t": sigma_hat_out,
        "vol_future_zscore": vol_future_z_out,
        "future_high_vol_flag_pct90": hi_pct_out,
        "future_high_vol_flag_z15": hi_z_out,
        "future_risk_level_num": risk_num_out,
        "future_risk_flag": risk_flag_out,
    }
    return pd.DataFrame(out)


def extend_hybrid_backtest_with_ensemble_spike(
    df: pd.DataFrame,
    bt: dict[str, Any],
    *,
    target: str = TARGET_COL_DAILY_MAX_PRICE,
    exog_method: str = "ets",
    univariate_structure: str = "stl_trend_season_error",
    exog_block1_kw: dict[str, Any] | None = None,
    origin_stride: int = 14,
    min_train_size: int = 500,
    use_garch_vol_spike: bool = False,
    spike_vol_prob_boost: float = 0.08,
    **train_kw: Any,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Usa el mismo corte y horizonte que ``temporal_backtest_last_window`` del híbrido:
    entrena el ensemble+spike solo con ``train`` y añade columnas al ``comparison_df``.

    Devuelve ``(comparison_extended, metrics)`` con MAE/RMSE de ``final_pred``,
    ``sarimax_pred`` y ``ensemble_pred``, y métricas cuantílicas del ensemble (pinball,
    cobertura, interval score) si hay filas alineadas.

    Parámetros opcionales de volatilidad GARCH (sin cambiar SARIMAX/VAR base):

    ``use_garch_vol_spike``
        Si es True, ajusta GARCH(1,1) sobre residuos log del ``block2`` del train, construye
        features de volatilidad alineadas al panel de features del ensemble y añade al
        pronóstico columnas ``sigma_hat_t``, ``vol_future_zscore``, banderas de alta volatilidad
        y ``future_risk_flag``.
    ``spike_vol_prob_boost``
        Incremento acotado a la probabilidad de pico cuando el horizonte futuro está marcado
        como alta volatilidad (regla heurística).
    """
    forecast_days = int(bt["forecast_df"].shape[0])
    d = normalize_column_names(df).sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    cutoff = pd.Timestamp(bt["cutoff_date"]).normalize()
    train = d[d["date"] <= cutoff].copy()

    b1: dict[str, Any] = {**(exog_block1_kw or {})}
    if "method" not in b1:
        b1["method"] = exog_method
    if "univariate_structure" not in b1:
        b1["univariate_structure"] = univariate_structure
    ex_f = block1_build_exog_future(train, days=forecast_days, **b1)
    train_kw_fit = dict(train_kw)
    b2_fit_kw, train_kw_fit = pop_sarimax_block2_kwargs(train_kw_fit)
    b2 = block2_sarimax_fit_forecast(
        train,
        ex_f,
        target=target,
        order=b2_fit_kw["order"],
        seasonal_order=b2_fit_kw["seasonal_order"],
        exog_cols=b2_fit_kw["exog_cols"],
        standardize_exog=b2_fit_kw["standardize_exog"],
    )

    if use_garch_vol_spike:
        try:
            from garch_intervals import fit_garch, forecast_garch_volatility
            from garch_volatility_spike import (
                VOL_SPIKE_FEATURE_NAMES,
                align_sigma_series,
                build_forecast_volatility_matrix,
                build_in_sample_volatility_features,
                conditional_sigma_in_sample,
                volatility_feature_matrix_for_spike,
            )
        except ImportError as e:
            logger.warning("use_garch_vol_spike=True pero import falló: %s", e)
            use_garch_vol_spike = False

    if use_garch_vol_spike:
        tr_pre = ensure_sarimax_calendar(normalize_column_names(train).sort_values("date").copy()).reset_index(
            drop=True
        )
        feat_df_align, _ = block3_build_direct_features(tr_pre, target=target, use_log_price_lags=True)
        resid = np.asarray(b2["residuals_in_sample_log"], dtype=float)
        g_fit, g_err = fit_garch(resid)
        if g_fit is None:
            logger.warning("GARCH vol-spike omitido: %s", g_err)
            bundle = train_hybrid_ensemble_spike(
                train,
                target=target,
                max_horizon=forecast_days,
                min_train_size=min_train_size,
                origin_stride=origin_stride,
                order=b2_fit_kw["order"],
                seasonal_order=b2_fit_kw["seasonal_order"],
                sarimax_exog_cols=b2_fit_kw["exog_cols"],
                sarimax_standardize_exog=b2_fit_kw["standardize_exog"],
                **train_kw_fit,
            )
            fc_e = ensemble_forecast_df(train, ex_f, bundle, b2, target=target)
        else:
            sig = conditional_sigma_in_sample(g_fit)
            sig_al = align_sigma_series(sig, len(feat_df_align), align="tail")
            vol_df = build_in_sample_volatility_features(sig_al)
            V = volatility_feature_matrix_for_spike(vol_df)
            bundle = train_hybrid_ensemble_spike(
                train,
                target=target,
                max_horizon=forecast_days,
                min_train_size=min_train_size,
                origin_stride=origin_stride,
                order=b2_fit_kw["order"],
                seasonal_order=b2_fit_kw["seasonal_order"],
                sarimax_exog_cols=b2_fit_kw["exog_cols"],
                sarimax_standardize_exog=b2_fit_kw["standardize_exog"],
                vol_feature_array=V,
                vol_feature_names=list(VOL_SPIKE_FEATURE_NAMES),
                **train_kw_fit,
            )
            sig_hat = forecast_garch_volatility(g_fit, forecast_days)
            st_clean = sig[np.isfinite(sig)]
            Vf = build_forecast_volatility_matrix(sig_hat, st_clean)
            fc_e = ensemble_forecast_df(
                train,
                ex_f,
                bundle,
                b2,
                target=target,
                vol_forecast_matrix=Vf,
                spike_vol_prob_boost=float(spike_vol_prob_boost),
            )
    else:
        bundle = train_hybrid_ensemble_spike(
            train,
            target=target,
            max_horizon=forecast_days,
            min_train_size=min_train_size,
            origin_stride=origin_stride,
            order=b2_fit_kw["order"],
            seasonal_order=b2_fit_kw["seasonal_order"],
            sarimax_exog_cols=b2_fit_kw["exog_cols"],
            sarimax_standardize_exog=b2_fit_kw["standardize_exog"],
            **train_kw_fit,
        )
        fc_e = ensemble_forecast_df(train, ex_f, bundle, b2, target=target)

    rename_q = {"q10": "q10_ensemble", "q50": "q50_ensemble", "q90": "q90_ensemble"}
    sub = fc_e.rename(columns=rename_q)
    left = bt["comparison_df"]
    # Evitar columnas duplicadas (p. ej. ``sarimax_pred`` en híbrido y en ensemble): si no,
    # pandas crea ``sarimax_pred_x`` / ``sarimax_pred_y`` y falla el acceso a ``sarimax_pred``.
    overlap = [c for c in sub.columns if c in left.columns and c != "date"]
    if overlap:
        sub = sub.drop(columns=overlap, errors="ignore")
    cmp = left.merge(sub, on="date", how="inner")

    y = cmp["y_true"].to_numpy(dtype=float)
    metrics: dict[str, float] = {}
    for key, pred_col in (
        ("hybrid_final", "final_pred"),
        ("hybrid_sarimax", "sarimax_pred"),
        ("ensemble_spike", "ensemble_pred"),
    ):
        p = cmp[pred_col].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(p)
        if int(np.sum(mask)) == 0:
            metrics[f"mae_{key}"] = float("nan")
            metrics[f"rmse_{key}"] = float("nan")
            continue
        ye, pe = y[mask], p[mask]
        metrics[f"mae_{key}"] = float(np.mean(np.abs(ye - pe)))
        metrics[f"rmse_{key}"] = float(np.sqrt(np.mean((ye - pe) ** 2)))

    bias_h = float(np.mean(cmp["final_pred"].to_numpy(dtype=float) - y))
    bias_e = float(np.mean(cmp["ensemble_pred"].to_numpy(dtype=float) - y))
    metrics["mean_bias_hybrid_pred_minus_actual"] = bias_h
    metrics["mean_bias_ensemble_pred_minus_actual"] = bias_e

    m_e = lgbm_quantile_oos_metrics(
        y,
        cmp["q10_ensemble"].to_numpy(dtype=float),
        cmp["q50_ensemble"].to_numpy(dtype=float),
        cmp["q90_ensemble"].to_numpy(dtype=float),
    )
    for k, v in m_e.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            metrics[f"ensemble_{k}"] = float(v)

    cov_h = float(
        np.mean(
            (y >= cmp["q10"].to_numpy(dtype=float))
            & (y <= cmp["q90"].to_numpy(dtype=float))
        )
    )
    metrics["hybrid_coverage_p10_p90_empirical"] = cov_h

    return cmp, metrics


def run_ensemble_spike_pipeline(
    df: pd.DataFrame,
    *,
    target: str = TARGET_COL_DAILY_MAX_PRICE,
    days: int = 30,
    exog_method: str = "ets",
    univariate_structure: str = "stl_trend_season_error",
    exog_block1_kw: dict[str, Any] | None = None,
    origin_stride: int = 14,
    min_train_size: int = 500,
    **train_kw: Any,
) -> dict[str, Any]:
    d = normalize_column_names(df).sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()

    b1: dict[str, Any] = {**(exog_block1_kw or {})}
    if "method" not in b1:
        b1["method"] = exog_method
    if "univariate_structure" not in b1:
        b1["univariate_structure"] = univariate_structure
    ex_f = block1_build_exog_future(d, days=days, **b1)
    train_kw_fit = dict(train_kw)
    b2_fit_kw, train_kw_fit = pop_sarimax_block2_kwargs(train_kw_fit)
    b2 = block2_sarimax_fit_forecast(
        d,
        ex_f,
        target=target,
        order=b2_fit_kw["order"],
        seasonal_order=b2_fit_kw["seasonal_order"],
        exog_cols=b2_fit_kw["exog_cols"],
        standardize_exog=b2_fit_kw["standardize_exog"],
    )
    bundle = train_hybrid_ensemble_spike(
        d,
        target=target,
        max_horizon=days,
        min_train_size=min_train_size,
        origin_stride=origin_stride,
        order=b2_fit_kw["order"],
        seasonal_order=b2_fit_kw["seasonal_order"],
        sarimax_exog_cols=b2_fit_kw["exog_cols"],
        sarimax_standardize_exog=b2_fit_kw["standardize_exog"],
        **train_kw_fit,
    )
    out = ensemble_forecast_df(
        d,
        ex_f,
        bundle,
        b2,
        target=target,
    )
    return {
        "forecast_table": out,
        "exog_future": ex_f,
        "sarimax_block": b2,
        "ensemble_bundle": bundle,
    }
