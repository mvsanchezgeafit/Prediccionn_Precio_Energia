"""
Pronóstico univariante de covariables diarias (ETS o ARIMA) para escenarios futuros.

Se usa para rellenar demanda, hidrología, ENSO alineado, etc., cuando no hay datos
reales — p. ej. 30 días tras el último día XM.

Opcionalmente, ``volatility_mode`` añade variabilidad día a día (bootstrap de innovaciones
o ruido gaussiano calibrado) sobre el modo **ets_arima**.

El modo ``stl_trend_season_error`` descompone con **STL** (tendencia + estacional + residuo in-sample).
En el horizonte: tendencia extrapolada + último ciclo estacional; el **residuo futuro** solo se añade
donde ``DEFAULT_STL_RESIDUAL_IN_FORECAST_BY_COL`` lo indica (p. ej. **sí** en demanda, **no** en hidrología
y ENSO) para evitar ruido día a día irreal en series lentas.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

from enso_data import ENSO_EL_NINO_THRESHOLD, ENSO_LA_NINA_THRESHOLD
from ml_features import ensure_sarimax_calendar

logger = logging.getLogger(__name__)

ExogMethod = Literal["ets", "arima"]
ExogVolatilityMode = Literal["none", "residual_bootstrap", "innovation_gaussian"]
ExogUnivariateStructure = Literal["ets_arima", "stl_trend_season_error"]

# Periodo estacional STL por columna (días, entero). Hidrología / aportes / ENSO: **365** para
# captar estacionalidad anual (épocas secas/húmedas) en datos diarios; la demanda sigue en **7**.
# Si el histórico es corto, :func:`_resolve_stl_period` baja automáticamente (p. ej. 182 → 90 → 30 → 7).
DEFAULT_STL_PERIOD_BY_COL: dict[str, int] = {
    "demanda_max_kwh": 7,
    "porc_vol_util_diario": 365,
    "vol_util_energia_sistema": 365,
    "enso_index": 365,
    "porc_aporte_value": 365,
    "porc_aporte_max": 365,
}

# Incluir remuestreo de residuos STL en el **horizonte** (alta frecuencia). Series lentas (embalse, ENSO)
# suelen verse mejor con ``False`` (solo tendencia + estacional proyectados).
DEFAULT_STL_RESIDUAL_IN_FORECAST_BY_COL: dict[str, bool] = {
    "demanda_max_kwh": True,
    "porc_vol_util_diario": False,
    "vol_util_energia_sistema": False,
    "enso_index": False,
    "porc_aporte_value": False,
    "porc_aporte_max": False,
}

# Columnas “físicas” a pronosticar (month_sin/cos salen del calendario en ensure_sarimax_calendar)
DEFAULT_EXOG_COLS = (
    "demanda_max_kwh",
    "porc_vol_util_diario",
    "vol_util_energia_sistema",
    "enso_index",
    "porc_aporte_value",
    "porc_aporte_max",
)

# Multiplicadores sobre ``volatility_strength`` global (Bloque 1): distinta dinámica por variable.
DEFAULT_VOL_STRENGTH_SCALE_BY_COL: dict[str, float] = {
    "demanda_max_kwh": 1.35,
    "enso_index": 0.0,
    "porc_vol_util_diario": 1.05,
    "vol_util_energia_sistema": 0.95,
    "porc_aporte_value": 1.25,
    "porc_aporte_max": 1.25,
}

# Ventana (días) para extrapolar tendencia STL; hidrología usa más historia (ciclos anuales).
DEFAULT_STL_TREND_FIT_WINDOW_BY_COL: dict[str, int] = {
    "demanda_max_kwh": 120,
    "porc_vol_util_diario": 365,
    "vol_util_energia_sistema": 365,
    "enso_index": 365,
    "porc_aporte_value": 365,
    "porc_aporte_max": 365,
}

# Amortiguamiento de tendencia por variable: hidrología y aportes necesitan decay alto para que
# el ciclo anual (seco dic–mar / húmedo abr–nov) domine sobre la extrapolación lineal.
DEFAULT_STL_TREND_DECAY_BY_COL: dict[str, float] = {
    "demanda_max_kwh": 0.0,
    "porc_vol_util_diario": 0.06,
    "vol_util_energia_sistema": 0.06,
    "enso_index": 0.0,
    "porc_aporte_value": 0.05,
    "porc_aporte_max": 0.05,
}

# Si True para una columna, trend_decay revierte hacia la MEDIA HISTÓRICA del trend (no hacia
# el último nivel). Fundamental para embalses/aportes cuya media anual es más baja que el último
# valor observado al final de la temporada húmeda.
DEFAULT_STL_TREND_REVERT_MEAN_BY_COL: dict[str, bool] = {
    "demanda_max_kwh": False,
    "porc_vol_util_diario": True,
    "vol_util_energia_sistema": True,
    "enso_index": False,
    "porc_aporte_value": True,
    "porc_aporte_max": True,
}


def resolve_volatility_strength_for_col(
    base_strength: float,
    col: str,
    scale_by_col: dict[str, float] | None = None,
) -> float:
    """Escala ``volatility_strength`` por tipo de exógena (demanda vs ENSO vs hidrología)."""
    scales = scale_by_col if scale_by_col is not None else DEFAULT_VOL_STRENGTH_SCALE_BY_COL
    mult = float(scales.get(col, 1.0))
    return float(base_strength) * mult


def _resolve_stl_period(desired: int, n: int) -> int:
    """
    Elige el mayor periodo estacional factible con ``n`` observaciones.

    STL (statsmodels) exige aprox. ``n >= max(3 * period, 56)``. Si no alcanza el periodo
    deseado (p. ej. 365 con menos de ~3 años), se prueba una escalera hacia abajo.
    """
    if n < 2:
        return 7
    d = max(2, int(desired))
    chain: list[int] = [d]
    for f in (365, 182, 120, 90, 60, 30, 14, 7):
        if f not in chain and f < d:
            chain.append(f)
    chain = sorted(set(chain), reverse=True)

    def _min_len(p: int) -> int:
        return max(3 * int(p), 56)

    for p in chain:
        if p < 2:
            continue
        if n >= _min_len(p):
            return p
    for p in (7, 5, 4, 3, 2):
        if n >= _min_len(p):
            return p
    return 7


def _clean_series(y: pd.Series) -> np.ndarray:
    s = pd.to_numeric(y, errors="coerce")
    s = s.interpolate(limit_direction="both").bfill().ffill()
    return s.astype(float).values


def _clip_exog_column(col: str, arr: np.ndarray) -> np.ndarray:
    """Límites físicos razonables tras añadir ruido."""
    a = np.asarray(arr, dtype=float)
    if col == "porc_vol_util_diario":
        return np.clip(a, 0.0, 1.0)
    if col in ("demanda_max_kwh", "vol_util_energia_sistema"):
        return np.maximum(a, 0.0)
    if col == "enso_index":
        return np.clip(a, -4.0, 4.0)
    if col in ("porc_aporte_value", "porc_aporte_max"):
        return np.maximum(a, 0.0)
    return a


def _apply_exog_volatility(
    y_hist: np.ndarray,
    fc_smooth: np.ndarray,
    col: str,
    *,
    mode: ExogVolatilityMode,
    strength: float,
    pool_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Añade variabilidad al pronóstico suavizado usando el comportamiento de innovaciones recientes.

    - ``residual_bootstrap``: remuestrea diferencias diarias centradas (últimos ``pool_days`` días).
    - ``innovation_gaussian``: ruido gaussiano con sigma = desv. de esas diferencias.
    """
    fc = np.asarray(fc_smooth, dtype=float).copy()
    if mode == "none" or strength <= 0:
        return fc
    y = np.asarray(y_hist, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 8:
        return fc

    tail = y[-min(int(pool_days), len(y)) :]
    diffs = np.diff(tail)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 5:
        return fc

    steps = len(fc)
    if mode == "residual_bootstrap":
        d0 = diffs - float(np.mean(diffs))
        noise = rng.choice(d0, size=steps, replace=True)
        fc = fc + float(strength) * noise
    elif mode == "innovation_gaussian":
        sigma = float(np.std(diffs))
        if not np.isfinite(sigma) or sigma < 1e-12:
            sigma = float(np.nanstd(tail)) * 0.05 or 1e-6
        fc = fc + float(strength) * rng.normal(0.0, sigma, size=steps)
    else:
        return fc_smooth

    return _clip_exog_column(col, fc)


def _forecast_one_univariate(
    y: np.ndarray,
    steps: int,
    *,
    method: ExogMethod,
) -> tuple[np.ndarray, str]:
    """
    Devuelve vector ``steps`` y etiqueta del método que ganó.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 5:
        v = float(np.nanmean(y)) if len(y) else 0.0
        return np.full(steps, v), "short_mean"

    if np.nanstd(y[-min(90, len(y)) :]) < 1e-12:
        v = float(np.nanmean(y[-min(30, len(y)) :]))
        return np.full(steps, v), "constant"

    def try_arima() -> tuple[np.ndarray, str] | None:
        for order in ((1, 1, 1), (0, 1, 1), (1, 0, 0)):
            try:
                fit = ARIMA(y, order=order).fit()
                fc = np.asarray(fit.forecast(steps), dtype=float)
                return fc, f"arima_{order[0]}{order[1]}{order[2]}"
            except Exception:
                continue
        return None

    def try_ets() -> tuple[np.ndarray, str] | None:
        candidates: list[tuple[str, dict[str, Any]]] = []
        if len(y) >= 56:
            candidates.append(
                (
                    "ets_add_s7",
                    {"seasonal_periods": 7, "trend": "add", "seasonal": "add"},
                )
            )
        candidates.extend(
            [
                ("ets_add", {"trend": "add", "seasonal": None}),
                ("ets_none", {"trend": None, "seasonal": None}),
            ]
        )
        for tag, kw in candidates:
            try:
                if kw.get("seasonal") is None:
                    model = ExponentialSmoothing(
                        y,
                        trend=kw.get("trend"),
                        seasonal=None,
                        initialization_method="estimated",
                    )
                else:
                    model = ExponentialSmoothing(
                        y,
                        trend=kw.get("trend"),
                        seasonal=kw["seasonal"],
                        seasonal_periods=kw["seasonal_periods"],
                        initialization_method="estimated",
                    )
                fit = model.fit(optimized=True, disp=False)
                fc = np.asarray(fit.forecast(steps), dtype=float)
                if np.all(np.isfinite(fc)):
                    return fc, tag
            except Exception as e:
                logger.debug("ETS %s: %s", tag, e)
        return None

    order_funcs = []
    if method == "arima":
        order_funcs = [try_arima, try_ets]
    else:
        order_funcs = [try_ets, try_arima]

    for fn in order_funcs:
        out = fn()
        if out is not None:
            return out

    tail = float(np.nanmean(y[-min(30, len(y)) :]))
    return np.full(steps, tail), "mean_tail"


def _forecast_one_stl_components(
    y: np.ndarray,
    steps: int,
    *,
    seasonal_period: int,
    rng: np.random.Generator,
    include_residual_in_forecast: bool = True,
    trend_decay: float = 0.0,
    trend_fit_window: int = 120,
    trend_revert_to_mean: bool = False,
) -> tuple[np.ndarray, str] | None:
    """
    Pronóstico vía STL: ``y ≈ tendencia + estacional + residuo``.

    - **Tendencia futura:** extrapolación lineal ajustada a los últimos puntos de la tendencia STL;
      si ``trend_decay > 0``, se amortigua exponencialmente.
      - ``trend_revert_to_mean=False`` (default): revierte hacia el último nivel de trend (aplana).
      - ``trend_revert_to_mean=True``: calendar-aware — la señal combinada (trend+seasonal)
        revierte hacia la **media histórica de y en cada posición del ciclo estacional**.
        Así el target para dic (seco) es naturalmente más bajo que para jul (húmedo).
    - **Estacional futuro:** repetición del último ciclo completo del componente estacional.
    - **Error futuro (opcional):** si ``include_residual_in_forecast``, remuestreo de residuos STL
      centrados; si no, pronóstico **solo** tendencia + estacional (trayectoria más suave).

    Si la serie es demasiado corta o STL falla, devuelve ``None`` (el llamador hace fallback a ETS/ARIMA).
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    p = int(seasonal_period)
    if p < 2 or steps < 1:
        return None
    min_len = max(3 * p, 56)
    if len(y) < min_len:
        return None

    try:
        if p <= 25:
            seasonal_win = min(2 * p + 1, 51)
        else:
            seasonal_win = min(max(51, 2 * int(np.sqrt(p)) + 1), 101)
        if seasonal_win % 2 == 0:
            seasonal_win += 1
        seasonal_win = min(seasonal_win, max(7, len(y) - 1))
        if seasonal_win % 2 == 0:
            seasonal_win -= 1
        seasonal_win = max(3, seasonal_win)
        use_robust = p < 90
        stl = STL(y, period=p, seasonal=seasonal_win, robust=use_robust).fit()
    except Exception as e:
        logger.debug("STL(period=%s) no aplicable: %s", p, e)
        return None

    trend = np.asarray(stl.trend, dtype=float)
    seasonal = np.asarray(stl.seasonal, dtype=float)
    resid = np.asarray(stl.resid, dtype=float)
    n = len(y)

    W = min(max(30, int(trend_fit_window)), n)
    x = np.arange(n - W, n, dtype=float)
    coef = np.polyfit(x, trend[-W:], 1)
    t_future = np.arange(n, n + steps, dtype=float)
    raw_trend_future = np.polyval(coef, t_future).astype(float)

    base = seasonal[-p:]
    seasonal_future = np.array([base[j % p] for j in range(steps)], dtype=float)

    td = float(trend_decay)
    if td > 0.0 and trend_revert_to_mean:
        # Calendar-aware: revert (trend+seasonal) toward the historical mean of y
        # at each seasonal position. This naturally encodes dry/wet cycles.
        pos_means = np.full(p, np.nanmean(y))
        for k in range(p):
            vals = y[k::p]
            if len(vals) > 0:
                pos_means[k] = float(np.nanmean(vals))
        seasonal_targets = np.array(
            [pos_means[(n + j) % p] for j in range(steps)], dtype=float
        )
        combined_raw = raw_trend_future + seasonal_future
        h1 = np.arange(1, steps + 1, dtype=float)
        w = np.exp(-td * h1)
        fc_base = seasonal_targets + w * (combined_raw - seasonal_targets)
    elif td > 0.0:
        anchor = float(trend[-1])
        if np.isfinite(anchor):
            h1 = np.arange(1, steps + 1, dtype=float)
            w = np.exp(-td * h1)
            trend_future = anchor + w * (raw_trend_future - anchor)
        else:
            trend_future = raw_trend_future
        fc_base = trend_future + seasonal_future
    else:
        fc_base = raw_trend_future + seasonal_future

    if include_residual_in_forecast:
        r = resid[np.isfinite(resid)]
        if len(r) < 3:
            err_future = np.zeros(steps)
        else:
            r0 = r - float(np.mean(r))
            err_future = rng.choice(r0, size=steps, replace=True).astype(float)
        fc = fc_base + err_future
        tag = f"stl_p{p}_trend_season_resid"
    else:
        fc = fc_base
        tag = f"stl_p{p}_trend_season_only"

    if not np.all(np.isfinite(fc)):
        return None
    return fc, tag


def build_exog_forecast_univariate(
    df_hist: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    *,
    method: ExogMethod = "ets",
    columns: tuple[str, ...] | None = None,
    univariate_structure: ExogUnivariateStructure = "stl_trend_season_error",
    stl_seasonal_period: dict[str, int] | None = None,
    stl_residual_in_forecast: dict[str, bool] | None = None,
    stl_trend_decay: float = 0.0,
    stl_trend_decay_by_col: dict[str, float] | None = None,
    stl_trend_revert_mean_by_col: dict[str, bool] | None = None,
    stl_trend_fit_window: int = 120,
    stl_trend_fit_window_by_col: dict[str, int] | None = None,
    volatility_mode: ExogVolatilityMode = "residual_bootstrap",
    volatility_strength: float = 1.0,
    volatility_strength_scale_by_col: dict[str, float] | None = None,
    volatility_pool_days: int = 252,
    random_state: int | None = 42,
    enso_flat_max_horizon: int = 30,
) -> pd.DataFrame:
    """
    Para cada columna numérica disponible en ``DEFAULT_EXOG_COLS``:

    - ``stl_trend_season_error`` (por defecto): **STL** (tendencia + estacional + residuo opcional en
      horizonte según ``stl_residual_in_forecast``); si STL no aplica, **fallback** a ETS/ARIMA.
    - ``ets_arima``: ETS o ARIMA univariante; con ``volatility_mode`` opcional.

    Con STL, ``volatility_mode`` (p. ej. ``residual_bootstrap``) se aplica **después** del pronóstico
    STL como capa adicional de variabilidad día a día (remuestreo de diferencias recientes).

    ``enso_flat_max_horizon``: si el horizonte ``<=`` este valor y existe ``enso_index``, se fuerza
    trayectoria **plana** al último valor observado (ENSO lento vs. 30 días). Use ``0`` para desactivar.

    Añade ``month_sin`` / ``month_cos`` y ``enso_regime_ord`` si aplica.
    """
    h = df_hist.sort_values("date").reset_index(drop=True)
    cols = list(columns) if columns else [c for c in DEFAULT_EXOG_COLS if c in h.columns]
    if not cols:
        raise ValueError("No hay columnas exógenas reconocidas en el histórico.")

    steps = len(future_dates)
    if steps < 1:
        raise ValueError("future_dates vacío.")

    rng = np.random.default_rng(random_state)
    period_map: dict[str, int] = {**DEFAULT_STL_PERIOD_BY_COL, **(stl_seasonal_period or {})}
    resid_map: dict[str, bool] = {
        **DEFAULT_STL_RESIDUAL_IN_FORECAST_BY_COL,
        **(stl_residual_in_forecast or {}),
    }
    trend_win_map: dict[str, int] = {
        **DEFAULT_STL_TREND_FIT_WINDOW_BY_COL,
        **(stl_trend_fit_window_by_col or {}),
    }
    decay_map: dict[str, float] = {
        **DEFAULT_STL_TREND_DECAY_BY_COL,
        **(stl_trend_decay_by_col or {}),
    }
    revert_mean_map: dict[str, bool] = {
        **DEFAULT_STL_TREND_REVERT_MEAN_BY_COL,
        **(stl_trend_revert_mean_by_col or {}),
    }
    rows: dict[str, Any] = {"date": future_dates.normalize()}
    meta_methods: dict[str, str] = {}
    stl_period_used: dict[str, int] = {}
    stl_resid_used: dict[str, bool] = {}

    use_stl = univariate_structure == "stl_trend_season_error"
    vol_eff: ExogVolatilityMode = volatility_mode

    for c in cols:
        y = _clean_series(h[c])
        tag = "unknown"
        if use_stl:
            desired_p = int(period_map.get(c, 7))
            pcol = _resolve_stl_period(desired_p, len(y))
            stl_period_used[c] = pcol
            if pcol != desired_p:
                logger.info(
                    "Exógena %s: STL periodo efectivo=%s (config %s, n=%s)",
                    c,
                    pcol,
                    desired_p,
                    len(y),
                )
            inc_res = bool(resid_map.get(c, False))
            stl_resid_used[c] = inc_res
            tw = int(trend_win_map.get(c, stl_trend_fit_window))
            td_col = float(decay_map.get(c, stl_trend_decay))
            rm_col = bool(revert_mean_map.get(c, False))
            stl_out = _forecast_one_stl_components(
                y,
                steps,
                seasonal_period=pcol,
                rng=rng,
                include_residual_in_forecast=inc_res,
                trend_decay=td_col,
                trend_fit_window=tw,
                trend_revert_to_mean=rm_col,
            )
            if stl_out is not None:
                fc, tag = stl_out
            else:
                fc, tag = _forecast_one_univariate(y, steps, method=method)
                tag = f"{tag} (fallback; STL no aplicable)"
                logger.warning("Exógena %s: STL no aplicable; fallback %s", c, tag)
        else:
            fc, tag = _forecast_one_univariate(y, steps, method=method)
            # ETS/ARIMA a veces elige un modelo casi sin dinámica → pronóstico ~constante aunque el
            # histórico varíe. En ese caso STL (misma period_map / resid_map) recupera t+S(+R).
            y_hist_std = float(np.nanstd(y[-min(120, len(y)) :])) if len(y) else 0.0
            fc_a = np.asarray(fc, dtype=float)
            fc_std = float(np.nanstd(fc_a)) if fc_a.size > 1 else 0.0
            if (
                steps >= 7
                and y_hist_std > 1e-6
                and (fc_std < 1e-6 or fc_std < 0.04 * y_hist_std)
            ):
                desired_p_fb = int(period_map.get(c, 7))
                pcol_fb = _resolve_stl_period(desired_p_fb, len(y))
                stl_period_used[c] = pcol_fb
                if pcol_fb != desired_p_fb:
                    logger.info(
                        "Exógena %s: STL periodo efectivo=%s (config %s, n=%s) [fallback vs ETS/ARIMA plano]",
                        c,
                        pcol_fb,
                        desired_p_fb,
                        len(y),
                    )
                inc_res_fb = bool(resid_map.get(c, False))
                stl_resid_used[c] = inc_res_fb
                tw_fb = int(trend_win_map.get(c, stl_trend_fit_window))
                td_col_fb = float(decay_map.get(c, stl_trend_decay))
                rm_col_fb = bool(revert_mean_map.get(c, False))
                stl_fb = _forecast_one_stl_components(
                    y,
                    steps,
                    seasonal_period=pcol_fb,
                    rng=rng,
                    include_residual_in_forecast=inc_res_fb,
                    trend_decay=td_col_fb,
                    trend_fit_window=tw_fb,
                    trend_revert_to_mean=rm_col_fb,
                )
                if stl_fb is not None:
                    fc, tag = stl_fb
                    tag = f"{tag} (fallback; ETS/ARIMA degenerado)"
                    logger.info("Exógena %s: %s", c, tag)

        strength_col = resolve_volatility_strength_for_col(
            float(volatility_strength),
            c,
            scale_by_col=volatility_strength_scale_by_col,
        )
        fc = _apply_exog_volatility(
            y,
            fc,
            c,
            mode=vol_eff,
            strength=strength_col,
            pool_days=int(volatility_pool_days),
            rng=rng,
        )
        if (
            c == "enso_index"
            and int(enso_flat_max_horizon) > 0
            and steps <= int(enso_flat_max_horizon)
            and len(y) > 0
        ):
            lv = float(y[-1]) if np.isfinite(y[-1]) else float(np.nanmedian(y))
            fc = np.full(steps, lv, dtype=float)
            tag = f"{tag}_enso_flat_h{steps}"
        rows[c] = _clip_exog_column(c, fc)
        meta_methods[c] = tag
        logger.info("Exógena %s: método=%s structure=%s vol=%s", c, tag, univariate_structure, vol_eff)

    out = pd.DataFrame(rows)
    if "enso_index" in out.columns:
        out["enso_regime_ord"] = out["enso_index"].apply(_enso_regime_ord)
    else:
        out["enso_regime_ord"] = 0.0
    out = ensure_sarimax_calendar(out)
    out.attrs["univariate_method_per_column"] = meta_methods
    out.attrs["exog_forecast_method"] = method
    out.attrs["exog_univariate_structure"] = univariate_structure
    used_stl_any = use_stl or bool(stl_period_used)
    out.attrs["stl_seasonal_period_by_column"] = stl_period_used if used_stl_any else {}
    out.attrs["stl_residual_in_forecast_by_column"] = stl_resid_used if used_stl_any else {}
    out.attrs["stl_trend_decay"] = float(stl_trend_decay) if used_stl_any else 0.0
    out.attrs["stl_trend_fit_window_by_column"] = (
        {c: int(trend_win_map.get(c, stl_trend_fit_window)) for c in cols} if used_stl_any else {}
    )
    out.attrs["exog_volatility"] = {
        "mode": vol_eff,
        "requested_mode": volatility_mode,
        "strength": float(volatility_strength),
        "strength_scale_by_col": dict(volatility_strength_scale_by_col or DEFAULT_VOL_STRENGTH_SCALE_BY_COL),
        "pool_days": int(volatility_pool_days),
        "random_state": random_state,
        "enso_flat_max_horizon": int(enso_flat_max_horizon),
    }
    return out


def summarize_exog_forecast_df(
    exog_future: pd.DataFrame,
    *,
    eps_flat: float = 1e-8,
) -> dict[str, Any]:
    """
    Métricas rápidas sobre ``exog_future`` (horizonte pronosticado): std por columna, NaNs,
    columnas con variación casi nula (posible trayectoria “plana”).
    """
    per: dict[str, Any] = {}
    flat_like: list[str] = []
    nan_total = 0
    for c in exog_future.columns:
        if c == "date":
            continue
        s = pd.to_numeric(exog_future[c], errors="coerce")
        nan_n = int(s.isna().sum())
        nan_total += nan_n
        if len(s) < 2:
            per[c] = {"n": int(len(s)), "std": 0.0, "n_nan": nan_n}
            if len(s) == 1:
                flat_like.append(c)
            continue
        std = float(s.std(ddof=1))
        if not np.isfinite(std):
            std = 0.0
        per[c] = {
            "n": int(len(s)),
            "std": std,
            "n_nan": nan_n,
            "min": float(np.nanmin(s.to_numpy(dtype=float))),
            "max": float(np.nanmax(s.to_numpy(dtype=float))),
        }
        if std < eps_flat:
            flat_like.append(c)
    return {
        "per_column": per,
        "flat_like_columns": flat_like,
        "nan_total": nan_total,
    }


def _enso_regime_ord(x: float) -> float:
    if pd.isna(x):
        return 0.0
    if x > ENSO_EL_NINO_THRESHOLD:
        return 1.0
    if x < ENSO_LA_NINA_THRESHOLD:
        return -1.0
    return 0.0


def build_exog_monthly_quantile_scenarios(
    df_hist: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> dict[float, pd.DataFrame]:
    """
    Escenarios de exógenas por **cuantil mensual** del histórico (mismo mes calendario).

    Útil para un abanico de SARIMAX: covariables “bajas / centrales / altas” típicas
    de cada mes, sin ETS/ARIMA.
    """
    h = df_hist.sort_values("date").copy()
    h["month"] = pd.to_datetime(h["date"]).dt.month
    exog_cols = [c for c in DEFAULT_EXOG_COLS if c in h.columns]
    if not exog_cols:
        raise ValueError("No hay columnas exógenas reconocidas en el histórico.")

    out_map: dict[float, pd.DataFrame] = {}
    for q in quantiles:
        gq = h.groupby("month")[exog_cols].quantile(q)
        rows: list[dict[str, Any]] = []
        for ts in future_dates:
            m = int(ts.month)
            row: dict[str, Any] = {"date": pd.Timestamp(ts).normalize()}
            for c in exog_cols:
                try:
                    if isinstance(gq, pd.Series):
                        val = gq.loc[m] if m in gq.index else np.nan
                    elif m in gq.index and c in gq.columns:
                        val = gq.loc[m, c]
                    else:
                        val = np.nan
                    row[c] = float(val) if pd.notna(val) else np.nan
                except (KeyError, TypeError, ValueError):
                    row[c] = np.nan
            rows.append(row)
        sdf = pd.DataFrame(rows)
        if "enso_index" in sdf.columns:
            sdf["enso_regime_ord"] = sdf["enso_index"].apply(_enso_regime_ord)
        else:
            sdf["enso_regime_ord"] = 0.0
        sdf = ensure_sarimax_calendar(sdf)
        out_map[float(q)] = sdf
    return out_map
