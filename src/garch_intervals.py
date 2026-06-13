"""
GARCH(1,1) en residuos in-sample + pronóstico de volatilidad e intervalos dinámicos.

- **SARIMAX:** `y_hat` en escala **log1p**; σ̂ del GARCH ajustado sobre residuos log; intervalos en log y
  transformación **expm1** a COP/kWh.
- **VAR:** residuos de la 1.ª ecuación (Δ precio) coinciden con el error de pronóstico a un paso del
  nivel bajo ŷ_t = y_{t-1} + \widehat{Δy}_t; σ̂ en COP (misma escala que el precio); bandas **nivel**.

Sin refitar el modelo de media: solo se usa el vector de residuos ya obtenido y el `y_hat` entregado.

**Notas:** sigue siendo un enfoque en dos etapas (media + volatilidad), pero el módulo ofrece dos
modos de simulación puntual: independiente por horizonte y recursivo con dependencia temporal
GARCH(1,1) por trayectoria.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:
    from arch import arch_model
except ImportError as e:  # pragma: no cover
    arch_model = None  # type: ignore[misc, assignment]
    _ARCH_IMPORT_ERROR = e
else:
    _ARCH_IMPORT_ERROR = None


def z_scores_for_confidence(levels: Sequence[float]) -> dict[float, float]:
    """z bilateral para intervalos simétricos: P(-z < Z < z) = nivel."""
    out: dict[float, float] = {}
    for cl in levels:
        cl = float(cl)
        if not (0.0 < cl < 1.0):
            raise ValueError(f"nivel de confianza inválido: {cl}")
        out[cl] = float(stats.norm.ppf(0.5 + 0.5 * cl))
    return out


def fit_garch(
    residuals: np.ndarray,
    *,
    p: int = 1,
    q: int = 1,
    min_obs: int = 200,
) -> tuple[Any | None, str | None]:
    """
    Ajusta GARCH(p,q) con media cero sobre ``residuals`` (in-sample).

    Returns
    -------
    (result, None) o (None, mensaje_error).
    """
    if arch_model is None:
        return None, f"arch no disponible: {_ARCH_IMPORT_ERROR}"
    r = np.asarray(residuals, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if len(r) < int(min_obs):
        return None, f"muy pocos residuos finitos: {len(r)} < {min_obs}"
    try:
        am = arch_model(r, mean="Zero", vol="Garch", p=int(p), q=int(q))
        return am.fit(disp="off"), None
    except Exception as e:
        return None, str(e)


def forecast_garch_volatility(garch_res: Any, horizon: int) -> np.ndarray:
    """
    Desviación condicional pronosticada σ̂_{T+1},…,σ̂_{T+H} (raíz de la varianza).

    Usa ``forecast(horizon=H, reindex=False)`` de ``arch``; toma los últimos H valores de varianza.
    """
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon debe ser >= 1")
    fc = garch_res.forecast(horizon=h, reindex=False)
    raw = np.asarray(fc.variance, dtype=float)
    # ``arch`` puede devolver (H, 1), (1, H) o un vector; evitar ``[:, 0]`` si la forma es (1, H).
    if raw.ndim == 2:
        if raw.shape[0] == 1:
            var = np.asarray(raw[0, :], dtype=float).ravel()
        elif raw.shape[1] == 1:
            var = np.asarray(raw[:, 0], dtype=float).ravel()
        else:
            var = raw.ravel()
    else:
        var = raw.ravel()
    if var.size < h:
        raise ValueError(
            f"varianza pronosticada corta: {var.size} < {h} (forma raw={getattr(raw, 'shape', None)})"
        )
    tail = var[-h:]
    tail = np.maximum(tail, 1e-18)
    return np.sqrt(tail)


def build_prediction_intervals(
    y_hat: np.ndarray,
    sigma: np.ndarray,
    z: float,
    *,
    scale: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye bandas simétricas ``y_hat ± z * sigma``.

    Parameters
    ----------
    y_hat, sigma
        Misma longitud H (broadcast si sigma escalar).
    scale
        ``"log1p"``: y_hat y sigma en escala log(precio+1); salida en **COP/kWh** vía ``exp - 1``.
        ``"level"``: y_hat y sigma en nivel COP; salida en COP.
    """
    y_hat = np.asarray(y_hat, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    if sigma.size == 1:
        sigma = np.full_like(y_hat, float(sigma[0]), dtype=float)
    if y_hat.shape != sigma.shape:
        raise ValueError(f"y_hat shape {y_hat.shape} != sigma shape {sigma.shape}")
    if scale == "log1p":
        lo = y_hat - z * sigma
        hi = y_hat + z * sigma
        return np.exp(lo) - 1.0, np.exp(hi) - 1.0
    if scale == "level":
        return y_hat - z * sigma, y_hat + z * sigma
    raise ValueError(f"scale desconocido: {scale}")


def add_interval_columns(
    df: pd.DataFrame,
    y_hat_col: str,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    confidence: float,
) -> pd.DataFrame:
    """Añade ``{y_hat_col}_lower_{pct}`` y ``{y_hat_col}_upper_{pct}`` (pct = 90, 95, 99)."""
    pct = int(round(float(confidence) * 100))
    lo_n = f"{y_hat_col}_lower_{pct}"
    hi_n = f"{y_hat_col}_upper_{pct}"
    out = df.copy()
    out[lo_n] = lower
    out[hi_n] = upper
    return out


def _garch11_params(garch_res: Any) -> tuple[float, float, float]:
    """Extrae parámetros (omega, alpha[1], beta[1]) de un ajuste arch GARCH(1,1)."""
    p = getattr(garch_res, "params", None)
    if p is None:
        raise ValueError("resultado GARCH sin params")
    if hasattr(p, "index"):
        names = list(p.index)
        get = lambda key: float(p[key]) if key in names else float("nan")
        omega = get("omega")
        alpha = get("alpha[1]")
        beta = get("beta[1]")
    else:
        arr = np.asarray(p, dtype=float).ravel()
        if arr.size < 3:
            raise ValueError(f"params GARCH insuficientes: {arr.size}")
        omega, alpha, beta = float(arr[0]), float(arr[1]), float(arr[2])
    if not (np.isfinite(omega) and np.isfinite(alpha) and np.isfinite(beta)):
        raise ValueError("no se pudieron extraer omega/alpha[1]/beta[1] del ajuste GARCH")
    return omega, alpha, beta


def simulate_garch11_sigma_paths(
    garch_res: Any,
    horizon: int,
    *,
    n_paths: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simula trayectorias GARCH(1,1) recursivas.

    Returns
    -------
    sigma_paths, eps_paths : ambos de forma (n_paths, horizon)
    """
    h = int(horizon)
    n = int(n_paths)
    if h < 1 or n < 1:
        raise ValueError("horizon y n_paths deben ser >= 1")
    omega, alpha, beta = _garch11_params(garch_res)
    rng = rng if rng is not None else np.random.default_rng()
    cond_vol = np.asarray(garch_res.conditional_volatility, dtype=float).ravel()
    cond_var = np.maximum(cond_vol ** 2, 1e-18)
    sigma2_prev = float(cond_var[-1])
    resid = np.asarray(getattr(garch_res, "resid", np.array([0.0])), dtype=float).ravel()
    eps_prev = float(resid[-1]) if resid.size else 0.0
    sigma_paths = np.empty((n, h), dtype=float)
    eps_paths = np.empty((n, h), dtype=float)
    for i in range(n):
        s2 = sigma2_prev
        e_prev = eps_prev
        for t in range(h):
            s2 = max(omega + alpha * (e_prev ** 2) + beta * s2, 1e-18)
            s = float(np.sqrt(s2))
            e_t = float(rng.normal(0.0, s))
            sigma_paths[i, t] = s
            eps_paths[i, t] = e_t
            e_prev = e_t
    return sigma_paths, eps_paths


def simulate_sarimax_garch_point_cop(
    y_hat_log_future: np.ndarray,
    sigma_hat_future: np.ndarray,
    *,
    n_paths: int = 2000,
    aggregate: Literal["mean_cop", "median_cop"] = "mean_cop",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Pronóstico puntual en **COP/kWh** a partir de SARIMAX (media en log1p) + ruido acoplado al GARCH.

    Para cada horizonte ``h`` se simula ``ε_h ~ N(0, σ̂_h²)`` con ``σ̂_h`` del pronóstico de varianza
    de ``arch`` (innovaciones **independientes** entre ``h``: aproximación exploratoria; no replica
    la correlación completa de un camino GARCH simulado paso a paso).

    Se forma ``y_log = y_hat_log_h + ε_h`` y se pasa a COP con ``expm1``. El pronóstico puntual es
    la **media** o la **mediana** sobre ``n_paths`` trayectorias en COP, por lo que puede diferir de
    ``expm1(y_hat_log)`` (SARIMAX determinístico) por la no linealidad de ``expm1``.
    """
    y_hat = np.asarray(y_hat_log_future, dtype=float).ravel()
    sig = np.asarray(sigma_hat_future, dtype=float).ravel()
    h = int(y_hat.size)
    if h < 1:
        raise ValueError("y_hat_log_future vacío")
    if sig.size == 1:
        sig = np.full(h, float(sig[0]), dtype=float)
    if sig.shape != y_hat.shape:
        raise ValueError(f"sigma_hat_future shape {sig.shape} != y_hat {y_hat.shape}")
    n = int(n_paths)
    if n < 1:
        raise ValueError("n_paths debe ser >= 1")
    rng = rng if rng is not None else np.random.default_rng()
    sig = np.maximum(sig, 1e-12)
    eps = rng.normal(0.0, 1.0, size=(n, h)) * sig
    y_log = y_hat + eps
    y_cop = np.expm1(y_log)
    if aggregate == "mean_cop":
        return np.mean(y_cop, axis=0).astype(float)
    if aggregate == "median_cop":
        return np.median(y_cop, axis=0).astype(float)
    raise ValueError(f"aggregate desconocido: {aggregate}")


def simulate_sarimax_garch_point_cop_path_dependent(
    y_hat_log_future: np.ndarray,
    garch_res: Any,
    *,
    n_paths: int = 2000,
    aggregate: Literal["mean_cop", "median_cop"] = "mean_cop",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Igual que ``simulate_sarimax_garch_point_cop`` pero con ruido recursivo GARCH(1,1) por trayectoria.
    """
    y_hat = np.asarray(y_hat_log_future, dtype=float).ravel()
    h = int(y_hat.size)
    if h < 1:
        raise ValueError("y_hat_log_future vacío")
    sig_paths, eps_paths = simulate_garch11_sigma_paths(
        garch_res, h, n_paths=n_paths, rng=rng
    )
    _ = sig_paths  # útil para depuración/diagnóstico, no requerido en salida puntual
    y_log = y_hat.reshape(1, -1) + eps_paths
    y_cop = np.expm1(y_log)
    if aggregate == "mean_cop":
        return np.mean(y_cop, axis=0).astype(float)
    if aggregate == "median_cop":
        return np.median(y_cop, axis=0).astype(float)
    raise ValueError(f"aggregate desconocido: {aggregate}")


def simulate_var_garch_point_cop(
    y_hat_level: np.ndarray,
    sigma_hat_future: np.ndarray,
    *,
    n_paths: int = 2000,
    aggregate: Literal["mean_cop", "median_cop"] = "mean_cop",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Pronóstico puntual **exploratorio** en COP: el pronóstico VAR en nivel se perturba en **log1p**
    con ruido Gaussiano cuya escala deriva de ``sigma_hat_future`` (σ GARCH en COP del horizonte).

    ``sig_l = clip(σ / max(y_hat, 1), …)`` mapea volatilidad en nivel a una desviación en log1p;
    la media (o mediana) sobre trayectorias de ``expm1(log1p(y_hat) + ε)`` **no** coincide en general
    con ``y_hat``, así que MAE/RMSE/MAPE pueden diferir del VAR puro sin alterar el ajuste VAR base.

    No sustituye un modelo conjunto VAR-GARCH; sirve para **comparar métricas puntuales** con el GARCH
    como modulador de incertidumbre en el punto reportado.
    """
    y_hat = np.asarray(y_hat_level, dtype=float).ravel()
    sig = np.asarray(sigma_hat_future, dtype=float).ravel()
    h = int(y_hat.size)
    if h < 1:
        raise ValueError("y_hat_level vacío")
    if sig.size == 1:
        sig = np.full(h, float(sig[0]), dtype=float)
    if sig.shape != y_hat.shape:
        raise ValueError(f"sigma_hat_future shape {sig.shape} != y_hat {y_hat.shape}")
    n = int(n_paths)
    if n < 1:
        raise ValueError("n_paths debe ser >= 1")
    rng = rng if rng is not None else np.random.default_rng()
    y_pos = np.maximum(y_hat, 0.0)
    yz = np.log1p(y_pos)
    sig_l = sig / np.maximum(y_pos, 1.0)
    sig_l = np.clip(sig_l, 1e-6, 0.5)
    eps = rng.normal(0.0, 1.0, size=(n, h)) * sig_l
    y_cop = np.expm1(yz + eps)
    if aggregate == "mean_cop":
        return np.mean(y_cop, axis=0).astype(float)
    if aggregate == "median_cop":
        return np.median(y_cop, axis=0).astype(float)
    raise ValueError(f"aggregate desconocido: {aggregate}")


def simulate_var_garch_point_cop_path_dependent(
    y_hat_level: np.ndarray,
    garch_res: Any,
    *,
    n_paths: int = 2000,
    aggregate: Literal["mean_cop", "median_cop"] = "mean_cop",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Variante recursiva: genera shocks GARCH(1,1) dependientes en el tiempo y luego mapea a COP.
    """
    y_hat = np.asarray(y_hat_level, dtype=float).ravel()
    h = int(y_hat.size)
    if h < 1:
        raise ValueError("y_hat_level vacío")
    n = int(n_paths)
    if n < 1:
        raise ValueError("n_paths debe ser >= 1")
    rng = rng if rng is not None else np.random.default_rng()
    y_pos = np.maximum(y_hat, 0.0)
    yz = np.log1p(y_pos)
    sig_paths, eps_paths = simulate_garch11_sigma_paths(garch_res, h, n_paths=n, rng=rng)
    sig_l = sig_paths / np.maximum(y_pos.reshape(1, -1), 1.0)
    sig_l = np.clip(sig_l, 1e-6, 0.5)
    eps_std = eps_paths / np.maximum(sig_paths, 1e-12)
    y_cop = np.expm1(yz.reshape(1, -1) + eps_std * sig_l)
    if aggregate == "mean_cop":
        return np.mean(y_cop, axis=0).astype(float)
    if aggregate == "median_cop":
        return np.median(y_cop, axis=0).astype(float)
    raise ValueError(f"aggregate desconocido: {aggregate}")


def var_endog_col_names(target: str) -> list[str]:
    """Series del VAR: precio + físicas + aportes + Fourier (misma familia que SARIMAX full)."""
    return [
        target,
        "demanda_max_kwh",
        "porc_vol_util_diario",
        "enso_index",
        "porc_aporte_value",
        "week_sin",
        "week_cos",
        "month_sin",
        "month_cos",
    ]


def _var_prepare_levels(
    train_df: pd.DataFrame,
    target: str,
    *,
    ensure_sarimax_calendar: Any,
    normalize_column_names: Any,
) -> tuple[pd.DataFrame, list[str]]:
    tr = ensure_sarimax_calendar(normalize_column_names(train_df)).sort_values("date").reset_index(drop=True)
    cols = [c for c in var_endog_col_names(target) if c in tr.columns]
    if target not in cols:
        raise ValueError(f"falta objetivo en columnas VAR: {cols}")
    y = tr[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    tr = tr.copy()
    tr[cols] = y
    return tr, cols


def _var_build_future_levels(
    projected_exog_future: pd.DataFrame,
    cols: list[str],
    target: str,
    h: int,
    *,
    ensure_sarimax_calendar: Any,
) -> pd.DataFrame:
    ex = projected_exog_future.copy()
    if "date" not in ex.columns:
        raise ValueError("projected_exog_future debe incluir columna date")
    ex["date"] = pd.to_datetime(ex["date"], errors="coerce").dt.normalize()
    ex = ensure_sarimax_calendar(ex).sort_values("date").reset_index(drop=True)
    needed = [c for c in cols if c != target]
    missing = [c for c in needed if c not in ex.columns]
    if missing:
        raise ValueError(f"projected_exog_future falta columnas: {missing}")
    fut = ex[["date"] + needed].head(int(h)).copy()
    for c in needed:
        fut[c] = pd.to_numeric(fut[c], errors="coerce").astype(float)
    if len(fut) < int(h):
        raise ValueError(f"projected_exog_future corto: {len(fut)} < h={h}")
    if fut[needed].isna().any().any():
        raise ValueError("projected_exog_future contiene NaN en columnas requeridas")
    return fut


def _var_forecast_endogenous(
    res: Any,
    y_levels: pd.DataFrame,
    cols: list[str],
    target: str,
    h: int,
) -> np.ndarray:
    d = y_levels[cols].diff().iloc[1:].dropna(how="any")
    p = int(res.k_ar)
    hist = d.values.astype(float)
    last = hist[-p:, :]
    fc_d = res.forecast(last, steps=int(h))
    y_T = float(y_levels[target].iloc[-1])
    d0 = np.asarray(fc_d, dtype=float)[:, 0]
    return y_T + np.cumsum(d0)


def _var_forecast_projected_b1(
    res: Any,
    y_levels: pd.DataFrame,
    cols: list[str],
    target: str,
    h: int,
    future_levels: pd.DataFrame,
) -> np.ndarray:
    """Pronóstico recursivo: solo Δprecio del VAR; resto fijado por B1 + calendario."""
    d = y_levels[cols].diff().iloc[1:].dropna(how="any")
    p = int(res.k_ar)
    hist = d.values.astype(float)
    target_idx = cols.index(target)
    prev_levels = {c: float(y_levels[c].iloc[-1]) for c in cols}
    price_level = prev_levels[target]
    preds: list[float] = []

    for t in range(int(h)):
        fc = res.forecast(hist[-p:], steps=1)
        fc_row = np.asarray(fc[0] if getattr(fc, "ndim", 1) > 1 else fc, dtype=float).ravel()
        price_diff = float(fc_row[target_idx])
        new_diff = np.zeros(len(cols), dtype=float)
        new_diff[target_idx] = price_diff

        row_fut = future_levels.iloc[t]
        for j, c in enumerate(cols):
            if j == target_idx:
                continue
            lvl = float(row_fut[c])
            new_diff[j] = lvl - prev_levels[c]
            prev_levels[c] = lvl

        price_level += price_diff
        prev_levels[target] = price_level
        preds.append(price_level)
        hist = np.vstack([hist, new_diff.reshape(1, -1)])

    return np.asarray(preds, dtype=float)


def var_forecast_and_fit(
    train_df: pd.DataFrame,
    h: int,
    target: str,
    *,
    maxlags: int = 14,
    ensure_sarimax_calendar: Any | None = None,
    normalize_column_names: Any | None = None,
    projected_exog_future: pd.DataFrame | None = None,
    exog_futr_source: Literal["projected", "endogenous"] = "projected",
) -> tuple[np.ndarray | None, Any | None, str | None]:
    """
    VAR en diferencias: pronóstico H pasos del **nivel** del precio + objeto ajustado.

    Con ``exog_futr_source=\"projected\"`` (defecto) y ``projected_exog_future`` (Bloque 1),
    las covariables futuras quedan fijadas (demanda, embalse, ENSO, aportes, Fourier) y solo
    se pronostica la ecuación del precio — alineado con SARIMAX / N-HiTS v6.

    ``ensure_sarimax_calendar`` y ``normalize_column_names`` deben ser las del proyecto
    (inyectadas desde el notebook para no importar ciclos).
    """
    if ensure_sarimax_calendar is None or normalize_column_names is None:
        raise ValueError("pasar ensure_sarimax_calendar y normalize_column_names")

    from statsmodels.tsa.api import VAR

    try:
        tr, cols = _var_prepare_levels(
            train_df,
            target,
            ensure_sarimax_calendar=ensure_sarimax_calendar,
            normalize_column_names=normalize_column_names,
        )
    except ValueError as e:
        return None, None, str(e)

    y = tr[cols]
    d = y.diff().iloc[1:].dropna(how="any")
    if len(d) < maxlags + 50:
        return None, None, f"muestra VAR corta tras diff: n={len(d)}"

    use_projected = str(exog_futr_source).lower() == "projected"
    if use_projected and projected_exog_future is None:
        return None, None, "exog_futr_source=projected requiere projected_exog_future (Bloque 1)"

    try:
        res = VAR(d).fit(maxlags=maxlags, ic="aic")
        if use_projected:
            fut_levels = _var_build_future_levels(
                projected_exog_future,  # type: ignore[arg-type]
                cols,
                target,
                int(h),
                ensure_sarimax_calendar=ensure_sarimax_calendar,
            )
            pred = _var_forecast_projected_b1(res, tr, cols, target, int(h), fut_levels)
        else:
            pred = _var_forecast_endogenous(res, tr, cols, target, int(h))
        return pred, res, None
    except Exception as e:
        return None, None, str(e)
