"""
Métricas de evaluación para precio máximo diario: estándar y foco en colas / extremos.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-9


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    mask = np.isfinite(diff)
    if not np.any(mask):
        return float("nan")
    sq = diff[mask] ** 2
    if not np.all(np.isfinite(sq)):
        return float("inf")
    return float(np.sqrt(np.mean(sq)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE en %; ignora filas donde |y_true| ≤ EPS (evita división por cero)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > EPS
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    sMAPE en % (escala simétrica); denom = (|y|+|ŷ|)/2, ignora filas con denom ≈ 0.
    Más estable que MAPE cuando hay valores cercanos a cero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y, p = y_true[mask], y_pred[mask]
    denom = (np.abs(y) + np.abs(p)) / 2.0
    mask2 = denom > EPS
    if not np.any(mask2):
        return float("nan")
    return float(np.mean(np.abs(y[mask2] - p[mask2]) / denom[mask2]) * 100.0)


def relative_improvement_pct(base_mae: float, new_mae: float) -> float:
    """
    Reducción relativa de MAE respecto al baseline (positivo = el nuevo modelo mejora).
    """
    if not np.isfinite(base_mae) or base_mae <= 0 or not np.isfinite(new_mae):
        return float("nan")
    return float((base_mae - new_mae) / base_mae * 100.0)


def metrics_standard(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "mae": mae(yt, yp),
        "rmse": rmse(yt, yp),
        "mape_pct": mape(yt, yp),
        "smape_pct": smape(yt, yp),
        "n": int(np.sum(mask)),
    }


def metrics_error_scaled_by_sigma(
    errors: np.ndarray,
    sigma: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    MAE/RMSE/MAPE «en unidades de σ»: ``e`` y ``σ`` deben estar en la **misma escala**
    (p. ej. residuo en log vs σ GARCH en log; o error en COP vs σ en COP).

    - ``mae_scaled`` = mean(|e|/σ)
    - ``rmse_scaled`` = sqrt(mean((e/σ)²)); bajo gaussianidad i.i.d. bien calibrada ≈ 1
    - ``mape_sigma_pct`` = mean(|e|/σ)·100 (interpretable como error medio en % de una σ condicional)
    """
    e = np.asarray(errors, dtype=float).ravel()
    s = np.asarray(sigma, dtype=float).ravel()
    if s.size == 1:
        s = np.full_like(e, float(s[0]), dtype=float)
    if e.shape != s.shape:
        raise ValueError(f"errors shape {e.shape} != sigma shape {s.shape}")
    mask = np.isfinite(e) & np.isfinite(s) & (np.abs(s) > eps)
    if not np.any(mask):
        return {
            "mae_scaled": float("nan"),
            "rmse_scaled": float("nan"),
            "mape_sigma_pct": float("nan"),
            "n": 0,
        }
    ee, ss = e[mask], s[mask]
    z = ee / ss
    return {
        "mae_scaled": float(np.mean(np.abs(z))),
        "rmse_scaled": float(np.sqrt(np.mean(z**2))),
        "mape_sigma_pct": float(np.mean(np.abs(z)) * 100.0),
        "n": int(np.sum(mask)),
    }


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """
    Pérdida pinball (check function) para el cuantil de nivel ``alpha`` ∈ (0,1).
    L_α(y,q) = (α - 𝟙{y<q})(y - q); promedio sobre muestras válidas.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y, q = y_true[mask], y_pred[mask]
    e = y - q
    loss = np.where(e >= 0, float(alpha) * e, (float(alpha) - 1.0) * e)
    return float(np.mean(loss))


def empirical_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Proporción de observaciones con y ∈ [lower, upper] (intervalos inclusivos)."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    y, lo, hi = y_true[mask], lower[mask], upper[mask]
    inside = (y >= lo) & (y <= hi)
    return float(np.mean(inside)) if inside.size else float("nan")


def interval_score_mean(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    alpha: float,
) -> float:
    """
    Interval score de Gneiting–Raftery para un intervalo central (1-α).

    S = (U-L) + (2/α)(L-Y)_+ + (2/α)(Y-U)_+.
    Para banda P10–P90 usar alpha=0.2 (cobertura nominal 80%).
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    y, L, U = y_true[mask], lower[mask], upper[mask]
    width = U - L
    pen = np.maximum(L - y, 0.0) + np.maximum(y - U, 0.0)
    s = width + (2.0 / float(alpha)) * pen
    return float(np.mean(s)) if s.size else float("nan")


def lgbm_quantile_oos_metrics(
    y_true: np.ndarray,
    pred_p10: np.ndarray,
    pred_p50: np.ndarray,
    pred_p90: np.ndarray,
) -> dict[str, float]:
    """
    Métricas fuera de muestra para pronóstico cuantílico recursivo (P10, P50, P90).
    Incluye pinball por cuantil, cobertura empírica de la banda central 80 % y
    interval score medio para [P10, P90].
    """
    y_true = np.asarray(y_true, dtype=float)
    p10 = np.asarray(pred_p10, dtype=float)
    p50 = np.asarray(pred_p50, dtype=float)
    p90 = np.asarray(pred_p90, dtype=float)
    alpha_band = 0.2  # 80 % central entre P10 y P90
    return {
        "pinball_p10": pinball_loss(y_true, p10, 0.1),
        "pinball_p50": pinball_loss(y_true, p50, 0.5),
        "pinball_p90": pinball_loss(y_true, p90, 0.9),
        "coverage_p10_p90_empirical": empirical_interval_coverage(y_true, p10, p90),
        "nominal_coverage_p10_p90": 0.8,
        "interval_score_p10_p90_mean": interval_score_mean(
            y_true, p10, p90, alpha=alpha_band
        ),
        "mae_median_vs_y": mae(y_true, p50),
        "rmse_median_vs_y": rmse(y_true, p50),
    }


def metrics_high_tail(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """
    Métricas solo en días con precio observado >= threshold (p. ej. percentil 95 del train).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true >= threshold)
    if not np.any(mask):
        return {
            "mae_tail": float("nan"),
            "rmse_tail": float("nan"),
            "mape_tail_pct": float("nan"),
            "smape_tail_pct": float("nan"),
            "n_tail": 0,
        }
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "mae_tail": mae(yt, yp),
        "rmse_tail": rmse(yt, yp),
        "mape_tail_pct": mape(yt, yp),
        "smape_tail_pct": smape(yt, yp),
        "n_tail": int(np.sum(mask)),
    }


def metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    p95_threshold: float | None = None,
) -> dict[str, Any]:
    """Informe completo: estándar + cola alta si se da umbral (p95 histórico)."""
    out: dict[str, Any] = metrics_standard(y_true, y_pred)
    if p95_threshold is not None:
        tail = metrics_high_tail(y_true, y_pred, p95_threshold)
        out.update(tail)
        out["p95_threshold_used"] = float(p95_threshold)
    return out


def dataframe_metrics(
    df: pd.DataFrame,
    *,
    actual_col: str = "y_true",
    pred_col: str = "y_pred",
    p95_from_train: float | None = None,
) -> dict[str, Any]:
    return metrics_report(
        df[actual_col].values,
        df[pred_col].values,
        p95_threshold=p95_from_train,
    )
