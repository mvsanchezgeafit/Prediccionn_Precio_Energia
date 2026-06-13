"""
Volatilidad condicional GARCH para regímenes de riesgo y **features** del clasificador de picos.

- σ_t in-sample: ``sqrt(conditional_volatility)`` del ajuste ``arch``.
- Umbrales de percentil (p50, p90) y z-score se calculan **solo** sobre la σ in-sample del entrenamiento
  (para pronóstico se reutilizan esas estadísticas → sin mirar σ futura observada).

Funciones añadidas para notebooks y diagnóstico: ``enrich_volatility_columns``, ``build_train_volatility_panel``,
``forecast_risk_features_dataframe``, ``spike_vol_conditional_probabilities``, y ``plot_price_vs_sigma`` con
``spike_mask`` opcional.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Orden fijo: debe coincidir con ``HybridSpikeEnsembleBundle.vol_dim`` / matrices apiladas
VOL_SPIKE_FEATURE_NAMES: tuple[str, ...] = (
    "garch_sigma_t",
    "garch_vol_zscore",
    "garch_roll_mean_sigma_30",
    "garch_roll_std_sigma_30",
    "garch_high_vol_pct90",
    "garch_high_vol_z15",
    "garch_risk_level_num",
)


def conditional_sigma_in_sample(garch_result: Any) -> np.ndarray:
    """σ_t = sqrt(h_t) in-sample del objeto ajustado ``arch``."""
    v = np.asarray(garch_result.conditional_volatility, dtype=float)
    return np.sqrt(np.maximum(v, 1e-18))


def align_sigma_series(sigma: np.ndarray, target_len: int, *, align: str = "tail") -> np.ndarray:
    """
    Alinea un vector σ (longitud del ajuste GARCH) a ``target_len`` filas (p. ej. ``feat_df``).

    ``tail``: rellena NaN al inicio y coloca σ al final (últimos ``len(sigma)`` días alineados al train).
    """
    s = np.asarray(sigma, dtype=float).ravel()
    n = int(target_len)
    if n <= 0:
        return s
    out = np.full(n, np.nan, dtype=float)
    if align == "tail":
        m = min(len(s), n)
        out[n - m :] = s[-m:]
    else:
        m = min(len(s), n)
        out[:m] = s[:m]
    return out


def build_in_sample_volatility_features(
    sigma: np.ndarray,
    *,
    roll_window: int = 30,
    z_threshold: float = 1.5,
    high_vol_percentile: float = 90.0,
) -> pd.DataFrame:
    """
    Tabla alineada día a día con:

    - ``sigma_t``
    - ``vol_zscore`` = (σ - mean(σ)) / std(σ) con media/desv **muestral in-sample** (solo σ finita)
    - media móvil y desv. móvil de σ
    - ``high_vol_flag_pct90``: σ > percentil(high_vol_percentile)
    - ``high_vol_flag_z15``: vol_zscore > z_threshold
    - ``risk_level`` ∈ {low, medium, high} y ``risk_level_num`` ∈ {0,1,2}
    """
    s = pd.Series(np.asarray(sigma, dtype=float))
    sigma_t = s.copy()
    mu = float(np.nanmean(sigma_t))
    sd = float(np.nanstd(sigma_t))
    if sd < 1e-12:
        sd = 1.0
    vol_z = (sigma_t - mu) / sd
    rm = sigma_t.rolling(int(roll_window), min_periods=max(3, int(roll_window) // 3)).mean()
    rs = sigma_t.rolling(int(roll_window), min_periods=max(3, int(roll_window) // 3)).std()
    thr_hi = float(np.nanpercentile(sigma_t.to_numpy(dtype=float), high_vol_percentile))
    thr_mid = float(np.nanpercentile(sigma_t.to_numpy(dtype=float), 50.0))
    hi_pct = (sigma_t > thr_hi).astype(float)
    hi_z = (vol_z > float(z_threshold)).astype(float)
    risk_num = np.where(
        sigma_t.to_numpy() < thr_mid,
        0,
        np.where(sigma_t.to_numpy() <= thr_hi, 1, 2),
    )
    risk_lbl = np.where(risk_num == 0, "low", np.where(risk_num == 1, "medium", "high"))
    return pd.DataFrame(
        {
            "sigma_t": sigma_t.to_numpy(dtype=float),
            "vol_zscore": vol_z.to_numpy(dtype=float),
            "rolling_mean_sigma": rm.to_numpy(dtype=float),
            "rolling_std_sigma": rs.to_numpy(dtype=float),
            "high_vol_flag_pct90": hi_pct.to_numpy(dtype=float),
            "high_vol_flag_z15": hi_z.to_numpy(dtype=float),
            "risk_level": risk_lbl,
            "risk_level_num": risk_num.astype(float),
        }
    )


def volatility_feature_matrix_for_spike(vol_df: pd.DataFrame) -> np.ndarray:
    """Matriz (n, 7) para LGBM; NaN → 0."""
    cols = [
        "sigma_t",
        "vol_zscore",
        "rolling_mean_sigma",
        "rolling_std_sigma",
        "high_vol_flag_pct90",
        "high_vol_flag_z15",
        "risk_level_num",
    ]
    X = vol_df[cols].to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)
    return X


def build_forecast_volatility_matrix(
    sigma_hat: np.ndarray,
    sigma_train: np.ndarray,
    *,
    roll_window: int = 30,
    z_threshold: float = 1.5,
    high_vol_percentile: float = 90.0,
) -> np.ndarray:
    """
    Características de **horizonte futuro** (una fila por paso h=1..H).

    Usa percentiles y (μ,σ) de ``sigma_train`` (in-sample) para z-scores y regímenes, sin usar σ futura observada.
    """
    sh = np.asarray(sigma_hat, dtype=float).ravel()
    st = np.asarray(sigma_train, dtype=float)
    st = st[np.isfinite(st)]
    mu = float(np.mean(st)) if st.size else 0.0
    sd = float(np.std(st)) if st.size else 1.0
    if sd < 1e-12:
        sd = 1.0
    thr_hi = float(np.percentile(st, high_vol_percentile)) if st.size else float("nan")
    thr_mid = float(np.percentile(st, 50.0)) if st.size else float("nan")

    z_hat = (sh - mu) / sd
    # rolling solo sobre la trayectoria **pronosticada** (interpretación local de riesgo OOS)
    sser = pd.Series(sh)
    rm = sser.rolling(int(roll_window), min_periods=max(3, int(roll_window) // 3)).mean().to_numpy(dtype=float)
    rs = sser.rolling(int(roll_window), min_periods=max(3, int(roll_window) // 3)).std().to_numpy(dtype=float)
    hi_pct = (sh > thr_hi).astype(float) if np.isfinite(thr_hi) else np.zeros_like(sh)
    hi_z = (z_hat > float(z_threshold)).astype(float)
    risk_num = np.where(sh < thr_mid, 0, np.where(sh <= thr_hi, 1, 2)).astype(float)

    mat = np.column_stack([sh, z_hat, rm, rs, hi_pct, hi_z, risk_num])
    mat = np.where(np.isfinite(mat), mat, 0.0)
    return mat


def future_risk_flag_from_matrix(vol_forecast_matrix: np.ndarray) -> np.ndarray:
    """
    Vector longitud H: 1.0 si ``high_vol_pct90`` o ``high_vol_z15`` en la fila de paso, si no 0.
    (columnas 4 y 5 en el esquema de :func:`build_forecast_volatility_matrix`).
    """
    m = np.asarray(vol_forecast_matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 6:
        return np.zeros(m.shape[0], dtype=float)
    return np.maximum(m[:, 4], m[:, 5])


def attach_volatility_columns_to_train_df(
    train_df: pd.DataFrame,
    vol_df: pd.DataFrame,
) -> pd.DataFrame:
    """Añade columnas diagnóstico al panel diario (misma longitud que ``train_df``)."""
    out = train_df.reset_index(drop=True).copy()
    for c in vol_df.columns:
        out[f"garch_{c}"] = vol_df[c].to_numpy()
    return out


def plot_price_vs_sigma(
    dates: pd.Series | np.ndarray,
    price: np.ndarray,
    sigma: np.ndarray,
    high_vol_mask: np.ndarray | None = None,
    *,
    spike_mask: np.ndarray | None = None,
    title: str = "Precio vs σ GARCH",
) -> Any:
    """
    Gráfico opcional: precio y volatilidad condicional; sombrea periodos ``high_vol_mask`` (True/1).
    Si ``spike_mask`` (booleano alineado), marca picos con triángulos naranjas.
    Devuelve la figura matplotlib.
    """
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 4), constrained_layout=True)
    x = pd.to_datetime(dates)
    ax1.plot(x, price, color="black", lw=1.2, label="Precio")
    ax1.set_ylabel("Precio")
    ax2 = ax1.twinx()
    ax2.plot(x, sigma, color="tab:blue", lw=1.0, alpha=0.85, label="σ GARCH")
    ax2.set_ylabel("σ")
    if high_vol_mask is not None:
        m = np.asarray(high_vol_mask).astype(bool)
        if np.any(m):
            ax1.scatter(
                np.asarray(x)[m],
                np.asarray(price, dtype=float)[m],
                color="red",
                s=14,
                alpha=0.55,
                label="Alta vol.",
                zorder=5,
            )
    if spike_mask is not None:
        sm = np.asarray(spike_mask).astype(bool)
        if np.any(sm):
            ax1.scatter(
                np.asarray(x)[sm],
                np.asarray(price, dtype=float)[sm],
                color="tab:orange",
                s=36,
                marker="^",
                alpha=0.75,
                label="Pico (regla)",
                zorder=6,
            )
    ax1.set_title(title)
    fig.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), bbox_transform=ax1.transAxes, fontsize=8)
    return fig


def enrich_volatility_columns(vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas de diagnóstico sin alterar las 7 usadas por ``volatility_feature_matrix_for_spike``:

    - ``vol_percentile``: rango percentil empírico de σ (0–1)
    - ``extreme_vol_flag``: σ > percentil 95 de la muestra in-sample
    - ``high_vol_flag``: copia de ``high_vol_flag_pct90``
    - ``zscore_flag``: copia de ``high_vol_flag_z15``
    """
    out = vol_df.copy()
    s = pd.Series(out["sigma_t"].to_numpy(dtype=float))
    out["vol_percentile"] = s.rank(pct=True, method="average").to_numpy(dtype=float)
    thr95 = float(np.nanpercentile(s.to_numpy(), 95.0))
    out["extreme_vol_flag"] = (s.to_numpy(dtype=float) > thr95).astype(float)
    out["high_vol_flag"] = out["high_vol_flag_pct90"].to_numpy(dtype=float).copy()
    out["zscore_flag"] = out["high_vol_flag_z15"].to_numpy(dtype=float).copy()
    return out


def build_train_volatility_panel(
    train_df: pd.DataFrame,
    garch_result: Any,
    *,
    date_col: str = "date",
    align: str = "tail",
    roll_window: int = 30,
    z_threshold: float = 1.5,
    high_vol_percentile: float = 90.0,
) -> pd.DataFrame:
    """
    Panel alineado al **train** (una fila por día): σ in-sample del GARCH y regímenes.

    ``garch_result`` debe estar ajustado sobre residuos cuya longitud se alinea por ``tail``
    a ``len(train_df)`` (mismo criterio que ``align_sigma_series``).
    """
    sig = conditional_sigma_in_sample(garch_result)
    n = int(len(train_df))
    sig_a = align_sigma_series(sig, n, align=align)
    base = build_in_sample_volatility_features(
        sig_a,
        roll_window=roll_window,
        z_threshold=z_threshold,
        high_vol_percentile=high_vol_percentile,
    )
    out = enrich_volatility_columns(base)
    out.insert(0, date_col, pd.to_datetime(train_df[date_col].values))
    return out


def forecast_risk_features_dataframe(
    sigma_hat: np.ndarray,
    sigma_insample_ref: np.ndarray,
    *,
    z_threshold: float = 1.5,
    high_vol_percentile: float = 90.0,
    extreme_vol_percentile: float = 95.0,
) -> pd.DataFrame:
    """
    Una fila por paso del horizonte (pronóstico OOS). Estadísticos (μ, σ, percentiles) solo de
    ``sigma_insample_ref`` → sin fuga de σ futura observada.

    Columnas: ``sigma_hat_t``, ``future_vol_level``, ``future_vol_zscore``, ``future_vol_percentile``,
    ``future_high_vol_flag``, ``future_extreme_vol_flag``, ``future_zscore_flag``,
    ``future_risk_level``, ``future_risk_level_num``.
    """
    sh = np.asarray(sigma_hat, dtype=float).ravel()
    st = np.asarray(sigma_insample_ref, dtype=float)
    st = st[np.isfinite(st)]
    if st.size == 0:
        return pd.DataFrame(
            {
                "sigma_hat_t": sh,
                "future_vol_level": sh,
                "future_vol_zscore": np.full_like(sh, np.nan),
                "future_vol_percentile": np.full_like(sh, np.nan),
                "future_high_vol_flag": np.zeros_like(sh, dtype=int),
                "future_extreme_vol_flag": np.zeros_like(sh, dtype=int),
                "future_zscore_flag": np.zeros_like(sh, dtype=int),
                "future_risk_level": np.full(sh.shape[0], "low", dtype=object),
                "future_risk_level_num": np.zeros_like(sh, dtype=int),
            }
        )
    mu = float(np.mean(st))
    sd = float(np.std(st))
    if sd < 1e-12:
        sd = 1.0
    thr50 = float(np.percentile(st, 50.0))
    thr90 = float(np.percentile(st, high_vol_percentile))
    thr95 = float(np.percentile(st, extreme_vol_percentile))
    fvz = (sh - mu) / sd
    vp = np.mean(st[:, None] <= sh[None, :], axis=0).astype(float)
    fh = (sh > thr90).astype(int)
    fe = (sh > thr95).astype(int)
    fz = (fvz > float(z_threshold)).astype(int)
    risk_num = np.where(sh < thr50, 0, np.where(sh <= thr90, 1, 2)).astype(int)
    risk_lbl = np.where(risk_num == 0, "low", np.where(risk_num == 1, "medium", "high"))
    return pd.DataFrame(
        {
            "sigma_hat_t": sh,
            "future_vol_level": sh,
            "future_vol_zscore": fvz,
            "future_vol_percentile": vp,
            "future_high_vol_flag": fh,
            "future_extreme_vol_flag": fe,
            "future_zscore_flag": fz,
            "future_risk_level": risk_lbl.astype(str),
            "future_risk_level_num": risk_num,
        }
    )


def spike_vol_conditional_probabilities(
    spike_binary: np.ndarray,
    high_vol_binary: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    Frecuencias empíricas P(pico | alta vol) y P(pico | no alta vol), con P(pico) marginal.

    ``spike_binary`` y ``high_vol_binary`` deben estar alineados (misma longitud).
    """
    s = np.asarray(spike_binary, dtype=float).ravel()
    h = np.asarray(high_vol_binary, dtype=float).ravel()
    n = min(s.size, h.size)
    if n == 0:
        return {
            "n": 0.0,
            "p_spike": float("nan"),
            "p_spike_given_high_vol": float("nan"),
            "p_spike_given_low_vol": float("nan"),
            "p_high_vol": float("nan"),
        }
    s = s[:n]
    h = h[:n]
    sb = (s > 0.5).astype(bool)
    hb = (h > 0.5).astype(bool)
    n_hi = int(np.sum(hb))
    n_lo = int(np.sum(~hb))
    return {
        "n": float(n),
        "p_spike": float(np.mean(sb)),
        "p_spike_given_high_vol": float(np.sum(sb & hb) / max(n_hi, eps)),
        "p_spike_given_low_vol": float(np.sum(sb & ~hb) / max(n_lo, eps)),
        "p_high_vol": float(np.mean(hb)),
    }
