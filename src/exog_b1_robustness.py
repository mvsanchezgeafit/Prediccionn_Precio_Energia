"""
Experimentos controlados del **Bloque 1** (proyección exógenas) para SARIMAX TVT.

- Barrido acotado de ``volatility_strength``, ``stl_trend_decay``, ``scenario_aggregation``.
- Diagnóstico **projected vs oracle** (oráculo solo diagnóstico; sin fuga en pipeline principal).
- Métricas: MAE/RMSE/MAPE, ``std_ratio``, correlación pred-real.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Literal

import numpy as np
import pandas as pd

from hybrid_direct_30d import block1_build_exog_future, block2_sarimax_fit_forecast, from_log_price
from ml_metrics import metrics_standard
from sarimax_exog_robustness import build_oracle_exog_future

ScenarioAgg = Literal["mean", "percentile_75", "single_path_seeded", "median"]


def dynamic_forecast_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    n = int(min(len(y), len(p)))
    if n == 0:
        return {"std_y": float("nan"), "std_pred": float("nan"), "std_ratio": float("nan")}
    ys = float(np.std(y[:n]))
    ps = float(np.std(p[:n]))
    ratio = ps / ys if ys > 1e-9 else float("nan")
    return {"std_y": ys, "std_pred": ps, "std_ratio": ratio}


def forecast_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    n = int(min(len(y), len(p)))
    if n < 2:
        return float("nan")
    m = np.isfinite(y[:n]) & np.isfinite(p[:n])
    if int(np.sum(m)) < 2:
        return float("nan")
    return float(np.corrcoef(y[:n][m], p[:n][m])[0, 1])


def _sarimax_pred_cop(
    train_df: pd.DataFrame,
    ex_f: pd.DataFrame,
    *,
    target: str,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    exog_cols: list[str],
    standardize_exog: bool,
    h_eff: int,
) -> np.ndarray:
    b2 = block2_sarimax_fit_forecast(
        train_df,
        ex_f,
        target=target,
        order=order,
        seasonal_order=seasonal_order,
        exog_cols=exog_cols,
        standardize_exog=standardize_exog,
    )
    return from_log_price(np.asarray(b2["sarimax_pred_future_log"], dtype=float))[:h_eff]


def evaluate_b1_sarimax_window(
    *,
    train_df: pd.DataFrame,
    future_panel: pd.DataFrame,
    cutoff_end: pd.Timestamp,
    y_true: np.ndarray,
    h_eff: int,
    target: str,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    exog_cols: list[str],
    standardize_exog: bool,
    b1_kw: dict[str, Any],
    mode: Literal["projected", "oracle"],
) -> dict[str, Any]:
    """Una ventana (val o test): exógenas projected u oracle + SARIMAX fijo."""
    if mode == "oracle":
        ex_f = build_oracle_exog_future(train_df, future_panel, h=h_eff, exog_cols=exog_cols)
    else:
        kw = {**b1_kw, "volatility_mode": b1_kw.get("volatility_mode", "residual_bootstrap")}
        ex_f = block1_build_exog_future(train_df, days=h_eff, **kw)
    pred = _sarimax_pred_cop(
        train_df,
        ex_f,
        target=target,
        order=order,
        seasonal_order=seasonal_order,
        exog_cols=exog_cols,
        standardize_exog=standardize_exog,
        h_eff=h_eff,
    )
    y = np.asarray(y_true, dtype=float).ravel()[: len(pred)]
    m = metrics_standard(y, pred)
    dyn = dynamic_forecast_stats(y, pred)
    return {
        "pred": pred,
        "exog_future": ex_f,
        "mae": m["mae"],
        "rmse": m["rmse"],
        "mape_pct": m["mape_pct"],
        "corr": forecast_correlation(y, pred),
        **dyn,
    }


def run_b1_config_case(
    *,
    tag: str,
    train_df: pd.DataFrame,
    train_val_df: pd.DataFrame,
    val_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    spl_date_train_end: pd.Timestamp,
    spl_date_val_end: pd.Timestamp,
    y_val: np.ndarray,
    y_test: np.ndarray,
    h_eff: int,
    target: str,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    exog_cols: list[str],
    standardize_exog: bool,
    b1_kw: dict[str, Any],
) -> dict[str, Any]:
    """Evalúa projected en Val y Test con una configuración B1."""
    rv = evaluate_b1_sarimax_window(
        train_df=train_df,
        future_panel=val_panel,
        cutoff_end=spl_date_train_end,
        y_true=y_val,
        h_eff=h_eff,
        target=target,
        order=order,
        seasonal_order=seasonal_order,
        exog_cols=exog_cols,
        standardize_exog=standardize_exog,
        b1_kw=b1_kw,
        mode="projected",
    )
    rt = evaluate_b1_sarimax_window(
        train_df=train_val_df,
        future_panel=test_panel,
        cutoff_end=spl_date_val_end,
        y_true=y_test,
        h_eff=h_eff,
        target=target,
        order=order,
        seasonal_order=seasonal_order,
        exog_cols=exog_cols,
        standardize_exog=standardize_exog,
        b1_kw=b1_kw,
        mode="projected",
    )
    row = {
        "tag": tag,
        "volatility_strength": float(b1_kw.get("volatility_strength", float("nan"))),
        "stl_trend_decay": float(b1_kw.get("stl_trend_decay", float("nan"))),
        "scenario_aggregation": str(b1_kw.get("scenario_aggregation", "")),
        "mae_val": rv["mae"],
        "rmse_val": rv["rmse"],
        "mape_val": rv["mape_pct"],
        "std_ratio_val": rv["std_ratio"],
        "corr_val": rv["corr"],
        "mae_test": rt["mae"],
        "rmse_test": rt["rmse"],
        "mape_test": rt["mape_pct"],
        "std_ratio_test": rt["std_ratio"],
        "corr_test": rt["corr"],
        "mae_mean": (float(rv["mae"]) + float(rt["mae"])) / 2.0,
        "std_ratio_mean": (float(rv["std_ratio"]) + float(rt["std_ratio"])) / 2.0,
    }
    row["_val_detail"] = rv
    row["_test_detail"] = rt
    return row


def run_b1_parameter_sweep(
    *,
    b1_base: dict[str, Any],
    volatility_strengths: tuple[float, ...],
    stl_trend_decays: tuple[float, ...],
    scenario_aggregations: tuple[str, ...],
    train_df: pd.DataFrame,
    train_val_df: pd.DataFrame,
    val_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    spl_date_train_end: pd.Timestamp,
    spl_date_val_end: pd.Timestamp,
    y_val: np.ndarray,
    y_test: np.ndarray,
    h_eff: int,
    target: str,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    exog_cols: list[str],
    standardize_exog: bool,
    include_baseline: bool = True,
    baseline_b1: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Barrido acotado B1 × SARIMAX fijo (sin tocar órdenes)."""
    rows: list[dict[str, Any]] = []
    if include_baseline and baseline_b1 is not None:
        rows.append(
            run_b1_config_case(
                tag="baseline_legacy",
                train_df=train_df,
                train_val_df=train_val_df,
                val_panel=val_panel,
                test_panel=test_panel,
                spl_date_train_end=spl_date_train_end,
                spl_date_val_end=spl_date_val_end,
                y_val=y_val,
                y_test=y_test,
                h_eff=h_eff,
                target=target,
                order=order,
                seasonal_order=seasonal_order,
                exog_cols=exog_cols,
                standardize_exog=standardize_exog,
                b1_kw=dict(baseline_b1),
            )
        )
    for vs, td, agg in product(volatility_strengths, stl_trend_decays, scenario_aggregations):
        kw = {**b1_base, "volatility_strength": float(vs), "stl_trend_decay": float(td), "scenario_aggregation": str(agg)}
        tag = f"vs{vs}_td{td}_{agg}"
        rows.append(
            run_b1_config_case(
                tag=tag,
                train_df=train_df,
                train_val_df=train_val_df,
                val_panel=val_panel,
                test_panel=test_panel,
                spl_date_train_end=spl_date_train_end,
                spl_date_val_end=spl_date_val_end,
                y_val=y_val,
                y_test=y_test,
                h_eff=h_eff,
                target=target,
                order=order,
                seasonal_order=seasonal_order,
                exog_cols=exog_cols,
                standardize_exog=standardize_exog,
                b1_kw=kw,
            )
        )
    out_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    df = pd.DataFrame(out_rows)
    df.attrs["detail_rows"] = rows
    return df


def run_projected_vs_oracle_diagnostic(
    *,
    b1_kw: dict[str, Any],
    train_df: pd.DataFrame,
    train_val_df: pd.DataFrame,
    val_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    y_val: np.ndarray,
    y_test: np.ndarray,
    h_eff: int,
    target: str,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    exog_cols: list[str],
    standardize_exog: bool,
) -> pd.DataFrame:
    """Compara projected vs oracle (solo diagnóstico)."""
    rows: list[dict[str, Any]] = []
    for ventana, tr, panel, y in (
        ("Val", train_df, val_panel, y_val),
        ("Test", train_val_df, test_panel, y_test),
    ):
        for mode in ("projected", "oracle"):
            ev = evaluate_b1_sarimax_window(
                train_df=tr,
                future_panel=panel,
                cutoff_end=pd.Timestamp(tr["date"].max()),
                y_true=y,
                h_eff=h_eff,
                target=target,
                order=order,
                seasonal_order=seasonal_order,
                exog_cols=exog_cols,
                standardize_exog=standardize_exog,
                b1_kw=b1_kw,
                mode=mode,  # type: ignore[arg-type]
            )
            rows.append(
                {
                    "Ventana": ventana,
                    "modo_exog": mode,
                    "MAE": ev["mae"],
                    "RMSE": ev["rmse"],
                    "MAPE_pct": ev["mape_pct"],
                    "std_ratio": ev["std_ratio"],
                    "corr": ev["corr"],
                }
            )
    return pd.DataFrame(rows)


def score_b1_candidate(row: pd.Series, *, baseline_mae_mean: float | None = None) -> float:
    """
    Menor = mejor. Balance error (MAE Val+Test) y amplitud (std_ratio hacia ~0.7–1.0).
    Penaliza std_ratio < 0.15 (muy plano) y MAE muy por encima del baseline.
    """
    mae = float(row.get("mae_mean", float("nan")))
    sr = float(row.get("std_ratio_mean", float("nan")))
    score = mae
    if np.isfinite(sr):
        if sr < 0.15:
            score += 25.0 * (0.15 - sr)
        elif sr > 1.25:
            score += 15.0 * (sr - 1.25)
        else:
            score -= 3.0 * min(sr, 1.0)
    if baseline_mae_mean is not None and np.isfinite(baseline_mae_mean) and np.isfinite(mae):
        if mae > baseline_mae_mean * 1.08:
            score += 8.0 * (mae / baseline_mae_mean - 1.08)
    return float(score)


def _spike_mask(y: np.ndarray, *, pct: float = 75.0) -> np.ndarray:
    """Días con |y − mediana| por encima del percentil ``pct`` (picos / valles)."""
    y = np.asarray(y, dtype=float).ravel()
    dev = np.abs(y - np.nanmedian(y))
    thr = float(np.nanpercentile(dev, pct)) if np.any(np.isfinite(dev)) else float("nan")
    if not np.isfinite(thr):
        return np.zeros(len(y), dtype=bool)
    return dev >= thr


def _mae_subset(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    m = mask[: min(len(y), len(p))]
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(y[: len(m)][m] - p[: len(m)][m])))


def interpret_b1_experiment(
    sweep_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    *,
    best_row: pd.Series | None = None,
) -> str:
    """
    Texto interpretativo: amplitud, suavización, sensibilidad B1, descomposición oracle vs projected.
    """
    df = sweep_df.copy()
    base = df[df["tag"] == "baseline_legacy"]
    if best_row is None:
        best_row, _ = recommend_b1_setup(df)
    lines: list[str] = ["=== Interpretación Bloque 1 (§1.c) ==="]

    if len(base):
        b = base.iloc[0]
        br = best_row
        lines.append(
            f"Antes (baseline_legacy): MAE Val/Test={b['mae_val']:.1f}/{b['mae_test']:.1f} | "
            f"std_ratio Val/Test={b['std_ratio_val']:.3f}/{b['std_ratio_test']:.3f}"
        )
        lines.append(
            f"Después (recomendado {br['tag']}): MAE Val/Test={br['mae_val']:.1f}/{br['mae_test']:.1f} | "
            f"std_ratio Val/Test={br['std_ratio_val']:.3f}/{br['std_ratio_test']:.3f}"
        )
        d_sr = float(br["std_ratio_mean"]) - float(b["std_ratio_mean"])
        d_mae = float(br["mae_mean"]) - float(b["mae_mean"])
        lines.append(
            f"Δ std_ratio_mean={d_sr:+.3f} | Δ MAE_mean={d_mae:+.2f} COP/kWh "
            f"({'más amplitud' if d_sr > 0.02 else 'amplitud similar' if abs(d_sr) <= 0.02 else 'menos amplitud'})"
        )

    sr = float(best_row.get("std_ratio_mean", float("nan")))
    if np.isfinite(sr):
        if sr < 0.15:
            lines.append("Amplitud: muy baja — trayectorias casi planas (std_pred << std_real).")
        elif sr < 0.35:
            lines.append("Amplitud: baja-moderada — algo de variación pero aún suavizada vs real.")
        elif sr < 0.75:
            lines.append("Amplitud: moderada — balance razonable entre suavizado y dinámica.")
        else:
            lines.append("Amplitud: alta — std(pred) se acerca o supera std(real); vigilar sobreajuste OOS.")

    sens_vs = df.groupby("volatility_strength", dropna=False)["std_ratio_mean"].mean()
    sens_td = df.groupby("stl_trend_decay", dropna=False)["std_ratio_mean"].mean()
    sens_ag = df.groupby("scenario_aggregation", dropna=False)["std_ratio_mean"].mean()
    if len(sens_vs) > 1:
        vs_best = sens_vs.idxmax()
        lines.append(
            f"Sensibilidad volatility_strength: std_ratio_mean sube con vs "
            f"(mejor media en vs={vs_best:.2f}, sr={sens_vs.max():.3f})."
        )
    if len(sens_td) > 1:
        td_best = sens_td.idxmax()
        lines.append(
            f"Sensibilidad stl_trend_decay: menor decay → más persistencia "
            f"(mejor media en td={td_best}, sr={sens_td.max():.3f})."
        )
    if len(sens_ag) > 1:
        ag_best = sens_ag.idxmax()
        lines.append(
            f"Sensibilidad scenario_aggregation: '{ag_best}' maximiza amplitud media "
            f"(sr={sens_ag.max():.3f}); mean tiende a suavizar vs p75/single_path."
        )

    for ventana in ("Val", "Test"):
        sub = oracle_df[oracle_df["Ventana"] == ventana]
        if len(sub) < 2:
            continue
        proj = sub[sub["modo_exog"] == "projected"].iloc[0]
        orac = sub[sub["modo_exog"] == "oracle"].iloc[0]
        gap = float(orac["MAE"]) - float(proj["MAE"])
        lines.append(
            f"Oracle vs projected ({ventana}): ΔMAE={gap:+.1f} "
            f"(oracle {'mejor' if gap < 0 else 'peor' if gap > 0 else 'igual'}). "
            f"std_ratio projected/oracle={proj['std_ratio']:.3f}/{orac['std_ratio']:.3f}."
        )
        if gap < -5:
            lines.append(f"  → Cuello de botella en B1 ({ventana}): mejorar proyección exógena ayudaría.")
        elif gap > 5 and float(orac["std_ratio"]) < 0.25:
            lines.append(f"  → Límite SARIMAX ({ventana}): incluso con exógenas reales la curva es plana.")
        else:
            lines.append(f"  → Error repartido entre B1 y SARIMAX lineal ({ventana}).")

    lines.append(
        "Estabilidad OOS: priorizar configs con MAE_test no mucho peor que baseline y std_ratio_test "
        "en [0.25, 0.85]; evitar std_ratio_test > 1.1 salvo evidencia consistente en val."
    )
    return "\n".join(lines)


def spike_mae_comparison(
    y_true: np.ndarray,
    pred_projected: np.ndarray,
    pred_oracle: np.ndarray,
    *,
    spike_pct: float = 75.0,
) -> dict[str, float]:
    """MAE en días pico (|y − mediana| alto) para projected vs oracle."""
    mask = _spike_mask(y_true, pct=spike_pct)
    return {
        "mae_spike_projected": _mae_subset(y_true, pred_projected, mask),
        "mae_spike_oracle": _mae_subset(y_true, pred_oracle, mask),
        "n_spike_days": int(np.sum(mask)),
    }


def recommend_b1_setup(sweep_df: pd.DataFrame) -> tuple[pd.Series, str]:
    """Selecciona mejor fila del barrido y texto interpretativo."""
    df = sweep_df.copy()
    base = df[df["tag"] == "baseline_legacy"]
    b_mae = float(base.iloc[0]["mae_mean"]) if len(base) else None
    df["score"] = df.apply(lambda r: score_b1_candidate(r, baseline_mae_mean=b_mae), axis=1)
    df = df.sort_values(["score", "mae_val", "std_ratio_mean"], ascending=[True, True, False])
    best = df.iloc[0]
    lines = [
        f"Recomendación B1: tag={best['tag']}",
        f"  volatility_strength={best.get('volatility_strength')} | stl_trend_decay={best.get('stl_trend_decay')} | agg={best.get('scenario_aggregation')}",
        f"  MAE Val/Test={best.get('mae_val'):.2f}/{best.get('mae_test'):.2f} | std_ratio Val/Test={best.get('std_ratio_val'):.3f}/{best.get('std_ratio_test'):.3f}",
    ]
    if b_mae is not None:
        delta = float(best["mae_mean"]) - b_mae
        lines.append(f"  ΔMAE_mean vs baseline_legacy: {delta:+.2f} COP/kWh")
    sr = float(best.get("std_ratio_mean", float("nan")))
    if np.isfinite(sr):
        if sr < 0.15:
            amp = "muy baja (curva plana)"
        elif sr < 0.35:
            amp = "baja-moderada"
        elif sr < 0.75:
            amp = "moderada"
        else:
            amp = "alta (más dinámica)"
        lines.append(f"  Amplitud OOS media: {amp} (std_ratio_mean={sr:.3f})")
    lines.append(
        "  Interpretación: si oracle >> projected en MAE pero std_ratio similar, el cuello es B1; "
        "si oracle también falla en picos, el límite es SARIMAX lineal."
    )
    return best, "\n".join(lines)


def plot_b1_price_comparison(
    dates: np.ndarray,
    y_true: np.ndarray,
    pred_projected: np.ndarray,
    pred_oracle: np.ndarray | None,
    *,
    title: str = "",
) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3.8), constrained_layout=True)
    ax.plot(dates, y_true, color="k", lw=2.0, label="Real")
    ax.plot(dates, pred_projected, color="C0", lw=1.4, label="SARIMAX projected-exog")
    if pred_oracle is not None:
        ax.plot(dates, pred_oracle, color="C3", lw=1.2, ls="--", alpha=0.9, label="SARIMAX oracle-exog (diag.)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (COP/kWh)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.35)
    if title:
        ax.set_title(title)
    return fig


def plot_exog_trajectories(ex_f: pd.DataFrame, *, cols: list[str] | None = None, title: str = "") -> Any:
    import matplotlib.pyplot as plt

    d = ex_f.sort_values("date").copy()
    use = cols or [c for c in d.columns if c != "date" and pd.api.types.is_numeric_dtype(d[c])][:4]
    fig, axes = plt.subplots(len(use), 1, figsize=(12, 2.2 * max(1, len(use))), sharex=True, constrained_layout=True)
    if len(use) == 1:
        axes = [axes]
    for ax, c in zip(axes, use):
        ax.plot(d["date"], pd.to_numeric(d[c], errors="coerce"), lw=1.2)
        ax.set_ylabel(c, fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Fecha")
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


CAL_FOURIER_COLS = frozenset({"month_sin", "month_cos", "week_sin", "week_cos"})

# Periodos STL deseados (referencia diagnóstico).
_B1_STL_EXPECTED_PERIOD: dict[str, int] = {
    "demanda_max_kwh": 7,
    "porc_vol_util_diario": 365,
    "vol_util_energia_sistema": 365,
    "enso_index": 365,
    "porc_aporte_value": 365,
    "porc_aporte_max": 365,
}


def b1_stl_period_report(ex_f: pd.DataFrame) -> pd.DataFrame:
    """Tabla diagnóstico: periodo STL efectivo vs deseado (¿365 en embalse?)."""
    used = ex_f.attrs.get("stl_seasonal_period_by_column", {}) or {}
    methods = ex_f.attrs.get("univariate_method_per_column", {}) or {}
    rows: list[dict[str, Any]] = []
    cols = sorted(set(used) | set(methods))
    for c in cols:
        eff = used.get(c)
        want = _B1_STL_EXPECTED_PERIOD.get(c)
        ok = eff == want if want is not None and eff is not None else None
        rows.append(
            {
                "columna": c,
                "periodo_deseado": want,
                "periodo_efectivo": eff,
                "ok_anual_365": ok,
                "metodo_b1": methods.get(c, ""),
            }
        )
    return pd.DataFrame(rows)


def b1_exog_amplitude_table(
    train_df: pd.DataFrame,
    ex_f: pd.DataFrame,
    cols: list[str],
    *,
    recent_days: int = 365,
) -> pd.DataFrame:
    """
    std(proyectado) / std(histórico reciente). ratio ≪ 1 → B1 aplasta señal.
    """
    tr = train_df.sort_values("date").copy()
    ex = ex_f.sort_values("date").copy()
    rows: list[dict[str, Any]] = []
    for c in cols:
        if c in CAL_FOURIER_COLS:
            continue
        if c not in tr.columns or c not in ex.columns:
            continue
        hist = pd.to_numeric(tr[c], errors="coerce")
        proj = pd.to_numeric(ex[c], errors="coerce")
        tail = hist.iloc[-min(int(recent_days), len(hist)) :]
        hs = float(tail.std())
        ps = float(proj.std())
        ratio = ps / hs if hs > 1e-12 else float("nan")
        flag = ""
        if np.isfinite(ratio):
            if ratio < 0.25:
                flag = "muy_plano"
            elif ratio < 0.5:
                flag = "suavizado"
            elif ratio > 1.5:
                flag = "sobre_volatil"
        rows.append(
            {
                "columna": c,
                "std_hist_reciente": hs,
                "std_proyectado": ps,
                "ratio_std": ratio,
                "diagnostico": flag,
            }
        )
    return pd.DataFrame(rows)


def plot_b1_long_history(
    train_df: pd.DataFrame,
    ex_f: pd.DataFrame,
    cols: list[str],
    *,
    hist_days: int = 1095,
    title: str = "",
) -> Any:
    """Histórico largo (2–3 años) + proyección B1 (p. ej. h=60)."""
    import matplotlib.pyplot as plt

    tr = train_df.sort_values("date").copy()
    tr["date"] = pd.to_datetime(tr["date"]).dt.normalize()
    ex = ex_f.sort_values("date").copy()
    ex["date"] = pd.to_datetime(ex["date"]).dt.normalize()
    last = tr["date"].max()
    th = tr[tr["date"] >= last - pd.Timedelta(days=int(hist_days))].copy()
    use = [c for c in cols if c not in CAL_FOURIER_COLS and c in th.columns and c in ex.columns]
    if not use:
        raise ValueError("Sin columnas físicas para graficar histórico largo B1.")

    fig, axes = plt.subplots(len(use), 1, figsize=(12, 2.4 * len(use)), sharex=True, constrained_layout=True)
    if len(use) == 1:
        axes = [axes]
    for ax, c in zip(axes, use):
        ax.plot(th["date"], pd.to_numeric(th[c], errors="coerce"), color="dimgray", lw=0.9, label="Histórico")
        ax.plot(
            ex["date"],
            pd.to_numeric(ex[c], errors="coerce"),
            color="C1",
            lw=1.3,
            ls="--",
            marker=".",
            ms=2,
            label="Proyección B1",
        )
        ax.axvline(last, color="k", ls=":", lw=0.8, alpha=0.7)
        ax.set_ylabel(c, fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Fecha")
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def plot_std_ratio_bar(sweep_df: pd.DataFrame, *, top_n: int = 12) -> Any:
    import matplotlib.pyplot as plt

    df = sweep_df.copy()
    if "score" not in df.columns:
        df["score"] = df.apply(lambda r: score_b1_candidate(r), axis=1)
    sub = df.nsmallest(int(top_n), "score")
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.35 * len(sub))), constrained_layout=True)
    labels = sub["tag"].astype(str)
    x = np.arange(len(sub))
    w = 0.35
    ax.barh(x - w / 2, sub["std_ratio_val"], height=w, label="Val", color="C0", alpha=0.85)
    ax.barh(x + w / 2, sub["std_ratio_test"], height=w, label="Test", color="C2", alpha=0.85)
    ax.axvline(1.0, color="k", ls=":", lw=0.8, alpha=0.6)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("std_ratio = std(pred)/std(real)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    return fig
