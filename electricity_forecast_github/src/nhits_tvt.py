"""
N-HiTS (Neural Hierarchical Interpolation for Time Series) para el protocolo TVT del notebook 10.

Usa ``neuralforecast`` (Nixtla). Objetivo ``log1p(precio)``; salida en COP con ``expm1``.
Exógenas futuras vía ``futr_exog_list``:

- ``exog_futr_source="real"``: valores observados en val/test (solo evaluación; ventaja informacional).
- ``exog_futr_source="projected"``: Bloque 1 (``block1_build_exog_future``), alineado con SARIMAX.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from hybrid_direct_30d import normalize_column_names

NHITS_DEFAULT_EXOG = (
    "demanda_max_kwh",
    "porc_vol_util_diario",
    "enso_index",
    "porc_aporte_value",
)


def nhits_calendar_columns(df: pd.DataFrame, *, use_dow: bool = True) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["date"])
    m = dt.dt.month.astype(float)
    out["month_sin"] = np.sin(2 * np.pi * m / 12.0).astype(np.float64)
    out["month_cos"] = np.cos(2 * np.pi * m / 12.0).astype(np.float64)
    if use_dow:
        dow = dt.dt.dayofweek.astype(float)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0).astype(np.float64)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0).astype(np.float64)
    return out


def nhits_resolve_exog(
    df: pd.DataFrame,
    *,
    sarimax_resolver: Callable[[pd.DataFrame], list[str]] | None = None,
    train_df: pd.DataFrame | None = None,
) -> list[str]:
    cal = {"month_sin", "month_cos", "dow_sin", "dow_cos"}
    if sarimax_resolver is not None and train_df is not None:
        resolved = [c for c in sarimax_resolver(train_df) if c in df.columns and c not in cal]
        if resolved:
            return resolved
    return [c for c in NHITS_DEFAULT_EXOG if c in df.columns]


def nhits_futr_exog_list(*, use_dow: bool = True) -> list[str]:
    cols = ["month_sin", "month_cos"]
    if use_dow:
        cols.extend(["dow_sin", "dow_cos"])
    return cols


def build_nhits_nf_frame(
    parts: list[pd.DataFrame],
    target: str,
    *,
    exog_cols: list[str],
    use_dow: bool = True,
    unique_id: str = "precio",
) -> pd.DataFrame:
    """Panel NeuralForecast: ``unique_id``, ``ds``, ``y`` (log1p), exógenas."""
    base = normalize_column_names(pd.concat(parts, ignore_index=True)).sort_values("date")
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    base = nhits_calendar_columns(base, use_dow=use_dow)
    xcal = nhits_futr_exog_list(use_dow=use_dow)
    need = [target, *exog_cols, *xcal]
    for c in need:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")
    base = base.dropna(subset=[c for c in need if c in base.columns]).reset_index(drop=True)
    px = np.maximum(base[target].to_numpy(dtype=float), 0.0)
    out = base[["date", target, *exog_cols, *xcal]].copy()
    out["y"] = np.log1p(px).astype(np.float64)
    out = out.rename(columns={"date": "ds"})
    out["unique_id"] = unique_id
    return out[["unique_id", "ds", "y", *exog_cols, *xcal]]


def amplitude_stats(y_cop: np.ndarray, pred_cop: np.ndarray) -> dict[str, float]:
    """std_real, std_pred, std_ratio para diagnóstico de volatilidad capturada."""
    y = np.asarray(y_cop, dtype=float).ravel()
    p = np.asarray(pred_cop, dtype=float).ravel()
    n = int(min(len(y), len(p)))
    if n == 0:
        return {"std_real": float("nan"), "std_pred": float("nan"), "std_ratio": float("nan")}
    ys = float(np.nanstd(y[:n]))
    ps = float(np.nanstd(p[:n]))
    ratio = ps / ys if ys > 1e-9 else float("nan")
    return {"std_real": ys, "std_pred": ps, "std_ratio": ratio}


def print_amplitude_diagnostic(
    y_cop: np.ndarray,
    pred_cop: np.ndarray,
    *,
    label: str,
    model_name: str,
) -> dict[str, float]:
    st = amplitude_stats(y_cop, pred_cop)
    print(
        f"N-HiTS amplitud ({label} / {model_name}): "
        f"std_real={st['std_real']:.2f} std_pred={st['std_pred']:.2f} std_ratio={st['std_ratio']:.3f}"
    )
    return st


def build_nhits_futr_df(
    df_hist: pd.DataFrame,
    *,
    h_fore: int,
    exog_cols: list[str],
    use_dow: bool = True,
    unique_id: str = "precio",
    exog_futr_source: str = "real",
    df_all: pd.DataFrame | None = None,
    projected_exog_future: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Construye ``futr_df`` para ``NeuralForecast.predict``.

    ``real``: filas OOS de ``df_all`` (exógenas observadas en val/test).
    ``projected``: Bloque 1 sobre fechas futuras (misma info que SARIMAX en OOS).
    """
    src = str(exog_futr_source).strip().lower()
    last_ds = pd.Timestamp(df_hist["ds"].max()).normalize()
    xcal = nhits_futr_exog_list(use_dow=use_dow)

    if src == "projected":
        if projected_exog_future is None:
            raise ValueError("projected_exog_future es obligatorio con exog_futr_source='projected'")
        exf = normalize_column_names(projected_exog_future.copy())
        exf["date"] = pd.to_datetime(exf["date"], errors="coerce").dt.normalize()
        exf = exf[exf["date"] > last_ds].head(int(h_fore)).copy()
        if len(exf) < int(h_fore):
            raise RuntimeError(
                f"futr projected corto ({len(exf)} < {h_fore}); revise block1_build_exog_future."
            )
        exf = nhits_calendar_columns(exf, use_dow=use_dow)
        futr = exf.rename(columns={"date": "ds"})
        for c in exog_cols + xcal:
            if c in futr.columns:
                futr[c] = pd.to_numeric(futr[c], errors="coerce")
        futr["unique_id"] = unique_id
        cols = ["unique_id", "ds", *exog_cols, *xcal]
        return futr[cols].copy()

    if src == "real":
        if df_all is None:
            raise ValueError("df_all es obligatorio con exog_futr_source='real'")
        futr = df_all[pd.to_datetime(df_all["ds"]).dt.normalize() > last_ds].head(int(h_fore)).copy()
        return futr

    raise ValueError(f"exog_futr_source desconocido: {exog_futr_source!r} (use 'real' o 'projected')")


def _resolve_nhits_cache_path(
    hist_parts: list[pd.DataFrame],
    *,
    cache_dir: Path,
    cache_schema: str,
    target: str,
    h_fore: int,
    n_hist: int,
    input_size: int,
    h_train: int,
    max_steps: int,
    exog_sig: tuple[str, ...],
    futr_sig: tuple[str, ...],
    scaler_type: str,
    n_blocks: tuple[int, ...],
) -> Path:
    stem = _cache_stem(
        hist_parts,
        schema=cache_schema,
        target=target,
        h_fore=h_fore,
        n_hist=n_hist,
        input_size=input_size,
        h_train=h_train,
        max_steps=max_steps,
        exog_sig=exog_sig,
        futr_sig=futr_sig,
        scaler_type=scaler_type,
        n_blocks=n_blocks,
    )
    return cache_dir / stem


def repredict_nhits_oos_from_cache(
    hist_parts: list[pd.DataFrame],
    oos_parts: list[pd.DataFrame],
    *,
    target: str,
    h_fore: int,
    label: str,
    exog_cols: list[str],
    cache_dir: Path,
    cache_schema: str,
    use_dow: bool = True,
    input_size: int = 180,
    h_train: int | None = None,
    max_steps: int = 1500,
    n_blocks: tuple[int, ...] = (2, 2, 2),
    scaler_type: str = "standard",
    exog_futr_source: str = "real",
    projected_exog_future: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga modelo N-HiTS desde caché (sin reentrenar) y pronostica con otra fuente de exógenas futuras.

    Útil para comparar ``real`` vs ``projected`` con los mismos pesos de red.
    """
    try:
        from neuralforecast import NeuralForecast
    except ImportError as e:
        raise ImportError("Instale neuralforecast: pip install neuralforecast>=1.7.0") from e

    h_fore = int(h_fore)
    if h_train is None:
        h_model = h_fore
    else:
        h_model = int(min(int(h_train), h_fore))
    h_model = max(1, h_model)

    df_hist = build_nhits_nf_frame(hist_parts, target, exog_cols=exog_cols, use_dow=use_dow)
    df_all = build_nhits_nf_frame(hist_parts + oos_parts, target, exog_cols=exog_cols, use_dow=use_dow)
    n_hist = len(df_hist)
    input_size_req = int(input_size)
    input_size_eff = min(input_size_req, max(2 * h_model + 1, n_hist - h_model - 3))
    input_size_eff = max(input_size_eff, 2 * h_model)
    futr_exog = list(exog_cols) + nhits_futr_exog_list(use_dow=use_dow)
    n_blocks_t = tuple(int(x) for x in n_blocks)
    scaler_use = str(scaler_type).strip().lower()

    cache_path = _resolve_nhits_cache_path(
        hist_parts,
        cache_dir=cache_dir,
        cache_schema=cache_schema,
        target=target,
        h_fore=h_fore,
        n_hist=n_hist,
        input_size=input_size,
        h_train=h_model,
        max_steps=max_steps,
        exog_sig=tuple(exog_cols),
        futr_sig=tuple(futr_exog),
        scaler_type=scaler_use,
        n_blocks=n_blocks_t,
    )
    if not cache_path.is_dir():
        raise FileNotFoundError(f"N-HiTS caché no encontrada para repredict: {cache_path}")

    print(
        f"N-HiTS repredict ({label}) — exog_futr={exog_futr_source} | caché={cache_path.name}"
    )
    nf = NeuralForecast.load(str(cache_path))
    futr = build_nhits_futr_df(
        df_hist,
        h_fore=h_fore,
        exog_cols=exog_cols,
        use_dow=use_dow,
        exog_futr_source=exog_futr_source,
        df_all=df_all,
        projected_exog_future=projected_exog_future,
    )
    return _predict_nhits_futr(nf, futr, h_fore=h_fore, label=label)


def _predict_nhits_futr(
    nf: Any,
    futr: pd.DataFrame,
    *,
    h_fore: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    import warnings

    h_fore = int(h_fore)
    if len(futr) < h_fore:
        raise RuntimeError(f"N-HiTS ({label}): futr_df corto ({len(futr)} < {h_fore})")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = nf.predict(futr_df=futr)
    model_name = [c for c in fc.columns if c not in ("unique_id", "ds")][-1]
    pred_log = fc[model_name].to_numpy(dtype=float).ravel()
    n_pred = int(pred_log.size)
    if n_pred < h_fore:
        pad_n = h_fore - n_pred
        print(
            f"N-HiTS ({label}) — AVISO padding: predict devolvió {n_pred} pasos < h_fore={h_fore}; "
            f"np.pad(..., mode='edge') añade {pad_n} día(s) constantes."
        )
        pred_log = np.pad(pred_log, (0, pad_n), mode="edge")
    elif n_pred > h_fore:
        print(f"N-HiTS ({label}) — predict devolvió {n_pred} pasos; se truncan a h_fore={h_fore}.")
    pred_log = pred_log[:h_fore]

    oos_dates = futr["ds"].to_numpy()[:h_fore]
    pred_cop = np.expm1(np.clip(pred_log, -20.0, 20.0))

    if "y" in futr.columns:
        y_log = futr["y"].to_numpy(dtype=float)[:h_fore]
        y_cop = np.expm1(np.clip(y_log, -20.0, 20.0))
        st = amplitude_stats(y_cop, pred_cop)
        print(
            f"N-HiTS ({label}) — std(y)={st['std_real']:.2f} std(pred)={st['std_pred']:.2f} "
            f"ratio={st['std_ratio']:.3f} | pred_log std={float(np.nanstd(pred_log)):.4f}"
        )
    return oos_dates, pred_cop


def _cache_stem(
    hist_parts: list[pd.DataFrame],
    *,
    schema: str,
    target: str,
    h_fore: int,
    n_hist: int,
    input_size: int,
    h_train: int,
    max_steps: int,
    exog_sig: tuple[str, ...],
    futr_sig: tuple[str, ...],
    scaler_type: str,
    n_blocks: tuple[int, ...],
) -> str:
    """Huella de caché en disco solo para N-HiTS (directorio ``nhits_tvt_*``). No afecta TVT parquet §1–§5."""
    tr = normalize_column_names(pd.concat(hist_parts, ignore_index=True)).sort_values("date")
    d0 = str(pd.to_datetime(tr["date"]).min().date())
    d1 = str(pd.to_datetime(tr["date"]).max().date())
    sig = "|".join(
        str(x)
        for x in (
            schema,
            target,
            int(h_fore),
            int(n_hist),
            d0,
            d1,
            int(input_size),
            int(h_train),
            int(max_steps),
            scaler_type,
            n_blocks,
            exog_sig,
            futr_sig,
        )
    )
    h = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:28]
    return f"h{int(h_fore)}_{h}"


def print_nhits_config_diagnostic(
    *,
    h_fore: int,
    h_model: int,
    input_size_requested: int,
    input_size_effective: int,
    max_steps: int,
    n_blocks: tuple[int, ...],
    scaler_type: str,
    val_size: int,
    early_stop_patience: int,
    n_hist: int,
    label: str,
) -> None:
    """Resumen técnico antes de fit (horizonte, padding, escalado)."""
    pad_risk = h_model < h_fore
    print(f"N-HiTS diagnóstico ({label}):")
    print(f"  h_fore (evaluación)     = {h_fore}")
    print(f"  h_model (entrenamiento) = {h_model}  {'[IGUAL a h_fore]' if h_model == h_fore else '[DISTINTO]'}")
    print(f"  input_size solicitado   = {input_size_requested}  | efectivo = {input_size_effective}")
    print(f"  max_steps               = {max_steps}  | early_stop_patience = {early_stop_patience}")
    print(f"  n_blocks                = {n_blocks}")
    print(f"  scaler_type             = {scaler_type}")
    print(f"  val_size (fit)          = {val_size}  | n_hist = {n_hist}")
    print(f"  loss (NeuralForecast)   = MSE en y (log1p) — cuantílico no aplica al NHITS base")
    if pad_risk:
        print(
            f"  AVISO: h_model ({h_model}) < h_fore ({h_fore}) → predict puede devolver <{h_fore} pasos "
            f"y se rellena con np.pad(..., mode='edge') (aplanamiento en cola del horizonte)."
        )
    else:
        print(f"  padding edge: no esperado si predict devuelve {h_fore} pasos (h_model == h_fore).")


def train_and_predict_nhits_oos(
    hist_parts: list[pd.DataFrame],
    oos_parts: list[pd.DataFrame],
    *,
    target: str,
    h_fore: int,
    label: str,
    exog_cols: list[str],
    use_dow: bool = True,
    input_size: int = 180,
    h_train: int | None = None,
    max_steps: int = 1500,
    learning_rate: float = 1e-3,
    n_pool_kernel_size: tuple[int, ...] = (2, 2, 1),
    n_blocks: tuple[int, ...] = (2, 2, 2),
    mlp_units: list | None = None,
    val_size: int = 60,
    early_stop_patience: int = 10,
    seed: int = 42,
    scaler_type: str = "standard",
    cache_dir: Path | None = None,
    cache_schema: str = "nhits_v5_h60_exp",
    use_disk_cache: bool = True,
    force_refit: bool = False,
    exog_futr_source: str = "real",
    projected_exog_future: pd.DataFrame | None = None,
    loss: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Entrena N-HiTS solo con ``hist_parts`` y pronostica ``h_fore`` pasos.

    ``exog_futr_source``:
    - ``real``: exógenas observadas en ``oos_parts`` (val/test).
    - ``projected``: ``projected_exog_future`` del Bloque 1 (comparación justa vs SARIMAX).

    Devuelve ``(fechas_oos, pred_cop)``.
    """
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS
    except ImportError as e:
        raise ImportError(
            "Instale neuralforecast: pip install neuralforecast>=1.7.0"
        ) from e

    h_fore = int(h_fore)
    # Experimental v5: horizonte de entrenamiento = horizonte de evaluación (sin cap en 14 días).
    if h_train is None:
        h_model = h_fore
    else:
        h_model = int(min(int(h_train), h_fore))
    h_model = max(1, h_model)

    df_hist = build_nhits_nf_frame(hist_parts, target, exog_cols=exog_cols, use_dow=use_dow)
    df_all = build_nhits_nf_frame(hist_parts + oos_parts, target, exog_cols=exog_cols, use_dow=use_dow)
    n_hist = len(df_hist)
    input_size_req = int(input_size)
    input_size_eff = min(input_size_req, max(2 * h_model + 1, n_hist - h_model - 3))
    input_size_eff = max(input_size_eff, 2 * h_model)
    need = int(input_size_eff + h_model + 5)
    if n_hist < need:
        raise ValueError(f"N-HiTS ({label}): train corto n={n_hist} < input_size+h+5 ({need})")

    futr_exog = list(exog_cols) + nhits_futr_exog_list(use_dow=use_dow)
    n_blocks_t = tuple(int(x) for x in n_blocks)
    scaler_use = str(scaler_type).strip().lower()
    print_nhits_config_diagnostic(
        h_fore=h_fore,
        h_model=h_model,
        input_size_requested=input_size_req,
        input_size_effective=input_size_eff,
        max_steps=int(max_steps),
        n_blocks=n_blocks_t,
        scaler_type=scaler_use,
        val_size=int(val_size),
        early_stop_patience=int(early_stop_patience),
        n_hist=n_hist,
        label=label,
    )
    print(
        f"N-HiTS ({label}) — n_hist={n_hist}, h_fore={h_fore}, h_model={h_model}, "
        f"input_size={input_size_eff}, exog_futr={exog_futr_source}, futr_exog={futr_exog}"
    )

    vs = min(int(val_size), max(1, n_hist // 10))
    vs = max(vs, h_model)
    vs = min(vs, max(h_model, n_hist - h_model - 1))
    mlp_kw = {} if mlp_units is None else {"mlp_units": mlp_units}
    nhits_kwargs = dict(
        h=h_model,
        input_size=int(input_size_eff),
        futr_exog_list=futr_exog,
        n_pool_kernel_size=n_pool_kernel_size,
        n_blocks=n_blocks_t,
        **mlp_kw,
        max_steps=int(max_steps),
        learning_rate=float(learning_rate),
        random_seed=int(seed),
        early_stop_patience_steps=int(early_stop_patience),
        scaler_type=scaler_use,
    )
    if loss is not None:
        nhits_kwargs["loss"] = loss
    try:
        models = [NHITS(**nhits_kwargs)]
    except TypeError as e:
        # Compatibilidad: algunas versiones no exponen `loss=` en NHITS.
        if loss is not None:
            print(f"N-HiTS ({label}) — AVISO: NHITS no acepta loss=... en esta versión; usando loss default. ({e})")
            nhits_kwargs.pop("loss", None)
            models = [NHITS(**nhits_kwargs)]
        else:
            raise
    nf = NeuralForecast(models=models, freq="D")

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = _resolve_nhits_cache_path(
            hist_parts,
            cache_dir=cache_dir,
            cache_schema=cache_schema,
            target=target,
            h_fore=h_fore,
            n_hist=n_hist,
            input_size=input_size,
            h_train=h_model,
            max_steps=max_steps,
            exog_sig=tuple(exog_cols),
            futr_sig=tuple(futr_exog),
            scaler_type=scaler_use,
            n_blocks=n_blocks_t,
        )

    use_cache = bool(use_disk_cache) and (not force_refit) and cache_path is not None and cache_path.is_dir()
    if use_cache:
        print("N-HiTS: cargando caché", cache_path)
        nf = NeuralForecast.load(str(cache_path))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nf.fit(df=df_hist, val_size=vs)
        if cache_path is not None and use_disk_cache:
            # NeuralForecast puede fallar si el directorio existe y no está vacío.
            # Preferimos overwrite explícito; fallback compatible con versiones antiguas.
            try:
                nf.save(str(cache_path), overwrite=True)
            except TypeError:
                if cache_path.exists():
                    shutil.rmtree(cache_path, ignore_errors=True)
                nf.save(str(cache_path))
            (cache_path / "_meta.json").write_text(
                json.dumps(
                    {
                        "label": label,
                        "h_fore": h_fore,
                        "h_model": h_model,
                        "n_hist": n_hist,
                        "input_size_eff": input_size_eff,
                        "max_steps": int(max_steps),
                        "n_blocks": list(n_blocks_t),
                        "scaler_type": scaler_use,
                        "cache_schema": cache_schema,
                        "exog_futr_source": str(exog_futr_source),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print("N-HiTS: modelo guardado en", cache_path)

    # Futuro: exógenas reales (val/test) o proyectadas (Bloque 1, alineado con SARIMAX)
    futr = build_nhits_futr_df(
        df_hist,
        h_fore=h_fore,
        exog_cols=exog_cols,
        use_dow=use_dow,
        exog_futr_source=exog_futr_source,
        df_all=df_all,
        projected_exog_future=projected_exog_future,
    )
    # Adjuntar y real si está disponible (diagnóstico de amplitud)
    if exog_futr_source == "real" and "y" in df_all.columns:
        last_ds = pd.Timestamp(df_hist["ds"].max()).normalize()
        oos_y = df_all[pd.to_datetime(df_all["ds"]).dt.normalize() > last_ds].head(h_fore)
        if len(oos_y) == len(futr):
            futr = futr.copy()
            futr["y"] = oos_y["y"].to_numpy(dtype=float)

    return _predict_nhits_futr(nf, futr, h_fore=h_fore, label=label)


def metrics_by_date(
    dates: np.ndarray,
    y_true: np.ndarray,
    results_df: pd.DataFrame,
    *,
    pred_col: str,
    tag: str,
    metrics_fn: Callable[..., dict[str, float]],
) -> dict[str, float]:
    left = pd.DataFrame(
        {
            "date": pd.to_datetime(np.asarray(dates).ravel(), errors="coerce").normalize(),
            "y_true": np.asarray(y_true, dtype=float).ravel(),
        }
    )
    r = results_df[["date", pred_col]].copy()
    r["date"] = pd.to_datetime(r["date"], errors="coerce").dt.normalize()
    r = r.drop_duplicates(subset=["date"], keep="last")
    m = left.merge(r, on="date", how="inner").sort_values("date")
    nexp = int(len(left))
    if len(m) != nexp:
        print(
            f"N-HiTS métricas ({tag}): alineadas {len(m)}/{nexp} por fecha "
            "(resto: sin pred por dropna en panel)."
        )
    if len(m) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape_pct": float("nan"), "n": 0}
    return metrics_fn(m["y_true"].to_numpy(), m[pred_col].astype(float).to_numpy())
