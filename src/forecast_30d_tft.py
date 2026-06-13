"""
TFT (Temporal Fusion Transformer, p50) para ventana **solo futura** alineada con ``exog_future``.

Usado por ``forecast_30d_all_models.run_forecast_30d_all``. Requiere torch, lightning y pytorch-forecasting.
Las filas futuras en ``df_all`` repiten el **último precio** solo como placeholder del ``TimeSeriesDataSet``
en ``predict()``; **no** se usan para ``val_loss``: el entrenamiento corta en ``training_cutoff`` y la
validación recorre la **cola real** del histórico (target observado), para no forzar un p50 artificialmente plano.
``sigma_feat`` se fija en 0 (sin panel GARCH del notebook 10).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from hybrid_direct_30d import normalize_column_names
from xm_config import PROJECT_ROOT

TFT_QUANTILES = (0.05, 0.5, 0.95)
TFT_Q50_IDX = 1
TFT_MAX_ENCODER = 90
TFT_MIN_ENCODER = 14
TFT_BATCH = 32
TFT_MAX_EPOCHS = 50
TFT_LR = 1e-3
TFT_HIDDEN = 64
TFT_ATT_HEAD = 4
TFT_DROPOUT = 0.1
TFT_HIDDEN_CONT = 32
TFT_SEED = 42


def tft_known_exog_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in ("demanda_max_kwh", "porc_vol_util_diario", "enso_index") if c in df.columns]


def _concat_reset(*parts: pd.DataFrame) -> pd.DataFrame:
    z = normalize_column_names(pd.concat(parts, ignore_index=True)).sort_values("date").reset_index(drop=True)
    z["date"] = pd.to_datetime(z["date"]).dt.normalize()
    z["time_idx"] = np.arange(len(z), dtype=np.int64)
    z["series_id"] = np.zeros(len(z), dtype=np.int64)
    return z


def _sigma_zero(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["sigma_feat"] = 0.0
    return out


def _build_frame(parts: list[pd.DataFrame], target: str) -> pd.DataFrame:
    base = _concat_reset(*parts)
    exog = tft_known_exog_columns(base)
    if target not in base.columns or len(exog) < 1:
        raise ValueError(f"TFT 30d: faltan target o covariables: target={target!r}, exog={exog}")
    for c in [target, *exog]:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    base = base.dropna(subset=[target, *exog]).reset_index(drop=True)
    base["time_idx"] = np.arange(len(base), dtype=np.int64)
    px = np.maximum(base[target].to_numpy(dtype=float), 0.0)
    base["target"] = np.log1p(px).astype(np.float64)
    base = _sigma_zero(base)
    if not bool(np.all(np.diff(base["time_idx"].to_numpy()) == 1)):
        raise ValueError("TFT 30d: time_idx debe ser consecutivo")
    if bool(base[["target", *exog, "sigma_feat"]].isna().any().any()):
        raise ValueError("TFT 30d: NaN en target/exógenas/sigma_feat")
    return base


def _cache_stem(d_hist: pd.DataFrame, *, target: str, days: int, n_hist: int, max_epochs: int) -> str:
    tr = normalize_column_names(d_hist.sort_values("date")).copy()
    d0 = str(pd.to_datetime(tr["date"]).min().date())
    d1 = str(pd.to_datetime(tr["date"]).max().date())
    exog = tft_known_exog_columns(tr)
    known_sig = tuple([*exog, "sigma_feat"])
    tft_ds_sig = (known_sig, ("target",), True, True, True)
    sig = "|".join(
        str(x)
        for x in (
            "f30",
            "val_real_tail",
            target,
            int(days),
            int(n_hist),
            d0,
            d1,
            int(TFT_MAX_ENCODER),
            int(TFT_MIN_ENCODER),
            int(max_epochs),
            int(TFT_HIDDEN),
            int(TFT_ATT_HEAD),
            int(TFT_HIDDEN_CONT),
            float(TFT_LR),
            str(TFT_QUANTILES),
            str(tft_ds_sig),
        )
    )
    h = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:28]
    return f"h{int(days)}_{h}"


def tft_forecast_p50_future(
    df_hist_raw: pd.DataFrame,
    exog_future: pd.DataFrame,
    target: str,
    days: int,
    *,
    max_epochs: int | None = None,
    use_disk_cache: bool = True,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Entrena (o carga .ckpt) un TFT sobre ``df_hist_raw`` y devuelve pronóstico p50 en COP para los
    próximos ``days`` días alineados con las primeras filas de ``exog_future``.
    """
    try:
        from lightning.pytorch import Trainer, seed_everything
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError:  # pragma: no cover
        from pytorch_lightning import Trainer, seed_everything
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss

    d = normalize_column_names(df_hist_raw.sort_values("date").copy())
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    ex = normalize_column_names(exog_future.copy())
    ex["date"] = pd.to_datetime(ex["date"]).dt.normalize()
    ex = ex.iloc[: int(days)].copy()
    exog_cols = tft_known_exog_columns(d)
    if len(exog_cols) < 1:
        raise ValueError("TFT 30d: el histórico no tiene covariables demanda/embalses/enso")
    miss = [c for c in exog_cols if c not in ex.columns]
    if miss:
        raise ValueError(f"TFT 30d: faltan columnas en exog_future: {miss}")
    fut = ex[["date", *exog_cols]].copy()
    last_px = float(pd.to_numeric(d[target].iloc[-1], errors="coerce"))
    if not np.isfinite(last_px):
        raise ValueError("TFT 30d: último precio histórico no finito")
    fut[target] = last_px

    known_reals = [*exog_cols, "sigma_feat"]
    unknown_reals = ["target"]

    df_hist = _build_frame([d], target)
    df_all = _build_frame([d, fut], target)
    n_hist = len(df_hist)
    need = int(TFT_MAX_ENCODER + int(days) + 5)
    if n_hist < need:
        raise ValueError(f"TFT 30d: historia n={n_hist} < encoder+horizonte+5 ({need})")

    me = int(max_epochs) if max_epochs is not None else int(TFT_MAX_EPOCHS)
    seed_everything(TFT_SEED, workers=False)

    # Validación SOLO sobre cola del histórico con target real. Antes se usaba ``df_all`` con el
    # futuro sintético donde ``target`` = último precio constante → ``val_loss`` empujaba el TFT a
    # pronósticos demasiado planos (p50 casi constante).
    hi = int(df_hist["time_idx"].iloc[-1])
    val_tail = int(days) + int(TFT_MAX_ENCODER) + 20
    training_cutoff = hi - val_tail
    min_train_last = int(need) - 1
    training_cutoff = max(training_cutoff, min_train_last)
    if training_cutoff >= hi:
        raise ValueError(
            "TFT 30d: historia insuficiente para separar entrenamiento y validación con target observado."
        )
    if hi - training_cutoff < int(days) + int(TFT_MIN_ENCODER):
        raise ValueError(
            "TFT 30d: cola de validación demasiado corta respecto al horizonte; amplíe el histórico."
        )
    train_df = df_hist.loc[df_hist["time_idx"] <= training_cutoff].copy()

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target",
        group_ids=["series_id"],
        max_encoder_length=int(TFT_MAX_ENCODER),
        max_prediction_length=int(days),
        min_prediction_length=1,
        min_encoder_length=int(TFT_MIN_ENCODER),
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    val_ds = TimeSeriesDataSet.from_dataset(
        training,
        df_hist,
        min_prediction_idx=int(training_cutoff) + 1,
        stop_randomization=True,
    )
    train_loader = training.to_dataloader(train=True, batch_size=int(TFT_BATCH), num_workers=0, shuffle=True)
    val_loader = val_ds.to_dataloader(train=False, batch_size=int(TFT_BATCH * 4), num_workers=0)
    pred_ds = TimeSeriesDataSet.from_dataset(training, df_all, predict=True, stop_randomization=True)
    pred_loader = pred_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

    root = cache_dir if cache_dir is not None else Path(PROJECT_ROOT) / "data" / "processed" / "notebook_models" / "tft_forecast_30d"
    root.mkdir(parents=True, exist_ok=True)
    stem = _cache_stem(d, target=target, days=int(days), n_hist=n_hist, max_epochs=me)
    ckpt = root / f"{stem}.ckpt"

    use_cache = bool(use_disk_cache) and ckpt.is_file()
    if use_cache:
        best = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
    else:
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=float(TFT_LR),
            hidden_size=int(TFT_HIDDEN),
            attention_head_size=int(TFT_ATT_HEAD),
            dropout=float(TFT_DROPOUT),
            hidden_continuous_size=int(TFT_HIDDEN_CONT),
            output_size=len(TFT_QUANTILES),
            loss=QuantileLoss(quantiles=list(TFT_QUANTILES)),
            log_interval=-1,
        )
        es = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        mc = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
        trainer = Trainer(
            max_epochs=int(me),
            accelerator="auto",
            enable_checkpointing=True,
            enable_model_summary=False,
            enable_progress_bar=False,
            callbacks=[es, mc],
            logger=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
        best_path = getattr(mc, "best_model_path", None) or ""
        if isinstance(best_path, str) and len(best_path) > 0:
            best = TemporalFusionTransformer.load_from_checkpoint(best_path)
        else:
            best = tft
        if use_disk_cache and isinstance(best_path, str) and len(best_path) > 0:
            shutil.copy2(best_path, ckpt)
            (root / f"{stem}.json").write_text(
                json.dumps({"target": target, "days": int(days), "n_hist": int(n_hist), "max_epochs": me}, indent=2),
                encoding="utf-8",
            )

    best.eval()
    yhat = best.predict(pred_loader, mode="prediction", return_x=False)
    if isinstance(yhat, (list, tuple)):
        yhat = yhat[0]
    arr = np.asarray(yhat, dtype=float)
    if arr.ndim == 3:
        arr = arr[:, :, int(TFT_Q50_IDX)]
    elif arr.ndim == 2 and arr.shape[-1] == len(TFT_QUANTILES):
        arr = arr[:, int(TFT_Q50_IDX)]
    pred_log = np.asarray(arr, dtype=float).ravel()[: int(days)]
    pred_cop = np.expm1(np.clip(pred_log, -20.0, 20.0))

    meta: dict = {
        "ok": True,
        "stem": stem,
        "n_hist": int(n_hist),
        "days": int(days),
        "known_reals": list(known_reals),
        "checkpoint_cached": bool(use_cache),
        "checkpoint_path": str(ckpt) if ckpt.is_file() else None,
        "nota": (
            "sigma_feat=0; futuro sintético solo en predict(). "
            "Entrenamiento hasta time_idx<=training_cutoff; val sobre cola real (no target constante)."
        ),
        "training_cutoff_time_idx": int(training_cutoff),
    }
    return pred_cop.astype(float), meta
