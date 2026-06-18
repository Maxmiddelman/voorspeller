from __future__ import annotations

import math
from dataclasses import dataclass

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from app.config import BATCH_SIZE, ENCODER_LENGTH, FORECAST_HORIZON, LEARNING_RATE, MAX_EPOCHS

TARGET = "net_kw"
GROUP_IDS = ["site_id"]

KNOWN_REAL_FEATURES = [
    "time_idx",
    "quarter_of_day",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "temperature_2m",
    "cloud_cover",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "wind_speed_10m",
    "precipitation",
]

UNKNOWN_REAL_FEATURES = [
    "net_kw",
    "pv_kw",
    "ev_kw",
    "battery_kw",
    "net_kw_lag_1",
    "net_kw_lag_4",
    "net_kw_lag_96",
    "net_kw_lag_672",
    "net_kw_roll_4",
    "net_kw_roll_24",
    "net_kw_roll_96",
]

REQUIRED_COLUMNS = sorted(set(["site_id", "timestamp", TARGET] + KNOWN_REAL_FEATURES + UNKNOWN_REAL_FEATURES))


def prepare_frame(df: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["site_id", "timestamp"]).reset_index(drop=True)

    # time_idx = kwartieren vanaf eerste timestamp per site
    first_ts = df.groupby("site_id")["timestamp"].transform("min")
    df["time_idx"] = ((df["timestamp"] - first_ts).dt.total_seconds() // 900).astype(int)

    # Categorieën moeten string zijn voor PyTorch Forecasting
    df["site_id"] = df["site_id"].astype(str)

    numeric_cols = [c for c in KNOWN_REAL_FEATURES + UNKNOWN_REAL_FEATURES if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if require_target:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.dropna(subset=[TARGET])

    # Kleine gaten niet laten crashen; grotere gaten moeten in Supabase al als invalid gefilterd worden
    for col in numeric_cols:
        df[col] = df[col].ffill().bfill().fillna(0.0)

    return df


def validate_training_data(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"forecast_training_view mist kolommen: {missing}")
    min_rows = ENCODER_LENGTH + FORECAST_HORIZON + 500
    if len(df) < min_rows:
        raise ValueError(f"Te weinig trainingsdata: {len(df)} rows. Minimaal ongeveer {min_rows} nodig.")


def make_dataset(df: pd.DataFrame) -> TimeSeriesDataSet:
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=GROUP_IDS,
        min_encoder_length=ENCODER_LENGTH,
        max_encoder_length=ENCODER_LENGTH,
        min_prediction_length=FORECAST_HORIZON,
        max_prediction_length=FORECAST_HORIZON,
        static_categoricals=["site_id"],
        time_varying_known_reals=KNOWN_REAL_FEATURES,
        time_varying_unknown_reals=UNKNOWN_REAL_FEATURES,
        target_normalizer=GroupNormalizer(groups=GROUP_IDS, transformation=None),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


def train_tft(df: pd.DataFrame, checkpoint_path: str) -> dict[str, float]:
    validate_training_data(df)
    df = prepare_frame(df, require_target=True)

    max_time_idx = df["time_idx"].max()
    training_cutoff = max_time_idx - FORECAST_HORIZON

    training_df = df[df.time_idx <= training_cutoff]
    validation_df = df[df.time_idx > training_cutoff - ENCODER_LENGTH]

    training = make_dataset(training_df)
    validation = TimeSeriesDataSet.from_dataset(training, validation_df, predict=False, stop_randomization=True)

    train_loader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    pl.seed_everything(42)
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=8, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        enable_checkpointing=False,
        logger=False,
    )

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LEARNING_RATE,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.15,
        hidden_continuous_size=32,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        optimizer="AdamW",
        reduce_on_plateau_patience=4,
    )

    trainer.fit(model, train_loader, val_loader)

    # Validatie-metrics op median forecast
    actuals = torch.cat([y[0] for x, y in iter(val_loader)]).detach().cpu().numpy().reshape(-1)
    preds = model.predict(val_loader, mode="prediction").detach().cpu().numpy().reshape(-1)
    n = min(len(actuals), len(preds))
    actuals, preds = actuals[:n], preds[:n]

    mae = float(np.mean(np.abs(actuals - preds)))
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    mape = float(np.mean(np.abs((actuals - preds) / np.maximum(np.abs(actuals), 1.0))) * 100)

    trainer.save_checkpoint(checkpoint_path)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def predict_tft(checkpoint_path: str, prediction_df: pd.DataFrame) -> list[dict]:
    df = prepare_frame(prediction_df, require_target=False)
    if len(df) < ENCODER_LENGTH + FORECAST_HORIZON:
        raise ValueError(f"forecast_prediction_input_view moet minimaal {ENCODER_LENGTH + FORECAST_HORIZON} rows bevatten")

    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)

    # Dataset reconstrueren op basis van recente encoder + toekomstige known features.
    # Omdat de checkpoint de dataset-parameters bevat, gebruiken we from_dataset via model.dataset_parameters.
    dataset = TimeSeriesDataSet.from_parameters(
        model.dataset_parameters,
        df,
        predict=True,
        stop_randomization=True,
    )
    loader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    raw = model.predict(loader, mode="quantiles", return_x=False)
    arr = raw.detach().cpu().numpy()[0]  # shape: horizon x quantiles

    future = df.sort_values("timestamp").tail(FORECAST_HORIZON).copy()
    rows = []
    for i, (_, r) in enumerate(future.iterrows()):
        rows.append({
            "timestamp": r["timestamp"].isoformat(),
            "confidence_low": float(arr[i, 0]),
            "predicted_net_kw": float(arr[i, 1]),
            "confidence_high": float(arr[i, 2]),
        })
    return rows
