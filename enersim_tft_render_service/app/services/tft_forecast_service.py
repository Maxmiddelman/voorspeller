from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.services.supabase_io import (
    download_model,
    fetch_prediction_frame,
    fetch_training_frame,
    get_active_model,
    save_model_metadata,
    upload_model,
    upsert_forecasts,
)
from app.ml.tft_model import predict_tft, train_tft

TMP_DIR = Path("/tmp/enersim-models")
TMP_DIR.mkdir(parents=True, exist_ok=True)


def train_tft_for_site(site_id: str) -> dict:
    df = fetch_training_frame(site_id)
    if df.empty:
        return {"site_id": site_id, "error": "Geen data in forecast_training_view"}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local_ckpt = str(TMP_DIR / f"{site_id}-tft-{ts}.ckpt")
    storage_path = f"tft/{site_id}/{ts}.ckpt"

    metrics = train_tft(df, local_ckpt)
    upload_model(local_ckpt, storage_path)
    metadata = save_model_metadata(site_id, storage_path, metrics)

    return {
        "site_id": site_id,
        "status": "trained",
        "model_type": "tft",
        "storage_path": storage_path,
        "metrics": metrics,
        "metadata": metadata,
    }


def forecast_tft_for_site(site_id: str) -> dict:
    active = get_active_model(site_id)
    if not active:
        return {"site_id": site_id, "error": "Geen actief TFT model. Run eerst /train."}

    local_ckpt = str(TMP_DIR / active["storage_path"].replace("/", "_"))
    if not Path(local_ckpt).exists():
        download_model(active["storage_path"], local_ckpt)

    df = fetch_prediction_frame(site_id)
    if df.empty:
        return {"site_id": site_id, "error": "Geen data in forecast_prediction_input_view"}

    rows = predict_tft(local_ckpt, df)
    upsert_forecasts(site_id, rows)

    return {
        "site_id": site_id,
        "status": "forecasted",
        "model_type": "tft",
        "horizon": len(rows),
        "forecast": rows,
    }
