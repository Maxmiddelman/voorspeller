from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import MODEL_BUCKET
from app.db import supabase


def paged_select(table: str, *, filters: list[tuple[str, str, Any]] | None = None, page_size: int = 1000) -> list[dict]:
    """Supabase REST geeft vaak max. 1000 rows terug; deze helper pagineert expliciet."""
    rows: list[dict] = []
    start = 0
    filters = filters or []

    while True:
        q = supabase.table(table).select("*")
        for col, op, value in filters:
            if op == "eq":
                q = q.eq(col, value)
            elif op == "gte":
                q = q.gte(col, value)
            elif op == "lte":
                q = q.lte(col, value)
            elif op == "gt":
                q = q.gt(col, value)
            elif op == "lt":
                q = q.lt(col, value)
            else:
                raise ValueError(f"Unsupported filter op: {op}")

        res = q.order("timestamp", desc=False).range(start, start + page_size - 1).execute()
        batch = res.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    return rows


def fetch_training_frame(site_id: str) -> pd.DataFrame:
    rows = paged_select("forecast_training_view", filters=[("site_id", "eq", site_id)])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def fetch_prediction_frame(site_id: str) -> pd.DataFrame:
    rows = paged_select("forecast_prediction_input_view", filters=[("site_id", "eq", site_id)])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def upload_model(local_path: str, storage_path: str) -> None:
    with open(local_path, "rb") as f:
        supabase.storage.from_(MODEL_BUCKET).upload(
            storage_path,
            f,
            file_options={"content-type": "application/octet-stream", "upsert": "true"},
        )


def download_model(storage_path: str, local_path: str) -> None:
    data = supabase.storage.from_(MODEL_BUCKET).download(storage_path)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(data)


def save_model_metadata(site_id: str, storage_path: str, metrics: dict[str, float]) -> dict:
    row = {
        "site_id": site_id,
        "model_type": "tft",
        "storage_path": storage_path,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "mape": metrics.get("mape"),
        "is_active": True,
    }
    supabase.table("forecast_models").update({"is_active": False}).eq("site_id", site_id).eq("model_type", "tft").execute()
    res = supabase.table("forecast_models").insert(row).execute()
    return (res.data or [row])[0]


def get_active_model(site_id: str) -> dict | None:
    res = (
        supabase.table("forecast_models")
        .select("*")
        .eq("site_id", site_id)
        .eq("model_type", "tft")
        .eq("is_active", True)
        .order("trained_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    return rows[0] if rows else None


def upsert_forecasts(site_id: str, rows: list[dict]) -> None:
    payload = []
    run_ts = datetime.now(timezone.utc).isoformat()
    for row in rows:
        payload.append({
            "site_id": site_id,
            "forecast_run_at": run_ts,
            "timestamp": row["timestamp"],
            "predicted_net_kw": float(row["predicted_net_kw"]),
            "confidence_low": float(row["confidence_low"]),
            "confidence_high": float(row["confidence_high"]),
            "model_type": "tft",
        })
    supabase.table("forecast_predictions_15m").upsert(payload, on_conflict="site_id,timestamp,model_type").execute()
