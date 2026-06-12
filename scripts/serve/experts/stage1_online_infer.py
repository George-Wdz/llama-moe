#!/usr/bin/env python
"""Run Stage1 rainfall inversion online and emit a compact JSON summary."""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


DEFAULT_CHECKPOINT_DIR = (
    "/home/wdz/BT/Stage1/model/checkpoints/pass_dataset_rain_retrieval_20260610_1420/"
    "stage1_cm_dm256_df512_eh8_el3_dl2_pl8_st4_bs32_lr0.0001_itr0"
)
STAGE1_ROOT = Path("/home/wdz/BT/Stage1/model")

sys.path.insert(0, str(STAGE1_ROOT))

from data.data_factory import (  # noqa: E402
    _optional_feature_keys,
    attach_train_dry_baseline,
    load_all_passes,
    split_passes_by_time,
)
from data.dataset import PassDataset, SatelliteIDMapper  # noqa: E402
from data.db import load_ground_weather, load_weather_station  # noqa: E402
from data.preprocessing import (  # noqa: E402
    IMAGE_WEATHER_COLS,
    merge_ground_weather,
    load_image_weather_predictions,
    segment_passes,
)
from models.patch_encoder_decoder import PatchEncoderDecoder  # noqa: E402


DEFAULT_DB_PATH = "/home/wdz/satellite_data/satellite_data.db"
DEFAULT_IMAGE_WEATHER_CSV = "/home/wdz/BT/Stage1/data/camera_labels/latest_weather_labels_slim.csv"


def _coerce_numeric(cfg: dict) -> None:
    b = cfg.get("dry_baseline", {})
    for k in (
        "rain_threshold",
        "image_rain_prob_threshold",
        "time_scale_hours",
        "time_weight",
        "position_weight",
    ):
        if isinstance(b.get(k), str):
            b[k] = float(b[k])
    t = cfg["training"]
    for k in (
        "lr",
        "weight_decay",
        "rainfall_loss_weight",
        "rain_threshold",
        "rainy_loss_weight",
        "rain_classification_loss_weight",
        "rain_classification_pos_weight",
        "grad_clip",
        "decay_fac",
    ):
        if k in t and isinstance(t[k], str):
            t[k] = float(t[k])
    for k in ("epochs", "batch_size", "warmup_epochs", "patience", "iterations", "seed", "tmax"):
        if k in t and isinstance(t[k], str):
            t[k] = int(t[k])


def _load_checkpoint(ckpt_dir: Path, device: torch.device):
    meta = torch.load(ckpt_dir / "meta.pt", map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(meta["cfg"])
    _coerce_numeric(cfg)

    sat_mapper = SatelliteIDMapper(known_ids=[])
    sat_mapper.id_to_idx = meta["sat_mapper"]
    sat_mapper.num_satellites = max(sat_mapper.id_to_idx.values()) + 1 if sat_mapper.id_to_idx else 1
    cfg["model"]["num_satellites"] = max(cfg["model"]["num_satellites"], sat_mapper.num_satellites)

    model = PatchEncoderDecoder(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_dir / "checkpoint.pth", map_location=device))
    model.eval()
    return cfg, model, sat_mapper, meta["scaler_X"], meta["scaler_y"]


def _read_recent_phy(db_path: str, link_cols: list[str], lookback_hours: float) -> pd.DataFrame:
    select_cols = ", ".join(link_cols)
    predicates = " AND ".join(f"{col} IS NOT NULL" for col in link_cols)
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        latest = pd.read_sql_query("SELECT max(localTime) AS latest FROM phy_data", conn)["latest"].iloc[0]
        if latest is None:
            return pd.DataFrame()
        latest_ts = pd.to_datetime(latest, format="ISO8601")
        start_ts = latest_ts - pd.Timedelta(hours=lookback_hours)
        query = f"""
            SELECT localTime, satelliteId, earthStationId, {select_cols}
            FROM phy_data
            WHERE localTime >= ? AND {predicates}
            ORDER BY localTime
        """
        df = pd.read_sql_query(query, conn, params=[start_ts.isoformat()])
    if df.empty:
        return df
    df["earthStationId"] = 0
    df["localTime"] = pd.to_datetime(df["localTime"], format="ISO8601")
    return df.set_index("localTime").sort_index()


def _read_recent_position(db_path: str, pos_cols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    select_cols = ", ".join(pos_cols)
    start = start - pd.Timedelta(minutes=10)
    end = end + pd.Timedelta(minutes=10)
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        query = f"""
            SELECT localTime, satId, {select_cols}
            FROM position_data
            WHERE localTime >= ? AND localTime <= ?
            ORDER BY localTime
        """
        df = pd.read_sql_query(query, conn, params=[start.isoformat(), end.isoformat()])
    if df.empty:
        return df
    df["localTime"] = pd.to_datetime(df["localTime"], format="ISO8601")
    return df.set_index("localTime")[pos_cols].sort_index()


def _attach_online_features(
    passes: list[dict],
    db_path: str,
    weather_cols: list[str],
    image_csv: str,
    image_tolerance: str,
) -> list[dict]:
    gw = load_ground_weather(db_path)
    ws = load_weather_station(db_path)
    ground_weather = merge_ground_weather(gw, ws)

    image_weather = None
    if image_csv and Path(image_csv).exists():
        image_weather = load_image_weather_predictions(image_csv)
    image_tol = pd.Timedelta(image_tolerance)

    out = []
    for p in passes:
        idx = pd.DatetimeIndex(p["timestamps"])
        gw_aligned = ground_weather[weather_cols].reindex(
            idx, method="nearest", tolerance=pd.Timedelta("60s")
        )
        if gw_aligned.isna().any().any():
            gw_aligned = gw_aligned.ffill().bfill()
        if gw_aligned.isna().any().any():
            continue

        meta = {
            "pass_start": idx[0],
            "pass_end": idx[-1],
            "weather_rows": 0,
            "rain_rate_mean": 0.0,
            "rain_rate_max": 0.0,
            "rainy_ratio": 0.0,
        }
        ws_in_range = ws.loc[idx[0] : idx[-1]]
        if len(ws_in_range):
            rain_rate = pd.to_numeric(ws_in_range["rainfall"], errors="coerce")
            meta.update({
                "weather_rows": int(len(ws_in_range)),
                "rain_rate_mean": float(rain_rate.mean()) if rain_rate.notna().any() else 0.0,
                "rain_rate_max": float(rain_rate.max()) if rain_rate.notna().any() else 0.0,
                "rainy_ratio": float((rain_rate > 0).mean()) if rain_rate.notna().any() else 0.0,
            })

        image_vec = np.zeros(len(IMAGE_WEATHER_COLS), dtype=np.float32)
        if image_weather is not None:
            center = idx[0] + (idx[-1] - idx[0]) / 2
            nearest_pos = image_weather.index.get_indexer([center], method="nearest", tolerance=image_tol)[0]
            if nearest_pos >= 0:
                row = image_weather.iloc[nearest_pos]
                image_vec = row[IMAGE_WEATHER_COLS].to_numpy(dtype=np.float32)
                meta["image_available"] = 1
                meta["image_time_delta_s"] = abs((image_weather.index[nearest_pos] - center).total_seconds())
            else:
                meta["image_available"] = 0
                meta["image_time_delta_s"] = None

        out.append({
            **p,
            "ground_weather": gw_aligned.values.astype(np.float32),
            "image_weather": np.repeat(image_vec.reshape(1, -1), len(idx), axis=0).astype(np.float32),
            "labels": np.zeros(3, dtype=np.float32),
            "label_meta": meta,
        })
    return out


def _build_latest_db_passes(cfg: dict, args: argparse.Namespace) -> list[dict]:
    feature_cfg = cfg.get("features", {})
    link_cols = list(feature_cfg.get("link", ["phyRssi", "rssi", "snr", "lastCniValue"]))
    pos_cols = list(feature_cfg.get("position", [
        "longitude", "latitude", "satAltitude", "posLongitude", "posLatitude", "altitude"
    ]))
    weather_cols = list(feature_cfg.get("ground_weather", ["temperature", "humidity", "pressure"]))

    phy = _read_recent_phy(args.db_path, link_cols, args.lookback_hours)
    if phy.empty:
        return []
    pos = _read_recent_position(args.db_path, pos_cols, phy.index.min(), phy.index.max())
    if pos.empty:
        return []

    passes = segment_passes(
        phy,
        pos,
        link_cols=link_cols,
        pos_cols=pos_cols,
        gap_threshold=args.pass_gap_threshold_s,
        min_points=args.min_pass_points,
    )
    passes = sorted(passes, key=lambda p: pd.DatetimeIndex(p["timestamps"])[-1], reverse=True)
    if args.max_passes > 0:
        passes = passes[: args.max_passes]
    return _attach_online_features(
        passes,
        args.db_path,
        weather_cols,
        args.image_weather_csv,
        args.image_tolerance,
    )


def _predict_rows(model, dataset: PassDataset, passes: list[dict], device: torch.device, batch_size: int) -> list[dict]:
    preds = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            sat_idx = batch["satellite_idx"].to(device).long()
            rain_pred, _, _ = model(features, mask, sat_idx)
            preds.extend(rain_pred.detach().cpu().numpy().reshape(-1).astype(float).tolist())

    rows = []
    for p, y_pred in zip(passes, preds):
        ts = pd.DatetimeIndex(p["timestamps"])
        meta = p.get("label_meta", {})
        y_true = None
        row = {
            "satellite_id": int(p["satellite_id"]),
            "pass_start": str(ts[0]),
            "pass_end": str(ts[-1]),
            "pred_rainfall_mm": float(y_pred),
            "true_rainfall_mm": y_true,
            "rain_rate_mean": float(meta.get("rain_rate_mean", 0.0) or 0.0),
            "rain_rate_max": float(meta.get("rain_rate_max", 0.0) or 0.0),
            "rainy_ratio": float(meta.get("rainy_ratio", 0.0) or 0.0),
        }
        image = p.get("image_weather")
        if image is not None:
            image = np.asarray(image, dtype=np.float64)
            if image.ndim == 2 and image.shape[1] >= 4:
                row["prob_sunny"] = float(np.nanmean(image[:, 0]))
                row["prob_cloudy"] = float(np.nanmean(image[:, 1]))
                row["prob_rain"] = float(np.nanmean(image[:, 2]))
                row["image_available"] = bool(np.nanmax(image[:, 3]) > 0)
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> dict:
    ckpt_dir = Path(args.checkpoint_dir)
    device = torch.device(args.device)
    cfg, model, sat_mapper, scaler_X, scaler_y = _load_checkpoint(ckpt_dir, device)

    passes = load_all_passes(cfg)
    train_passes, val_passes, test_passes = split_passes_by_time(
        passes,
        cfg["data"]["data_split"],
        val_strategy=cfg["data"].get("val_strategy", "time"),
        seed=cfg["training"].get("seed", 42),
    )
    train_passes, val_passes, test_passes = attach_train_dry_baseline(train_passes, val_passes, test_passes, cfg)

    if args.source == "db":
        selected = _build_latest_db_passes(cfg, args)
        if args.satellite_id is not None:
            selected = [p for p in selected if int(p["satellite_id"]) == args.satellite_id]
        _, _, selected = attach_train_dry_baseline(train_passes, [], selected, cfg)
    else:
        split_map = {"train": train_passes, "val": val_passes, "test": test_passes, "all": train_passes + val_passes + test_passes}
        selected = list(split_map[args.split])
        if args.satellite_id is not None:
            selected = [p for p in selected if int(p["satellite_id"]) == args.satellite_id]
        selected = sorted(selected, key=lambda p: str(pd.DatetimeIndex(p["timestamps"])[-1]), reverse=args.latest_first)
        if args.max_passes > 0:
            selected = selected[: args.max_passes]

    dataset = PassDataset(
        selected,
        sat_mapper,
        max_len=cfg["model"]["max_seq_len"],
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        fit_scalers=False,
        extra_feature_keys=_optional_feature_keys(cfg),
        target_names=list(cfg["targets"]["primary"]) + list(cfg["targets"].get("auxiliary", [])),
    )
    rows = _predict_rows(model, dataset, selected, device, args.batch_size)
    pred_values = [r["pred_rainfall_mm"] for r in rows]
    rows_by_rain = sorted(rows, key=lambda r: r["pred_rainfall_mm"], reverse=True)

    return {
        "expert": "stage1_inversion",
        "mode": "online_model_forward",
        "source": args.source,
        "task": "pass_level_rainfall_inversion",
        "checkpoint_dir": str(ckpt_dir),
        "device": str(device),
        "split": args.split,
        "rows": len(rows),
        "pred_rainfall_mm_mean": round(float(np.mean(pred_values)), 6) if pred_values else None,
        "pred_rainfall_mm_max": round(float(np.max(pred_values)), 6) if pred_values else None,
        "recent_passes": rows[: args.limit],
        "highest_predicted_rain_passes": rows_by_rain[: args.limit],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--source", choices=("db", "cached_split"), default="db")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--lookback-hours", type=float, default=48.0)
    parser.add_argument("--pass-gap-threshold-s", type=float, default=60.0)
    parser.add_argument("--min-pass-points", type=int, default=10)
    parser.add_argument("--image-weather-csv", default=DEFAULT_IMAGE_WEATHER_CSV)
    parser.add_argument("--image-tolerance", default="10min")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--satellite-id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-passes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latest-first", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
