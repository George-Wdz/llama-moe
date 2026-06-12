#!/usr/bin/env python
"""Run Stage2 GPT4TS long-term forecasting online and emit JSON."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


STAGE2_ROOT = Path("/home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting")
DEFAULT_ROOT_PATH = STAGE2_ROOT / "datasets/weather"
DEFAULT_DATA_PATH = "26051801_2026_05_24_21_48_52_utf8.csv"
DEFAULT_DB_PATH = "/home/wdz/satellite_data/satellite_data.db"
DEFAULT_CHECKPOINT = (
    STAGE2_ROOT
    / "checkpoints/weather_custom_GPT4TS_6_512_336_10min_100_sl336_ll48_pl336_dm768_nh4_el3_gl6_df768_ebtimeF_itr0/checkpoint.pth"
)

sys.path.insert(0, str(STAGE2_ROOT))

from data_provider.data_loader import Dataset_Pred  # noqa: E402
from models.GPT4TS import GPT4TS  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def _build_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_id="online_stage2_forecast",
        checkpoints=str(DEFAULT_CHECKPOINT.parent.parent),
        root_path=str(Path(args.root_path)),
        data_path=args.data_path,
        data="custom",
        features=args.features,
        freq=args.freq,
        target=args.target,
        embed="timeF",
        percent=100,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        label_len=48,
        batch_size=1,
        num_workers=0,
        gpt_layers=6,
        is_gpt=1,
        e_layers=3,
        d_model=768,
        n_heads=4,
        d_ff=768,
        dropout=0.3,
        enc_in=args.enc_in,
        c_out=args.c_out,
        patch_size=16,
        kernel_size=25,
        pretrain=1,
        freeze=1,
        model="GPT4TS",
        stride=8,
        max_len=-1,
        hid_dim=16,
        tmax=10,
        itr=1,
        cos=0,
        learning_rate=0.0001,
        decay_fac=0.9,
        train_epochs=10,
        lradj="type3",
        patience=3,
        loss_func="mse",
    )


def _prediction_dates(csv_path: Path, freq: str, pred_len: int) -> list[str]:
    df = pd.read_csv(csv_path)
    last_date = pd.to_datetime(df["date"].iloc[-1])
    dates = pd.date_range(last_date, periods=pred_len + 1, freq=freq)[1:]
    return [str(d) for d in dates]


def _write_latest_db_history(args: argparse.Namespace) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if args.source != "db":
        return Path(args.root_path) / args.data_path, None

    rows_needed = max(args.history_rows, args.seq_len)
    with sqlite3.connect(f"file:{args.db_path}?mode=ro", uri=True) as conn:
        query = """
            SELECT datetime AS date,
                   temperature AS "大气温度(℃)",
                   humidity AS "大气湿度(%RH)",
                   pressure AS "数字气压(hPa)",
                   wind_speed AS "超声波风速(m/s)",
                   wind_direction AS "超声波风向(°)",
                   rainfall AS "雨量(mm)"
            FROM weather_station
            WHERE temperature IS NOT NULL
              AND humidity IS NOT NULL
              AND pressure IS NOT NULL
              AND wind_speed IS NOT NULL
              AND wind_direction IS NOT NULL
              AND rainfall IS NOT NULL
            ORDER BY datetime DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=[rows_needed])

    if len(df) < args.seq_len:
        raise ValueError(f"weather_station has only {len(df)} valid rows, need seq_len={args.seq_len}")

    df["date"] = pd.to_datetime(df["date"], format="ISO8601")
    df = df.sort_values("date").tail(rows_needed)
    tmpdir = tempfile.TemporaryDirectory(prefix="stage2_online_")
    out = Path(tmpdir.name) / "latest_weather_history.csv"
    df.to_csv(out, index=False)
    return out, tmpdir


def run(args: argparse.Namespace) -> dict:
    history_csv, tmpdir = _write_latest_db_history(args)
    root_path = history_csv.parent
    data_path = history_csv.name
    checkpoint = Path(args.checkpoint)
    csv_path = root_path / data_path
    device = torch.device(args.device)
    args.root_path = str(root_path)
    args.data_path = data_path
    model_args = _build_args(args)

    old_cwd = Path.cwd()
    os.chdir(STAGE2_ROOT)
    try:
        pred_data = Dataset_Pred(
            root_path=model_args.root_path,
            data_path=model_args.data_path,
            flag="pred",
            size=[model_args.seq_len, model_args.label_len, model_args.pred_len],
            features=model_args.features,
            target=model_args.target,
            timeenc=1,
            freq=model_args.freq,
        )
        pred_loader = DataLoader(pred_data, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        model = GPT4TS(model_args, device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()

        with torch.no_grad():
            batch_x, _, _, _ = next(iter(pred_loader))
            batch_x = batch_x.float().to(device)
            outputs = model(batch_x[:, -model_args.seq_len :, :], 0)
            outputs = outputs[:, -model_args.pred_len :, :].detach().cpu().numpy()[0]
    finally:
        os.chdir(old_cwd)

    if hasattr(pred_data, "scaler") and hasattr(pred_data.scaler, "inverse_transform"):
        outputs_denorm = pred_data.scaler.inverse_transform(outputs)
    else:
        outputs_denorm = outputs

    feature_names = getattr(pred_data, "feature_names", [args.target])
    target_idx = feature_names.index(args.target) if args.target in feature_names else len(feature_names) - 1
    target_pred_raw = outputs_denorm[:, target_idx].astype(float)
    target_pred = np.maximum(target_pred_raw, 0.0)
    dates = _prediction_dates(csv_path, args.freq, model_args.pred_len)

    steps = []
    for idx, value in enumerate(target_pred[: args.limit]):
        steps.append({
            "step": idx + 1,
            "time": dates[idx] if idx < len(dates) else None,
            "pred_rainfall_mm": round(float(value), 6),
            "raw_pred_rainfall_mm": round(float(target_pred_raw[idx]), 6),
        })

    return {
        "expert": "stage2_forecast",
        "mode": "online_model_forward",
        "source": args.source,
        "task": "long_term_rainfall_forecasting",
        "checkpoint": str(checkpoint),
        "data_path": str(csv_path),
        "history_rows": args.history_rows,
        "device": str(device),
        "features": args.features,
        "target": args.target,
        "freq": args.freq,
        "seq_len": model_args.seq_len,
        "horizon_steps": model_args.pred_len,
        "pred_rainfall_mm_mean": round(float(np.mean(target_pred)), 6),
        "pred_rainfall_mm_total": round(float(np.sum(target_pred)), 6),
        "pred_rainfall_mm_max": round(float(np.max(target_pred)), 6),
        "pred_rainfall_max_step": int(np.argmax(target_pred)) + 1,
        "forecast_head": steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source", choices=("db", "csv"), default="db")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--root-path", default=str(DEFAULT_ROOT_PATH))
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--features", choices=("S", "M", "MS"), default="M")
    parser.add_argument("--target", default="雨量(mm)")
    parser.add_argument("--freq", default="10min")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--pred-len", type=int, default=336)
    parser.add_argument("--history-rows", type=int, default=512)
    parser.add_argument("--enc-in", type=int, default=6)
    parser.add_argument("--c-out", type=int, default=6)
    parser.add_argument("--limit", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
