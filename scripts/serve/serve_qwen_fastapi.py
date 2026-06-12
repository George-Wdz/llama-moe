#!/usr/bin/env python
"""FastAPI service for Qwen text generation and weather expert context."""

import argparse
import base64
import csv
import sqlite3
import io
import json
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_DIR = "/home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct"
DEFAULT_VISION_DIR = "/home/wdz/LLaMA-Factory/leo_model/vision"
DEFAULT_STAGE1_CHECKPOINT_DIR = (
    "/home/wdz/BT/Stage1/model/checkpoints/pass_dataset_rain_retrieval_20260610_1420/"
    "stage1_cm_dm256_df512_eh8_el3_dl2_pl8_st4_bs32_lr0.0001_itr0"
)
DEFAULT_STAGE2_CHECKPOINT = (
    "/home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting/checkpoints/"
    "weather_custom_GPT4TS_6_512_336_10min_100_sl336_ll48_pl336_dm768_nh4_el3_gl6_df768_ebtimeF_itr0/checkpoint.pth"
)
DEFAULT_STAGE2_ROOT_PATH = "/home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting/datasets/weather"
DEFAULT_STAGE2_DATA_PATH = "26051801_2026_05_24_21_48_52_utf8.csv"
DEFAULT_SENSOR_DB_PATH = "/home/wdz/satellite_data/satellite_data.db"
SERVE_DIR = Path(__file__).resolve().parent
DEFAULT_STAGE1_SCRIPT = str(SERVE_DIR / "experts/stage1_online_infer.py")
DEFAULT_STAGE2_SCRIPT = str(SERVE_DIR / "experts/stage2_online_forecast.py")
LLAMA_FACTORY_ROOT = "/home/wdz/LLaMA-Factory"
WEATHER_LABEL_ZH = {
    "sunny": "晴天",
    "cloudy": "多云",
    "rain": "下雨",
    "rainy": "下雨",
}

sys.path.insert(0, LLAMA_FACTORY_ROOT)
from leo_model.vision.models import WeatherClassifier  # noqa: E402


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=80, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.1)
    image_base64: Optional[str] = None
    image_name: Optional[str] = None
    task_mode: str = Field(default="auto")


class WeatherClassifyRequest(BaseModel):
    image_base64: str
    image_name: Optional[str] = None


class Stage1InversionRequest(BaseModel):
    split: Optional[str] = None
    satellite_id: Optional[int] = None
    limit: int = Field(default=5, ge=1, le=50)


class Stage2ForecastRequest(BaseModel):
    limit: int = Field(default=24, ge=1, le=336)


class GenerateResponse(BaseModel):
    prompt: str
    model_prompt: str
    generated_text: str
    full_text: str
    input_tokens: int
    output_tokens: int
    image_received: bool = False
    image_name: Optional[str] = None
    image_bytes: int = 0
    modality_status: str = "text_only"
    weather: Optional[dict] = None
    stage1_inversion: Optional[dict] = None
    stage2_forecast: Optional[dict] = None
    route_decision: Optional[dict] = None


def _latest_weight(weights_dir: Path) -> Path:
    candidates = sorted(weights_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise ValueError(f"no *.pt found under {weights_dir}")
    return candidates[0]


def _preprocess_image(img: Image.Image, image_size: int) -> torch.Tensor:
    img = img.convert("RGB")
    img = img.resize((image_size, image_size), resample=Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(3, 1, 1)
    return (x - mean) / std


def _safe_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _latest_stage1_predictions() -> Path:
    base = Path("/home/wdz/BT/Stage1/analysis/satellite_weather_diff")
    candidates = []
    for p in base.glob("runs/*/*_predictions.csv"):
        if "_test_predictions" not in p.name:
            candidates.append(p)
    for p in base.glob("*predictions*.csv"):
        if "_test_predictions" not in p.name and "legacy" not in str(p):
            candidates.append(p)
    if not candidates:
        candidates = [p for p in base.glob("**/*predictions*.csv") if "_test_predictions" not in p.name]
    if not candidates:
        raise ValueError(f"no Stage1 predictions csv found under {base}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _latest_stage2_result_dir() -> Path:
    base = Path("/home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting/results")
    preferred = base / "weather_custom_GPT4TS_6_512_336_10min_100"
    if preferred.exists():
        return preferred
    candidates = [p for p in base.glob("*") if p.is_dir() and list(p.glob("summary_statistics*.csv"))]
    if not candidates:
        raise ValueError(f"no Stage2 result directory with summary_statistics*.csv found under {base}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _run_json_command(cmd: list[str], timeout: int) -> dict:
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "error": "expert_command_failed",
            "returncode": proc.returncode,
            "cmd": cmd,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return {
        "error": "expert_json_not_found",
        "cmd": cmd,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-2000:],
    }


class Stage1InversionRunner:
    """Polls the sensor DB and keeps the latest Stage1 pass inversion state."""

    def __init__(
        self,
        script: str,
        checkpoint_dir: str,
        db_path: str,
        device: str,
        max_passes: int,
        timeout: int,
        lookback_hours: float,
        poll_interval_s: float,
        stale_after_s: float,
    ):
        self.script = Path(script).expanduser()
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        self.db_path = Path(db_path).expanduser()
        self.device = device
        self.max_passes = max_passes
        self.timeout = timeout
        self.lookback_hours = lookback_hours
        self.poll_interval_s = poll_interval_s
        self.stale_after_s = stale_after_s
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.state = {
            "expert": "stage1_inversion",
            "mode": "background_polling",
            "source": "db",
            "status": "not_started",
            "message": "Stage1 worker not started",
            "pred_rainfall_mm": None,
            "updated_at": None,
        }
        print(
            f"Stage1 online inversion script: {self.script}, checkpoint={self.checkpoint_dir}, device={device}",
            flush=True,
        )

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, name="stage1-online-worker", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def _set_state(self, state: dict) -> None:
        state.setdefault("expert", "stage1_inversion")
        state.setdefault("mode", "background_polling")
        state.setdefault("source", "db")
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        with self.lock:
            self.state = state

    def _latest_phy_time(self) -> Optional[datetime]:
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
            row = conn.execute("SELECT max(localTime) FROM phy_data").fetchone()
        if not row or not row[0]:
            return None
        return datetime.fromisoformat(str(row[0]).replace("Z", "+00:00")).replace(tzinfo=None)

    def _no_latest_state(self, latest_time: Optional[datetime], reason: str) -> dict:
        age_s = None
        if latest_time is not None:
            age_s = max(0.0, (datetime.now() - latest_time).total_seconds())
        return {
            "status": "no_latest_satellite_pass",
            "message": "无最新卫星过境",
            "pred_rainfall_mm": None,
            "latest_phy_time": latest_time.isoformat(sep=" ", timespec="seconds") if latest_time else None,
            "latest_phy_age_s": round(age_s, 3) if age_s is not None else None,
            "reason": reason,
        }

    def _run_once(self) -> dict:
        latest_time = self._latest_phy_time()
        if latest_time is None:
            return self._no_latest_state(None, "phy_data_empty")

        age_s = (datetime.now() - latest_time).total_seconds()
        if age_s > self.stale_after_s:
            return self._no_latest_state(latest_time, "phy_data_stale")

        cmd = [
            sys.executable,
            str(self.script),
            "--checkpoint-dir",
            str(self.checkpoint_dir),
            "--device",
            self.device,
            "--source",
            "db",
            "--db-path",
            str(self.db_path),
            "--lookback-hours",
            str(self.lookback_hours),
            "--split",
            "test",
            "--limit",
            "5",
            "--batch-size",
            "128",
        ]
        if self.max_passes > 0:
            cmd.extend(["--max-passes", str(self.max_passes)])
        result = _run_json_command(cmd, self.timeout)
        if result.get("error"):
            result["status"] = "error"
            result["message"] = "Stage1反演失败"
            return result

        recent = result.get("recent_passes") or []
        if not recent:
            return self._no_latest_state(latest_time, "no_valid_pass_after_preprocessing")

        latest_pass = recent[0]
        pred = latest_pass.get("pred_rainfall_mm")
        status = "active"
        pass_end = latest_pass.get("pass_end")
        if pass_end:
            pass_end_dt = datetime.fromisoformat(str(pass_end).replace("Z", "+00:00")).replace(tzinfo=None)
            if (latest_time - pass_end_dt).total_seconds() > self.stale_after_s:
                status = "completed"

        return {
            **result,
            "mode": "background_polling",
            "status": status,
            "message": "最新卫星过境反演已更新",
            "pred_rainfall_mm": pred,
            "latest_phy_time": latest_time.isoformat(sep=" ", timespec="seconds"),
            "latest_phy_age_s": round(max(0.0, age_s), 3),
        }

    def _run_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._set_state(self._run_once())
            except Exception as exc:  # Keep the API alive even if the worker hits bad sensor rows.
                self._set_state({
                    "status": "error",
                    "message": "Stage1反演失败",
                    "error": repr(exc),
                    "pred_rainfall_mm": None,
                })
            self.stop_event.wait(self.poll_interval_s)

    def tick(self) -> dict:
        state = self._run_once()
        self._set_state(state)
        return self.summarize()

    def summarize(
        self,
        split: Optional[str] = None,
        satellite_id: Optional[int] = None,
        limit: int = 5,
    ) -> dict:
        with self.lock:
            state = dict(self.state)
        if limit >= 0 and "recent_passes" in state:
            state["recent_passes"] = state.get("recent_passes", [])[:limit]
        if limit >= 0 and "highest_predicted_rain_passes" in state:
            state["highest_predicted_rain_passes"] = state.get("highest_predicted_rain_passes", [])[:limit]
        return state


class Stage2ForecastRunner:
    """Runs the Stage2 checkpoint online in an isolated Python process."""

    def __init__(
        self,
        script: str,
        checkpoint: str,
        root_path: str,
        data_path: str,
        db_path: str,
        device: str,
        timeout: int,
        seq_len: int,
        pred_len: int,
        history_rows: int,
        freq: str,
    ):
        self.script = Path(script).expanduser()
        self.checkpoint = Path(checkpoint).expanduser()
        self.root_path = Path(root_path).expanduser()
        self.data_path = data_path
        self.db_path = Path(db_path).expanduser()
        self.device = device
        self.timeout = timeout
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.history_rows = history_rows
        self.freq = freq
        print(
            f"Stage2 online forecast script: {self.script}, checkpoint={self.checkpoint}, device={device}",
            flush=True,
        )

    def summarize(self, limit: int = 24) -> dict:
        cmd = [
            sys.executable,
            str(self.script),
            "--checkpoint",
            str(self.checkpoint),
            "--root-path",
            str(self.root_path),
            "--data-path",
            self.data_path,
            "--source",
            "db",
            "--db-path",
            str(self.db_path),
            "--device",
            self.device,
            "--seq-len",
            str(self.seq_len),
            "--pred-len",
            str(self.pred_len),
            "--history-rows",
            str(self.history_rows),
            "--freq",
            self.freq,
            "--limit",
            str(limit),
        ]
        return _run_json_command(cmd, self.timeout)


class WeatherVisionRunner:
    def __init__(self, vision_dir: str, weights: str, device: str):
        self.vision_dir = Path(vision_dir).expanduser()
        self.device = device
        weights_dir = self.vision_dir / "weights"
        self.weight_path = Path(weights).expanduser() if weights else _latest_weight(weights_dir)

        print(
            f"Loading weather vision model from {self.weight_path} on {device}...",
            flush=True,
        )
        ckpt = torch.load(self.weight_path, map_location="cpu")
        self.class_names = ckpt.get("class_names", [])
        if not self.class_names:
            raise ValueError("weather checkpoint does not contain class_names")
        self.image_size = int(ckpt.get("image_size", 224))
        self.model = WeatherClassifier(
            num_classes=len(self.class_names),
            dropout=float(ckpt.get("dropout", 0.2)),
            resnet_width=int(ckpt.get("resnet_width", 32)),
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print(
            f"Weather vision model loaded. classes={self.class_names}",
            flush=True,
        )

    @torch.inference_mode()
    def predict_base64(self, image_base64: str, image_name: Optional[str] = None) -> dict:
        raw = base64.b64decode(image_base64)
        with Image.open(io.BytesIO(raw)) as img:
            pixel_values = _preprocess_image(img, self.image_size).unsqueeze(0).to(self.device)

        logits = self.model(pixel_values)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()
        pred_idx = int(torch.argmax(probs).item())
        pred_label = str(self.class_names[pred_idx])
        confidence = float(probs[pred_idx].item())
        result = {
            "image_name": image_name,
            "pred_label": pred_label,
            "pred_label_zh": WEATHER_LABEL_ZH.get(pred_label, pred_label),
            "pred_idx": pred_idx,
            "confidence": round(confidence, 6),
            "weight_path": str(self.weight_path),
        }
        for idx, name in enumerate(self.class_names):
            result[f"prob_{name}"] = round(float(probs[idx].item()), 6)
        return result


class ModelRunner:
    def __init__(
        self,
        model_dir: str,
        device: str,
        dtype: str,
        device_map: str,
        vision_runner: Optional[WeatherVisionRunner] = None,
        stage1_runner: Optional[Stage1InversionRunner] = None,
        stage2_runner: Optional[Stage2ForecastRunner] = None,
    ):
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.device_map = device_map
        self.vision_runner = vision_runner
        self.stage1_runner = stage1_runner
        self.stage2_runner = stage2_runner
        self.lock = threading.Lock()
        print(
            f"Loading Qwen from {model_dir} with device={device}, device_map={device_map}, dtype={dtype}...",
            flush=True,
        )

        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

        print("Loading tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        print("Tokenizer loaded.", flush=True)
        print("Loading model weights...", flush=True)
        if device_map and device_map.lower() != "none":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )
            self.input_device = next(self.model.parameters()).device
            print(f"Model loaded with device_map={device_map}; input_device={self.input_device}.", flush=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            self.input_device = torch.device(device)
            print("Model weights loaded.", flush=True)
            print(f"Moving model to {device}...", flush=True)
            self.model.to(device)
        self.model.eval()
        print("Model loaded.", flush=True)

    def _route_experts(self, prompt: str, task_mode: str) -> dict:
        mode = (task_mode or "auto").strip().lower()
        if mode in ("text", "none", "off"):
            return {
                "task_mode": mode,
                "stage1": False,
                "stage2": False,
                "reason": "manual_text_only",
            }
        if mode in ("stage1", "inversion"):
            return {
                "task_mode": mode,
                "stage1": True,
                "stage2": False,
                "reason": "manual_stage1",
            }
        if mode in ("stage2", "forecast"):
            return {
                "task_mode": mode,
                "stage1": False,
                "stage2": True,
                "reason": "manual_stage2",
            }
        if mode in ("fusion", "all"):
            return {
                "task_mode": mode,
                "stage1": True,
                "stage2": True,
                "reason": "manual_fusion",
            }

        text = prompt.lower()
        stage1_keywords = (
            "stage1",
            "反演",
            "链路",
            "link",
            "当前天气",
            "当前降雨",
            "当前雨量",
            "rainfall inversion",
        )
        stage2_keywords = (
            "stage2",
            "未来",
            "预测",
            "长期",
            "forecast",
            "long-term",
            "long term",
            "tomorrow",
            "future",
        )
        use_stage1 = any(k in text for k in stage1_keywords)
        use_stage2 = any(k in text for k in stage2_keywords)
        return {
            "task_mode": mode or "auto",
            "stage1": use_stage1,
            "stage2": use_stage2,
            "reason": "keyword_auto",
            "stage1_keywords": [k for k in stage1_keywords if k in text],
            "stage2_keywords": [k for k in stage2_keywords if k in text],
        }

    def _build_multimodal_prompt(
        self,
        prompt: str,
        weather: Optional[dict],
        stage1_inversion: Optional[dict],
        stage2_forecast: Optional[dict],
    ) -> str:
        if not weather and not stage1_inversion and not stage2_forecast:
            return prompt

        blocks = []
        if weather:
            prob_lines = []
            for key in sorted(k for k in weather if k.startswith("prob_")):
                prob_lines.append(f"- {key}: {weather[key]}")
            probs = "\n".join(prob_lines)
            blocks.append(
                "视觉天气分类专家结果:\n"
                f"- pred_label: {weather['pred_label']}\n"
                f"- pred_label_zh: {weather.get('pred_label_zh', weather['pred_label'])}\n"
                f"- confidence: {weather['confidence']}\n"
                f"{probs}"
            )
        if stage1_inversion:
            if stage1_inversion.get("status") == "no_latest_satellite_pass":
                stage1_prompt = {
                    "message": "无最新卫星过境",
                    "pred_rainfall_mm": None,
                }
                blocks.append(
                    "Stage1 降雨反演专家结果:\n"
                    f"{json.dumps(stage1_prompt, ensure_ascii=False, indent=2)}"
                )
                stage1_inversion = None
            else:
                recent_passes = stage1_inversion.get("recent_passes", [])
                latest_rainfall = None
                if recent_passes:
                    latest_rainfall = recent_passes[0].get("pred_rainfall_mm")
                if latest_rainfall is None:
                    latest_rainfall = stage1_inversion.get("pred_rainfall_mm")
                if latest_rainfall is None:
                    latest_rainfall = stage1_inversion.get("pred_rainfall_mm_mean")
                stage1_prompt = {
                    "pred_rainfall_mm": latest_rainfall,
                }
                blocks.append(
                    "Stage1 降雨反演专家结果:\n"
                    f"{json.dumps(stage1_prompt, ensure_ascii=False, indent=2)}"
                )
        if stage2_forecast:
            stage2_prompt = {
                "expert": stage2_forecast.get("expert"),
                "mode": stage2_forecast.get("mode"),
                "source": stage2_forecast.get("source"),
                "freq": stage2_forecast.get("freq"),
                "seq_len": stage2_forecast.get("seq_len"),
                "horizon_steps": stage2_forecast.get("horizon_steps"),
                "pred_rainfall_mm_mean": stage2_forecast.get("pred_rainfall_mm_mean"),
                "pred_rainfall_mm_total": stage2_forecast.get("pred_rainfall_mm_total"),
                "pred_rainfall_mm_max": stage2_forecast.get("pred_rainfall_mm_max"),
                "pred_rainfall_max_step": stage2_forecast.get("pred_rainfall_max_step"),
                "forecast_head": stage2_forecast.get("forecast_head", [])[:6],
            }
            blocks.append(
                "Stage2 长期降雨预测专家结果:\n"
                f"{json.dumps(stage2_prompt, ensure_ascii=False, indent=2)}"
            )
        evidence = "\n\n".join(blocks)
        user_prompt = prompt.strip() or "请根据专家结果，简要说明当前输入反映的天气情况。"
        return (
            "你会收到在线气象专家的结构化输出。"
            "只允许基于这些专家结果回答，不要补充没有依据的细节。"
            "除非用户明确要求英文，否则一律用中文回答。"
            "回答要简洁，优先给出结论和关键数值。\n\n"
            f"{evidence}\n\n"
            "用户问题:\n"
            f"{user_prompt}\n\n"
            "回答:"
        )

    @torch.inference_mode()
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        weather = None
        if request.image_base64 and self.vision_runner is not None:
            weather = self.vision_runner.predict_base64(
                request.image_base64, request.image_name
            )

        route_decision = self._route_experts(request.prompt, request.task_mode)
        route_decision["vision"] = bool(request.image_base64 and self.vision_runner is not None)
        use_stage1 = bool(route_decision.get("stage1"))
        use_stage2 = bool(route_decision.get("stage2"))
        stage1_inversion = None
        stage2_forecast = None
        if use_stage1 and self.stage1_runner is not None:
            stage1_inversion = self.stage1_runner.summarize(split="test", limit=3)
        if use_stage2 and self.stage2_runner is not None:
            stage2_forecast = self.stage2_runner.summarize(limit=12)

        if (
            stage1_inversion
            and stage1_inversion.get("status") == "no_latest_satellite_pass"
            and not stage2_forecast
            and not weather
        ):
            return GenerateResponse(
                prompt=request.prompt,
                model_prompt='{"message":"无最新卫星过境","pred_rainfall_mm":null}',
                generated_text="无最新卫星过境",
                full_text="无最新卫星过境",
                input_tokens=0,
                output_tokens=0,
                image_received=bool(request.image_base64),
                image_name=request.image_name,
                image_bytes=int(len(request.image_base64) * 3 / 4) if request.image_base64 else 0,
                modality_status="text_only+stage1_inversion",
                weather=weather,
                stage1_inversion=stage1_inversion,
                stage2_forecast=stage2_forecast,
                route_decision=route_decision,
            )

        model_prompt = self._build_multimodal_prompt(
            request.prompt,
            weather,
            stage1_inversion,
            stage2_forecast,
        )
        prompt_for_model = model_prompt
        if getattr(self.tokenizer, "chat_template", None):
            prompt_for_model = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": model_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(prompt_for_model, return_tensors="pt").to(self.input_device)
        do_sample = request.temperature > 0

        # One GPU can only serve a small number of concurrent generations well.
        # Serialize generation first; add batching/replicas after the API is stable.
        with self.lock:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=do_sample,
                temperature=request.temperature if do_sample else None,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0, input_len:]
        image_bytes = 0
        if request.image_base64:
            image_bytes = int(len(request.image_base64) * 3 / 4)
        if request.image_base64 and weather is not None:
            modality_status = "image_weather_classifier_connected"
        elif request.image_base64:
            modality_status = "image_received_vision_disabled"
        else:
            modality_status = "text_only"
        if stage1_inversion and stage2_forecast:
            modality_status = f"{modality_status}+stage1_inversion+stage2_forecast"
        elif stage1_inversion:
            modality_status = f"{modality_status}+stage1_inversion"
        elif stage2_forecast:
            modality_status = f"{modality_status}+stage2_forecast"
        return GenerateResponse(
            prompt=request.prompt,
            model_prompt=model_prompt,
            generated_text=self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ),
            full_text=self.tokenizer.decode(output_ids[0], skip_special_tokens=True),
            input_tokens=int(input_len),
            output_tokens=int(generated_ids.shape[-1]),
            image_received=bool(request.image_base64),
            image_name=request.image_name,
            image_bytes=image_bytes,
            modality_status=modality_status,
            weather=weather,
            stage1_inversion=stage1_inversion,
            stage2_forecast=stage2_forecast,
            route_decision=route_decision,
        )


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Qwen 在线推理</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #6b7280;
      --line: #d8dee8;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --warn: #92400e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    main {
      width: min(1120px, calc(100vw - 32px));
      margin: 24px auto;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 16px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    h1 {
      width: min(1120px, calc(100vw - 32px));
      margin: 24px auto 0;
      font-size: 22px;
      font-weight: 650;
    }
    label {
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    select, textarea, input[type="number"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      font: inherit;
      background: #fff;
    }
    textarea {
      min-height: 150px;
      resize: vertical;
      line-height: 1.5;
      border: 0;
      padding: 0;
      outline: none;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 12px;
    }
    .field { margin-top: 14px; }
    button {
      width: 100%;
      height: 42px;
      margin-top: 16px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    button:hover { background: var(--accent-strong); }
    button:disabled {
      cursor: wait;
      opacity: 0.65;
    }
    .composer {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }
    .composer-tools {
      display: flex;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid var(--line);
    }
    .file-tools {
      display: flex;
      gap: 8px;
      align-items: center;
      min-width: 0;
    }
    .small-button {
      width: auto;
      height: 34px;
      margin-top: 0;
      padding: 0 12px;
      background: #eef2f7;
      color: var(--text);
      font-weight: 600;
    }
    .small-button:hover {
      background: #e4e9f1;
    }
    .file-name {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 170px;
    }
    #image {
      display: none;
    }
    .preview {
      width: 100%;
      min-height: 88px;
      margin-top: 10px;
      border: 1px dashed var(--line);
      border-radius: 6px;
      display: grid;
      place-items: center;
      overflow: hidden;
      color: var(--muted);
      background: #fbfcfd;
    }
    .preview img {
      width: 100%;
      max-height: 220px;
      object-fit: contain;
      height: auto;
      display: block;
    }
    .status {
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      min-height: 20px;
    }
    .notice {
      color: var(--warn);
      font-size: 13px;
      line-height: 1.5;
      margin-top: 10px;
    }
    .answer {
      min-height: 420px;
      white-space: pre-wrap;
      line-height: 1.6;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
      background: #fbfcfd;
    }
    .meta {
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
      white-space: pre-wrap;
    }
    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <h1>Qwen 在线推理</h1>
  <main>
    <section>
      <div class="field" style="margin-top: 0">
        <label for="prompt">输入内容</label>
        <div class="composer">
          <textarea id="prompt" placeholder="输入问题，也可以附加一张图片一起发送">请根据已有专家结果，分析未来长期降雨趋势。</textarea>
          <div class="preview" id="preview">未选择图片</div>
          <div class="composer-tools">
            <div class="file-tools">
              <input id="image" type="file" accept="image/*" />
              <button class="small-button" id="chooseImage" type="button">选择图片</button>
              <button class="small-button" id="clearImage" type="button">清除</button>
            </div>
            <div class="file-name" id="fileName">无图片</div>
          </div>
        </div>
      </div>

      <div class="field">
        <label for="taskMode">专家路由</label>
        <select id="taskMode">
          <option value="auto" selected>自动判断</option>
          <option value="text">只用文本</option>
          <option value="stage1">Stage1 反演</option>
          <option value="stage2">Stage2 长期预测</option>
          <option value="fusion">Stage1 + Stage2</option>
        </select>
      </div>

      <div class="notice">
        当前图片会先进入在线天气分类模型，分类概率会作为视觉上下文传给 Qwen。
      </div>

      <div class="row">
        <div>
          <label for="maxNewTokens">输出 tokens</label>
          <input id="maxNewTokens" type="number" min="1" max="1024" value="80" />
        </div>
        <div>
          <label for="temperature">temperature</label>
          <input id="temperature" type="number" min="0" step="0.1" value="0.0" />
        </div>
      </div>

      <button id="send">发送</button>
      <div class="status" id="status"></div>
    </section>

    <section>
      <label>模型输出</label>
      <div class="answer" id="answer"></div>
      <div class="meta" id="meta"></div>
    </section>
  </main>

  <script>
    const imageInput = document.getElementById("image");
    const chooseImage = document.getElementById("chooseImage");
    const clearImage = document.getElementById("clearImage");
    const fileName = document.getElementById("fileName");
    const preview = document.getElementById("preview");
    const send = document.getElementById("send");
    const statusEl = document.getElementById("status");
    const answer = document.getElementById("answer");
    const meta = document.getElementById("meta");
    let imageBase64 = null;
    let imageName = null;

    chooseImage.addEventListener("click", () => imageInput.click());
    clearImage.addEventListener("click", () => {
      imageInput.value = "";
      imageBase64 = null;
      imageName = null;
      fileName.textContent = "无图片";
      preview.textContent = "未选择图片";
    });

    imageInput.addEventListener("change", () => {
      const file = imageInput.files[0];
      imageBase64 = null;
      imageName = null;
      if (!file) {
        preview.textContent = "未选择图片";
        fileName.textContent = "无图片";
        return;
      }
      imageName = file.name;
      fileName.textContent = file.name;
      const reader = new FileReader();
      reader.onload = () => {
        const dataUrl = reader.result;
        imageBase64 = String(dataUrl).split(",")[1] || "";
        preview.innerHTML = "";
        const img = document.createElement("img");
        img.src = dataUrl;
        img.alt = file.name;
        preview.appendChild(img);
      };
      reader.readAsDataURL(file);
    });

    send.addEventListener("click", async () => {
      send.disabled = true;
      statusEl.textContent = "生成中...";
      answer.textContent = "";
      meta.textContent = "";
      try {
        const payload = {
          prompt: document.getElementById("prompt").value,
          max_new_tokens: Number(document.getElementById("maxNewTokens").value),
          temperature: Number(document.getElementById("temperature").value),
          task_mode: document.getElementById("taskMode").value,
          image_base64: imageBase64,
          image_name: imageName
        };
        const res = await fetch("/generate", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || data.message || JSON.stringify(data));
        }
        answer.textContent = data.generated_text || "";
        meta.textContent =
          `input_tokens: ${data.input_tokens}\n` +
          `output_tokens: ${data.output_tokens}\n` +
          `modality_status: ${data.modality_status}\n` +
          `image_received: ${data.image_received}\n` +
          `image_name: ${data.image_name || ""}\n` +
          `image_bytes: ${data.image_bytes || 0}\n\n` +
          `route_decision:\n${JSON.stringify(data.route_decision || {}, null, 2)}\n\n` +
          `weather:\n${JSON.stringify(data.weather || {}, null, 2)}\n\n` +
          `stage1_inversion:\n${JSON.stringify(data.stage1_inversion || {}, null, 2)}\n\n` +
          `stage2_forecast:\n${JSON.stringify(data.stage2_forecast || {}, null, 2)}`;
        statusEl.textContent = "完成";
      } catch (err) {
        statusEl.textContent = "请求失败";
        answer.textContent = String(err);
      } finally {
        send.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def create_app(runner: ModelRunner) -> FastAPI:
    app = FastAPI(title="Qwen Inference API", version="0.1.0")

    @app.on_event("startup")
    def startup():
        if runner.stage1_runner is not None:
            runner.stage1_runner.start()

    @app.on_event("shutdown")
    def shutdown():
        if runner.stage1_runner is not None:
            runner.stage1_runner.stop()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_dir": runner.model_dir,
            "device": runner.device,
            "device_map": runner.device_map,
            "dtype": runner.dtype,
            "cuda_device_count": torch.cuda.device_count(),
            "vision_enabled": runner.vision_runner is not None,
            "vision_weight_path": (
                str(runner.vision_runner.weight_path)
                if runner.vision_runner is not None
                else None
            ),
            "vision_classes": (
                runner.vision_runner.class_names
                if runner.vision_runner is not None
                else None
            ),
            "stage1_enabled": runner.stage1_runner is not None,
            "stage1_checkpoint_dir": (
                str(runner.stage1_runner.checkpoint_dir)
                if runner.stage1_runner is not None
                else None
            ),
            "stage1_device": runner.stage1_runner.device if runner.stage1_runner is not None else None,
            "stage1_db_path": str(runner.stage1_runner.db_path) if runner.stage1_runner is not None else None,
            "stage1_lookback_hours": runner.stage1_runner.lookback_hours if runner.stage1_runner is not None else None,
            "stage1_poll_interval_s": runner.stage1_runner.poll_interval_s if runner.stage1_runner is not None else None,
            "stage1_stale_after_s": runner.stage1_runner.stale_after_s if runner.stage1_runner is not None else None,
            "stage1_status": (
                runner.stage1_runner.summarize(limit=0).get("status")
                if runner.stage1_runner is not None
                else None
            ),
            "stage2_enabled": runner.stage2_runner is not None,
            "stage2_checkpoint": (
                str(runner.stage2_runner.checkpoint)
                if runner.stage2_runner is not None
                else None
            ),
            "stage2_data_path": (
                str(runner.stage2_runner.db_path)
                if runner.stage2_runner is not None
                else None
            ),
            "stage2_device": runner.stage2_runner.device if runner.stage2_runner is not None else None,
            "stage2_seq_len": runner.stage2_runner.seq_len if runner.stage2_runner is not None else None,
            "stage2_pred_len": runner.stage2_runner.pred_len if runner.stage2_runner is not None else None,
            "stage2_history_rows": runner.stage2_runner.history_rows if runner.stage2_runner is not None else None,
            "stage2_freq": runner.stage2_runner.freq if runner.stage2_runner is not None else None,
        }

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest):
        return runner.generate(request)

    @app.post("/classify_weather")
    def classify_weather(request: WeatherClassifyRequest):
        if runner.vision_runner is None:
            return {"error": "vision_disabled"}
        return runner.vision_runner.predict_base64(
            request.image_base64, request.image_name
        )

    @app.post("/stage1_inversion")
    def stage1_inversion(request: Stage1InversionRequest):
        if runner.stage1_runner is None:
            return {"error": "stage1_disabled"}
        return runner.stage1_runner.summarize(
            split=request.split,
            satellite_id=request.satellite_id,
            limit=request.limit,
        )

    @app.get("/stage1_inversion")
    def stage1_inversion_get(limit: int = 5, split: Optional[str] = None, satellite_id: Optional[int] = None):
        if runner.stage1_runner is None:
            return {"error": "stage1_disabled"}
        return runner.stage1_runner.summarize(
            split=split,
            satellite_id=satellite_id,
            limit=limit,
        )

    @app.get("/stage1/status")
    def stage1_status():
        if runner.stage1_runner is None:
            return {"error": "stage1_disabled"}
        return runner.stage1_runner.summarize(limit=3)

    @app.post("/stage1/tick")
    def stage1_tick():
        if runner.stage1_runner is None:
            return {"error": "stage1_disabled"}
        return runner.stage1_runner.tick()

    @app.post("/stage2_forecast")
    def stage2_forecast(request: Stage2ForecastRequest):
        if runner.stage2_runner is None:
            return {"error": "stage2_disabled"}
        return runner.stage2_runner.summarize(limit=request.limit)

    @app.get("/stage2_forecast")
    def stage2_forecast_get(limit: int = 24):
        if runner.stage2_runner is None:
            return {"error": "stage2_disabled"}
        return runner.stage2_runner.summarize(limit=limit)

    return app


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map for Qwen. Use 'auto' for multi-GPU or 'none' for --device single-device loading.",
    )
    parser.add_argument("--vision-dir", default=DEFAULT_VISION_DIR)
    parser.add_argument("--vision-weights", default="")
    parser.add_argument("--stage1-script", default=DEFAULT_STAGE1_SCRIPT)
    parser.add_argument("--stage1-checkpoint-dir", default=DEFAULT_STAGE1_CHECKPOINT_DIR)
    parser.add_argument("--stage1-device", default="cpu")
    parser.add_argument("--stage1-max-passes", type=int, default=32)
    parser.add_argument("--stage1-timeout", type=int, default=120)
    parser.add_argument("--stage1-lookback-hours", type=float, default=48.0)
    parser.add_argument("--stage1-poll-interval-s", type=float, default=30.0)
    parser.add_argument("--stage1-stale-after-s", type=float, default=300.0)
    parser.add_argument("--stage2-script", default=DEFAULT_STAGE2_SCRIPT)
    parser.add_argument("--stage2-checkpoint", default=DEFAULT_STAGE2_CHECKPOINT)
    parser.add_argument("--stage2-root-path", default=DEFAULT_STAGE2_ROOT_PATH)
    parser.add_argument("--stage2-data-path", default=DEFAULT_STAGE2_DATA_PATH)
    parser.add_argument("--sensor-db-path", default=DEFAULT_SENSOR_DB_PATH)
    parser.add_argument("--stage2-device", default="cpu")
    parser.add_argument("--stage2-timeout", type=int, default=180)
    parser.add_argument("--stage2-seq-len", type=int, default=512)
    parser.add_argument("--stage2-pred-len", type=int, default=336)
    parser.add_argument("--stage2-history-rows", type=int, default=512)
    parser.add_argument("--stage2-freq", default="10min")
    parser.add_argument(
        "--vision-device",
        default="",
        help="Device for weather image classifier. Defaults to --device.",
    )
    parser.add_argument(
        "--disable-vision",
        action="store_true",
        help="Disable online weather image classification.",
    )
    parser.add_argument(
        "--disable-stage1",
        action="store_true",
        help="Disable Stage1 rainfall inversion expert.",
    )
    parser.add_argument(
        "--disable-stage2",
        action="store_true",
        help="Disable Stage2 long-term forecasting expert.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    return parser.parse_args()


def main():
    import uvicorn

    args = parse_args()
    vision_runner = None
    if not args.disable_vision:
        vision_runner = WeatherVisionRunner(
            args.vision_dir,
            args.vision_weights,
            args.vision_device or args.device,
        )
    stage1_runner = None
    if not args.disable_stage1:
        stage1_runner = Stage1InversionRunner(
            args.stage1_script,
            args.stage1_checkpoint_dir,
            args.sensor_db_path,
            args.stage1_device,
            args.stage1_max_passes,
            args.stage1_timeout,
            args.stage1_lookback_hours,
            args.stage1_poll_interval_s,
            args.stage1_stale_after_s,
        )
    stage2_runner = None
    if not args.disable_stage2:
        stage2_runner = Stage2ForecastRunner(
            args.stage2_script,
            args.stage2_checkpoint,
            args.stage2_root_path,
            args.stage2_data_path,
            args.sensor_db_path,
            args.stage2_device,
            args.stage2_timeout,
            args.stage2_seq_len,
            args.stage2_pred_len,
            args.stage2_history_rows,
            args.stage2_freq,
        )
    runner = ModelRunner(
        args.model_dir,
        args.device,
        args.dtype,
        args.device_map,
        vision_runner,
        stage1_runner,
        stage2_runner,
    )
    app = create_app(runner)
    print(f"Serving Qwen FastAPI on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
