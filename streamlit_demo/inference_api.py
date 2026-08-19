# -*- coding: utf-8 -*-
"""
inference_api.py — Streamlit 데모 추론 API

Model 1 : CGMPatchTCN 대체 휴리스틱 (horizon)
Model 2 : TBR/TAR 위험 휴리스틱
Model 3 : train_model3_top20.py LightGBM (next-day mean glucose) — 실제 모델
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from model3_inference import predict_next_day_mean_glucose

HORIZON_MINUTES = (15, 30, 45, 60, 75, 90, 105, 120)
HYPO_MGDL = 70.0
HYPER_MGDL = 180.0
TIR_LOW, TIR_HIGH = 70.0, 180.0


def _last_n(arr: np.ndarray, n: int = 4) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return arr[-n:] if len(arr) >= n else arr


def _scenario_label(data: pd.DataFrame) -> str:
    if "scenario" not in data.columns:
        return "unknown"
    return str(data["scenario"].iloc[-1])


def _fallback_next_day_mean(
    data: pd.DataFrame,
    hyper_prob: float,
    hypo_prob: float,
    horizon: dict[str, Any] | None = None,
) -> float:
    """Model3 artifact 없을 때만 사용하는 휴리스틱 fallback."""
    g = data["glucose"].astype(float).to_numpy()
    observed_mean = float(np.mean(g))
    last_g = float(g[-1])
    recent_slope = float(g[-1] - g[-4]) if len(g) >= 4 else 0.0
    scenario = _scenario_label(data)
    horizon_mean = float(np.mean(horizon["predictions"])) if horizon else last_g
    base = 0.45 * observed_mean + 0.30 * last_g + 0.15 * horizon_mean + 0.10 * (last_g + recent_slope * 2)
    if "Hyper" in scenario:
        base += 18
    elif "Hypo" in scenario:
        base -= 12
    elif "Post-Exercise" in scenario:
        base -= 8 + hypo_prob * 15
    elif "Meal Spike" in scenario:
        base += 22 + hyper_prob * 20
    base += hyper_prob * 12 - hypo_prob * 10
    return float(np.clip(base, 55, 260))


def predict_glucose_horizon(data: pd.DataFrame) -> dict[str, Any]:
    """
    Model 1 — 미래 2시간(15분 × 8 horizon) 혈당 예측 (데모 휴리스틱).
    """
    g = data["glucose"].astype(float).to_numpy()
    anchor = float(g[-1])
    anchor_time = pd.Timestamp(data["timestamp"].iloc[-1])

    recent = _last_n(g, 4)
    velocity = float(np.mean(np.diff(recent))) if len(recent) > 1 else 0.0
    recent_mean = float(np.mean(recent))
    scenario = _scenario_label(data)

    preds: list[float] = []
    for i, mins in enumerate(HORIZON_MINUTES, 1):
        steps_ahead = i
        raw = anchor + velocity * steps_ahead * 0.85
        pull = (recent_mean - raw) * 0.08 * steps_ahead
        noise = np.random.RandomState(int(mins)).normal(0, 2.5)
        val = raw + pull + noise

        if "Hyper" in scenario:
            val += 0.15 * steps_ahead
        elif "Hypo" in scenario:
            val -= 0.35 * steps_ahead
        elif "Post-Exercise" in scenario:
            val -= 0.25 * steps_ahead
        elif "Meal Spike" in scenario:
            val += 0.20 * steps_ahead

        preds.append(float(np.clip(val, 40, 400)))

    return {
        "anchor_glucose": anchor,
        "anchor_time": anchor_time,
        "horizons_min": list(HORIZON_MINUTES),
        "predictions": preds,
        "scenario_hint": scenario,
    }


def get_glucose_at_minute(data: pd.DataFrame, minutes: int) -> dict[str, Any]:
    """입력 minutes와 가장 가까운 Horizon 예측값 반환."""
    horizon = predict_glucose_horizon(data)
    mins_arr = np.array(horizon["horizons_min"], dtype=int)
    idx = int(np.argmin(np.abs(mins_arr - minutes)))
    return {
        "requested_minutes": minutes,
        "matched_minutes": int(mins_arr[idx]),
        "glucose_mgdl": float(horizon["predictions"][idx]),
    }


def predict_risk_warning(data: pd.DataFrame) -> dict[str, float | str | bool]:
    """
    Model 2 (TBR/TAR 위험, 휴리스틱) + Model 3 (내일 평균 혈당, LightGBM).

    Returns
    -------
    dict
        hyper_prob, hypo_prob, risk_score,
        predicted_next_day_mean, next_day_model_source, next_day_hyper_flag
    """
    g = data["glucose"].astype(float).to_numpy()
    steps = data["steps"].astype(float).to_numpy() if "steps" in data.columns else np.zeros_like(g)

    pct_hyper = float(np.mean(g > HYPER_MGDL))
    pct_hypo = float(np.mean(g < HYPO_MGDL))
    recent_slope = float(g[-1] - g[-4]) if len(g) >= 4 else 0.0
    last_g = float(g[-1])
    low_steps = float(np.mean(steps[-8:] < 100)) if len(steps) >= 8 else 0.0
    high_steps = float(np.mean(steps[-8:] > 500)) if len(steps) >= 8 else 0.0

    hyper_prob = (
        0.45 * pct_hyper
        + 0.25 * np.clip(recent_slope / 80.0, 0, 1)
        + 0.20 * np.clip((last_g - 140) / 80.0, 0, 1)
        + 0.10 * (1 - low_steps)
    )
    hyper_prob = float(np.clip(hyper_prob, 0.0, 1.0))

    hypo_prob = (
        0.40 * pct_hypo
        + 0.30 * np.clip(-recent_slope / 60.0, 0, 1)
        + 0.20 * np.clip((HYPO_MGDL - last_g + 20) / 50.0, 0, 1)
        + 0.10 * np.clip(np.mean(steps[-8:]) / 600.0, 0, 1)
        + 0.10 * high_steps
    )
    hypo_prob = float(np.clip(hypo_prob, 0.0, 1.0))

    scenario = _scenario_label(data)
    if "Hyper" in scenario:
        hyper_prob = float(np.clip(hyper_prob + 0.25, 0, 1))
        hypo_prob = float(np.clip(hypo_prob - 0.15, 0, 1))
    elif "Hypo" in scenario:
        hypo_prob = float(np.clip(hypo_prob + 0.30, 0, 1))
        hyper_prob = float(np.clip(hyper_prob - 0.10, 0, 1))
    elif "Post-Exercise" in scenario:
        hypo_prob = float(np.clip(hypo_prob + 0.35, 0, 1))
        hyper_prob = float(np.clip(hyper_prob - 0.08, 0, 1))
    elif "Meal Spike" in scenario:
        hyper_prob = float(np.clip(hyper_prob + 0.30, 0, 1))
        hypo_prob = float(np.clip(hypo_prob - 0.12, 0, 1))

    risk_score = float(max(hyper_prob, hypo_prob))
    horizon = predict_glucose_horizon(data)

    m3 = predict_next_day_mean_glucose(data)
    if m3["model_source"] != "unavailable" and np.isfinite(m3["predicted_next_day_mean"]):
        predicted_next_day_mean = float(m3["predicted_next_day_mean"])
        next_day_model_source = str(m3["model_source"])
        next_day_hyper_flag = bool(m3["hyper_flag"])
    else:
        predicted_next_day_mean = _fallback_next_day_mean(data, hyper_prob, hypo_prob, horizon)
        next_day_model_source = "heuristic_fallback"
        next_day_hyper_flag = predicted_next_day_mean > 140.0

    return {
        "hyper_prob": hyper_prob,
        "hypo_prob": hypo_prob,
        "risk_score": risk_score,
        "predicted_next_day_mean": predicted_next_day_mean,
        "next_day_model_source": next_day_model_source,
        "next_day_hyper_flag": next_day_hyper_flag,
    }
