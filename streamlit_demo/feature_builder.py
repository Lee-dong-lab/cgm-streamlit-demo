# -*- coding: utf-8 -*-
"""
feature_builder.py — 12h 시나리오 CGM → Model3 Top-20 일별 피처 변환

train_model3_top20.py 의 TOP20_FEATURES 형식에 맞춰
Streamlit 데모 입력을 LightGBM 추론용 벡터로 변환한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOP20_FEATURES = [
    "mean_glucose",
    "roll3_mean_glucose",
    "d_cv",
    "steps",
    "roll3_d_tar",
    "diff_mean_from_prev_day",
    "lag1_sleep_total_min",
    "lag1_steps",
    "lag1_mean_glucose",
    "diff_steps_from_prev_day",
    "diff_d_cv_from_prev_day",
    "lag1_d_cv",
    "sleep_rem_min",
    "sleep_light_min",
    "sleep_deep_min",
    "roll3_steps",
    "diff_sleep_from_prev_day",
    "sleep_total_min",
    "d_tir",
    "diff_d_tir_from_prev_day",
]

TIR_LOW, TIR_HIGH = 70.0, 180.0

# 시나리오별 수면 기본값 (dataset_b 중앙값 근사)
_SLEEP_DEFAULTS = {
    "Stable": dict(total=408, rem=95, light=210, deep=75),
    "Hyper": dict(total=385, rem=88, light=200, deep=70),
    "Hypo": dict(total=395, rem=90, light=205, deep=72),
    "Post-Exercise": dict(total=420, rem=100, light=215, deep=80),
    "Meal Spike": dict(total=380, rem=85, light=195, deep=68),
}


def _sleep_for_scenario(scenario: str) -> dict[str, float]:
    for key, val in _SLEEP_DEFAULTS.items():
        if key in scenario:
            return val
    return _SLEEP_DEFAULTS["Stable"]


def _ada_fractions(glucose: np.ndarray) -> tuple[float, float, float]:
    g = glucose.astype(float)
    n = max(len(g), 1)
    d_tir = float(np.mean((g >= TIR_LOW) & (g <= TIR_HIGH)))
    d_tar = float(np.mean(g > TIR_HIGH))
    d_tbr = float(np.mean(g < TIR_LOW))
    return d_tir, d_tar, d_tbr


def build_top20_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    12h(15분) 시계열 → Model3 Top-20 1행 DataFrame.

    전반 6h = lag1(어제) proxy, 전체 12h = 오늘 관측으로 diff/lag/roll 근사.
    steps 는 12h 합 × 2 로 일일 걸음수 추정.
    """
    g = data["glucose"].astype(float).to_numpy()
    steps_12h = float(data["steps"].astype(float).sum()) if "steps" in data.columns else 0.0
    steps_day = steps_12h * 2.0

    mid = max(len(g) // 2, 1)
    g_early = g[:mid]
    g_late = g[mid:] if len(g) > mid else g

    mean_glucose = float(np.mean(g))
    lag1_mean_glucose = float(np.mean(g_early))
    diff_mean = mean_glucose - lag1_mean_glucose

    d_cv = float(np.std(g) / np.mean(g) * 100.0) if np.mean(g) > 0 else 0.0
    lag1_d_cv = float(np.std(g_early) / np.mean(g_early) * 100.0) if np.mean(g_early) > 0 else d_cv
    diff_d_cv = d_cv - lag1_d_cv

    d_tir, d_tar, _d_tbr = _ada_fractions(g)
    lag1_d_tir, lag1_d_tar, _ = _ada_fractions(g_early)
    diff_d_tir = d_tir - lag1_d_tir

    lag1_steps = steps_day * 0.92
    diff_steps = steps_day - lag1_steps

    roll3_mean_glucose = (lag1_mean_glucose * 2 + mean_glucose) / 3.0
    roll3_d_tar = (lag1_d_tar * 2 + d_tar) / 3.0
    roll3_steps = (lag1_steps * 2 + steps_day) / 3.0

    scenario = str(data.get("scenario", pd.Series(["안정(Stable)"])).iloc[-1])
    sleep = _sleep_for_scenario(scenario)
    lag1_sleep = sleep["total"] * 0.98
    diff_sleep = sleep["total"] - lag1_sleep

    row = {
        "mean_glucose": mean_glucose,
        "roll3_mean_glucose": roll3_mean_glucose,
        "d_cv": d_cv,
        "steps": steps_day,
        "roll3_d_tar": roll3_d_tar,
        "diff_mean_from_prev_day": diff_mean,
        "lag1_sleep_total_min": lag1_sleep,
        "lag1_steps": lag1_steps,
        "lag1_mean_glucose": lag1_mean_glucose,
        "diff_steps_from_prev_day": diff_steps,
        "diff_d_cv_from_prev_day": diff_d_cv,
        "lag1_d_cv": lag1_d_cv,
        "sleep_rem_min": sleep["rem"],
        "sleep_light_min": sleep["light"],
        "sleep_deep_min": sleep["deep"],
        "roll3_steps": roll3_steps,
        "diff_sleep_from_prev_day": diff_sleep,
        "sleep_total_min": sleep["total"],
        "d_tir": d_tir,
        "diff_d_tir_from_prev_day": diff_d_tir,
    }
    return pd.DataFrame([row], columns=TOP20_FEATURES)
