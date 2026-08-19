# -*- coding: utf-8 -*-
"""
data_generator.py — Streamlit 데모용 더미 CGM 시계열 생성기
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SCENARIOS = (
    "안정(Stable)",
    "고혈당(Hyper)",
    "저혈당(Hypo)",
    "운동 후 급락(Post-Exercise Drop)",
    "식후 급등(Meal Spike)",
)
SLOT_MINUTES = 15
LOOKBACK_HOURS = 12
N_SLOTS = LOOKBACK_HOURS * 60 // SLOT_MINUTES  # 48 × 15min = 12h


def generate_scenario_data(scenario: str, seed: int | None = 42) -> pd.DataFrame:
    """
    과거 12시간(15분 간격) 더미 시계열 생성.

    Parameters
    ----------
    scenario : SCENARIOS 중 하나

    Returns
    -------
    DataFrame
        columns: timestamp, glucose, steps, heart_rate, scenario
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")

    rng = np.random.RandomState(seed)
    now = datetime.now().replace(second=0, microsecond=0)
    timestamps = [now - timedelta(minutes=SLOT_MINUTES * (N_SLOTS - 1 - i)) for i in range(N_SLOTS)]
    t = np.linspace(0, 1, N_SLOTS)

    if scenario == "안정(Stable)":
        base = 115 + 12 * np.sin(2 * np.pi * t) + rng.normal(0, 4, N_SLOTS)
        glucose = np.clip(base, 85, 155)
        steps = np.clip(rng.poisson(180, N_SLOTS) + 40 * np.sin(2 * np.pi * t), 0, 600)
        heart_rate = np.clip(72 + rng.normal(0, 3, N_SLOTS), 60, 95)

    elif scenario == "고혈당(Hyper)":
        meal_rise = 60 * (1 / (1 + np.exp(-12 * (t - 0.35))))
        base = 110 + meal_rise + 15 * t + rng.normal(0, 6, N_SLOTS)
        glucose = np.clip(base, 140, 280)
        steps = np.clip(rng.poisson(120, N_SLOTS), 0, 400)
        heart_rate = np.clip(78 + 8 * t + rng.normal(0, 4, N_SLOTS), 65, 110)

    elif scenario == "저혈당(Hypo)":
        drop = 90 * (t ** 1.8)
        base = 145 - drop + rng.normal(0, 5, N_SLOTS)
        glucose = np.clip(base, 45, 130)
        steps = np.clip(rng.poisson(250, N_SLOTS) + 30 * (1 - t), 0, 800)
        heart_rate = np.clip(80 - 12 * t + rng.normal(0, 4, N_SLOTS), 55, 100)

    elif scenario == "운동 후 급락(Post-Exercise Drop)":
        # 초반 정상 → 후반 steps 급증 + 혈당 60 이하 급락
        exercise_onset = 0.55
        exercise_intensity = 1 / (1 + np.exp(-18 * (t - exercise_onset)))
        base = 118 + 8 * np.sin(2 * np.pi * t * 0.8) + rng.normal(0, 3, N_SLOTS)
        drop = 75 * exercise_intensity
        glucose = np.clip(base - drop, 48, 140)
        steps = np.clip(
            rng.poisson(150, N_SLOTS) + 700 * exercise_intensity + rng.normal(0, 30, N_SLOTS),
            0, 1200,
        ).astype(int)
        heart_rate = np.clip(72 + 35 * exercise_intensity + rng.normal(0, 4, N_SLOTS), 65, 155)

    else:  # 식후 급등(Meal Spike)
        # 중반 이후 급격한 식후 스파이크 → 220+ mg/dL
        meal_time = 0.48
        spike = 130 * (1 / (1 + np.exp(-16 * (t - meal_time))))
        base = 105 + spike + rng.normal(0, 5, N_SLOTS)
        glucose = np.clip(base, 90, 265)
        # 스파이크 구간에서 peak를 220+로 보장
        peak_idx = int(np.argmax(glucose))
        if glucose[peak_idx] < 220:
            glucose[peak_idx:] = np.clip(glucose[peak_idx:] + (220 - glucose[peak_idx]), 90, 265)
        steps = np.clip(rng.poisson(100, N_SLOTS), 0, 350)
        heart_rate = np.clip(74 + 6 * (glucose - 100) / 40 + rng.normal(0, 3, N_SLOTS), 65, 115)

    return pd.DataFrame({
        "timestamp": timestamps,
        "glucose": np.round(glucose, 1),
        "steps": steps.astype(int),
        "heart_rate": np.round(heart_rate, 0).astype(int),
        "scenario": scenario,
    })
