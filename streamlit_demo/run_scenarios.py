#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_scenarios.py — 3가지 시나리오 CLI 구동 예시

웹 UI 없이 데이터 생성 → 추론 파이프라인을 한 번에 검증할 때 사용.

  python run_scenarios.py
"""
from __future__ import annotations

from data_generator import SCENARIOS, generate_scenario_data
from inference_api import get_glucose_at_minute, predict_glucose_horizon, predict_risk_warning

RISK_THRESHOLD = 0.5


def main() -> None:
    print("=" * 60)
    print("CGM Streamlit Demo — 시나리오 일괄 구동")
    print("=" * 60)

    for scenario in SCENARIOS:
        df = generate_scenario_data(scenario)
        risk = predict_risk_warning(df)
        horizon = predict_glucose_horizon(df)

        signal = "🔴 위험" if risk["risk_score"] >= RISK_THRESHOLD else "🟢 안전"
        g_min, g_max = df["glucose"].min(), df["glucose"].max()

        print(f"\n[{scenario}]")
        print(f"  데이터: {len(df)} rows | glucose {g_min:.0f}–{g_max:.0f} mg/dL")
        print(f"  Model 1 앵커: {horizon['anchor_glucose']:.1f} mg/dL")
        print(f"  Model 1 2h 예측: {horizon['predictions'][-1]:.1f} mg/dL")
        print(
            f"  Model 2 TAR {risk['hyper_prob']:.0%} | TBR {risk['hypo_prob']:.0%} "
            f"| 내일 평균 {risk['predicted_next_day_mean']:.0f} mg/dL "
            f"({risk.get('next_day_model_source', '?')}) | {signal}"
        )
        g30 = get_glucose_at_minute(df, 30)
        print(f"  get_glucose_at_minute(30) → {g30['glucose_mgdl']:.0f} mg/dL (matched {g30['matched_minutes']}m)")

    print("\n" + "=" * 60)
    print("웹 데모 실행: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
