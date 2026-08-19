# -*- coding: utf-8 -*-
"""
app.py — CGM 혈당 관리 Streamlit 시뮬레이션 데모

실행:
  streamlit run app.py
"""
from __future__ import annotations

import re
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_generator import SCENARIOS, generate_scenario_data
from inference_api import (
    get_glucose_at_minute,
    predict_glucose_horizon,
    predict_risk_warning,
)

st.set_page_config(
    page_title="CGM AI Demo",
    page_icon="🩸",
    layout="wide",
)

RISK_THRESHOLD = 0.5
TIR_LOW, TIR_HIGH = 70.0, 180.0


def _glucose_range_label(value: float) -> str:
    if value < TIR_LOW:
        return "저혈당 주의 구간"
    if value > TIR_HIGH:
        return "고혈당 주의 구간"
    return "안정 범위 내"


def _traffic_light_ui(risk: dict[str, float]) -> None:
    """Model 2 — 위험 신호등 + 알림."""
    score = risk["risk_score"]
    is_alert = score >= RISK_THRESHOLD

    col_icon, col_msg = st.columns([1, 5])
    with col_icon:
        st.markdown(
            f"<h1 style='text-align:center;margin:0;'>{'🔴' if is_alert else '🟢'}</h1>",
            unsafe_allow_html=True,
        )
    with col_msg:
        if is_alert:
            if risk["hypo_prob"] >= risk["hyper_prob"]:
                st.error(
                    f"**저혈당 위험 알림** — 내일 TBR 위험 확률 **{risk['hypo_prob']:.0%}** "
                    f"(고혈당 {risk['hyper_prob']:.0%})"
                )
            else:
                st.error(
                    f"**고혈당 위험 알림** — 내일 TAR 위험 확률 **{risk['hyper_prob']:.0%}** "
                    f"(저혈당 {risk['hypo_prob']:.0%})"
                )
        else:
            st.success(
                f"**안전** — 통합 위험 점수 **{score:.0%}** "
                f"(TAR {risk['hyper_prob']:.0%} · TBR {risk['hypo_prob']:.0%})"
            )


def _plot_glucose_horizon(df: pd.DataFrame, pred: dict) -> go.Figure:
    """Model 1 — 과거 실선 + 미래 예측 점선."""
    past_t = pd.to_datetime(df["timestamp"])
    past_g = df["glucose"].astype(float)

    anchor_t = pred["anchor_time"]
    fut_t = [anchor_t + timedelta(minutes=m) for m in pred["horizons_min"]]
    fut_g = pred["predictions"]

    fig = go.Figure()

    fig.add_hrect(y0=0, y1=70, fillcolor="rgba(255,0,0,0.08)", line_width=0)
    fig.add_hrect(y0=180, y1=400, fillcolor="rgba(255,200,0,0.10)", line_width=0)

    fig.add_trace(go.Scatter(
        x=past_t, y=past_g,
        mode="lines+markers",
        name="과거 관측 (12h)",
        line=dict(color="#1f77b4", width=2.5),
        marker=dict(size=4),
    ))

    bridge_t = [past_t.iloc[-1], fut_t[0]]
    bridge_g = [past_g.iloc[-1], fut_g[0]]
    fig.add_trace(go.Scatter(
        x=bridge_t, y=bridge_g,
        mode="lines",
        line=dict(color="#d62728", width=2, dash="dot"),
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=fut_t, y=fut_g,
        mode="lines+markers",
        name="미래 예측 (2h)",
        line=dict(color="#d62728", width=2.5, dash="dash"),
        marker=dict(size=7, symbol="diamond"),
    ))

    fig.add_vline(
        x=anchor_t, line_width=1.5, line_dash="dot", line_color="gray",
        annotation_text="현재(t=0)", annotation_position="top",
    )

    fig.update_layout(
        title="Model 1 — 혈당 Horizon 예측 (과거 12h → 미래 2h)",
        xaxis_title="시간",
        yaxis_title="혈당 (mg/dL)",
        yaxis=dict(range=[max(40, past_g.min() - 20), min(400, max(past_g.max(), max(fut_g)) + 30)]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=480,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
    )
    return fig


def _extract_minutes(text: str) -> int | None:
    """질문에서 'N분 뒤' / 'N시간 뒤' 패턴 추출."""
    text = text.strip().lower()

    hour_match = re.search(r"(\d+)\s*(?:시간|hour|h|hr)", text)
    if hour_match:
        return int(hour_match.group(1)) * 60

    min_match = re.search(r"(\d+)\s*(?:분|minute|min|m)\b", text)
    if min_match:
        return int(min_match.group(1))

    if re.search(r"1\s*시간|한\s*시간|one\s*hour", text):
        return 60
    if re.search(r"30\s*분|삼\s*십\s*분|half\s*hour", text):
        return 30

    if re.search(r"몇\s*분|분\s*뒤|뒤\s*혈당|예측", text):
        return 30

    return None


def _is_next_day_mean_question(text: str) -> bool:
    text = text.strip().lower()
    patterns = (
        r"내일.*평균",
        r"다음\s*날.*평균",
        r"다음날.*평균",
        r"내일\s*혈당",
        r"predicted.*mean",
        r"next\s*day.*mean",
    )
    return any(re.search(p, text) for p in patterns)


def _is_horizon_question(text: str) -> bool:
    text = text.strip().lower()
    keywords = ("분 뒤", "시간 뒤", "후 혈당", "후 예측", "예측해", "얼마야", "얼마", "minute", "hour")
    return any(k in text for k in keywords) or _extract_minutes(text) is not None


def _chatbot_reply(
    user_text: str,
    df: pd.DataFrame,
    risk: dict[str, float],
    horizon: dict,
) -> str:
    """질문 유형 파싱 후 답변 생성."""
    if _is_next_day_mean_question(user_text):
        mean_g = risk["predicted_next_day_mean"]
        src = risk.get("next_day_model_source", "Model3")
        src_note = " (LightGBM Top-20)" if "Model3" in str(src) else ""
        return (
            f"선택하신 시나리오 기준 **내일 예상 평균 혈당은 {mean_g:.0f} mg/dL**입니다{src_note}. "
            f"({_glucose_range_label(mean_g)})"
        )

    if _is_horizon_question(user_text):
        minutes = _extract_minutes(user_text)
        if minutes is None:
            minutes = 60 if "시간" in user_text else 30

        result = get_glucose_at_minute(df, minutes)
        g = result["glucose_mgdl"]
        matched = result["matched_minutes"]
        range_label = _glucose_range_label(g)

        if matched != minutes:
            return (
                f"현재 데이터 기반으로 **{minutes}분 뒤** 예상 혈당은 "
                f"**{g:.0f} mg/dL**입니다 (가장 가까운 {matched}분 horizon 사용). "
                f"({range_label})"
            )
        return (
            f"현재 데이터 기반으로 **{minutes}분 뒤** 예상 혈당은 "
            f"**{g:.0f} mg/dL**입니다. ({range_label})"
        )

    anchor = horizon["anchor_glucose"]
    return (
        f"현재 앵커 혈당은 **{anchor:.0f} mg/dL**입니다. "
        "아래와 같은 질문을 해보실 수 있습니다:\n\n"
        "- \"30분 뒤 혈당 얼마야?\" / \"1시간 뒤 예측해줘\"\n"
        "- \"내일 평균 혈당 알려줘\" / \"다음날 평균\""
    )


# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.header("데모 설정")
scenario = st.sidebar.selectbox(
    "환자 시나리오 선택",
    options=list(SCENARIOS),
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "딥러닝 학습 없이 **더미 데이터 + 휴리스틱 추론**으로 "
    "UI/UX 시뮬레이션을 확인하는 데모입니다."
)

# ── Main ─────────────────────────────────────────────────────────────────
st.title("🩸 CGM AI 혈당 관리 시뮬레이션")
st.caption(f"선택 시나리오: **{scenario}**")

df = generate_scenario_data(scenario)
horizon = predict_glucose_horizon(df)
risk = predict_risk_warning(df)

tab_dashboard, tab_chat = st.tabs(["📊 대시보드", "💬 AI 혈당 챗봇"])

with tab_dashboard:
    st.subheader("Model 2 · 내일 TBR/TAR 위험 예측")
    _traffic_light_ui(risk)

    m1, m2, m3 = st.columns(3)
    with m1:
        src = risk.get("next_day_model_source", "unknown")
        src_label = "LightGBM Top-20" if "Model3" in str(src) else str(src)
        st.metric(
            "내일 예상 평균 혈당",
            f"{risk['predicted_next_day_mean']:.0f} mg/dL",
            help=f"Model 3 ({src_label})",
        )
        st.caption(f"Model 3 · {src_label}")
    with m2:
        st.metric("TAR 위험 확률", f"{risk['hyper_prob']:.0%}")
    with m3:
        st.metric("TBR 위험 확률", f"{risk['hypo_prob']:.0%}")

    st.markdown("---")

    st.subheader("Model 1 · 2시간 혈당 Horizon 예측")
    fig = _plot_glucose_horizon(df, horizon)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("현재 혈당 (앵커)", f"{horizon['anchor_glucose']:.0f} mg/dL")
    with c2:
        st.metric("2h 예측 최솟값", f"{min(horizon['predictions']):.0f} mg/dL")
    with c3:
        st.metric("2h 예측 최댓값", f"{max(horizon['predictions']):.0f} mg/dL")

    with st.expander("입력 데이터 미리보기 (과거 12h)"):
        st.dataframe(df.tail(12), use_container_width=True)

    with st.expander("Horizon 예측값 상세"):
        pred_df = pd.DataFrame({
            "minutes_ahead": horizon["horizons_min"],
            "predicted_glucose": horizon["predictions"],
        })
        st.dataframe(pred_df, use_container_width=True)

with tab_chat:
    st.subheader("AI 혈당 예측 챗봇")
    st.caption(
        "예: \"30분 뒤 혈당 얼마야?\", \"1시간 뒤 예측해줘\", \"내일 평균 혈당 알려줘\""
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "안녕하세요! CGM AI 예측 챗봇입니다. "
                    "몇 분 뒤 혈당이나 내일 평균 혈당에 대해 질문해 주세요."
                ),
            }
        ]

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("혈당 예측에 대해 질문해 보세요..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        reply = _chatbot_reply(prompt, df, risk, horizon)
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()
