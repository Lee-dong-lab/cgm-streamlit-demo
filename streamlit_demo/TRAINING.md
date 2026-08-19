# CGM AI 모델 학습 · 추론 가이드

본 문서는 `streamlit_demo/` 웹 데모와 연동되는 **3개 모델**의 학습 방법과 추론 구조를 정리합니다.

---

## 모델 개요

| 모델 | 역할 | 알고리즘 | 학습 스크립트 | 데모 연동 |
|------|------|----------|---------------|-----------|
| **Model 1** | 2시간 혈당 Horizon (15~120분) | **CGMPatchTCN** (PyTorch) | `최종 모델/train_model1_deep_sliding_window.py` | 휴리스틱 (추후 체크포인트 연동 가능) |
| **Model 2** | 내일 TBR/TAR 위험 확률 | **LightGBM** (+ CatBoost/AutoGluon 옵션) | `최종 모델/train_tar_tbr_2track_sota.py` | 휴리스틱 |
| **Model 3** | **내일 평균 혈당** 회귀 | **LightGBM Top-20** | `train_model3_top20.py` | **실제 모델 적용 ✅** |

---

## 사전 준비

```bash
cd "/home/당뇨 실험"
source .venv/bin/activate   # 또는 ../.venv/bin/python 직접 사용

# 공통 데이터 파이프라인 (Dataset A/B 생성)
python build_final_pipeline.py
```

산출물: `processed/dataset_b_daily_level.csv` 등

---

## Model 3 — 내일 평균 혈당 (LightGBM Top-20) ⭐

### 목적
- **타깃**: `target_next_day_mean_glucose` (= 같은 visit 내 다음날 `mean_glucose`)
- **출력**: 내일 예상 평균 혈당 (mg/dL, 연속값)
- **부가**: >140 mg/dL 이진 분류 (Clarke Grid, ROC, Optimal Threshold)

### 학습 실행

```bash
cd "/home/당뇨 실험"
.venv/bin/python train_model3_top20.py
```

### 학습 설정 (요약)

| 항목 | 값 |
|------|-----|
| 알고리즘 | `LGBMRegressor` |
| n_estimators | 300 |
| learning_rate | 0.03 |
| CV | GroupKFold(5) — `participant_id` 기준 |
| 피처 | Feature Importance **Top 20** |
| 가중치 | y ≤ 140 샘플 weight = 1.5 (FP 억제) |
| Optimal thr | Recall ≥ 0.85 조건에서 Specificity 최대 |

### Top-20 피처

```
mean_glucose, roll3_mean_glucose, d_cv, steps, roll3_d_tar,
diff_mean_from_prev_day, lag1_sleep_total_min, lag1_steps,
lag1_mean_glucose, diff_steps_from_prev_day, diff_d_cv_from_prev_day,
lag1_d_cv, sleep_rem_min, sleep_light_min, sleep_deep_min,
roll3_steps, diff_sleep_from_prev_day, sleep_total_min, d_tir,
diff_d_tir_from_prev_day
```

### 학습 산출물

| 파일 | 설명 |
|------|------|
| `processed/model3_top20_lgb.joblib` | **Streamlit 추론용** full-fit LightGBM |
| `processed/model3_top20_summary.csv` | OOF RMSE, Clarke A+B, AUC 등 |
| `processed/model3_top20_fold_metrics.csv` | Fold별 지표 |
| `processed/model3_top20_threshold_comparison.csv` | Fixed 140 vs Optimal thr |
| `figures_top20/*.png` | Clarke, ROC, CM, FI, SHAP |

### Streamlit 연동

```
streamlit_demo/feature_builder.py  → 12h 시나리오 → Top-20 벡터
streamlit_demo/model3_inference.py → joblib 로드 → predict
streamlit_demo/inference_api.py    → predict_risk_warning() 에 통합
```

artifact 없으면 휴리스틱 fallback. **반드시 `train_model3_top20.py` 실행 후 데모 구동.**

---

## Model 1 — 2h Horizon (CGMPatchTCN)

### 학습

```bash
cd "/home/당뇨 실험/최종 모델"
../.venv/bin/python train_model1_deep_sliding_window.py
```

### 구조
- **입력**: 15분 CGM 슬라이딩 윈도우 `[B, SEQ_LEN, N_FEATURES]`
- **아키텍처**: Patch 임베딩 + Temporal Attention + Dilated 1D-TCN
- **출력**: 8 horizon (15, 30, …, 120분)
- **손실**: Weighted Huber (저/고혈당 구간 가중)

### 데모 상태
- 현재 `inference_api.predict_glucose_horizon()` 은 **휴리스틱**
- 실제 모델 연동 시: `processed/` 체크포인트 로드 후 `predict_glucose_horizon` 교체

---

## Model 2 — TBR/TAR 위험 (LightGBM)

### 학습

```bash
cd "/home/당뇨 실험/최종 모델"
../.venv/bin/python train_tar_tbr_2track_sota.py --tbr-metric d_tbr_70 --tar-metric d_tar_180
../.venv/bin/python train_tir_2track_sota.py
```

### 구조
- **Track 1**: 일별 조기 경보 (`dataset_b`)
- **Track 2**: 방문 간 초기 상태
- **모델**: LightGBM (기본), CatBoost/AutoGluon 옵션
- **임계값**: Youden J (TBR/TAR), Recall≥0.85 (TIR)

### 데모 상태
- `predict_risk_warning()` 의 TAR/TBR 확률은 **휴리스틱**
- Model 3 LightGBM 예측값으로 **내일 평균 혈당**만 실제 모델 사용

---

## Streamlit 데모 실행

```bash
# 1) Model 3 학습 (최초 1회 또는 재학습 시)
cd "/home/당뇨 실험"
.venv/bin/python train_model3_top20.py

# 2) 데모 실행
./streamlit_demo/run.sh
# 또는
cd streamlit_demo && ../.venv/bin/streamlit run app.py
```

### CLI 시나리오 검증

```bash
cd streamlit_demo
../.venv/bin/python run_scenarios.py
```

출력 예:
```
Model 2 TAR 62% | TBR 0% | 내일 평균 178 mg/dL | 🔴 위험
```

---

## 재학습 체크리스트

- [ ] `build_final_pipeline.py` 실행 → Dataset B 최신화
- [ ] `train_model3_top20.py` → `model3_top20_lgb.joblib` 갱신
- [ ] (선택) Model 1 CGMPatchTCN, Model 2 TAR/TBR 재학습
- [ ] `run_scenarios.py` 로 5시나리오 sanity check
- [ ] `streamlit run app.py` → 내일 평균 혈당 metric · 챗봇 확인

---

## 참고 성능 (Model 3 Top-20, OOF)

`processed/model3_top20_summary.csv` 기준:

- RMSE ≈ 23.5 mg/dL
- MAE ≈ 16.3 mg/dL
- Clarke Zone A+B ≈ 99%
- AUC (>140) ≈ 0.90

※ Streamlit 데모는 **12h 합성 CGM → Top-20 근사 변환**을 사용하므로, 실제 임상 데이터 추론과 수치가 다를 수 있습니다.
