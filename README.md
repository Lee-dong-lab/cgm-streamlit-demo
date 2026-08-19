# CGM AI Streamlit Demo

당뇨 CGM 혈당 관리 **시뮬레이션 데모** — Model 1(horizon), Model 2(TBR/TAR), Model 3(내일 평균 혈당 LightGBM).

## 로컬 실행

```bash
cd streamlit_demo
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud 배포

| 설정 | 값 |
|------|-----|
| Main file | `streamlit_demo/app.py` |
| Requirements | `requirements.txt` |

## 모델

- **Model 3**: LightGBM Top-20 (`artifacts/model3_top20_lgb.joblib`)
- 학습: `train_model3_top20.py` — 자세한 내용은 `streamlit_demo/TRAINING.md`
