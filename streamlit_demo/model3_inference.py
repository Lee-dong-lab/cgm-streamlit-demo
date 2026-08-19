# -*- coding: utf-8 -*-
"""
model3_inference.py — Model3 Top-20 LightGBM (next-day mean glucose) 추론
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from feature_builder import TOP20_FEATURES, build_top20_features

_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DEMO_DIR.parent
_ARTIFACT_CANDIDATES = (
    _DEMO_DIR / "artifacts" / "model3_top20_lgb.joblib",
    _PROJECT_ROOT / "processed" / "model3_top20_lgb.joblib",
)
DEFAULT_ARTIFACT = next((p for p in _ARTIFACT_CANDIDATES if p.exists()), _ARTIFACT_CANDIDATES[0])


@lru_cache(maxsize=1)
def load_model3_artifact(path: str | None = None) -> dict[str, Any] | None:
    artifact_path = Path(path) if path else DEFAULT_ARTIFACT
    if not artifact_path.exists():
        return None
    return joblib.load(artifact_path)


def predict_next_day_mean_glucose(
    data: pd.DataFrame,
    artifact_path: str | None = None,
) -> dict[str, Any]:
    """
    Model 3 — 내일 평균 혈당 (LightGBM Top-20).

    Returns
    -------
    dict
        predicted_next_day_mean, hyper_flag, model_source, features_used
    """
    bundle = load_model3_artifact(artifact_path)
    features_df = build_top20_features(data)

    if bundle is None:
        return {
            "predicted_next_day_mean": float("nan"),
            "hyper_flag": False,
            "model_source": "unavailable",
            "features_used": TOP20_FEATURES,
            "error": f"artifact not found: {DEFAULT_ARTIFACT}",
        }

    model = bundle["model"]
    meta = bundle["meta"]
    feat_cols = meta["features"]
    X = features_df[feat_cols]
    pred = float(model.predict(X)[0])
    label_thr = float(meta.get("label_threshold", 140.0))
    opt_thr = float(meta.get("optimal_threshold", label_thr))

    return {
        "predicted_next_day_mean": pred,
        "hyper_flag": pred > label_thr,
        "hyper_flag_opt": pred > opt_thr,
        "label_threshold": label_thr,
        "optimal_threshold": opt_thr,
        "model_source": meta.get("model_name", "Model3_top20"),
        "features_used": feat_cols,
        "feature_values": features_df.iloc[0].to_dict(),
    }
