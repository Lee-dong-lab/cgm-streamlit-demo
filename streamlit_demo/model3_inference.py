# -*- coding: utf-8 -*-
"""
model3_inference.py — Model3 Top-20 LightGBM (next-day mean glucose) 추론

Streamlit Cloud(Python 3.14) 호환: sklearn joblib 대신 native Booster(.txt) 사용.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from feature_builder import TOP20_FEATURES, build_top20_features

_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DEMO_DIR.parent
_ARTIFACT_DIR = _DEMO_DIR / "artifacts"

_BOOSTER_CANDIDATES = (
    _ARTIFACT_DIR / "model3_top20_lgb.txt",
    _PROJECT_ROOT / "processed" / "model3_top20_lgb.txt",
)
_META_CANDIDATES = (
    _ARTIFACT_DIR / "model3_top20_meta.json",
    _PROJECT_ROOT / "processed" / "model3_top20_meta.json",
)


def _first_existing(paths: tuple[Path, ...]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


@lru_cache(maxsize=1)
def load_model3_bundle() -> dict[str, Any] | None:
    """Native LightGBM Booster + JSON meta 로드."""
    booster_path = _first_existing(_BOOSTER_CANDIDATES)
    meta_path = _first_existing(_META_CANDIDATES)
    if booster_path is None or meta_path is None:
        return None

    booster = lgb.Booster(model_file=str(booster_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {"booster": booster, "meta": meta}


def _predict_booster(booster: lgb.Booster, X: pd.DataFrame) -> float:
    """Booster.predict — sklearn wrapper 없이 직접 추론."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = booster.predict(arr)
    return float(out[0])


def predict_next_day_mean_glucose(
    data: pd.DataFrame,
    artifact_path: str | None = None,  # kept for API compat; uses default bundle
) -> dict[str, Any]:
    """
    Model 3 — 내일 평균 혈당 (LightGBM Top-20).

    Returns
    -------
    dict
        predicted_next_day_mean, hyper_flag, model_source, features_used
    """
    _ = artifact_path  # native bundle only in demo deploy
    bundle = load_model3_bundle()
    features_df = build_top20_features(data)

    if bundle is None:
        return {
            "predicted_next_day_mean": float("nan"),
            "hyper_flag": False,
            "model_source": "unavailable",
            "features_used": TOP20_FEATURES,
            "error": "Model3 artifact not found (model3_top20_lgb.txt + meta.json)",
        }

    meta = bundle["meta"]
    feat_cols = meta["features"]
    X = features_df[feat_cols]

    try:
        pred = _predict_booster(bundle["booster"], X)
    except Exception as exc:
        return {
            "predicted_next_day_mean": float("nan"),
            "hyper_flag": False,
            "model_source": "unavailable",
            "features_used": feat_cols,
            "error": f"LightGBM predict failed: {exc}",
        }

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
