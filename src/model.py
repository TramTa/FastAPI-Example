from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from src.features import build_features, fetch_recent_features, f2

ARTIFACT_DIR = Path("model_repos")

def predict_proba(ticker: str, lookback: int = 200) -> float:
    pre = joblib.load(ARTIFACT_DIR / "preproc.joblib")
    lr = joblib.load(ARTIFACT_DIR / f"lr_{ticker}.joblib")
    rf = joblib.load(ARTIFACT_DIR / f"rf_{ticker}.joblib")
    xgb = joblib.load(ARTIFACT_DIR / f"xgb_{ticker}.joblib")

    X = fetch_recent_features(ticker, lookback)
    scaler = pre["scaler"]
    Xs = scaler.transform(X)
    p_lr = lr.predict_proba(Xs)[:,1]
    p_rf = rf.predict_proba(Xs)[:,1]
    p_xg = xgb.predict_proba(Xs)[:,1]
    p = f2(p_lr, p_rf, p_xg) 
    return float(p)
