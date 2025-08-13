from __future__ import annotations
from fastapi import FastAPI, HTTPException
from app.schemas import StockRequest, StockResponse
from src.model import predict_proba

app = FastAPI(title="Stock Movement API")

@app.get("/")
def root():
    return {"message": "POST /predict with {ticker, lookback}."}

@app.post("/predict", response_model=StockResponse)
def predict(req: StockRequest) -> StockResponse:
    try:
        p = predict_proba(req.ticker.upper(), req.lookback)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    pred = int(p >= 0.5)
    return StockResponse(ticker=req.ticker.upper(), prob_up=p, prediction=pred)
