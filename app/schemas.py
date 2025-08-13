from pydantic import BaseModel, Field

class StockRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g., AAPL")
    lookback: int = Field(200, ge=50, le=500, description="Days of history lookback")

class StockResponse(BaseModel):
    ticker: str
    prob_up: float
    prediction: int
