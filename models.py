from pydantic import BaseModel
from typing import List

class Financials(BaseModel):
    price: float
    market_cap: float

class ResearchReport(BaseModel):
    company: str
    summary: str
    financials: Financials
    sentiment: str
    competitors: List[str]
    recommendations: str