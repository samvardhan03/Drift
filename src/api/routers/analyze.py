from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_analyze
from src.api.models import PortfolioAnalyzeRequest, PortfolioAnalysisResponse
from src.showcase.services import MockAnalyze

router = APIRouter(prefix="/portfolio", tags=["portfolio-analysis"])


@router.post("/analyze", response_model=PortfolioAnalysisResponse)
def analyze_portfolio(
    req: PortfolioAnalyzeRequest,
    svc: MockAnalyze = Depends(get_analyze),
) -> PortfolioAnalysisResponse:
    return svc.get()
