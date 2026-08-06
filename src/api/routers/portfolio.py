from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_portfolio
from src.api.models import OptimiseRequest, OptimiseResponse
from src.showcase.services import MockPortfolio

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.post("/optimise", response_model=OptimiseResponse)
def optimise(
    req: OptimiseRequest,
    svc: MockPortfolio = Depends(get_portfolio),
) -> OptimiseResponse:
    return svc.get()
