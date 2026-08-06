from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_risk
from src.api.models import RiskRequest, RiskResponse, StressResponse
from src.showcase.services import MockRisk

router = APIRouter(prefix="/risk", tags=["risk"])


@router.post("/decompose", response_model=RiskResponse)
def decompose(
    req: RiskRequest,
    svc: MockRisk = Depends(get_risk),
) -> RiskResponse:
    return svc.decompose()


@router.post("/stress", response_model=StressResponse)
def stress_test(
    req: RiskRequest,
    svc: MockRisk = Depends(get_risk),
) -> StressResponse:
    return svc.stress()
