from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_signals
from src.api.models import SignalRequest, SignalResponse
from src.showcase.services import MockSignals

router = APIRouter(prefix="/signals", tags=["signals"])


@router.post("/compute", response_model=SignalResponse)
def compute_signals(
    req: SignalRequest,
    svc: MockSignals = Depends(get_signals),
) -> SignalResponse:
    return svc.get()
