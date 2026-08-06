from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_screener
from src.api.screener_models import ScreenRequest, ScreenResponse
from src.showcase.services import MockScreener

router = APIRouter(prefix="/screen", tags=["screener"])


@router.get("",  response_model=ScreenResponse)
@router.post("", response_model=ScreenResponse)
def screen(
    req:  ScreenRequest = ScreenRequest(),
    svc:  MockScreener  = Depends(get_screener),
) -> ScreenResponse:
    return svc.get()
