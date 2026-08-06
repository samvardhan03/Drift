from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.api.deps import get_backtest
from src.api.models import BacktestRequest, BacktestResponse
from src.showcase.services import MockBacktest

router = APIRouter(prefix="/backtest", tags=["backtest"])

_jobs: dict[str, dict] = {}


def _run_backtest(job_id: str, svc: MockBacktest) -> None:
    _jobs[job_id]["status"] = "done"
    _jobs[job_id]["result"] = svc.get()


@router.post("/run", status_code=202)
def run_backtest(
    req:    BacktestRequest,
    tasks:  BackgroundTasks,
    svc:    MockBacktest = Depends(get_backtest),
) -> dict:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "result": None}
    tasks.add_task(_run_backtest, job_id, svc)
    return {"job_id": job_id, "status": "queued"}


@router.get("/{job_id}")
def get_backtest_result(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    if job["status"] != "done":
        return {"job_id": job_id, "status": job["status"]}
    return {"job_id": job_id, "status": "done", "result": job["result"]}
