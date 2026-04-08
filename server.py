"""
FastAPI server — exposes OpenEnv HTTP API for HealthEnv
Endpoints: POST /reset  POST /step  GET /state
"""
from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from health_env import Action, HealthEnv

app = FastAPI(title="HealthEnv — Pediatric Triage OpenEnv", version="1.0.0")

# One env instance per task; task selected via env var or query param
_ENVS: Dict[str, HealthEnv] = {}


def _get_env(task: str) -> HealthEnv:
    if task not in _ENVS:
        _ENVS[task] = HealthEnv(task=task)
    return _ENVS[task]


DEFAULT_TASK = os.getenv("HEALTH_ENV_TASK", "easy_fever")
VALID_TASKS = ["easy_fever", "medium_asthma", "hard_meningitis_risk"]


class ResetRequest(BaseModel):
    task: str = DEFAULT_TASK


class StepRequest(BaseModel):
    task: str = DEFAULT_TASK
    message: str


@app.post("/reset")
async def reset(req: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    if req.task not in VALID_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task. Choose from {VALID_TASKS}")
    env = _get_env(req.task)
    result = env.reset()
    return result


@app.post("/step")
async def step(req: StepRequest) -> Dict[str, Any]:
    if req.task not in VALID_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task. Choose from {VALID_TASKS}")
    env = _get_env(req.task)
    try:
        result = env.step({"message": req.message})
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/state")
async def state(task: str = DEFAULT_TASK) -> Dict[str, Any]:
    if task not in VALID_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task. Choose from {VALID_TASKS}")
    env = _get_env(task)
    return env.state()


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": VALID_TASKS,
        "descriptions": {
            "easy_fever": "4-year-old low-grade fever — home care",
            "medium_asthma": "7-year-old asthma flare — urgent care",
            "hard_meningitis_risk": "2-year-old meningitis warning signs — ER immediately",
        },
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
