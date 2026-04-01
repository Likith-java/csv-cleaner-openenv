"""
main.py — FastAPI server for the CSV Cleaner OpenEnv environment.

Endpoints
---------
POST /reset          Start a new episode. Body: {"task_id": "task1"}
POST /step           Take one action.    Body: Action JSON
GET  /state          Get current episode state.
GET  /tasks          List all available tasks.
GET  /health         Liveness check.

The validator script pings POST /reset — it must return HTTP 200.
"""

from __future__ import annotations

import sys
import os

# Make sure server/ is on the path when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import CSVCleanerEnv
from models import Action, EpisodeState, Observation, Reward

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "CSV Cleaner — OpenEnv Environment",
    description = (
        "An OpenEnv environment where AI agents learn to clean messy CSV data. "
        "Three tasks of increasing difficulty: fill nulls (easy), "
        "deduplicate + normalize dates (medium), full pipeline clean (hard)."
    ),
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# One environment instance per server process.
# For production you'd use a session ID → env map, but for the hackathon
# a single global instance is correct and matches the validator's expectations.
_env = CSVCleanerEnv()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task1"


class StepResponse(BaseModel):
    observation: Observation
    reward:      Reward


class TaskInfo(BaseModel):
    id:         str
    name:       str
    difficulty: str
    goal:       str
    columns:    list
    max_steps:  int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset(body: ResetRequest = ResetRequest()) -> Observation:
    """
    Start a fresh episode.

    - Resets the CSV to its original dirty state.
    - Returns the initial Observation (dirty rows + goal).
    - Default task is 'task1' if no body is provided.

    The pre-submission validator sends POST /reset with an empty body {}
    and expects HTTP 200 — the default task_id handles this gracefully.
    """
    try:
        obs = _env.reset(task_id=body.task_id or "task1")
        return obs
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    """
    Apply one cleaning action to the current CSV state.

    Returns the updated Observation and a Reward (score 0.0–1.0).
    Call /reset first — calling /step before /reset returns HTTP 400.

    Example body:
        {"action_type": "fill_nulls", "column": "age", "value": 0}
    """
    try:
        obs, reward = _env.step(action)
        return StepResponse(observation=obs, reward=reward)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@app.get("/state", response_model=EpisodeState)
def state() -> EpisodeState:
    """
    Return the full internal episode state.

    Includes the current rows, step number, score history, and done flag.
    Call /reset first — calling /state before /reset returns HTTP 400.
    """
    try:
        return _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/tasks", response_model=list[TaskInfo])
def tasks() -> list:
    """
    List all available tasks with their IDs, difficulty, and goals.

    Useful for the inference script to discover which tasks to run.
    """
    return _env.list_tasks()


@app.get("/")
def root() -> Dict[str, Any]:
    """Root endpoint — returns environment info and available endpoints."""
    return {
        "environment": "CSV Cleaner",
        "version":     "1.0.0",
        "description": "OpenEnv environment for AI-powered CSV data cleaning.",
        "endpoints": {
            "POST /reset":  "Start a new episode",
            "POST /step":   "Take a cleaning action",
            "GET  /state":  "Get current episode state",
            "GET  /tasks":  "List available tasks",
            "GET  /health": "Liveness check",
            "GET  /docs":   "Interactive API documentation (Swagger UI)",
        },
        "tasks": ["task1 (easy)", "task2 (medium)", "task3 (hard)"],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = 7860,       # HuggingFace Spaces default port
        reload  = False,
        workers = 1,
    )