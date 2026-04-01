"""
models.py — Typed Pydantic models for the CSV Cleaner OpenEnv environment.

These are the three core types required by the OpenEnv spec:
  - Observation  : what the agent sees each step
  - Action       : what the agent can do
  - Reward       : the scored result of an action
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action types the agent can perform
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    FILL_NULLS        = "fill_nulls"         # Fill missing values in a column
    REMOVE_DUPLICATES = "remove_duplicates"  # Drop duplicate rows
    NORMALIZE_COLUMN  = "normalize_column"   # Standardize format of a column
    CAST_COLUMN       = "cast_column"        # Change dtype of a column
    REMOVE_OUTLIERS   = "remove_outliers"    # Drop rows where value is out of range
    NOOP              = "noop"               # Do nothing (agent is stuck / done)


# ---------------------------------------------------------------------------
# Action — what the agent sends to step()
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    One operation the agent wants to apply to the current CSV state.

    Examples
    --------
    Fill nulls in 'age' with 0:
        Action(action_type="fill_nulls", column="age", value=0)

    Remove duplicate rows:
        Action(action_type="remove_duplicates")

    Normalize 'phone' column to digits-only:
        Action(action_type="normalize_column", column="phone", format="digits_only")

    Cast 'joined_date' to ISO date string:
        Action(action_type="cast_column", column="joined_date", dtype="date")

    Remove rows where 'age' is outside 0–120:
        Action(action_type="remove_outliers", column="age", min_val=0, max_val=120)
    """

    action_type: ActionType = Field(
        ..., description="The cleaning operation to perform."
    )
    column: Optional[str] = Field(
        None, description="Target column name (required for most actions)."
    )
    value: Optional[Any] = Field(
        None, description="Fill value for fill_nulls action."
    )
    format: Optional[str] = Field(
        None,
        description=(
            "Format string for normalize_column. "
            "Supported: 'digits_only', 'lowercase', 'strip', 'title_case', 'iso_date'."
        ),
    )
    dtype: Optional[str] = Field(
        None,
        description=(
            "Target dtype for cast_column. "
            "Supported: 'int', 'float', 'str', 'date'."
        ),
    )
    min_val: Optional[float] = Field(
        None, description="Lower bound for remove_outliers."
    )
    max_val: Optional[float] = Field(
        None, description="Upper bound for remove_outliers."
    )


# ---------------------------------------------------------------------------
# Observation — what the agent receives from reset() and step()
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    The agent's view of the world at a given step.

    `rows` is the current (possibly still-dirty) CSV as a list of dicts.
    `columns` lists column names in order.
    `goal` describes in plain English what the agent must achieve.
    `task_id` identifies which of the 3 tasks is running.
    `step_number` counts how many actions have been taken this episode.
    `max_steps` is the episode step limit.
    `done` is True when the episode has ended.
    `info` carries optional diagnostic hints (e.g. last error message).
    """

    rows: List[Dict[str, Any]] = Field(
        ..., description="Current CSV state as list of row dicts."
    )
    columns: List[str] = Field(
        ..., description="Column names in order."
    )
    goal: str = Field(
        ..., description="Plain-English description of what the agent must do."
    )
    task_id: str = Field(
        ..., description="Which task is running: 'task1', 'task2', or 'task3'."
    )
    step_number: int = Field(
        0, description="Number of actions taken so far this episode."
    )
    max_steps: int = Field(
        20, description="Maximum steps allowed before the episode terminates."
    )
    done: bool = Field(
        False, description="True if the episode is over."
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional diagnostic information (errors, hints, etc.).",
    )


# ---------------------------------------------------------------------------
# Reward — returned alongside Observation from step()
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """
    Scores the result of the agent's last action.

    `score`   — float in [0.0, 1.0]. 1.0 = perfectly clean CSV.
    `delta`   — change in score vs. previous step (positive = improvement).
    `reason`  — human-readable explanation of what changed.
    `done`    — True when the episode should end (perfect score or max steps).
    """

    score: float = Field(
        ..., ge=0.0, le=1.0, description="Current cleanliness score (0.0–1.0)."
    )
    delta: float = Field(
        0.0, description="Score change from previous step."
    )
    reason: str = Field(
        ..., description="Why this score was given."
    )
    done: bool = Field(
        False, description="True when the episode is complete."
    )


# ---------------------------------------------------------------------------
# EpisodeState — internal state stored by the environment (returned by state())
# ---------------------------------------------------------------------------

class EpisodeState(BaseModel):
    """Full internal state of a running episode."""

    task_id: str
    rows: List[Dict[str, Any]]
    columns: List[str]
    step_number: int = 0
    max_steps: int = 20
    done: bool = False
    last_score: float = 0.0
    history: List[str] = Field(default_factory=list)