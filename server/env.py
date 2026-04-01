"""
env.py — Core environment logic for the CSV Cleaner OpenEnv environment.

Implements the three required OpenEnv methods:
  reset(task_id)  → Observation
  step(action)    → (Observation, Reward)
  state()         → EpisodeState

Action execution is handled here: each ActionType maps to a function that
mutates a copy of the current rows and returns the updated list.
"""

from __future__ import annotations

import copy
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from models import Action, ActionType, EpisodeState, Observation, Reward
from tasks import get_task, list_tasks

# Perfect score threshold — end episode when we hit this
PERFECT_SCORE = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_null(val: Any) -> bool:
    if val is None:
        return True
    return str(val).strip().lower() in ("", "none", "nan", "n/a", "na", "null", "-")


def _parse_date(val: Any) -> Optional[str]:
    """Try common date formats → YYYY-MM-DD. Returns None on failure."""
    if _is_null(val):
        return None
    val = str(val).strip()
    formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y",
        "%d %B %Y", "%Y%m%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(val, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _digits_only(val: Any) -> str:
    return re.sub(r"\D", "", str(val))


# ---------------------------------------------------------------------------
# Action executors
# Each function takes the current rows + action, returns updated rows + message
# ---------------------------------------------------------------------------

def _exec_fill_nulls(
    rows: List[Dict], action: Action
) -> Tuple[List[Dict], str]:
    if not action.column:
        return rows, "fill_nulls requires a 'column' parameter."
    if action.value is None:
        return rows, "fill_nulls requires a 'value' parameter."

    col   = action.column
    fill  = action.value
    count = 0
    for row in rows:
        if col not in row:
            continue
        if _is_null(row[col]):
            row[col] = fill
            count += 1
    return rows, f"Filled {count} null(s) in '{col}' with {repr(fill)}."


def _exec_remove_duplicates(
    rows: List[Dict], action: Action
) -> Tuple[List[Dict], str]:
    seen   = []
    unique = []
    for row in rows:
        key = tuple(sorted((k, str(v)) for k, v in row.items()))
        if key not in seen:
            seen.append(key)
            unique.append(row)
    removed = len(rows) - len(unique)
    return unique, f"Removed {removed} duplicate row(s)."


def _exec_normalize_column(
    rows: List[Dict], action: Action
) -> Tuple[List[Dict], str]:
    if not action.column:
        return rows, "normalize_column requires a 'column' parameter."
    if not action.format:
        return rows, (
            "normalize_column requires a 'format' parameter. "
            "Supported: digits_only, lowercase, strip, title_case, iso_date."
        )

    col    = action.column
    fmt    = action.format.lower().strip()
    count  = 0
    errors = 0

    for row in rows:
        if col not in row or _is_null(row[col]):
            continue
        val = str(row[col])
        if fmt == "digits_only":
            row[col] = _digits_only(val)
            count += 1
        elif fmt == "lowercase":
            row[col] = val.lower()
            count += 1
        elif fmt == "strip":
            row[col] = val.strip()
            count += 1
        elif fmt == "title_case":
            row[col] = val.title()
            count += 1
        elif fmt == "iso_date":
            parsed = _parse_date(val)
            if parsed:
                row[col] = parsed
                count += 1
            else:
                errors += 1
        else:
            return rows, f"Unknown format '{fmt}'."

    msg = f"Normalized {count} value(s) in '{col}' (format={fmt})."
    if errors:
        msg += f" {errors} value(s) could not be parsed."
    return rows, msg


def _exec_cast_column(
    rows: List[Dict], action: Action
) -> Tuple[List[Dict], str]:
    if not action.column:
        return rows, "cast_column requires a 'column' parameter."
    if not action.dtype:
        return rows, "cast_column requires a 'dtype' parameter. Supported: int, float, str, date."

    col   = action.column
    dtype = action.dtype.lower().strip()
    count = 0
    errors = 0

    for row in rows:
        if col not in row or _is_null(row[col]):
            continue
        val = row[col]
        try:
            if dtype == "int":
                row[col] = int(float(str(val)))
            elif dtype == "float":
                row[col] = float(str(val))
            elif dtype == "str":
                row[col] = str(val)
            elif dtype == "date":
                parsed = _parse_date(val)
                row[col] = parsed if parsed else val
            else:
                return rows, f"Unknown dtype '{dtype}'."
            count += 1
        except (ValueError, TypeError):
            errors += 1

    msg = f"Cast {count} value(s) in '{col}' to {dtype}."
    if errors:
        msg += f" {errors} value(s) failed to cast."
    return rows, msg


def _exec_remove_outliers(
    rows: List[Dict], action: Action
) -> Tuple[List[Dict], str]:
    if not action.column:
        return rows, "remove_outliers requires a 'column' parameter."
    if action.min_val is None or action.max_val is None:
        return rows, "remove_outliers requires 'min_val' and 'max_val' parameters."

    col     = action.column
    lo, hi  = action.min_val, action.max_val
    kept    = []
    removed = 0

    for row in rows:
        val = row.get(col)
        if _is_null(val):
            kept.append(row)
            continue
        try:
            num = float(val)
            if lo <= num <= hi:
                kept.append(row)
            else:
                removed += 1
        except (TypeError, ValueError):
            kept.append(row)  # non-numeric rows are kept

    return kept, f"Removed {removed} outlier row(s) where '{col}' outside [{lo}, {hi}]."


def _exec_noop(
    rows: List[Dict], action: Action
) -> Tuple[List[Dict], str]:
    return rows, "No operation performed."


# Map ActionType → executor function
_EXECUTORS = {
    ActionType.FILL_NULLS:        _exec_fill_nulls,
    ActionType.REMOVE_DUPLICATES: _exec_remove_duplicates,
    ActionType.NORMALIZE_COLUMN:  _exec_normalize_column,
    ActionType.CAST_COLUMN:       _exec_cast_column,
    ActionType.REMOVE_OUTLIERS:   _exec_remove_outliers,
    ActionType.NOOP:              _exec_noop,
}


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class CSVCleanerEnv:
    """
    Single-episode CSV cleaning environment.

    Usage
    -----
        env = CSVCleanerEnv()
        obs = env.reset("task1")
        obs, reward = env.step(Action(action_type="fill_nulls", column="age", value=0))
        current_state = env.state()
    """

    def __init__(self) -> None:
        self._episode: Optional[EpisodeState] = None

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task1") -> Observation:
        """
        Start a fresh episode for the given task.
        Returns the initial Observation (dirty CSV + goal).
        """
        task = get_task(task_id)

        self._episode = EpisodeState(
            task_id    = task_id,
            rows       = copy.deepcopy(task["dirty"]),
            columns    = list(task["columns"]),
            step_number= 0,
            max_steps  = task["max_steps"],
            done       = False,
            last_score = 0.0,
            history    = [],
        )

        return self._make_observation(info={})

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: Action) -> Tuple[Observation, Reward]:
        """
        Apply one action to the current CSV state.
        Returns updated Observation and a Reward with a 0.0–1.0 score.
        """
        if self._episode is None:
            raise RuntimeError("Call reset() before step().")
        if self._episode.done:
            raise RuntimeError("Episode is already done. Call reset() to start a new one.")

        ep = self._episode

        # --- Execute the action ---
        executor = _EXECUTORS.get(action.action_type, _exec_noop)
        updated_rows, action_msg = executor(copy.deepcopy(ep.rows), action)
        ep.rows = updated_rows
        ep.step_number += 1

        # --- Score the new state ---
        task        = get_task(ep.task_id)
        new_score, grade_reason = task["grader"](ep.rows)
        delta       = round(new_score - ep.last_score, 4)
        ep.last_score = new_score

        # --- Determine if episode is done ---
        done = (new_score >= PERFECT_SCORE) or (ep.step_number >= ep.max_steps)
        ep.done = done

        # --- Build reward ---
        reward = Reward(
            score  = new_score,
            delta  = delta,
            reason = f"{action_msg} | {grade_reason}",
            done   = done,
        )

        # --- Log to history ---
        ep.history.append(
            f"Step {ep.step_number}: {action.action_type.value} → "
            f"score={new_score:.4f} (Δ{delta:+.4f})"
        )

        info = {"action_message": action_msg, "grade_detail": grade_reason}
        obs  = self._make_observation(info=info)
        return obs, reward

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------

    def state(self) -> EpisodeState:
        """Return the full internal episode state (OpenEnv spec requirement)."""
        if self._episode is None:
            raise RuntimeError("Call reset() before state().")
        return self._episode

    # ------------------------------------------------------------------
    # list_tasks() — convenience helper for the /tasks endpoint
    # ------------------------------------------------------------------

    def list_tasks(self):
        return list_tasks()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(self, info: Dict) -> Observation:
        ep   = self._episode
        task = get_task(ep.task_id)
        return Observation(
            rows        = copy.deepcopy(ep.rows),
            columns     = list(ep.columns),
            goal        = task["goal"],
            task_id     = ep.task_id,
            step_number = ep.step_number,
            max_steps   = ep.max_steps,
            done        = ep.done,
            info        = info,
        )