"""
inference.py — Baseline agent for the CSV Cleaner OpenEnv environment.

MANDATORY environment variables (set before running):
    API_BASE_URL   The API endpoint for the LLM  (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier           (e.g. meta-llama/Llama-3.3-70B-Instruct)
    HF_TOKEN       Your HuggingFace API key

Usage:
    python inference.py                          # runs all 3 tasks against local server
    python inference.py --base-url http://...    # override server URL
    TASK_ID=task2 python inference.py            # run a single task

The script must be named inference.py and placed in the project root.
All LLM calls use the OpenAI client pointed at API_BASE_URL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — read from environment variables (required by spec)
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY:      str = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME:   str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

# Where the OpenEnv server is running
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://0.0.0.0:7860")

MAX_TOKENS:  int   = 256
TEMPERATURE: float = 0.0   # deterministic for reproducibility
FALLBACK_ACTION    = {"action_type": "noop"}

# ---------------------------------------------------------------------------
# System prompt — tells the LLM exactly what to do
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a data cleaning agent. You will be shown a dirty CSV dataset and a cleaning goal.

Your job is to output ONE JSON action per turn to clean the data step by step.

Available actions (pick exactly one):

1. Fill null values in a column:
   {"action_type": "fill_nulls", "column": "<col>", "value": <value>}

2. Remove duplicate rows:
   {"action_type": "remove_duplicates"}

3. Normalize a column format:
   {"action_type": "normalize_column", "column": "<col>", "format": "<fmt>"}
   Formats: "digits_only", "lowercase", "strip", "title_case", "iso_date"

4. Cast a column to a different type:
   {"action_type": "cast_column", "column": "<col>", "dtype": "<type>"}
   Types: "int", "float", "str", "date"

5. Remove outlier rows by value range:
   {"action_type": "remove_outliers", "column": "<col>", "min_val": <num>, "max_val": <num>}

6. Do nothing:
   {"action_type": "noop"}

Rules:
- Output ONLY a valid JSON object. No explanation, no markdown, no extra text.
- Tackle ONE issue at a time.
- Look at the goal carefully and the current state of the rows.
- When the goal is fully achieved, output {"action_type": "noop"}.
"""

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> Dict[str, Any]:
    """Call POST /reset on the environment server."""
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    """Call POST /step on the environment server."""
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json=action,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_tasks() -> List[Dict[str, Any]]:
    """Call GET /tasks to list available tasks."""
    resp = requests.get(f"{ENV_BASE_URL}/tasks", timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(
    observation: Dict[str, Any],
    last_reward:  Optional[Dict[str, Any]],
    step:         int,
) -> str:
    """Build the user message shown to the LLM each step."""

    rows    = observation.get("rows", [])
    columns = observation.get("columns", [])
    goal    = observation.get("goal", "")
    task_id = observation.get("task_id", "")
    done    = observation.get("done", False)

    # Show at most 12 rows so the prompt doesn't explode in size
    display_rows = rows[:12]
    rows_json    = json.dumps(display_rows, indent=2)
    total_rows   = len(rows)

    reward_info = ""
    if last_reward:
        reward_info = (
            f"\nLast action result:\n"
            f"  Score : {last_reward.get('score', 0):.4f}\n"
            f"  Delta : {last_reward.get('delta', 0):+.4f}\n"
            f"  Reason: {last_reward.get('reason', '')}\n"
        )

    prompt = f"""Task: {task_id}  |  Step: {step}
Goal:
{goal}
{reward_info}
Current CSV state ({total_rows} rows, columns: {columns}):
{rows_json}
{"[... showing first 12 rows only ...]" if total_rows > 12 else ""}

What is the next single cleaning action to take?
Output ONLY a JSON object."""

    return prompt.strip()


# ---------------------------------------------------------------------------
# LLM call + action parser
# ---------------------------------------------------------------------------

def call_llm(user_prompt: str) -> str:
    """Call the LLM and return the raw response text."""
    try:
        completion = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
            stream      = False,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"  [LLM ERROR] {exc}")
        return ""


def parse_action(response_text: str) -> Dict[str, Any]:
    """
    Extract a JSON action from the LLM response.
    Tries strict JSON parse first, then falls back to regex extraction.
    """
    if not response_text:
        return FALLBACK_ACTION

    # 1. Try direct parse (model returned clean JSON)
    try:
        parsed = json.loads(response_text.strip())
        if isinstance(parsed, dict) and "action_type" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Try extracting a JSON object from within the text
    match = re.search(r"\{[^{}]*\"action_type\"[^{}]*\}", response_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "action_type" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    print(f"  [PARSE WARN] Could not extract action from: {response_text[:120]!r}")
    return FALLBACK_ACTION


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> float:
    """
    Run one full episode on the given task.
    Returns the final score (0.0–1.0).
    """
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id}")
    print(f"{'='*60}")

    # Reset environment
    try:
        observation = env_reset(task_id)
    except Exception as exc:
        print(f"  [ERROR] Failed to reset environment: {exc}")
        return 0.0

    print(f"  Goal: {observation.get('goal', '')[:120]}...")
    print(f"  Max steps: {observation.get('max_steps', '?')}")
    print()

    last_reward: Optional[Dict[str, Any]] = None
    final_score: float = 0.0
    step = 0

    max_steps = observation.get("max_steps", 20)

    for step in range(1, max_steps + 1):
        if observation.get("done", False):
            print(f"  Episode done at step {step - 1}.")
            break

        # Build prompt and call LLM
        user_prompt = build_user_prompt(observation, last_reward, step)
        raw_response = call_llm(user_prompt)
        action = parse_action(raw_response)

        print(f"  Step {step:2d} | action: {json.dumps(action)}")

        # Send action to environment
        try:
            result      = env_step(action)
            observation = result["observation"]
            reward      = result["reward"]
            last_reward = reward
            final_score = reward.get("score", 0.0)
            delta       = reward.get("delta", 0.0)
            reason      = reward.get("reason", "")[:80]
            done        = reward.get("done", False)

            print(
                f"          score={final_score:.4f}  "
                f"delta={delta:+.4f}  "
                f"{'DONE' if done else ''}  {reason}"
            )

            if done:
                break

        except Exception as exc:
            print(f"  [ERROR] step failed: {exc}")
            break

        # Small delay to avoid hammering the LLM API
        time.sleep(0.5)

    print(f"\n  Final score for {task_id}: {final_score:.4f}")
    return final_score


# ---------------------------------------------------------------------------
# Main — run all tasks and report
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CSV Cleaner OpenEnv baseline agent")
    parser.add_argument(
        "--base-url",
        default=ENV_BASE_URL,
        help="Environment server base URL (default: http://0.0.0.0:7860)",
    )
    parser.add_argument(
        "--task",
        default=os.getenv("TASK_ID", ""),
        help="Run a single task ID (task1/task2/task3). Default: run all.",
    )
    args = parser.parse_args()

    global ENV_BASE_URL
    ENV_BASE_URL = args.base_url.rstrip("/")

    # Validate env vars
    if not API_KEY:
        print(
            "ERROR: HF_TOKEN (or API_KEY) environment variable is not set.\n"
            "Export it before running:\n"
            "  export HF_TOKEN=hf_...",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Model      : {MODEL_NAME}")
    print(f"API URL    : {API_BASE_URL}")
    print(f"Server URL : {ENV_BASE_URL}")

    # Discover tasks from the environment
    try:
        all_tasks = env_tasks()
        task_ids  = [t["id"] for t in all_tasks]
    except Exception as exc:
        print(f"Could not fetch tasks from server: {exc}")
        task_ids = ["task1", "task2", "task3"]

    # Filter to a single task if requested
    if args.task:
        if args.task not in task_ids:
            print(f"ERROR: Unknown task '{args.task}'. Available: {task_ids}")
            sys.exit(1)
        task_ids = [args.task]

    # Run all selected tasks
    scores: Dict[str, float] = {}
    for task_id in task_ids:
        scores[task_id] = run_task(task_id)

    # Final summary
    print(f"\n{'='*60}")
    print("  BASELINE RESULTS")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        bar   = "█" * int(score * 20)
        empty = "░" * (20 - int(score * 20))
        print(f"  {task_id}  [{bar}{empty}]  {score:.4f}")

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n  Average score: {avg:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()