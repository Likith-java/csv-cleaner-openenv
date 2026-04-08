"""
inference.py — Baseline agent for the CSV Cleaner OpenEnv environment.

MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

# Auto-install missing dependencies so the validator can run inference.py directly
def _ensure_deps():
    required = ["openai", "requests"]
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure_deps()

from openai import OpenAI
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME:   str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
API_KEY:      str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "https://Likith-java-csv-cleaner-openenv.hf.space")

BENCHMARK          = "csv-cleaner"
MAX_TOKENS:  int   = 256
TEMPERATURE: float = 0.0
SUCCESS_THRESHOLD  = 0.5
FALLBACK_ACTION    = {"action_type": "noop"}

# ---------------------------------------------------------------------------
# Structured log helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# System prompt
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

6. Do nothing (only when task is complete):
   {"action_type": "noop"}

Rules:
- Output ONLY a valid JSON object. No explanation, no markdown, no extra text.
- Tackle ONE issue at a time.
- When the goal is fully achieved, output {"action_type": "noop"}.
"""

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# Environment HTTP helpers — using urllib (stdlib, no extra deps)
# ---------------------------------------------------------------------------

def _http_post(url: str, body: Dict) -> Dict:
    data = json.dumps(body).encode("utf-8")
    req  = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get(url: str) -> Any:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def env_reset(task_id: str) -> Dict[str, Any]:
    return _http_post(f"{ENV_BASE_URL}/reset", {"task_id": task_id})


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    return _http_post(f"{ENV_BASE_URL}/step", action)


def env_tasks() -> List[Dict[str, Any]]:
    return _http_get(f"{ENV_BASE_URL}/tasks")

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(
    observation: Dict[str, Any],
    last_reward: Optional[float],
    step: int,
) -> str:
    rows    = observation.get("rows", [])
    columns = observation.get("columns", [])
    goal    = observation.get("goal", "")
    task_id = observation.get("task_id", "")

    display_rows = rows[:12]
    rows_json    = json.dumps(display_rows, indent=2)
    total_rows   = len(rows)
    reward_line  = f"\nLast reward: {last_reward:.4f}\n" if last_reward is not None else ""

    return f"""Task: {task_id}  |  Step: {step}
Goal:
{goal}
{reward_line}
Current CSV state ({total_rows} rows, columns: {columns}):
{rows_json}
{"[... showing first 12 rows only ...]" if total_rows > 12 else ""}

What is the next single cleaning action to take?
Output ONLY a JSON object.""".strip()

# ---------------------------------------------------------------------------
# LLM call + action parser
# ---------------------------------------------------------------------------

def call_llm(user_prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return ""


def parse_action(response_text: str) -> Dict[str, Any]:
    if not response_text:
        return FALLBACK_ACTION
    try:
        parsed = json.loads(response_text.strip())
        if isinstance(parsed, dict) and "action_type" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\"action_type\"[^{}]*\}", response_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "action_type" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    print(f"[DEBUG] Could not parse action from: {response_text[:120]!r}", flush=True)
    return FALLBACK_ACTION

# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> float:
    global ENV_BASE_URL

    try:
        observation = env_reset(task_id)
    except Exception as exc:
        print(f"[DEBUG] Failed to reset {task_id}: {exc}", flush=True)
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

    max_steps    = observation.get("max_steps", 20)
    last_reward: Optional[float] = None
    final_score: float = 0.0
    rewards:     List[float] = []
    steps_taken: int = 0
    success:     bool = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, max_steps + 1):
            if observation.get("done", False):
                break

            user_prompt  = build_user_prompt(observation, last_reward, step)
            raw_response = call_llm(user_prompt)
            action       = parse_action(raw_response)
            action_str   = json.dumps(action)

            error_msg: Optional[str] = None
            try:
                result      = env_step(action)
                observation = result["observation"]
                reward_obj  = result["reward"]
                reward      = float(reward_obj.get("score", 0.0))
                done        = bool(reward_obj.get("done", False))
                final_score = reward
                last_reward = reward
                rewards.append(reward)
                steps_taken = step
            except Exception as exc:
                error_msg = str(exc)
                done      = False
                reward    = 0.0
                rewards.append(reward)
                steps_taken = step
                print(f"[DEBUG] step failed: {exc}", flush=True)

            log_step(step=step, action=action_str, reward=reward,
                     done=done, error=error_msg)

            if done:
                break

            time.sleep(0.3)

        success = final_score >= SUCCESS_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken,
                score=final_score, rewards=rewards)

    return final_score

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global ENV_BASE_URL

    parser = argparse.ArgumentParser(description="CSV Cleaner OpenEnv baseline agent")
    parser.add_argument(
        "--base-url",
        default=ENV_BASE_URL,
        help="Environment server base URL",
    )
    parser.add_argument(
        "--task",
        default=os.getenv("TASK_ID", ""),
        help="Run a single task (task1/task2/task3). Default: all.",
    )
    args = parser.parse_args()
    ENV_BASE_URL = args.base_url.rstrip("/")

    if not API_KEY:
        print("ERROR: HF_TOKEN is not set.", file=sys.stderr)
        sys.exit(1)

    print(f"[DEBUG] model={MODEL_NAME} api={API_BASE_URL} server={ENV_BASE_URL}", flush=True)

    try:
        all_tasks = env_tasks()
        task_ids  = [t["id"] for t in all_tasks]
    except Exception as exc:
        print(f"[DEBUG] Could not fetch tasks: {exc}", flush=True)
        task_ids = ["task1", "task2", "task3"]

    if args.task:
        if args.task not in task_ids:
            print(f"ERROR: Unknown task '{args.task}'.", file=sys.stderr)
            sys.exit(1)
        task_ids = [args.task]

    scores: Dict[str, float] = {}
    for task_id in task_ids:
        scores[task_id] = run_task(task_id)

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