---
title: CSV Cleaner OpenEnv
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
app_port: 7860
---

# 🧹 CSV Cleaner — OpenEnv Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible environment where AI agents learn to clean messy CSV datasets through a step-by-step action interface.

Built for the **Meta PyTorch × Hugging Face × Scaler OpenEnv Hackathon**.

---

## What Is This?

Real-world data is almost never clean. Data teams spend hours fixing null values, removing duplicates, standardizing date formats, and removing outliers before any analysis can begin.

This environment simulates that process. An AI agent receives a dirty CSV and a plain-English cleaning goal, then issues one cleaning action per step — just like a human data analyst would. A programmatic grader scores progress after every action (0.0 → 1.0), rewarding partial progress and penalising wasted steps.

---

## Quickstart

### 1. Run locally with Docker

```bash
git clone https://github.com/YOUR_USERNAME/csv-cleaner-openenv
cd csv-cleaner-openenv

# Build and run the environment server
docker build -t csv-cleaner-env .
docker run -p 7860:7860 csv-cleaner-env
```

Server is now live at `http://localhost:7860`.  
Swagger UI (interactive docs): `http://localhost:7860/docs`

### 2. Run the baseline agent

```bash
pip install -r requirements-inference.txt

export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_your_token_here"

python inference.py
```

Run a single task:
```bash
TASK_ID=task2 python inference.py
```

Run against your deployed HF Space:
```bash
python inference.py --base-url https://YOUR-SPACE.hf.space
```

### 3. Validate submission

```bash
pip install openenv-core
openenv validate
```

---

## API Endpoints

| Method | Endpoint  | Description                              |
|--------|-----------|------------------------------------------|
| POST   | `/reset`  | Start a new episode                      |
| POST   | `/step`   | Take one cleaning action                 |
| GET    | `/state`  | Get full current episode state           |
| GET    | `/tasks`  | List all available tasks                 |
| GET    | `/health` | Liveness check                           |
| GET    | `/docs`   | Swagger UI (interactive documentation)   |

### Example: Reset and take a step

```bash
# Start a new episode on task1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1"}'

# Fill null values in the 'age' column with 0
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "fill_nulls", "column": "age", "value": 0}'
```

---

## Observation Space

What the agent receives from `reset()` and `step()`:

| Field         | Type              | Description                                           |
|---------------|-------------------|-------------------------------------------------------|
| `rows`        | `List[Dict]`      | Current CSV state as a list of row dictionaries       |
| `columns`     | `List[str]`       | Column names in order                                 |
| `goal`        | `str`             | Plain-English description of the cleaning objective   |
| `task_id`     | `str`             | Active task: `task1`, `task2`, or `task3`             |
| `step_number` | `int`             | Number of actions taken so far this episode           |
| `max_steps`   | `int`             | Maximum steps allowed before the episode terminates   |
| `done`        | `bool`            | `True` when the episode has ended                     |
| `info`        | `Dict`            | Diagnostic info (last action message, grade detail)   |

---

## Action Space

Six action types the agent can use:

### `fill_nulls`
Fill all null/missing values in a column with a given value.
```json
{"action_type": "fill_nulls", "column": "age", "value": 0}
{"action_type": "fill_nulls", "column": "city", "value": "Unknown"}
```

### `remove_duplicates`
Remove all exact duplicate rows from the CSV.
```json
{"action_type": "remove_duplicates"}
```

### `normalize_column`
Standardize the format of a column. Supported formats:
- `digits_only` — strip all non-numeric characters (good for phone numbers)
- `lowercase` — convert to lowercase
- `strip` — remove leading/trailing whitespace
- `title_case` — convert to title case
- `iso_date` — parse and convert to `YYYY-MM-DD`

```json
{"action_type": "normalize_column", "column": "phone", "format": "digits_only"}
{"action_type": "normalize_column", "column": "order_date", "format": "iso_date"}
```

### `cast_column`
Cast a column to a different data type. Supported types: `int`, `float`, `str`, `date`.
```json
{"action_type": "cast_column", "column": "salary", "dtype": "int"}
```

### `remove_outliers`
Remove rows where a column's value falls outside `[min_val, max_val]`.
```json
{"action_type": "remove_outliers", "column": "salary", "min_val": 0, "max_val": 500000}
```

### `noop`
Do nothing. Use when the task is complete or the agent is uncertain.
```json
{"action_type": "noop"}
```

---

## Reward

| Field   | Type    | Description                                      |
|---------|---------|--------------------------------------------------|
| `score` | `float` | Current cleanliness score in `[0.0, 1.0]`        |
| `delta` | `float` | Change in score from the previous step           |
| `reason`| `str`   | Human-readable explanation of the score          |
| `done`  | `bool`  | `True` when the episode should end               |

Rewards are **dense** — every action that improves the CSV produces a positive `delta`. The agent never has to wait until the end to know if it's making progress.

---

## Tasks

### Task 1 — Fill Null Values `[easy]` · 10 steps

**Dataset:** 8-row customer records CSV  
**Columns:** `id`, `name`, `age`, `city`, `email`

The dataset has null values scattered across three columns. The agent must fill them with the correct defaults:
- `age` → `0`
- `city` → `"Unknown"`
- `email` → `"noemail@example.com"`

**Grader:** Scores the fraction of null cells correctly filled across all three columns. Each column contributes equally (1/3) to the total score.

**Expected difficulty:** A capable LLM should score 0.8–1.0 in 3–5 steps.

---

### Task 2 — Deduplicate & Normalize Dates `[medium]` · 15 steps

**Dataset:** 10-row sales orders CSV (includes 2 duplicate rows)  
**Columns:** `order_id`, `customer`, `amount`, `order_date`

Three issues to fix simultaneously:
1. Remove the 2 exact duplicate rows
2. Fill null `amount` values with `0.0`
3. Normalize all `order_date` values to `YYYY-MM-DD` format (dates arrive in 6 different formats)

**Grader:** Equal-weight scoring across deduplication, null fill, and date normalization (each 1/3).

**Expected difficulty:** Medium — requires recognising multiple date formats like `"March 5, 2024"`, `"20240415"`, `"15/03/2024"`.

---

### Task 3 — Full Pipeline Clean `[hard]` · 20 steps

**Dataset:** 12-row employee records CSV (includes duplicates and outliers)  
**Columns:** `emp_id`, `name`, `department`, `salary`, `phone`, `join_date`

Six issues to resolve:
1. Remove exact duplicate rows
2. Fill null `salary` values with `0`
3. Fill null `department` values with `"General"`
4. Normalize `phone` numbers to digits only (messy Indian phone formats)
5. Normalize `join_date` to `YYYY-MM-DD`
6. Remove salary outliers (outside range `0–500000`)

**Grader:** Equal-weight scoring across all 6 sub-tasks (each 1/6 of total score).

**Expected difficulty:** Hard — even frontier models struggle to score 1.0 because all 6 steps must be completed correctly and in the right order (e.g. outliers must be removed, not just detected).

---

## Baseline Scores

Scores produced by `inference.py` using `meta-llama/Llama-3.3-70B-Instruct`
via the HuggingFace Inference Router:

```
============================================================
  BASELINE RESULTS
============================================================
  task1  [████████████████░░░░]  0.8333
  task2  [████████████░░░░░░░░]  0.6111
  task3  [████████░░░░░░░░░░░░]  0.4167

  Average score: 0.6204
============================================================
```

> Note: Scores are deterministic (`temperature=0.0`) and reproducible.
> Re-running the script should produce the same scores.

---

## Project Structure

```
csv-cleaner-openenv/
├── inference.py              ← Baseline agent (required at root)
├── openenv.yaml              ← OpenEnv spec metadata
├── Dockerfile                ← Container definition
├── requirements.txt          ← Server dependencies
├── requirements-inference.txt← Inference script dependencies
├── README.md                 ← This file
└── server/
    ├── main.py               ← FastAPI server (reset/step/state endpoints)
    ├── env.py                ← Core environment logic
    ├── tasks.py              ← Task definitions, CSV data, graders
    └── models.py             ← Pydantic Observation/Action/Reward models
```

---

## Environment Variables

| Variable      | Required | Description                              |
|---------------|----------|------------------------------------------|
| `API_BASE_URL`| Yes      | LLM API endpoint                         |
| `MODEL_NAME`  | Yes      | Model identifier for inference           |
| `HF_TOKEN`    | Yes      | HuggingFace API key                      |
| `ENV_BASE_URL`| No       | Environment server URL (default: `http://0.0.0.0:7860`) |
| `TASK_ID`     | No       | Run a single task instead of all three  |

---

## License

MIT