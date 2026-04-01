"""
tasks.py — Task definitions for the CSV Cleaner OpenEnv environment.

Three tasks of increasing difficulty, each with:
  - A dirty CSV (initial state the agent starts from)
  - A clean CSV (ground truth used by the grader — agent never sees this)
  - A plain-English goal string
  - A grader function that scores current state 0.0 → 1.0

Grading philosophy
------------------
Each grader breaks the score into sub-components (null fix, dedup, normalize,
etc.) so partial progress is always rewarded. The agent never gets 0.0 unless
it has done absolutely nothing useful.
"""

from __future__ import annotations

import copy
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_date(val: Any) -> str | None:
    """Try to parse a messy date string into YYYY-MM-DD. Return None on failure."""
    if val is None or str(val).strip() in ("", "None", "NaN", "nan"):
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


def _is_null(val: Any) -> bool:
    if val is None:
        return True
    return str(val).strip().lower() in ("", "none", "nan", "n/a", "na", "null", "-")


def _digits_only(val: Any) -> str:
    return re.sub(r"\D", "", str(val))


def _rows_as_frozensets(rows: List[Dict]) -> List[frozenset]:
    return [frozenset((k, str(v)) for k, v in row.items()) for row in rows]


# ---------------------------------------------------------------------------
# TASK 1 — Easy: Fill null values
# ---------------------------------------------------------------------------

TASK1_GOAL = (
    "Clean this CSV of customer records. "
    "Fill all missing 'age' values with 0, "
    "fill all missing 'city' values with 'Unknown', "
    "and fill all missing 'email' values with 'noemail@example.com'. "
    "Do not remove any rows."
)

TASK1_DIRTY: List[Dict[str, Any]] = [
    {"id": 1, "name": "Alice",   "age": 30,   "city": "Mumbai",    "email": "alice@example.com"},
    {"id": 2, "name": "Bob",     "age": None,  "city": "Delhi",     "email": "bob@example.com"},
    {"id": 3, "name": "Carol",   "age": 25,   "city": None,        "email": "carol@example.com"},
    {"id": 4, "name": "David",   "age": None,  "city": "Pune",      "email": None},
    {"id": 5, "name": "Eva",     "age": 22,   "city": None,        "email": None},
    {"id": 6, "name": "Frank",   "age": 45,   "city": "Chennai",   "email": "frank@example.com"},
    {"id": 7, "name": "Grace",   "age": None,  "city": None,        "email": "grace@example.com"},
    {"id": 8, "name": "Hank",    "age": 38,   "city": "Kolkata",   "email": None},
]

TASK1_CLEAN: List[Dict[str, Any]] = [
    {"id": 1, "name": "Alice",   "age": 0,    "city": "Mumbai",             "email": "alice@example.com"},
    {"id": 2, "name": "Bob",     "age": 0,    "city": "Delhi",              "email": "bob@example.com"},
    {"id": 3, "name": "Carol",   "age": 25,   "city": "Unknown",            "email": "carol@example.com"},
    {"id": 4, "name": "David",   "age": 0,    "city": "Pune",               "email": "noemail@example.com"},
    {"id": 5, "name": "Eva",     "age": 22,   "city": "Unknown",            "email": "noemail@example.com"},
    {"id": 6, "name": "Frank",   "age": 45,   "city": "Chennai",            "email": "frank@example.com"},
    {"id": 7, "name": "Grace",   "age": 0,    "city": "Unknown",            "email": "grace@example.com"},
    {"id": 8, "name": "Hank",    "age": 38,   "city": "Kolkata",            "email": "noemail@example.com"},
]

TASK1_COLUMNS = ["id", "name", "age", "city", "email"]


def grade_task1(rows: List[Dict[str, Any]]) -> Tuple[float, str]:
    """
    Score = average fraction of null cells correctly filled across 3 columns.
    Each column contributes equally (1/3 of total score).
    """
    if not rows:
        return 0.0, "No rows found."

    cols = {
        "age":   {"fill": 0,                      "correct": 0, "total_nulls": 0},
        "city":  {"fill": "Unknown",               "correct": 0, "total_nulls": 0},
        "email": {"fill": "noemail@example.com",   "correct": 0, "total_nulls": 0},
    }

    clean_by_id = {r["id"]: r for r in TASK1_CLEAN}

    for col, cfg in cols.items():
        for dirty_row in TASK1_DIRTY:
            if _is_null(dirty_row[col]):
                cfg["total_nulls"] += 1
                agent_row = next((r for r in rows if r.get("id") == dirty_row["id"]), None)
                if agent_row is None:
                    continue
                expected = clean_by_id[dirty_row["id"]][col]
                actual   = agent_row.get(col)
                if str(actual).strip() == str(expected).strip():
                    cfg["correct"] += 1

    scores = []
    details = []
    for col, cfg in cols.items():
        if cfg["total_nulls"] == 0:
            scores.append(1.0)
        else:
            s = cfg["correct"] / cfg["total_nulls"]
            scores.append(s)
            details.append(f"{col}: {cfg['correct']}/{cfg['total_nulls']} nulls fixed")

    final = sum(scores) / len(scores)
    reason = "; ".join(details) if details else "All nulls correctly filled."
    return round(final, 4), reason


# ---------------------------------------------------------------------------
# TASK 2 — Medium: Remove duplicates + normalize dates + fill nulls
# ---------------------------------------------------------------------------

TASK2_GOAL = (
    "Clean this CSV of sales orders. "
    "1) Remove exact duplicate rows. "
    "2) Fill missing 'amount' values with 0.0. "
    "3) Normalize the 'order_date' column to YYYY-MM-DD format. "
    "The final table should have no duplicate rows, no null amounts, "
    "and all dates in ISO format."
)

TASK2_DIRTY: List[Dict[str, Any]] = [
    {"order_id": 1,  "customer": "Alice",   "amount": 250.0,  "order_date": "15/03/2024"},
    {"order_id": 2,  "customer": "Bob",     "amount": None,   "order_date": "2024-03-20"},
    {"order_id": 3,  "customer": "Carol",   "amount": 80.5,   "order_date": "March 5, 2024"},
    {"order_id": 4,  "customer": "David",   "amount": 320.0,  "order_date": "01/04/2024"},
    {"order_id": 2,  "customer": "Bob",     "amount": None,   "order_date": "2024-03-20"},   # duplicate
    {"order_id": 5,  "customer": "Eva",     "amount": None,   "order_date": "10-04-2024"},
    {"order_id": 6,  "customer": "Frank",   "amount": 150.0,  "order_date": "20240415"},
    {"order_id": 3,  "customer": "Carol",   "amount": 80.5,   "order_date": "March 5, 2024"}, # duplicate
    {"order_id": 7,  "customer": "Grace",   "amount": 90.0,   "order_date": "22/04/2024"},
    {"order_id": 8,  "customer": "Hank",    "amount": None,   "order_date": "2024-04-30"},
]

TASK2_CLEAN: List[Dict[str, Any]] = [
    {"order_id": 1,  "customer": "Alice",   "amount": 250.0,  "order_date": "2024-03-15"},
    {"order_id": 2,  "customer": "Bob",     "amount": 0.0,    "order_date": "2024-03-20"},
    {"order_id": 3,  "customer": "Carol",   "amount": 80.5,   "order_date": "2024-03-05"},
    {"order_id": 4,  "customer": "David",   "amount": 320.0,  "order_date": "2024-04-01"},
    {"order_id": 5,  "customer": "Eva",     "amount": 0.0,    "order_date": "2024-04-10"},
    {"order_id": 6,  "customer": "Frank",   "amount": 150.0,  "order_date": "2024-04-15"},
    {"order_id": 7,  "customer": "Grace",   "amount": 90.0,   "order_date": "2024-04-22"},
    {"order_id": 8,  "customer": "Hank",    "amount": 0.0,    "order_date": "2024-04-30"},
]

TASK2_COLUMNS = ["order_id", "customer", "amount", "order_date"]


def grade_task2(rows: List[Dict[str, Any]]) -> Tuple[float, str]:
    """
    Score has 3 equal components (each worth ~0.333):
      1. Deduplication  — correct number of rows
      2. Null fills     — missing amounts replaced with 0.0
      3. Date normalize — all dates in YYYY-MM-DD format
    """
    if not rows:
        return 0.0, "No rows found."

    details = []

    # --- Component 1: Deduplication (0.0–1.0) ---
    expected_count = len(TASK2_CLEAN)
    actual_count   = len(rows)
    if actual_count == expected_count:
        dedup_score = 1.0
    elif actual_count < expected_count:
        # Penalise for removing too many rows
        dedup_score = max(0.0, actual_count / expected_count)
    else:
        # Still has duplicates
        excess = actual_count - expected_count
        dedup_score = max(0.0, 1.0 - (excess / len(TASK2_DIRTY)))
    details.append(f"dedup: {actual_count} rows (expected {expected_count})")

    # Build a lookup for unique order_ids present in agent rows
    agent_by_id: Dict[int, Dict] = {}
    for r in rows:
        oid = r.get("order_id")
        if oid not in agent_by_id:
            agent_by_id[oid] = r

    clean_by_id = {r["order_id"]: r for r in TASK2_CLEAN}

    # --- Component 2: Null fills ---
    null_ids   = [r["order_id"] for r in TASK2_DIRTY if _is_null(r.get("amount"))]
    null_fixed = 0
    for oid in null_ids:
        agent_row = agent_by_id.get(oid)
        if agent_row and not _is_null(agent_row.get("amount")):
            try:
                if float(agent_row["amount"]) == 0.0:
                    null_fixed += 1
            except (TypeError, ValueError):
                pass
    null_score = null_fixed / len(null_ids) if null_ids else 1.0
    details.append(f"nulls fixed: {null_fixed}/{len(null_ids)}")

    # --- Component 3: Date normalization ---
    date_correct = 0
    for clean_row in TASK2_CLEAN:
        oid = clean_row["order_id"]
        agent_row = agent_by_id.get(oid)
        if agent_row is None:
            continue
        parsed = _parse_date(agent_row.get("order_date"))
        if parsed == clean_row["order_date"]:
            date_correct += 1
    date_score = date_correct / len(TASK2_CLEAN)
    details.append(f"dates normalized: {date_correct}/{len(TASK2_CLEAN)}")

    final = (dedup_score + null_score + date_score) / 3.0
    return round(final, 4), "; ".join(details)


# ---------------------------------------------------------------------------
# TASK 3 — Hard: Full pipeline on employee dataset
# ---------------------------------------------------------------------------

TASK3_GOAL = (
    "Clean this employee dataset completely. You must: "
    "1) Remove duplicate rows. "
    "2) Fill missing 'salary' values with 0. "
    "3) Fill missing 'department' values with 'General'. "
    "4) Normalize the 'phone' column to digits only (remove spaces, dashes, brackets). "
    "5) Normalize the 'join_date' column to YYYY-MM-DD format. "
    "6) Remove rows where 'salary' is outside the range 0–500000 (outliers). "
    "All 6 steps are required for a perfect score."
)

TASK3_DIRTY: List[Dict[str, Any]] = [
    {"emp_id": 1,  "name": "Alice",   "department": "Engineering",  "salary": 95000,  "phone": "98765-43210",  "join_date": "01/06/2020"},
    {"emp_id": 2,  "name": "Bob",     "department": None,           "salary": 72000,  "phone": "(22) 4567 8901","join_date": "15/08/2019"},
    {"emp_id": 3,  "name": "Carol",   "department": "HR",           "salary": None,   "phone": "91-98001-12345","join_date": "March 3, 2021"},
    {"emp_id": 4,  "name": "David",   "department": "Finance",      "salary": 850000, "phone": "044-23456789",  "join_date": "2022-01-10"},  # outlier salary
    {"emp_id": 5,  "name": "Eva",     "department": "Engineering",  "salary": 110000, "phone": "9900112233",    "join_date": "20230515"},
    {"emp_id": 6,  "name": "Frank",   "department": None,           "salary": 68000,  "phone": "+91 98765 12345","join_date": "10-07-2018"},
    {"emp_id": 7,  "name": "Grace",   "department": "Marketing",    "salary": None,   "phone": "080-45671234",  "join_date": "2021-11-22"},
    {"emp_id": 8,  "name": "Hank",    "department": "HR",           "salary": 54000,  "phone": "7788990011",    "join_date": "05/05/2017"},
    {"emp_id": 5,  "name": "Eva",     "department": "Engineering",  "salary": 110000, "phone": "9900112233",    "join_date": "20230515"},  # duplicate
    {"emp_id": 9,  "name": "Iris",    "department": "Finance",      "salary": -5000,  "phone": "9123456789",    "join_date": "2020-09-01"},  # outlier salary
    {"emp_id": 10, "name": "Jack",    "department": "Engineering",  "salary": 99000,  "phone": "98 76543210",   "join_date": "July 19, 2022"},
    {"emp_id": 3,  "name": "Carol",   "department": "HR",           "salary": None,   "phone": "91-98001-12345","join_date": "March 3, 2021"}, # duplicate
]

TASK3_CLEAN: List[Dict[str, Any]] = [
    {"emp_id": 1,  "name": "Alice",   "department": "Engineering",  "salary": 95000,  "phone": "9876543210",  "join_date": "2020-06-01"},
    {"emp_id": 2,  "name": "Bob",     "department": "General",      "salary": 72000,  "phone": "2245678901",  "join_date": "2019-08-15"},
    {"emp_id": 3,  "name": "Carol",   "department": "HR",           "salary": 0,      "phone": "919800112345","join_date": "2021-03-03"},
    # emp_id 4 removed (salary outlier 850000)
    {"emp_id": 5,  "name": "Eva",     "department": "Engineering",  "salary": 110000, "phone": "9900112233",  "join_date": "2023-05-15"},
    {"emp_id": 6,  "name": "Frank",   "department": "General",      "salary": 68000,  "phone": "919876512345","join_date": "2018-07-10"},
    {"emp_id": 7,  "name": "Grace",   "department": "Marketing",    "salary": 0,      "phone": "08045671234", "join_date": "2021-11-22"},
    {"emp_id": 8,  "name": "Hank",    "department": "HR",           "salary": 54000,  "phone": "7788990011",  "join_date": "2017-05-05"},
    # emp_id 9 removed (salary outlier -5000)
    {"emp_id": 10, "name": "Jack",    "department": "Engineering",  "salary": 99000,  "phone": "9876543210",  "join_date": "2022-07-19"},
]

TASK3_COLUMNS = ["emp_id", "name", "department", "salary", "phone", "join_date"]


def grade_task3(rows: List[Dict[str, Any]]) -> Tuple[float, str]:
    """
    Score has 6 equal components (each worth ~0.167):
      1. Deduplication
      2. Outlier removal (salary out of 0–500000)
      3. Null salary fill → 0
      4. Null department fill → 'General'
      5. Phone normalization → digits only
      6. Date normalization → YYYY-MM-DD
    """
    if not rows:
        return 0.0, "No rows found."

    details  = []
    agent_by_id: Dict[int, Dict] = {}
    for r in rows:
        eid = r.get("emp_id")
        if eid not in agent_by_id:
            agent_by_id[eid] = r

    clean_by_id = {r["emp_id"]: r for r in TASK3_CLEAN}
    dirty_by_id: Dict[int, List[Dict]] = {}
    for r in TASK3_DIRTY:
        dirty_by_id.setdefault(r["emp_id"], []).append(r)

    # --- 1. Deduplication ---
    expected_count = len(TASK3_CLEAN)
    actual_unique  = len(agent_by_id)
    dedup_score    = 1.0 if actual_unique == expected_count else max(
        0.0, 1.0 - abs(actual_unique - expected_count) / len(TASK3_DIRTY)
    )
    details.append(f"dedup: {actual_unique} unique rows (expected {expected_count})")

    # --- 2. Outlier removal (emp_id 4 salary=850000, emp_id 9 salary=-5000) ---
    outlier_ids      = {4, 9}
    outliers_removed = sum(1 for oid in outlier_ids if oid not in agent_by_id)
    outlier_score    = outliers_removed / len(outlier_ids)
    details.append(f"outliers removed: {outliers_removed}/{len(outlier_ids)}")

    # --- 3. Null salary fill ---
    null_salary_ids = [3, 7]  # emp_ids with null salary in dirty data
    salary_fixed    = 0
    for eid in null_salary_ids:
        r = agent_by_id.get(eid)
        if r and not _is_null(r.get("salary")):
            try:
                if float(r["salary"]) == 0.0:
                    salary_fixed += 1
            except (TypeError, ValueError):
                pass
    null_salary_score = salary_fixed / len(null_salary_ids)
    details.append(f"null salaries fixed: {salary_fixed}/{len(null_salary_ids)}")

    # --- 4. Null department fill ---
    null_dept_ids = [2, 6]  # emp_ids with null department
    dept_fixed    = 0
    for eid in null_dept_ids:
        r = agent_by_id.get(eid)
        if r and str(r.get("department", "")).strip() == "General":
            dept_fixed += 1
    null_dept_score = dept_fixed / len(null_dept_ids)
    details.append(f"null departments fixed: {dept_fixed}/{len(null_dept_ids)}")

    # --- 5. Phone normalization ---
    phone_correct = 0
    checkable     = [r for r in TASK3_CLEAN if r["emp_id"] in agent_by_id]
    for clean_row in checkable:
        agent_row = agent_by_id[clean_row["emp_id"]]
        agent_phone = _digits_only(agent_row.get("phone", ""))
        clean_phone = _digits_only(clean_row["phone"])
        if agent_phone == clean_phone:
            phone_correct += 1
    phone_score = phone_correct / len(checkable) if checkable else 0.0
    details.append(f"phones normalized: {phone_correct}/{len(checkable)}")

    # --- 6. Date normalization ---
    date_correct = 0
    for clean_row in checkable:
        agent_row = agent_by_id[clean_row["emp_id"]]
        parsed = _parse_date(agent_row.get("join_date"))
        if parsed == clean_row["join_date"]:
            date_correct += 1
    date_score = date_correct / len(checkable) if checkable else 0.0
    details.append(f"dates normalized: {date_correct}/{len(checkable)}")

    components = [
        dedup_score, outlier_score, null_salary_score,
        null_dept_score, phone_score, date_score,
    ]
    final = sum(components) / len(components)
    return round(final, 4), "; ".join(details)


# ---------------------------------------------------------------------------
# Task registry — used by env.py to look up tasks by ID
# ---------------------------------------------------------------------------

TASKS = {
    "task1": {
        "id":       "task1",
        "name":     "Fill Null Values",
        "difficulty": "easy",
        "goal":     TASK1_GOAL,
        "dirty":    TASK1_DIRTY,
        "clean":    TASK1_CLEAN,
        "columns":  TASK1_COLUMNS,
        "grader":   grade_task1,
        "max_steps": 10,
    },
    "task2": {
        "id":       "task2",
        "name":     "Deduplicate & Normalize Dates",
        "difficulty": "medium",
        "goal":     TASK2_GOAL,
        "dirty":    TASK2_DIRTY,
        "clean":    TASK2_CLEAN,
        "columns":  TASK2_COLUMNS,
        "grader":   grade_task2,
        "max_steps": 15,
    },
    "task3": {
        "id":       "task3",
        "name":     "Full Pipeline Clean",
        "difficulty": "hard",
        "goal":     TASK3_GOAL,
        "dirty":    TASK3_DIRTY,
        "clean":    TASK3_CLEAN,
        "columns":  TASK3_COLUMNS,
        "grader":   grade_task3,
        "max_steps": 20,
    },
}


def get_task(task_id: str) -> dict:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks() -> List[Dict]:
    return [
        {
            "id":         t["id"],
            "name":       t["name"],
            "difficulty": t["difficulty"],
            "goal":       t["goal"],
            "columns":    t["columns"],
            "max_steps":  t["max_steps"],
        }
        for t in TASKS.values()
    ]