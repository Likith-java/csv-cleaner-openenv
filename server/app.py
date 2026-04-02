"""
server/app.py — App factory for the CSV Cleaner OpenEnv environment.
This file is required by openenv validate for multi-mode deployment.
The actual FastAPI app lives in main.py; this re-exports it.
"""
from server.main import app

__all__ = ["app"]
