# -----------------------------------------------------------------------
# CSV Cleaner — OpenEnv Environment
# -----------------------------------------------------------------------
# Build:  docker build -t csv-cleaner-env .
# Run:    docker run -p 7860:7860 csv-cleaner-env
# -----------------------------------------------------------------------

FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers stdout/stderr
# so docker logs show up immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install dependencies first (separate layer = faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy server source
COPY server/ ./server/

# Copy OpenEnv metadata
COPY openenv.yaml .

# HuggingFace Spaces runs as a non-root user — create one to match
RUN useradd -m -u 1000 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Run from server/ so all relative imports (models, tasks, env) resolve
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--app-dir", "/app/server"]