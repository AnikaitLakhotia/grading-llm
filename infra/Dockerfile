# infra/Dockerfile
# Multi-stage build for a small runtime image
FROM python:3.11-slim AS builder

# Install build deps (if pip installing extras)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# copy only dependency files first for caching
COPY pyproject.toml requirements.txt requirements-dev.txt /app/

RUN python -m pip install --upgrade pip
# Install runtime deps (fastapi, uvicorn) and minimal deps. Avoid heavy dev deps.
RUN pip install "uvicorn[standard]" fastapi pydantic scikit-learn pandas

# Copy source
COPY . /app

# Final image
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy installed site-packages from builder? Simpler: pip install minimal runtime again
RUN python -m pip install --upgrade pip
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages || true

# Copy app
COPY . /app

EXPOSE 8000

# default envs
ENV APP_ENV=production
ENV USE_MOCK_LLM=1
ENV LOG_JSON=1
ENV MAX_FULL_TEXT_CHARS=20000

# Run uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
