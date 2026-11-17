# Serving image for the ADR-013 reference model (ADR-015).
#
# Generic by design: code and pinned dependencies only. The model bundle written
# by scripts/export_model_bundle.py is mounted read-only at /bundle at run time.
# No mlflow, no training stack, no curl, no secrets, no data.
#
# Build from the repo root:
#   docker build -f docker/api/Dockerfile -t email-priority-api:local .
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM needs the OpenMP runtime, which slim images do not ship.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home \
        --shell /usr/sbin/nologin app

WORKDIR /app

# Hashed, pinned lock compiled from requirements-serving.in (see its header).
COPY requirements-serving.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements-serving.lock

# Only the packages in the import closure of src.serving.app. Root-owned, so the
# non-root runtime user cannot modify them.
COPY src/__init__.py ./src/__init__.py
COPY src/features ./src/features
COPY src/labels ./src/labels
COPY src/serving ./src/serving

ENV MODEL_SOURCE=bundle \
    MODEL_BUNDLE_DIR=/bundle

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
