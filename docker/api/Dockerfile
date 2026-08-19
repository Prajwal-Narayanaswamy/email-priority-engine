FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency spec first for layer caching
COPY pyproject.toml .

# Install production dependencies only (no dev extras)
RUN uv pip install --system -e .

# Copy source code
COPY src/ ./src/

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000 8001

CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
