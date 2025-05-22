# Multi-stage build. The runtime image carries only the virtualenv and the
# application, not the build toolchain.
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependency metadata is copied first so the install layer is cached and does
# not re-run on every source change.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --upgrade pip && pip install ".[api,mcp]"


FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

# Run as a non-root user. A container compromise should not land on root.
RUN useradd --create-home --shell /bin/bash --uid 1000 app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app src/ ./src/

USER app
EXPOSE 8000

# Hits the liveness endpoint so an unhealthy container is replaced rather than
# left accepting traffic it cannot serve.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["enrich-api"]
