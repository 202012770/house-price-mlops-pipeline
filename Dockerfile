FROM python:3.10-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml constraints.txt ./
COPY src ./src
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -c constraints.txt .

FROM python:3.10-slim

# Override for networks where deb.debian.org is unreachable, e.g.:
#   docker build --build-arg APT_MIRROR=mirrors.tuna.tsinghua.edu.cn -t house-price-api .
ARG APT_MIRROR=deb.debian.org

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=7860 \
    HPP_PROJECT_ROOT=/app

RUN sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY configs ./configs
COPY models ./models
COPY data/reference ./data/reference

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn house_price.serving.api:app --host 0.0.0.0 --port ${PORT}"]
