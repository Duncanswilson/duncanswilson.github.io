# Build natively on either linux/arm64 (Oracle A1) or linux/amd64.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 flyreward \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /state flyreward \
    && mkdir -p /state/data /state/live \
    && chown -R 10001:10001 /state

WORKDIR /app
COPY pyproject.toml ./
COPY requirements-live-lock.txt ./
COPY flyreward ./flyreward
COPY visualizer ./visualizer
RUN python -m pip install --no-cache-dir -c requirements-live-lock.txt '.[live]'

USER 10001:10001
EXPOSE 8000
STOPSIGNAL SIGTERM

# Exec form preserves signal delivery. There is deliberately no ENTRYPOINT:
# diagnostic commands replace this complete command without inherited arguments.
CMD ["python", "-m", "flyreward.server", "--host", "0.0.0.0", "--port", "8000", "--data-dir", "/state/data", "--state-dir", "/state/live", "--viewer-dir", "/app/visualizer", "--download"]
