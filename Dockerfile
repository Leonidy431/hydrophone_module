# DiveGuard BlueOS Extension — Raspberry Pi 3 (linux/arm/v7) / arm64 / amd64
# Build: docker buildx build --platform linux/arm/v7 -t diveguard:latest .
FROM python:3.11-slim-bookworm

# Single layer for system deps; libatlas for numpy BLAS on armv7,
# libasound2 for pyalsaaudio, curl for HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libatlas3-base libasound2 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-rpi.txt .
RUN pip install --no-cache-dir -r requirements-rpi.txt

COPY audio_wal.py blueos_extension.py dsp_bridge.py \
     phases_1_4_implementation.py propeller_classifier.py \
     threat_assessment.py sensor_fusion.py diver_alert_controller.py \
     main_integration.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Non-root; /data volumes owned by app user
RUN groupadd -g 3000 diveguard && useradd -u 1000 -g 3000 -m diveguard \
    && mkdir -p /data/wal /data/logs && chown -R diveguard:diveguard /data /app
USER diveguard

VOLUME ["/data/wal", "/data/logs"]
EXPOSE 8734

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8734/v1/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
