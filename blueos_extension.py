"""
DiveGuard BlueOS Extension — FastAPI service for Raspberry Pi 3.

BlueOS integration contract:
- GET /register_service  -> extension metadata (BlueOS Helper convention)
- GET /v1/health         -> liveness (used by Docker HEALTHCHECK)
- GET /v1/status         -> pipeline stats (ring buffer, WAL, threshold)
- GET /v1/detections     -> recent threat detections (bounded list)
- POST /v1/simulate      -> inject one synthetic frame (dev/bench only,
                            enabled by DIVEGUARD_DEV=1)

Design constraints (RPi3, 1GB RAM, ARMv7):
- The acquisition loop runs as an asyncio background task; blocking calls
  (ALSA read, ZMQ DSP round-trip) are pushed to a single worker thread via
  run_in_executor so the event loop never stalls telemetry.
- MAVLink STATUSTEXT alerts go through MAVLink2REST with a hard timeout and
  fail-soft error flagging (sensor loss must not crash the service).
- SIGTERM triggers phased shutdown: stop intake -> drain ring to WAL ->
  fsync -> exit (fits the 90s Docker stop_grace_period).

Env configuration (.env / compose):
    DIVEGUARD_PORT              default 8734
    DIVEGUARD_WAL_DIR           default /data/wal
    DIVEGUARD_MAVLINK2REST_URL  default http://host.docker.internal:6040
    DIVEGUARD_ALERT_SEVERITY    default 4 (MAV_SEVERITY_WARNING)
    DIVEGUARD_DEV               default 0
    DIVEGUARD_LOG_LEVEL         default INFO
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
import urllib.request
from collections import deque
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from audio_wal import AudioWAL, RingBuffer, WALFrame

LOG_LEVEL = os.environ.get("DIVEGUARD_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='{"ts":"%(asctime)s","lvl":"%(levelname)s",'
           '"mod":"%(name)s","msg":"%(message)s"}',
)
logger = logging.getLogger("diveguard.ext")

PORT = int(os.environ.get("DIVEGUARD_PORT", "8734"))
WAL_DIR = os.environ.get("DIVEGUARD_WAL_DIR", "/data/wal")
MAVLINK2REST_URL = os.environ.get(
    "DIVEGUARD_MAVLINK2REST_URL", "http://host.docker.internal:6040")
ALERT_SEVERITY = int(os.environ.get("DIVEGUARD_ALERT_SEVERITY", "4"))
DEV_MODE = os.environ.get("DIVEGUARD_DEV", "0") == "1"

SERVICE_METADATA = {
    "name": "DiveGuard Propeller Detector",
    "description": "Hydroacoustic propeller detection (LOFAR/DEMON) "
                   "with diver alerts",
    "icon": "mdi-waveform",
    "company": "NPO Laboratoria K",
    "version": "1.0.0",
    "webpage": "https://github.com/leonidy431/hydrophone_module",
    "api": "/docs",
}


class Detection(BaseModel):
    timestamp_ms: float
    propeller_score: float = Field(ge=0.0, le=1.0)
    threat_level: str
    max_bpf_power_db: float


class SimulateRequest(BaseModel):
    propeller_score: float = Field(0.9, ge=0.0, le=1.0)
    threat_level: str = "HIGH"
    max_bpf_power_db: float = 72.0


class MavlinkNotifier:
    """STATUSTEXT via MAVLink2REST. Fail-soft: errors set a flag, never raise."""

    def __init__(self, base_url: str, timeout_s: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.last_error: Optional[str] = None
        self.sent_count = 0

    def _post_sync(self, text: str) -> None:
        # STATUSTEXT text field: max 50 chars, NUL-padded array.
        chars = [ord(c) for c in text[:50]]
        chars += [0] * (50 - len(chars))
        body = json.dumps({
            "header": {"system_id": 255, "component_id": 240, "sequence": 0},
            "message": {
                "type": "STATUSTEXT",
                "severity": {"type": f"MAV_SEVERITY_{'WARNING' if ALERT_SEVERITY >= 4 else 'ALERT'}"},
                "text": chars,
                "id": 0, "chunk_seq": 0,
            },
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/mavlink", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s):
            pass

    async def notify(self, text: str) -> bool:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._post_sync, text)
            self.sent_count += 1
            self.last_error = None
            return True
        except Exception as e:  # network down != mission abort
            self.last_error = str(e)
            logger.warning("MAVLink2REST notify failed: %s", e)
            return False


class DiveGuardService:
    def __init__(self) -> None:
        self.ring = RingBuffer(capacity_frames=4096, high_watermark=0.95)
        self.wal = AudioWAL(WAL_DIR)
        self.mavlink = MavlinkNotifier(MAVLINK2REST_URL)
        self.detections: deque[Detection] = deque(maxlen=200)
        self.started_at = time.time()
        self.sensor_ok = True
        self.sensor_last_error: Optional[str] = None
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._pipeline = None  # lazy: DSPPipeline needs the C++ core socket
        self._reader = None

    # ---------- acquisition (worker-thread blocking calls) ----------

    def _open_hardware_sync(self) -> None:
        from dsp_bridge import ALSAHydrophoneReader, DSPPipeline
        self._reader = ALSAHydrophoneReader(
            device_name=os.environ.get("DIVEGUARD_ALSA_DEVICE", "default"))
        try:
            self._reader.open()
        except Exception as e:
            self.sensor_ok = False
            self.sensor_last_error = f"ALSA open: {e}"
            logger.error("Hydrophone unavailable, running degraded: %s", e)
        self._pipeline = DSPPipeline(
            os.environ.get("DIVEGUARD_DSP_ENDPOINT",
                           "ipc:///tmp/diveguard_dsp.ipc"))
        try:
            self._pipeline.connect()
        except Exception as e:
            self.sensor_ok = False
            self.sensor_last_error = f"DSP connect: {e}"
            logger.error("DSP core unavailable, running degraded: %s", e)

    def _acquire_one_sync(self):
        """One blocking read+process cycle. Never raises."""
        try:
            frame = self._reader.read_frame() if self._reader else None
            if frame is None:
                return None
            wal_frame = WALFrame(frame.timestamp_ms, bytes(frame.samples))
            if not self.ring.push(wal_frame):
                logger.warning("Ring overflow, frame evicted "
                               "(dropped=%d)", self.ring.dropped_frames)
            result = None
            if self._pipeline is not None:
                result = self._pipeline.process_audio(frame)
            self.sensor_ok = True
            return result
        except Exception as e:
            self.sensor_ok = False
            self.sensor_last_error = str(e)
            logger.error("Acquisition error (degraded, will retry): %s", e)
            return None

    async def acquisition_loop(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._open_hardware_sync)
        backoff = 0.05
        while not self._stop.is_set():
            result = await loop.run_in_executor(None, self._acquire_one_sync)
            if result is not None and result.propeller_score >= 0.7:
                await self._register_detection(Detection(
                    timestamp_ms=result.timestamp_ms,
                    propeller_score=result.propeller_score,
                    threat_level=result.threat_level,
                    max_bpf_power_db=result.max_bpf_power_db,
                ))
            backoff = 0.05 if self.sensor_ok else min(backoff * 2, 5.0)
            await asyncio.sleep(backoff if not self.sensor_ok else 0.01)

    async def flush_loop(self) -> None:
        """Drain ring to WAL in batches; emergency path above watermark."""
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            try:
                if self.ring.above_watermark:
                    await loop.run_in_executor(
                        None, self.wal.emergency_flush, self.ring)
                elif len(self.ring) >= self.wal.batch_size:
                    batch = self.ring.pop_batch(self.wal.batch_size)
                    await loop.run_in_executor(
                        None, self.wal.write_batch, batch)
            except Exception:
                logger.exception("WAL flush error")
            await asyncio.sleep(0.2)

    async def _register_detection(self, det: Detection) -> None:
        self.detections.append(det)
        await self.mavlink.notify(
            f"DIVEGUARD {det.threat_level} score={det.propeller_score:.2f}")

    # ---------- lifecycle ----------

    async def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self.acquisition_loop(), name="acquire"),
            asyncio.create_task(self.flush_loop(), name="flush"),
        ]
        logger.info("DiveGuard service started (dev=%s, wal=%s)",
                    DEV_MODE, WAL_DIR)

    async def shutdown(self) -> None:
        """Phased: stop intake, drain ring fully, fsync, close."""
        logger.info("Shutdown phase 1: stopping intake")
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        logger.info("Shutdown phase 2: draining ring (%d frames)",
                    len(self.ring))
        loop = asyncio.get_running_loop()
        drain_deadline = time.monotonic() + 55.0
        while len(self.ring) and time.monotonic() < drain_deadline:
            batch = self.ring.pop_batch(self.wal.batch_size)
            await loop.run_in_executor(None, self.wal.write_batch, batch)

        logger.info("Shutdown phase 3: closing WAL "
                    "(%d frames written, %d emergency flushes)",
                    self.wal.frames_written, self.wal.emergency_flush_count)
        await loop.run_in_executor(None, self.wal.close)
        if self._pipeline is not None:
            try:
                self._pipeline.shutdown()
            except Exception:
                logger.exception("DSP pipeline shutdown error")
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                logger.exception("ALSA close error")
        logger.info("Shutdown complete")

    def status(self) -> dict:
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "sensor_ok": self.sensor_ok,
            "sensor_last_error": self.sensor_last_error,
            "ring": self.ring.stats(),
            "wal": {
                "frames_written": self.wal.frames_written,
                "emergency_flush_count": self.wal.emergency_flush_count,
            },
            "mavlink": {
                "sent": self.mavlink.sent_count,
                "last_error": self.mavlink.last_error,
            },
            "detections_cached": len(self.detections),
            "dev_mode": DEV_MODE,
        }


service = DiveGuardService()
app = FastAPI(title="DiveGuard BlueOS Extension", version="1.0.0")


@app.on_event("startup")
async def _startup() -> None:
    await service.start()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(
                sig, lambda: asyncio.create_task(service.shutdown()))
        except (NotImplementedError, RuntimeError, ValueError):
            # Non-unix envs / event loop in a non-main thread (TestClient).
            # uvicorn's own SIGTERM handling still drives the shutdown hook.
            pass


@app.on_event("shutdown")
async def _shutdown() -> None:
    if not service._stop.is_set():
        await service.shutdown()


@app.get("/register_service")
async def register_service() -> dict:
    return SERVICE_METADATA


@app.get("/v1/health")
async def health() -> dict:
    return {"status": "ok" if service.sensor_ok else "degraded",
            "ring_fill": service.ring.fill_ratio}


@app.get("/v1/status")
async def status() -> dict:
    return service.status()


@app.get("/v1/detections")
async def detections(limit: int = 50) -> list[Detection]:
    limit = max(1, min(limit, 200))
    return list(service.detections)[-limit:]


@app.post("/v1/simulate")
async def simulate(req: SimulateRequest) -> dict:
    if not DEV_MODE:
        raise HTTPException(status_code=403, detail="dev mode disabled")
    det = Detection(timestamp_ms=time.time() * 1000.0,
                    propeller_score=req.propeller_score,
                    threat_level=req.threat_level,
                    max_bpf_power_db=req.max_bpf_power_db)
    await service._register_detection(det)
    return {"accepted": True, "mavlink_ok": service.mavlink.last_error is None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL.lower())
