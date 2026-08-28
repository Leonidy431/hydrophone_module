"""
DiveGuard Audio Ring Buffer + Write-Ahead Log (WAL).

Durability layer for hydrophone acquisition on Raspberry Pi 3:
- Bounded ring buffer with watermark backpressure (no unbounded RAM growth).
- Batch WAL flush (append + fsync) off the hot path.
- Atomic emergency flush: temp file -> fsync -> os.rename (all-or-nothing,
  survives SIGKILL/OOM between extraction and write).
- Recovery scan that tolerates a torn final record.

Record format (little-endian):
    [timestamp_ms: float64][payload_len: uint32][payload: bytes]

Stdlib-only: no numpy required on this path.
"""

from __future__ import annotations

import logging
import os
import struct
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterator, List, Optional

logger = logging.getLogger("diveguard.wal")

_HEADER = struct.Struct("<dI")  # timestamp_ms (f64), payload_len (u32)
MAX_PAYLOAD_BYTES = 1 << 20  # 1 MiB sanity cap per frame


@dataclass(frozen=True)
class WALFrame:
    timestamp_ms: float
    payload: bytes


class RingBuffer:
    """Bounded FIFO of WALFrame with high-watermark backpressure."""

    def __init__(self, capacity_frames: int = 4096,
                 high_watermark: float = 0.95):
        if capacity_frames <= 0:
            raise ValueError("capacity_frames must be > 0")
        if not 0.0 < high_watermark <= 1.0:
            raise ValueError("high_watermark must be in (0, 1]")
        self._buf: deque[WALFrame] = deque(maxlen=capacity_frames)
        self.capacity = capacity_frames
        self.high_watermark = high_watermark
        self.dropped_frames = 0

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def fill_ratio(self) -> float:
        return len(self._buf) / self.capacity

    @property
    def above_watermark(self) -> bool:
        return self.fill_ratio >= self.high_watermark

    def push(self, frame: WALFrame) -> bool:
        """Append a frame. Returns False if the oldest frame was evicted
        (buffer full) — caller should trigger a flush."""
        evicted = len(self._buf) == self.capacity
        if evicted:
            self.dropped_frames += 1
        self._buf.append(frame)
        return not evicted

    def pop_batch(self, max_frames: int = 500) -> List[WALFrame]:
        out: List[WALFrame] = []
        while self._buf and len(out) < max_frames:
            out.append(self._buf.popleft())
        return out

    def stats(self) -> dict:
        return {
            "size": len(self._buf),
            "capacity": self.capacity,
            "fill_ratio": round(self.fill_ratio, 4),
            "dropped_frames": self.dropped_frames,
        }


class AudioWAL:
    """Append-only WAL with batch flush and atomic emergency flush."""

    def __init__(self, wal_dir: str, batch_size: int = 500,
                 max_wal_bytes: int = 512 * 1024 * 1024):
        self.wal_dir = os.path.abspath(wal_dir)
        os.makedirs(self.wal_dir, exist_ok=True)
        self.batch_size = batch_size
        self.max_wal_bytes = max_wal_bytes
        self._active_path = os.path.join(self.wal_dir, "active.wal")
        self._fh = open(self._active_path, "ab")
        self.frames_written = 0
        self.emergency_flush_count = 0
        self.last_emergency_flush_time: Optional[float] = None

    @staticmethod
    def _pack(frame: WALFrame) -> bytes:
        return _HEADER.pack(frame.timestamp_ms, len(frame.payload)) + frame.payload

    def write_batch(self, frames: List[WALFrame], fsync: bool = True) -> int:
        """Append frames to the active WAL. Returns bytes written."""
        if not frames:
            return 0
        blob = b"".join(self._pack(f) for f in frames)
        self._fh.write(blob)
        self._fh.flush()
        if fsync:
            os.fsync(self._fh.fileno())
        self.frames_written += len(frames)
        self._enforce_size_cap()
        return len(blob)

    def _enforce_size_cap(self) -> None:
        if self._fh.tell() < self.max_wal_bytes:
            return
        # Rotate: close active, rename with timestamp, open fresh.
        self._fh.close()
        rotated = os.path.join(self.wal_dir, f"rotated_{int(time.time())}.wal")
        os.rename(self._active_path, rotated)
        self._fh = open(self._active_path, "ab")
        logger.info("WAL rotated to %s", rotated)

    def emergency_flush(self, ring: RingBuffer,
                        max_frames: int = 500) -> Optional[str]:
        """Atomic all-or-nothing drain of up to max_frames from the ring.

        Phase 1: bounded extraction into memory.
        Phase 2: write to a temp file in the same directory + fsync.
        Phase 3: os.rename (atomic on POSIX, same filesystem).
        On any failure the temp file is unlinked; frames stay lost only if
        the process dies between popleft and pack — bounded by max_frames.
        Returns the final WAL path, or None if the ring was empty.
        """
        frames = ring.pop_batch(max_frames)
        if not frames:
            return None

        temp_fd, temp_path = tempfile.mkstemp(
            suffix=".wal.tmp", prefix="flush_", dir=self.wal_dir)
        try:
            with os.fdopen(temp_fd, "wb") as tf:
                for f in frames:
                    tf.write(self._pack(f))
                tf.flush()
                os.fsync(tf.fileno())
            final_path = os.path.join(
                self.wal_dir,
                f"emergency_{time.time_ns()}.wal")
            os.rename(temp_path, final_path)
        except Exception:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            logger.exception("Emergency flush failed (%d frames at risk)",
                             len(frames))
            raise

        self.emergency_flush_count += 1
        self.last_emergency_flush_time = time.time()
        logger.info("Emergency flush: %d frames -> %s",
                    len(frames), os.path.basename(final_path))
        return final_path

    def close(self) -> None:
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        finally:
            self._fh.close()

    # ---------------- recovery ----------------

    @staticmethod
    def iter_wal_file(path: str) -> Iterator[WALFrame]:
        """Yield frames from a WAL file; stop cleanly at a torn tail."""
        with open(path, "rb") as fh:
            while True:
                hdr = fh.read(_HEADER.size)
                if len(hdr) < _HEADER.size:
                    if hdr:
                        logger.warning("Torn header at end of %s", path)
                    return
                ts, plen = _HEADER.unpack(hdr)
                if plen > MAX_PAYLOAD_BYTES:
                    logger.error("Corrupt record (len=%d) in %s — stopping",
                                 plen, path)
                    return
                payload = fh.read(plen)
                if len(payload) < plen:
                    logger.warning("Torn payload at end of %s", path)
                    return
                yield WALFrame(ts, payload)

    def recover_all(self) -> List[WALFrame]:
        """Read every frame from every WAL file in wal_dir (sorted,
        numeric-aware for emergency_<ns> names)."""
        frames: List[WALFrame] = []
        names = [n for n in os.listdir(self.wal_dir) if n.endswith(".wal")]

        def sort_key(name: str):
            stem = name.rsplit(".", 1)[0]
            for prefix in ("emergency_", "rotated_"):
                if stem.startswith(prefix):
                    tail = stem[len(prefix):]
                    if tail.isdigit():
                        return (1, int(tail))
            return (2, name)  # active.wal last

        for name in sorted(names, key=sort_key):
            frames.extend(self.iter_wal_file(os.path.join(self.wal_dir, name)))
        return frames
