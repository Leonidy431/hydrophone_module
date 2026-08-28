"""Tests: RingBuffer, AudioWAL atomic emergency flush, recovery,
and the AdaptiveThresholdModule dB/score unit-bug regression."""

import os
import struct

import pytest

from audio_wal import MAX_PAYLOAD_BYTES, AudioWAL, RingBuffer, WALFrame


def make_frame(i: int, size: int = 32) -> WALFrame:
    return WALFrame(float(i), bytes([i % 256]) * size)


class TestRingBuffer:
    def test_push_pop_fifo(self):
        rb = RingBuffer(capacity_frames=10)
        for i in range(5):
            assert rb.push(make_frame(i))
        batch = rb.pop_batch(3)
        assert [f.timestamp_ms for f in batch] == [0.0, 1.0, 2.0]
        assert len(rb) == 2

    def test_overflow_evicts_oldest_and_counts(self):
        rb = RingBuffer(capacity_frames=3)
        for i in range(3):
            assert rb.push(make_frame(i))
        assert rb.push(make_frame(3)) is False
        assert rb.dropped_frames == 1
        assert [f.timestamp_ms for f in rb.pop_batch(10)] == [1.0, 2.0, 3.0]

    def test_watermark(self):
        rb = RingBuffer(capacity_frames=100, high_watermark=0.95)
        for i in range(94):
            rb.push(make_frame(i))
        assert not rb.above_watermark
        rb.push(make_frame(94))
        assert rb.above_watermark

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            RingBuffer(capacity_frames=0)
        with pytest.raises(ValueError):
            RingBuffer(capacity_frames=10, high_watermark=0.0)


class TestAudioWAL:
    def test_write_batch_and_recover(self, tmp_path):
        wal = AudioWAL(str(tmp_path))
        frames = [make_frame(i) for i in range(10)]
        wal.write_batch(frames)
        wal.close()
        recovered = AudioWAL(str(tmp_path)).recover_all()
        assert len(recovered) == 10
        assert recovered[0].payload == frames[0].payload
        assert recovered[-1].timestamp_ms == 9.0

    def test_emergency_flush_atomic_no_tmp_left(self, tmp_path):
        wal = AudioWAL(str(tmp_path))
        rb = RingBuffer(capacity_frames=100)
        for i in range(60):
            rb.push(make_frame(i))
        path = wal.emergency_flush(rb, max_frames=50)
        assert path is not None and os.path.exists(path)
        assert len(rb) == 10  # 50 drained
        assert not [n for n in os.listdir(tmp_path) if n.endswith(".tmp")]
        assert wal.emergency_flush_count == 1
        recovered = list(AudioWAL.iter_wal_file(path))
        assert len(recovered) == 50
        wal.close()

    def test_emergency_flush_empty_ring(self, tmp_path):
        wal = AudioWAL(str(tmp_path))
        assert wal.emergency_flush(RingBuffer(10)) is None
        wal.close()

    def test_recovery_tolerates_torn_tail(self, tmp_path):
        wal = AudioWAL(str(tmp_path))
        wal.write_batch([make_frame(i) for i in range(5)])
        wal.close()
        # Append a torn record: header promising 100 bytes, only 10 present
        with open(tmp_path / "active.wal", "ab") as fh:
            fh.write(struct.pack("<dI", 99.0, 100) + b"x" * 10)
        recovered = AudioWAL(str(tmp_path)).recover_all()
        assert len(recovered) == 5  # torn tail skipped, no crash

    def test_recovery_stops_on_corrupt_length(self, tmp_path):
        wal = AudioWAL(str(tmp_path))
        wal.write_batch([make_frame(1)])
        wal.close()
        with open(tmp_path / "active.wal", "ab") as fh:
            fh.write(struct.pack("<dI", 5.0, MAX_PAYLOAD_BYTES + 1))
            fh.write(b"y" * 64)
        recovered = AudioWAL(str(tmp_path)).recover_all()
        assert len(recovered) == 1

    def test_recovery_numeric_sort_of_emergency_files(self, tmp_path):
        """Lexical sort would order emergency_9 after emergency_10."""
        wal = AudioWAL(str(tmp_path))
        for ts, name in [(2.0, "emergency_9.wal"),
                         (1.0, "emergency_10.wal")]:
            with open(tmp_path / name, "wb") as fh:
                f = WALFrame(ts, b"z")
                fh.write(struct.pack("<dI", f.timestamp_ms, 1) + f.payload)
        recovered = wal.recover_all()
        # numeric order: 9 then 10 -> timestamps [2.0, 1.0]
        assert [f.timestamp_ms for f in recovered] == [2.0, 1.0]
        wal.close()

    def test_rotation(self, tmp_path):
        wal = AudioWAL(str(tmp_path), max_wal_bytes=1024)
        big = [WALFrame(float(i), b"a" * 200) for i in range(10)]
        wal.write_batch(big)
        wal.close()
        names = os.listdir(tmp_path)
        assert any(n.startswith("rotated_") for n in names)
        assert "active.wal" in names


class TestAdaptiveThresholdRegression:
    """Bug: dB values clamped into score scale -> constant 0.85."""

    def test_threshold_stays_in_score_scale_and_varies(self):
        from dsp_bridge import AdaptiveThresholdModule
        m = AdaptiveThresholdModule(baseline_window_size=200)
        # Quiet stable bay: ~45 dB noise floor
        for _ in range(150):
            m.feed_noise_sample(45.0 + 0.1)
        quiet_threshold = m.get_threshold()
        assert 0.60 <= quiet_threshold <= 0.85
        # Old bug: any realistic dB input pinned threshold at exactly 0.85
        assert quiet_threshold < 0.85
        # Noise spike well above baseline must raise the threshold
        for _ in range(5):
            m.feed_noise_sample(70.0)
        assert m.get_threshold() >= quiet_threshold
        assert m.get_threshold() <= 0.85

    def test_noise_floor_reported_in_db(self):
        from dsp_bridge import AdaptiveThresholdModule
        m = AdaptiveThresholdModule()
        assert m.get_noise_floor_db() is None
        for v in (40.0, 50.0):
            m.feed_noise_sample(v)
        assert m.get_noise_floor_db() == pytest.approx(45.0)

    def test_deque_bounded(self):
        from dsp_bridge import AdaptiveThresholdModule
        m = AdaptiveThresholdModule(baseline_window_size=50)
        for i in range(500):
            m.feed_noise_sample(float(i % 10))
        assert len(m.background_power_samples) == 50
