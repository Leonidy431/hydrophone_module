"""Tests for the BlueOS extension HTTP surface (no hardware, no network:
MavlinkNotifier is monkeypatched; acquisition loop is not started —
endpoints are exercised against the service object directly)."""

import os

os.environ.setdefault("DIVEGUARD_DEV", "1")
os.environ.setdefault("DIVEGUARD_WAL_DIR", "/tmp/diveguard_test_wal")

import pytest
from fastapi.testclient import TestClient

import blueos_extension as ext


@pytest.fixture()
def client(monkeypatch, tmp_path):
    # No real MAVLink2REST in tests
    async def fake_notify(self, text):
        self.sent_count += 1
        return True
    monkeypatch.setattr(ext.MavlinkNotifier, "notify", fake_notify)

    # Don't start hardware loops: stub service.start
    async def fake_start(self):
        return None
    monkeypatch.setattr(ext.DiveGuardService, "start", fake_start)

    with TestClient(ext.app) as c:
        yield c


def test_register_service(client):
    r = client.get("/register_service")
    assert r.status_code == 200
    body = r.json()
    assert body["name"].startswith("DiveGuard")
    assert body["api"] == "/docs"


def test_health(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "degraded")


def test_status_shape(client):
    r = client.get("/v1/status")
    assert r.status_code == 200
    body = r.json()
    for key in ("uptime_s", "sensor_ok", "ring", "wal", "mavlink"):
        assert key in body
    assert body["ring"]["capacity"] == 4096


def test_simulate_and_detections(client):
    r = client.post("/v1/simulate", json={"propeller_score": 0.92,
                                          "threat_level": "HIGH",
                                          "max_bpf_power_db": 75.0})
    assert r.status_code == 200
    assert r.json()["accepted"] is True

    r = client.get("/v1/detections")
    assert r.status_code == 200
    dets = r.json()
    assert len(dets) >= 1
    assert dets[-1]["propeller_score"] == pytest.approx(0.92)


def test_simulate_validation(client):
    r = client.post("/v1/simulate", json={"propeller_score": 1.5})
    assert r.status_code == 422  # pydantic bound


def test_detections_limit_clamped(client):
    r = client.get("/v1/detections?limit=100000")
    assert r.status_code == 200
    assert len(r.json()) <= 200


def test_statustext_payload_format():
    """STATUSTEXT text must be exactly 50 chars, NUL-padded."""
    captured = {}

    def fake_urlopen(req, timeout):
        import json as _json

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        captured["body"] = _json.loads(req.data.decode())
        captured["timeout"] = timeout
        return _Resp()

    notifier = ext.MavlinkNotifier("http://example.invalid:6040")
    import unittest.mock as mock
    with mock.patch.object(ext.urllib.request, "urlopen", fake_urlopen):
        notifier._post_sync("DIVEGUARD HIGH score=0.92")

    text = captured["body"]["message"]["text"]
    assert len(text) == 50
    assert text[-1] == 0
    assert captured["body"]["message"]["type"] == "STATUSTEXT"
    assert captured["timeout"] == pytest.approx(2.0)
