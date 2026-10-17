"""Batch endpoints run items concurrently, keep order, isolate failures."""

import threading
import time

from fastapi.testclient import TestClient


def test_batch_is_parallel_and_isolates_failures(monkeypatch):
    monkeypatch.delenv("ACCESS_PASSWORD", raising=False)
    monkeypatch.setenv("GEN_DAILY_LIMIT", "0")
    monkeypatch.setenv("BATCH_CONCURRENCY", "3")
    from app import main
    calls = {"n": 0}
    lock = threading.Lock()

    def slow_pipeline(params=None):
        with lock:
            calls["n"] += 1
            k = calls["n"]
        time.sleep(0.4)
        if k == 2:
            raise RuntimeError("boom")
        return {"ticket": {}, "script": {}, "review": {"overall_score": 80}, "params": {}}

    monkeypatch.setattr(main, "run_pipeline", slow_pipeline)
    t = time.time()
    r = TestClient(main.app).post("/api/batch?n=3").json()
    wall = time.time() - t
    assert wall < 0.9, wall  # sequential would be >= 1.2s
    assert r["stats"]["total_success"] == 2 and r["stats"]["total_failed"] == 1
    assert r["errors"][0]["error"] == "boom"
