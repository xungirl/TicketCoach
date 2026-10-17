"""HTTP-level tests: auth gates, coach endpoint, memory write on evaluate."""

import json

import pytest
from fastapi.testclient import TestClient

from core import llm, memory
from core.llm import ChatResult


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ACCESS_PASSWORD", "1024")
    monkeypatch.setenv("AUTH_SECRET", "test")
    monkeypatch.setattr(memory, "_MEMORY", memory.Memory(":memory:"))
    from app.main import app
    return TestClient(app)


def login(client, name="小王"):
    r = client.post("/api/login", json={"name": name, "access_code": "1024"})
    assert r.status_code == 200
    return {"X-Trainee-Token": r.json()["token"], "X-Access-Token": "1024"}


def test_login_rejects_wrong_code(client):
    assert client.post("/api/login", json={"name": "a", "access_code": "x"}).status_code == 401


def test_coach_requires_trainee_token(client):
    r = client.post("/api/coach", json={"message": "hi"}, headers={"X-Access-Token": "1024"})
    assert r.status_code == 401


def test_coach_end_to_end_with_fake_llm(client, monkeypatch):
    calls = []

    def fake_chat(messages, **kw):
        calls.append(kw.get("tools") is not None)
        if len(calls) == 1:
            tc = {"id": "c1", "name": "get_trainee_profile", "arguments": "{}"}
            return ChatResult("", "fake", 1, 5, 5, [tc],
                              {"role": "assistant", "content": "",
                               "tool_calls": [{"id": "c1", "type": "function",
                                               "function": {"name": "get_trainee_profile", "arguments": "{}"}}]})
        return ChatResult("你还没有对练记录，先练一个吧。", "fake", 1, 5, 5, [],
                          {"role": "assistant", "content": "x"})

    monkeypatch.setattr(llm, "chat", fake_chat)
    r = client.post("/api/coach", json={"message": "我哪里弱？"}, headers=login(client))
    body = r.json()
    assert r.status_code == 200, body
    assert body["tool_trace"][0]["tool"] == "get_trainee_profile"
    assert "对练记录" in body["answer"]


def test_evaluate_writes_trainee_memory(client, monkeypatch):
    from app import main
    monkeypatch.setattr(main, "evaluate_session", lambda script, t: {
        "overall_score": 55, "dimensions": [{"name": "共情", "score": 4, "comment": ""}],
        "missed_challenge_points": ["没有道歉"]})
    h = login(client)
    r = client.post("/api/evaluate", headers=h, json={
        "script": {"title": "ECS 愤怒客户"}, "business_type": "云服务器ECS/实例连接",
        "transcript": [{"role": "customer", "text": "连不上!"}, {"role": "agent", "text": "您好"}]})
    assert r.json()["saved_to_profile"] is True
    p = client.get("/api/me/profile", headers=h).json()
    assert p["sessions_count"] == 1 and p["recurring_missed_points"] == ["没有道歉"]
    # another trainee sees nothing
    assert client.get("/api/me/profile", headers=login(client, "小李")).json()["sessions_count"] == 0
    assert client.delete("/api/me", headers=h).json()["ok"]
    assert client.get("/api/me/profile", headers=h).json()["sessions_count"] == 0


def test_metrics_requires_access(client):
    assert client.get("/api/metrics").status_code == 401
    assert "models" in client.get("/api/metrics", headers={"X-Access-Token": "1024"}).json()


def test_request_id_header(client):
    r = client.get("/api/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["x-request-id"] == "abc123"
