"""Gateway behavior under failure, with a fake client (no network, no cost)."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from core import llm

REQ = httpx.Request("POST", "https://fake/v1/chat/completions")


def api_error(cls, status, code=None):
    body = {"code": code, "message": code or cls.__name__}
    return cls(code or cls.__name__, response=httpx.Response(status, request=REQ), body=body)


def ok_response(text="好的", pt=10, ct=5):
    msg = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=ct))


class FakeClient:
    """behavior: {model: exception or response}; records the models it was asked for."""

    def __init__(self, behavior, calls):
        self.behavior, self.calls = behavior, calls
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, **kw):
        self.calls.append(model)
        out = self.behavior.get(model, ok_response(f"from {model}"))
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "primary")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "backup,gemini:gem")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.delenv("LLM_CHAOS_RATE", raising=False)
    llm._dead_until.clear()
    llm.METRICS.reset()
    calls, behavior = [], {}
    monkeypatch.setattr(llm, "get_client", lambda provider="default": FakeClient(behavior, calls))
    return SimpleNamespace(calls=calls, behavior=behavior)


def chat():
    return llm.chat([{"role": "user", "content": "hi"}], step="test")


def test_happy_path_uses_primary(fake):
    r = chat()
    assert r.model == "primary" and fake.calls == ["primary"]
    snap = llm.METRICS.snapshot()
    assert snap["models"]["primary"]["prompt_tokens"] == 10 and snap["fallbacks"] == 0


def test_retired_model_falls_back_and_is_skipped_next_time(fake):
    fake.behavior["primary"] = api_error(openai.NotFoundError, 404, "model_not_found")
    assert chat().model == "backup"
    assert chat().model == "backup"
    # second request must not pay for the dead model again (circuit breaker)
    assert fake.calls == ["primary", "backup", "backup"]


def test_account_in_arrears_skips_whole_provider(fake):
    arrears = api_error(openai.BadRequestError, 400, "Arrearage")
    fake.behavior["primary"] = arrears
    fake.behavior["backup"] = arrears  # would also fail; must not even be tried
    r = chat()
    assert r.model == "gemini:gem"
    assert fake.calls == ["primary", "gem"]


def test_plain_bad_request_is_not_retried(fake):
    fake.behavior["primary"] = api_error(openai.BadRequestError, 400, "InvalidParameter")
    with pytest.raises(RuntimeError):
        chat()
    assert fake.calls == ["primary"]


def test_all_down_then_single_probe(fake):
    for m in ("primary", "backup", "gem"):
        fake.behavior[m] = api_error(openai.NotFoundError, 404, "model_not_found")
    with pytest.raises(llm.LLMUnavailable):
        chat()
    fake.calls.clear()
    with pytest.raises(llm.LLMUnavailable):
        chat()
    assert fake.calls == ["primary"]  # half-open: one probe, not three failures


def test_transient_error_does_not_trip_breaker(fake):
    fake.behavior["primary"] = api_error(openai.InternalServerError, 503)
    assert chat().model == "backup"
    del fake.behavior["primary"]
    assert chat().model == "primary"  # recovered: 5xx is not sticky


def test_chaos_only_hits_first_attempt(fake, monkeypatch):
    monkeypatch.setenv("LLM_CHAOS_RATE", "1.0")
    assert chat().model == "backup"
    assert llm.METRICS.snapshot()["fallbacks"] == 1


def test_cost_estimate_from_configured_prices(fake, monkeypatch):
    monkeypatch.setenv("LLM_PRICES", '{"primary": [2.0, 8.0]}')
    chat()
    # 10 prompt * 2/1M + 5 completion * 8/1M
    assert llm.METRICS.snapshot()["total_cost_cny"] == pytest.approx(6e-5)


def test_chain_parsing(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "a")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "b, a ,gemini:g,unknown:z")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert llm.model_chain() == [("default", "a"), ("default", "b"), ("gemini", "g")]
