"""
LLM gateway — the single place every model call goes through.

Responsibilities (kept out of the pipeline / agent code on purpose):
- Model fallback chain: LLM_MODEL first, then LLM_FALLBACK_MODELS. A retired
  model (404), exhausted quota (403), rate limit, timeout or 5xx moves on to
  the next model instead of failing the user's request.
- Circuit breaker: a model that returned 404/403 is skipped for a cooldown
  period, so later requests don't pay its latency again.
- Usage accounting: tokens, latency, fallbacks and estimated cost per model.
- Structured JSON logs (one line per call) and the eval trace hook.
- Chaos mode (LLM_CHAOS_RATE) to inject failures and prove the fallbacks work.
"""

import contextvars
import datetime
import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import openai
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("ticketcoach.llm")

# ---------------------------------------------------------------------------
# Request context (set by the API middleware, read by logs)
# ---------------------------------------------------------------------------

request_id_var: contextvars.ContextVar = contextvars.ContextVar("request_id", default="-")

# ---------------------------------------------------------------------------
# Trace hook (for the eval harness; no-op in normal operation)
# ---------------------------------------------------------------------------
# When a collector (any object with .append) is installed via set_trace_collector,
# every LLM call records one event dict into it. When it is None (default),
# behavior is exactly as before.

_trace_collector: contextvars.ContextVar = contextvars.ContextVar(
    "ticketcoach_trace_collector", default=None
)


def set_trace_collector(collector) -> None:
    """Install (or remove, with None) a list-like trace collector."""
    _trace_collector.set(collector)


def record_trace(event: dict) -> None:
    collector = _trace_collector.get()
    if collector is not None:
        event.setdefault("ts", datetime.datetime.now().isoformat(timespec="seconds"))
        collector.append(event)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def primary_model() -> str:
    return os.environ.get("LLM_MODEL", "qwen-plus")


def _providers() -> dict:
    """OpenAI-compatible providers we can route to. The default one is LLM_*;
    Gemini joins the chain only when GEMINI_API_KEY is set."""
    providers = {
        "default": {
            "api_key": os.environ.get("LLM_API_KEY"),
            "base_url": os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        }
    }
    if os.environ.get("GEMINI_API_KEY"):
        providers["gemini"] = {
            "api_key": os.environ["GEMINI_API_KEY"],
            "base_url": os.environ.get("GEMINI_BASE_URL",
                                       "https://generativelanguage.googleapis.com/v1beta/openai/"),
        }
    return providers


def model_chain(primary: Optional[str] = None) -> list:
    """[(provider, model), ...]: primary first, then fallbacks, de-duplicated.

    LLM_FALLBACK_MODELS entries are "model" (default provider) or
    "provider:model", e.g. "qwen-plus,gemini:gemini-2.5-flash". A cross-provider
    entry is what saves you when a whole account is down (arrears, revoked key).
    """
    default_fb = "qwen-plus,qwen-flash"
    if os.environ.get("GEMINI_API_KEY"):
        default_fb += ",gemini:" + os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    entries = [primary or primary_model()] + os.environ.get("LLM_FALLBACK_MODELS", default_fb).split(",")
    providers = _providers()
    chain = []
    for e in (x.strip() for x in entries):
        if not e:
            continue
        prov, _, model = e.rpartition(":") if ":" in e else ("default", "", e)
        if prov in providers and (prov, model) not in chain:
            chain.append((prov, model))
    return chain


def _prices() -> dict:
    """{model: [input_per_1M, output_per_1M]} in CNY, from LLM_PRICES (JSON).

    Deliberately no built-in prices: they change often, so copy the current
    numbers from the Bailian console instead of trusting stale defaults.
    """
    try:
        return json.loads(os.environ.get("LLM_PRICES", "{}"))
    except json.JSONDecodeError:
        return {}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    price = _prices().get(model)
    if not price:
        return None
    return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000


_clients: dict = {}
_client_lock = threading.Lock()


def get_client(provider: str = "default") -> openai.OpenAI:
    """Shared client per provider (connection pooling across requests)."""
    cfg = _providers().get(provider) or {}
    if not cfg.get("api_key"):
        raise ValueError("LLM_API_KEY environment variable is not set. Please check your .env file.")
    with _client_lock:
        cached = _clients.get(provider)
        if cached is None or cached.api_key != cfg["api_key"]:
            cached = _clients[provider] = openai.OpenAI(
                api_key=cfg["api_key"],
                base_url=cfg["base_url"],
                # Stronger models need more headroom for long generations.
                timeout=float(os.environ.get("LLM_TIMEOUT", "180")),
                # SDK-level retries with backoff (429 / 5xx / connection errors)
                # on the SAME model, before we fall back to the next one.
                max_retries=int(os.environ.get("LLM_MAX_RETRIES", "1")),
            )
        return cached


# ---------------------------------------------------------------------------
# Failure classification + circuit breaker
# ---------------------------------------------------------------------------

# Errors where trying another model can help. A plain BadRequest is our own
# fault (bad payload) and is raised immediately — unless it is an account-level
# error (see is_account_error), where another PROVIDER can still help.
FALLBACK_ERRORS = (
    openai.NotFoundError,          # model retired / renamed
    openai.PermissionDeniedError,  # free quota exhausted, model not enabled
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

# 404/403 won't fix themselves in seconds — skip that model for a while.
_STICKY_ERRORS = (openai.NotFoundError, openai.PermissionDeniedError)
_dead_until: dict = {}  # key: "provider:model", or "provider:*" for a whole account
_dead_lock = threading.Lock()


def is_account_error(e: Exception) -> bool:
    """The whole provider account is unusable (overdue bill, bad/revoked key).
    DashScope reports arrears as a 400 with code "Arrearage", so a status code
    alone can't tell it apart from a malformed request."""
    if isinstance(e, openai.AuthenticationError):
        return True
    body = getattr(e, "body", None)
    code = body.get("code") if isinstance(body, dict) else None
    return code in {"Arrearage", "InvalidApiKey", "AccountSuspended"} or "Arrearage" in str(e)


def _is_dead(provider: str, model: str) -> bool:
    now = time.time()
    with _dead_lock:
        return (_dead_until.get(f"{provider}:{model}", 0) > now
                or _dead_until.get(f"{provider}:*", 0) > now)


def _mark_dead(provider: str, model: str = "*") -> None:
    cooldown = float(os.environ.get("LLM_MODEL_COOLDOWN_S", "600"))
    with _dead_lock:
        _dead_until[f"{provider}:{model}"] = time.time() + cooldown


class ChaosError(openai.APIConnectionError):
    """Injected failure (LLM_CHAOS_RATE) — behaves like a network error."""

    def __init__(self, model: str):
        super().__init__(message=f"chaos: injected failure for {model}", request=None)


def _maybe_chaos(model: str, depth: int) -> None:
    rate = float(os.environ.get("LLM_CHAOS_RATE", "0"))
    # Only the first model tried is sabotaged, so a healthy fallback can recover.
    if rate > 0 and depth == 0 and random.random() < rate:
        raise ChaosError(model)


# ---------------------------------------------------------------------------
# Usage metrics (in-memory; exposed at /api/metrics)
# ---------------------------------------------------------------------------

class Metrics:
    """Thread-safe counters per model + latency samples for p50/p95."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.started = time.time()
        self.models: dict = {}
        self.steps: dict = {}
        self.fallbacks = 0
        self.failures = 0

    def _bucket(self, table: dict, key: str) -> dict:
        return table.setdefault(key, {"calls": 0, "errors": 0, "prompt_tokens": 0,
                                      "completion_tokens": 0, "cost_cny": 0.0,
                                      "latencies_ms": []})

    def record(self, model: str, step: str, latency_ms: int, prompt_tokens: int = 0,
               completion_tokens: int = 0, error: bool = False):
        cost = estimate_cost(model, prompt_tokens, completion_tokens) or 0.0
        with self._lock:
            for b in (self._bucket(self.models, model), self._bucket(self.steps, step)):
                b["calls"] += 1
                b["errors"] += int(error)
                b["prompt_tokens"] += prompt_tokens
                b["completion_tokens"] += completion_tokens
                b["cost_cny"] += cost
                if not error:
                    b["latencies_ms"] = (b["latencies_ms"] + [latency_ms])[-500:]

    def record_fallback(self):
        with self._lock:
            self.fallbacks += 1

    def record_failure(self):
        with self._lock:
            self.failures += 1

    @staticmethod
    def _pct(samples: list, q: float) -> Optional[int]:
        if not samples:
            return None
        s = sorted(samples)
        return s[min(len(s) - 1, int(q * len(s)))]

    def snapshot(self) -> dict:
        def summarize(table):
            out = {}
            for k, b in table.items():
                out[k] = {kk: vv for kk, vv in b.items() if kk != "latencies_ms"}
                out[k]["cost_cny"] = round(b["cost_cny"], 6)
                out[k]["p50_ms"] = self._pct(b["latencies_ms"], 0.5)
                out[k]["p95_ms"] = self._pct(b["latencies_ms"], 0.95)
            return out

        with self._lock:
            models = summarize(self.models)
            return {
                "since": datetime.datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
                "models": models,
                "steps": summarize(self.steps),
                "total_cost_cny": round(sum(m["cost_cny"] for m in models.values()), 6),
                "total_tokens": sum(m["prompt_tokens"] + m["completion_tokens"] for m in models.values()),
                "fallbacks": self.fallbacks,
                "failed_requests": self.failures,
                "cooling_down": [k for k, t in list(_dead_until.items()) if t > time.time()],
                "prices_configured": sorted(_prices()),
            }


METRICS = Metrics()


def _log(level: int, msg: str, **fields):
    fields.setdefault("request_id", request_id_var.get())
    log.log(level, msg, extra={"fields": fields})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ChatResult:
    content: str
    model: str
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: list = field(default_factory=list)
    message: Optional[dict] = None  # assistant message, ready to append to history


class LLMUnavailable(RuntimeError):
    """Every model in the chain failed."""


def _label(provider: str, model: str) -> str:
    return model if provider == "default" else f"{provider}:{model}"


def _classify(e: Exception) -> Optional[str]:
    """'account' / 'model' / 'transient' -> fall back; None -> raise (our bug)."""
    if is_account_error(e):
        return "account"
    if isinstance(e, _STICKY_ERRORS):
        return "model"
    if isinstance(e, FALLBACK_ERRORS):
        return "transient"
    return None


def _on_failure(e: Exception, kind: str, provider: str, model: str, step: str,
                latency: int, has_next, **extra) -> str:
    """Shared bookkeeping for a failed attempt; returns a short error string."""
    label = _label(provider, model)
    METRICS.record(label, step, latency, error=True)
    if kind == "account":
        _mark_dead(provider)          # skip every model on this account
    elif kind == "model":
        _mark_dead(provider, model)
    has_next = has_next()
    if has_next:
        METRICS.record_fallback()
    _log(logging.WARNING, "llm_call_failed", step=step, model=label, latency_ms=latency,
         failure=kind, error_type=type(e).__name__, error=str(e)[:300],
         will_fallback=has_next, **extra)
    return f"{label}: {type(e).__name__}({kind})"


def _attempts(chain: list):
    """Yield (depth, provider, model, has_next) over entries not cooling down.

    Circuit breaker: if EVERY entry is cooling down, send a single probe to the
    first one (half-open) instead of burning a failed call per entry.
    """
    live = [(p, m) for p, m in chain if not _is_dead(p, m)]
    if not live:
        yield 0, chain[0][0], chain[0][1], lambda: False
        return
    for depth, (p, m) in enumerate(live):
        # An earlier attempt in this request may have just killed the account.
        if depth > 0 and _is_dead(p, m):
            continue
        rest = live[depth + 1:]
        # Callable, so it is evaluated AFTER this attempt's failure is recorded.
        yield depth, p, m, lambda rest=rest: any(not _is_dead(pp, mm) for pp, mm in rest)


def chat(messages: list, *, step: str = "unknown", json_mode: bool = False,
         temperature: float = 0.8, tools: Optional[list] = None,
         model: Optional[str] = None) -> ChatResult:
    """One chat completion with fallbacks. Raises LLMUnavailable if all fail."""
    errors = []
    for depth, provider, m, has_next in _attempts(model_chain(model)):
        label = _label(provider, m)
        kwargs = {"model": m, "messages": messages, "temperature": temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools

        event = {"event": "llm_call", "step": step, "model": label, "json_mode": json_mode,
                 "temperature": temperature, "messages": messages}
        start = time.time()
        try:
            _maybe_chaos(m, depth)
            resp = get_client(provider).chat.completions.create(**kwargs)
        except openai.OpenAIError as e:
            latency = int((time.time() - start) * 1000)
            kind = _classify(e)
            event.update(error=str(e), latency_ms=latency)
            record_trace(event)
            if kind is None:
                # Malformed request: another model won't help.
                METRICS.record(label, step, latency, error=True)
                METRICS.record_failure()
                _log(logging.ERROR, "llm_call_error", step=step, model=label, error=str(e)[:300])
                raise RuntimeError(f"LLM API error: {e}")
            errors.append(_on_failure(e, kind, provider, m, step, latency, has_next))
            continue

        latency = int((time.time() - start) * 1000)
        msg = resp.choices[0].message
        pt = getattr(resp.usage, "prompt_tokens", 0) or 0
        ct = getattr(resp.usage, "completion_tokens", 0) or 0
        METRICS.record(label, step, latency, pt, ct)
        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in (msg.tool_calls or [])
        ]
        assistant = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        event.update(raw_response=msg.content or "", latency_ms=latency,
                     prompt_tokens=pt, completion_tokens=ct, tool_calls=tool_calls)
        record_trace(event)
        _log(logging.INFO, "llm_call", step=step, model=label, latency_ms=latency,
             prompt_tokens=pt, completion_tokens=ct, cost_cny=estimate_cost(label, pt, ct),
             fallback_depth=depth, tool_calls=[t["name"] for t in tool_calls] or None)
        return ChatResult(content=msg.content or "", model=label, latency_ms=latency,
                          prompt_tokens=pt, completion_tokens=ct,
                          tool_calls=tool_calls, message=assistant)

    METRICS.record_failure()
    _log(logging.ERROR, "llm_unavailable", step=step, errors=errors)
    raise LLMUnavailable("All models failed: " + " | ".join(errors))


def stream(messages: list, *, step: str = "unknown", temperature: float = 0.8,
           model: Optional[str] = None) -> Iterator[str]:
    """Streaming chat with fallbacks.

    Falls back only while nothing has been sent yet: once the first token is
    out, switching models mid-sentence would produce a garbled reply.
    """
    errors = []
    for depth, provider, m, has_next in _attempts(model_chain(model)):
        label = _label(provider, m)
        start = time.time()
        first_token_ms = None
        pt = ct = 0
        try:
            _maybe_chaos(m, depth)
            resp = get_client(provider).chat.completions.create(
                model=m, messages=messages, temperature=temperature, stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in resp:
                if getattr(chunk, "usage", None):
                    pt = chunk.usage.prompt_tokens or 0
                    ct = chunk.usage.completion_tokens or 0
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and getattr(delta, "content", None):
                    if first_token_ms is None:
                        first_token_ms = int((time.time() - start) * 1000)
                    yield delta.content
        except openai.OpenAIError as e:
            latency = int((time.time() - start) * 1000)
            kind = _classify(e)
            if kind is None:
                METRICS.record(label, step, latency, error=True)
                METRICS.record_failure()
                raise RuntimeError(f"LLM API error: {e}")
            sent = first_token_ms is not None
            errors.append(_on_failure(e, kind, provider, m, step, latency,
                                      lambda: has_next() and not sent, tokens_sent=sent))
            if first_token_ms is not None:
                METRICS.record_failure()
                raise LLMUnavailable(f"stream broke mid-reply on {label}") from e
            continue

        latency = int((time.time() - start) * 1000)
        METRICS.record(label, step, latency, pt, ct)
        _log(logging.INFO, "llm_stream", step=step, model=label, latency_ms=latency,
             first_token_ms=first_token_ms, prompt_tokens=pt, completion_tokens=ct,
             cost_cny=estimate_cost(label, pt, ct), fallback_depth=depth)
        return

    METRICS.record_failure()
    raise LLMUnavailable("All models failed: " + " | ".join(errors))


def embed(texts: list, *, dimensions: Optional[int] = None) -> list:
    """Embed a batch of texts with EMBED_MODEL on the default provider.

    No cross-provider fallback on purpose: vectors from different embedding
    models live in different spaces, so a query embedded by model B can't be
    searched against an index built by model A. Callers degrade to keyword
    search instead (see core.kb).
    """
    client = get_client()
    model = os.environ.get("EMBED_MODEL", "text-embedding-v4")
    dims = dimensions or int(os.environ.get("EMBED_DIM", "512"))
    out = []
    for i in range(0, len(texts), 10):  # DashScope caps a request at 10 inputs
        start = time.time()
        resp = client.embeddings.create(model=model, input=texts[i:i + 10], dimensions=dims)
        pt = getattr(resp.usage, "prompt_tokens", 0) or 0
        METRICS.record(model, "embed", int((time.time() - start) * 1000), pt, 0)
        out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return out
