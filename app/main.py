"""
TicketCoach FastAPI Application
Serves the frontend and exposes the pipeline API endpoints.
"""

import contextvars
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Query, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core import auth, llm
from core.agent import ToolContext, coach_reply
from core.kb import get_kb
from core.memory import get_memory
from core.obs import setup_logging
from core.pipeline import (
    run_pipeline,
    chat_reply,
    chat_reply_stream,
    agent_reply_stream,
    evaluate_session,
    ticket_to_script,
    review_script,
    coerce_ticket,
    BUSINESS_TYPES,
    EMOTIONS,
    ISSUE_CATEGORIES,
    DIFFICULTIES,
)

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TicketCoach API",
    description="Customer service ticket → AI training script generator",
    version="1.0.0",
)

setup_logging()
log = logging.getLogger("ticketcoach.api")


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Tag every request with an id (echoed in X-Request-ID and in every log
    line it produces) and emit one access-log line with status + latency."""
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    token = llm.request_id_var.set(rid)
    start = time.time()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        if request.url.path.startswith("/api/"):
            log.info("http_request", extra={"fields": {
                "request_id": rid, "method": request.method, "path": request.url.path,
                "status": status, "latency_ms": int((time.time() - start) * 1000),
            }})
        llm.request_id_var.reset(token)


# Allow all origins for demo / development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # wildcard origin + credentials is rejected by browsers; we use neither
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Pre-generated script library (enterprise pattern: generate offline, serve
# instantly). Seeded from data/library.json; in-memory, append via POST.
# ---------------------------------------------------------------------------
LIBRARY_PATH = Path(__file__).parent.parent / "data" / "library.json"


def _load_library() -> list:
    try:
        with open(LIBRARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


LIBRARY = _load_library()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    business_type: Optional[str] = None
    emotion: Optional[str] = None
    issue_category: Optional[str] = None
    difficulty: Optional[str] = None


class ChatTurn(BaseModel):
    role: str  # "customer" or "agent"
    text: str


class ChatRequest(BaseModel):
    actor_prompt: str
    history: list[ChatTurn] = []


class EvaluateRequest(BaseModel):
    script: dict
    transcript: list[ChatTurn]
    business_type: Optional[str] = None  # stored in trainee memory when logged in


class AgentStreamRequest(BaseModel):
    script: dict
    history: list[ChatTurn] = []


class ScriptFromTicketRequest(BaseModel):
    ticket_text: Optional[str] = None   # raw pasted ticket text
    ticket: Optional[dict] = None       # or an already-structured ticket
    review: bool = True                 # also run quality check


class LoginRequest(BaseModel):
    name: str
    access_code: Optional[str] = None


class CoachTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class CoachRequest(BaseModel):
    message: str
    history: list[CoachTurn] = []


class BatchScriptsRequest(BaseModel):
    content: str                        # raw file content (frontend reads the file)
    format: str = "auto"                # json | jsonl | csv | auto
    review: bool = False                # batch defaults to skip review (faster/cheaper)


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------
# If ACCESS_PASSWORD is set (e.g. in production), the expensive LLM endpoints
# require a matching X-Access-Token header. If it's unset (e.g. local dev),
# access is open. This protects your API tokens from anyone who finds the URL.

def require_access(x_access_token: Optional[str] = Header(default=None)):
    expected = os.environ.get("ACCESS_PASSWORD")
    if expected and x_access_token != expected:
        raise HTTPException(status_code=401, detail="访问口令错误或缺失")


def require_trainee(x_trainee_token: Optional[str] = Header(default=None)) -> str:
    """Resolve the signed trainee token to a trainee id (401 if missing/forged)."""
    tid = auth.verify_token(x_trainee_token)
    if not tid:
        raise HTTPException(status_code=401, detail="请先登录（输入姓名和访问口令）")
    return tid


def optional_trainee(x_trainee_token: Optional[str] = Header(default=None)) -> Optional[str]:
    return auth.verify_token(x_trainee_token)


# ---------------------------------------------------------------------------
# Daily generation quota
# ---------------------------------------------------------------------------
# Caps the number of full pipeline runs per day (generate counts 1, batch
# counts n) to protect API spend on a public link. Set GEN_DAILY_LIMIT=0 (or
# leave unset) to disable. In-memory counter, resets at UTC date change; with
# Cloud Run --max-instances 1 it's a reliable per-day cap. The real money
# backstop is a spending limit on the LLM provider side.

_quota_lock = threading.Lock()
_quota_state = {"day": "", "count": 0}


def enforce_quota(n: int = 1):
    limit = int(os.environ.get("GEN_DAILY_LIMIT", "0"))
    if limit <= 0:
        return
    today = time.strftime("%Y-%m-%d", time.gmtime())
    with _quota_lock:
        if _quota_state["day"] != today:
            _quota_state["day"] = today
            _quota_state["count"] = 0
        if _quota_state["count"] + n > limit:
            raise HTTPException(
                status_code=429,
                detail=f"今日生成额度已用完（每日上限 {limit} 次），请明天再试",
            )
        _quota_state["count"] += n


# Per-trainee daily cap on coach turns (each turn = 1-7 LLM calls).
_coach_quota: dict = {}


def enforce_coach_quota(trainee_id: str):
    limit = int(os.environ.get("COACH_DAILY_LIMIT", "50"))
    if limit <= 0:
        return
    key = (trainee_id, time.strftime("%Y-%m-%d", time.gmtime()))
    with _quota_lock:
        if len(_coach_quota) > 10000:
            _coach_quota.clear()
        if _coach_quota.get(key, 0) >= limit:
            raise HTTPException(status_code=429, detail=f"今日教练提问次数已达上限（{limit} 次）")
        _coach_quota[key] = _coach_quota.get(key, 0) + 1


# ---------------------------------------------------------------------------
# Bounded parallelism for batch endpoints
# ---------------------------------------------------------------------------
# LLM calls are I/O-bound, so threads are enough. Results keep input order;
# each item's exception is returned (not raised) so one failure doesn't sink
# the batch. contextvars are copied so logs keep the request id.

def _parallel_map(fn, items: list) -> list:
    """[(result, None) | (None, exception)] in input order."""
    workers = max(1, int(os.environ.get("BATCH_CONCURRENCY", "3")))

    def safe(item):
        try:
            return fn(item), None
        except Exception as e:
            return None, e

    if workers == 1 or len(items) <= 1:
        return [safe(i) for i in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        futures = [pool.submit(contextvars.copy_context().run, safe, i) for i in items]
        return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_landing():
    """Serve the marketing/landing page."""
    landing = STATIC_DIR / "landing.html"
    if not landing.exists():
        raise HTTPException(status_code=404, detail="landing.html not found")
    return FileResponse(str(landing))


@app.get("/app")
async def serve_app():
    """Serve the functional app page."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_path))


@app.get("/api/options")
async def get_options():
    """Return available dropdown options for the frontend."""
    return {
        "business_types": BUSINESS_TYPES,
        "emotions": EMOTIONS,
        "issue_categories": ISSUE_CATEGORIES,
        "difficulties": DIFFICULTIES,
    }


@app.post("/api/generate", dependencies=[Depends(require_access)])
def generate(body: GenerateRequest):
    """
    Run the 3-step pipeline to generate ticket, script, and review.
    All body fields are optional; missing fields will be randomized.
    Returns: {"ticket": ..., "script": ..., "review": ..., "params": ...}
    """
    enforce_quota(1)
    try:
        params = {
            "business_type": body.business_type or None,
            "emotion": body.emotion or None,
            "issue_category": body.issue_category or None,
            "difficulty": body.difficulty or None,
        }
        result = run_pipeline(params)
        return JSONResponse(content=result)
    except ValueError as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"配置错误: {str(e)}"},
        )
    except RuntimeError as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"LLM 调用失败: {str(e)}"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"生成失败: {str(e)}"},
        )


@app.post("/api/batch", dependencies=[Depends(require_access)])
def batch_generate(n: int = Query(default=3, ge=1, le=5)):
    """
    Run n pipelines (BATCH_CONCURRENCY at a time) and return results + basic stats.
    Stats include: avg_score, business_type distribution, difficulty distribution.
    """
    enforce_quota(n)
    results = []
    errors = []

    started = time.time()
    for i, (result, err) in enumerate(_parallel_map(lambda _: run_pipeline(), list(range(n)))):
        if err is None:
            results.append(result)
        else:
            print(f"[batch] Pipeline {i + 1} failed: {err}")
            errors.append({"index": i + 1, "error": str(err)})

    # Compute stats
    scores = [
        r["review"].get("overall_score", 0)
        for r in results
        if isinstance(r.get("review", {}).get("overall_score"), (int, float))
    ]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0

    business_dist: dict[str, int] = {}
    difficulty_dist: dict[str, int] = {}

    for r in results:
        bt = r.get("params", {}).get("business_type", "未知")
        diff = r.get("params", {}).get("difficulty", "未知")
        business_dist[bt] = business_dist.get(bt, 0) + 1
        difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1

    stats = {
        "total_requested": n,
        "total_success": len(results),
        "total_failed": len(errors),
        "avg_score": avg_score,
        "business_type_distribution": business_dist,
        "difficulty_distribution": difficulty_dist,
        "wall_time_s": round(time.time() - started, 1),
    }

    return JSONResponse(content={
        "results": results,
        "stats": stats,
        "errors": errors,
    })


# ---------------------------------------------------------------------------
# Roleplay chat engine + session evaluation
# ---------------------------------------------------------------------------

@app.post("/api/chat", dependencies=[Depends(require_access)])
def chat(body: ChatRequest):
    """
    One turn of live roleplay. The LLM replies as the customer, driven by the
    script's actor_prompt. Stateless: the frontend sends the full history each
    time. Returns: {"reply": "<customer's next message>"}.
    """
    try:
        history = [t.model_dump() for t in body.history]
        reply = chat_reply(body.actor_prompt, history)
        return {"reply": reply}
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": f"LLM 调用失败: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"对话失败: {str(e)}"})


@app.post("/api/chat-stream", dependencies=[Depends(require_access)])
def chat_stream(body: ChatRequest):
    """
    Streaming roleplay turn: the customer's reply is streamed token-by-token
    (text/plain chunks) for a real-time typing effect. Same inputs as /api/chat.
    """
    history = [t.model_dump() for t in body.history]
    rid = llm.request_id_var.get()  # the body streams after the middleware returns

    def generate_stream():
        llm.request_id_var.set(rid)
        try:
            for piece in chat_reply_stream(body.actor_prompt, history):
                yield piece
        except Exception as e:
            yield f"\n[对话出错：{str(e)}]"

    return StreamingResponse(generate_stream(), media_type="text/plain; charset=utf-8")


@app.post("/api/agent-stream", dependencies=[Depends(require_access)])
def agent_stream(body: AgentStreamRequest):
    """
    Streaming MODEL-AGENT turn for AI-vs-AI demo roleplay: an exemplary support
    agent responds (guided by the script), streamed token-by-token.
    """
    history = [t.model_dump() for t in body.history]
    rid = llm.request_id_var.get()  # the body streams after the middleware returns

    def generate_stream():
        llm.request_id_var.set(rid)
        try:
            for piece in agent_reply_stream(body.script, history):
                yield piece
        except Exception as e:
            yield f"\n[客服出错：{str(e)}]"

    return StreamingResponse(generate_stream(), media_type="text/plain; charset=utf-8")


@app.post("/api/evaluate", dependencies=[Depends(require_access)])
def evaluate(body: EvaluateRequest, trainee_id: Optional[str] = Depends(optional_trainee)):
    """
    Score the trainee's (agent's) performance after a roleplay session, using
    the script's scoring_criteria. Returns a review-style dict.
    """
    try:
        transcript = [t.model_dump() for t in body.transcript]
        if not any(t["role"] == "agent" for t in transcript):
            return JSONResponse(status_code=400, content={"error": "对练记录中没有客服发言，无法评分"})
        result = evaluate_session(body.script, transcript)
        if trainee_id:
            # Long-term memory: the coach agent reads this back later.
            try:
                get_memory().record_session(trainee_id, body.script, result,
                                            business_type=body.business_type or "")
                result["saved_to_profile"] = True
            except Exception as e:
                log.warning("memory_write_failed", extra={"fields": {"error": str(e)[:200]}})
        return JSONResponse(content=result)
    except ValueError as e:
        return JSONResponse(status_code=500, content={"error": f"评分解析失败: {str(e)}"})
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": f"LLM 调用失败: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"评分失败: {str(e)}"})


# ---------------------------------------------------------------------------
# Real-ticket ingestion → script (+ export-ready output)
# ---------------------------------------------------------------------------

@app.post("/api/script-from-ticket", dependencies=[Depends(require_access)])
def script_from_ticket(body: ScriptFromTicketRequest):
    """
    Generate a training script from a REAL ticket (pasted text or structured).
    Returns {"ticket": <normalized>, "script": ..., "review": <optional>}.
    For teams that already have real tickets (skips the AI ticket-generation step).
    """
    enforce_quota(1)
    try:
        if body.ticket:
            ticket = coerce_ticket(body.ticket)
        elif body.ticket_text and body.ticket_text.strip():
            ticket = coerce_ticket(body.ticket_text.strip())
        else:
            return JSONResponse(status_code=400, content={"error": "请提供工单文本或工单数据"})

        script = ticket_to_script(ticket)
        result = {"ticket": ticket, "script": script}
        if body.review:
            result["review"] = review_script(ticket, script)
        return JSONResponse(content=result)
    except ValueError as e:
        return JSONResponse(status_code=500, content={"error": f"解析失败: {str(e)}"})
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": f"LLM 调用失败: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"生成失败: {str(e)}"})


def _parse_tickets_payload(content: str, fmt: str) -> list:
    """Parse uploaded file content into a list of ticket items (dict or str)."""
    import csv
    import io

    content = content.strip()
    if not content:
        return []

    fmt = (fmt or "auto").lower()
    if fmt == "auto":
        if content[:1] in "[{":
            fmt = "jsonl" if ("\n" in content and content[:1] == "{") else "json"
        else:
            fmt = "csv" if ("," in content.splitlines()[0]) else "text"

    if fmt == "json":
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    if fmt == "jsonl":
        return [json.loads(ln) for ln in content.splitlines() if ln.strip()]
    if fmt == "csv":
        return list(csv.DictReader(io.StringIO(content)))
    # plain text: split on blank lines, each block is one ticket
    return [b.strip() for b in content.split("\n\n") if b.strip()]


@app.post("/api/batch-scripts", dependencies=[Depends(require_access)])
def batch_scripts(body: BatchScriptsRequest):
    """
    Batch-generate scripts from an uploaded file of real tickets.
    Accepts JSON array / JSONL / CSV / plain text. Capped to protect API spend.
    Returns {"results": [{ticket, script, review?}], "count": n, "errors": [...]}.
    """
    MAX_ITEMS = 10
    try:
        items = _parse_tickets_payload(body.content, body.format)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"文件解析失败: {str(e)}"})

    if not items:
        return JSONResponse(status_code=400, content={"error": "文件为空或无法识别工单"})
    if len(items) > MAX_ITEMS:
        items = items[:MAX_ITEMS]

    enforce_quota(len(items))

    def one(item):
        ticket = coerce_ticket(item)
        script = ticket_to_script(ticket)
        entry = {"ticket": ticket, "script": script}
        if body.review:
            entry["review"] = review_script(ticket, script)
        return entry

    results, errors = [], []
    for i, (entry, err) in enumerate(_parallel_map(one, items)):
        if err is None:
            results.append(entry)
        else:
            errors.append({"index": i + 1, "error": str(err)})

    return JSONResponse(content={
        "count": len(results),
        "total_input": len(items),
        "capped_at": MAX_ITEMS,
        "results": results,
        "errors": errors,
    })


# ---------------------------------------------------------------------------
# Script library (instant retrieval — no LLM call)
# ---------------------------------------------------------------------------

@app.get("/api/library")
def library_list():
    """List library entries as lightweight summaries (instant, no LLM)."""
    items = []
    for i, entry in enumerate(LIBRARY):
        script = entry.get("script", {})
        ticket = entry.get("ticket", {})
        review = entry.get("review", {})
        params = entry.get("params", {})
        items.append({
            "id": i,
            "title": script.get("title", "未命名剧本"),
            "business_type": ticket.get("business_type") or params.get("business_type", ""),
            "difficulty": params.get("difficulty", ""),
            "score": review.get("overall_score", 0),
        })
    return {"items": items, "count": len(items)}


@app.get("/api/library/{idx}")
def library_get(idx: int):
    """Fetch one full library entry instantly (no LLM call)."""
    if idx < 0 or idx >= len(LIBRARY):
        return JSONResponse(status_code=404, content={"error": "剧本不存在"})
    return JSONResponse(content=LIBRARY[idx])


@app.post("/api/library", dependencies=[Depends(require_access)])
def library_add(entry: dict):
    """Append a generated {ticket, script, review} to the in-memory library."""
    if not entry.get("script"):
        return JSONResponse(status_code=400, content={"error": "缺少 script 字段"})
    LIBRARY.append(entry)
    return {"id": len(LIBRARY) - 1, "count": len(LIBRARY)}


# ---------------------------------------------------------------------------
# Trainee login + coach agent + memory
# ---------------------------------------------------------------------------

@app.post("/api/login")
def login(body: LoginRequest):
    """Exchange (name, access code) for a signed trainee token."""
    expected = os.environ.get("ACCESS_PASSWORD")
    if expected and body.access_code != expected:
        raise HTTPException(status_code=401, detail="访问口令错误")
    name = body.name.strip()[:40]
    try:
        tid = auth.trainee_id_for(name)
    except ValueError:
        raise HTTPException(status_code=400, detail="请输入姓名")
    return {"token": auth.issue_token(tid), "trainee_id": tid, "name": name}


@app.post("/api/coach")
def coach(body: CoachRequest, trainee_id: str = Depends(require_trainee)):
    """One turn with the coach agent (tool calling + RAG + memory)."""
    msg = body.message.strip()
    if not msg:
        return JSONResponse(status_code=400, content={"error": "请输入问题"})
    enforce_coach_quota(trainee_id)
    ctx = ToolContext(trainee_id=trainee_id, kb=get_kb(), memory=get_memory(),
                      library=LIBRARY, business_types=BUSINESS_TYPES)
    try:
        return coach_reply(ctx, msg[:1000], [t.model_dump() for t in body.history])
    except (RuntimeError, llm.LLMUnavailable) as e:
        return JSONResponse(status_code=503, content={"error": f"模型暂时不可用，请稍后再试（{str(e)[:200]}）"})


@app.get("/api/me/profile")
def my_profile(trainee_id: str = Depends(require_trainee)):
    """What the coach remembers about me."""
    return get_memory().profile(trainee_id, BUSINESS_TYPES)


@app.delete("/api/me")
def forget_me(trainee_id: str = Depends(require_trainee)):
    """Delete all stored sessions and notes for the current trainee."""
    get_memory().forget(trainee_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Metrics (tokens / cost / latency / fallbacks since process start)
# ---------------------------------------------------------------------------

@app.get("/api/metrics", dependencies=[Depends(require_access)])
def metrics():
    """Per-model and per-step usage: calls, errors, tokens, est. cost, p50/p95."""
    return llm.METRICS.snapshot()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "TicketCoach"}


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
