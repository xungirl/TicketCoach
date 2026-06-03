"""
TicketCoach FastAPI Application
Serves the frontend and exposes the pipeline API endpoints.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.pipeline import (
    run_pipeline,
    chat_reply,
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


class ScriptFromTicketRequest(BaseModel):
    ticket_text: Optional[str] = None   # raw pasted ticket text
    ticket: Optional[dict] = None       # or an already-structured ticket
    review: bool = True                 # also run quality check


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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index():
    """Serve the main frontend page."""
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
    Run n pipelines sequentially and return results + basic stats.
    Stats include: avg_score, business_type distribution, difficulty distribution.
    """
    enforce_quota(n)
    results = []
    errors = []

    for i in range(n):
        try:
            print(f"[batch] Running pipeline {i + 1}/{n}...")
            result = run_pipeline()
            results.append(result)
        except Exception as e:
            error_msg = str(e)
            print(f"[batch] Pipeline {i + 1} failed: {error_msg}")
            errors.append({"index": i + 1, "error": error_msg})

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


@app.post("/api/evaluate", dependencies=[Depends(require_access)])
def evaluate(body: EvaluateRequest):
    """
    Score the trainee's (agent's) performance after a roleplay session, using
    the script's scoring_criteria. Returns a review-style dict.
    """
    try:
        transcript = [t.model_dump() for t in body.transcript]
        if not any(t["role"] == "agent" for t in transcript):
            return JSONResponse(status_code=400, content={"error": "对练记录中没有客服发言，无法评分"})
        result = evaluate_session(body.script, transcript)
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

    results, errors = [], []
    for i, item in enumerate(items):
        try:
            ticket = coerce_ticket(item)
            script = ticket_to_script(ticket)
            entry = {"ticket": ticket, "script": script}
            if body.review:
                entry["review"] = review_script(ticket, script)
            results.append(entry)
        except Exception as e:
            errors.append({"index": i + 1, "error": str(e)})

    return JSONResponse(content={
        "count": len(results),
        "total_input": len(items),
        "capped_at": MAX_ITEMS,
        "results": results,
        "errors": errors,
    })


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
