"""
TicketCoach FastAPI Application
Serves the frontend and exposes the pipeline API endpoints.
"""

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.pipeline import (
    run_pipeline,
    chat_reply,
    evaluate_session,
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


@app.post("/api/generate")
def generate(body: GenerateRequest):
    """
    Run the 3-step pipeline to generate ticket, script, and review.
    All body fields are optional; missing fields will be randomized.
    Returns: {"ticket": ..., "script": ..., "review": ..., "params": ...}
    """
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


@app.post("/api/batch")
def batch_generate(n: int = Query(default=3, ge=1, le=20)):
    """
    Run n pipelines sequentially and return results + basic stats.
    Stats include: avg_score, business_type distribution, difficulty distribution.
    """
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

@app.post("/api/chat")
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


@app.post("/api/evaluate")
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
