"""
Coach-agent eval: tool selection, grounding, memory, safety — hard checks only.

    venv/bin/python -m evals.coach_eval --run-id coach-baseline
    venv/bin/python -m evals.coach_eval --chaos 0.5 --run-id coach-chaos   # failure drill
    venv/bin/python -m evals.coach_eval --model qwen-flash --run-id coach-flash

Each case runs against a fresh in-memory trainee (seeded with sessions/notes
when the case says so), using the real KB, real tools and the configured LLM
chain. Checks are deterministic, so a score change means behavior changed,
not that a judge model had a mood.

Output: evals/runs/<run_id>/coach_results.jsonl + a summary table.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import llm  # noqa: E402
from core.agent import ToolContext, coach_reply  # noqa: E402
from core.kb import KnowledgeBase  # noqa: E402
from core.memory import Memory  # noqa: E402
from core.pipeline import BUSINESS_TYPES  # noqa: E402

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset" / "coach.jsonl"


def seed(memory: Memory, tid: str, case: dict) -> None:
    for s in case.get("seed_sessions", []):
        memory.record_session(tid, {"title": s["title"]}, {
            "overall_score": s["score"],
            "dimensions": [{"name": k, "score": v} for k, v in s["dims"].items()],
            "missed_challenge_points": s["missed"]}, business_type=s["business_type"])
    for n in case.get("seed_notes", []):
        memory.add_note(tid, n)


def check(case: dict, out: dict) -> list:
    """List of (check_name, passed)."""
    e, results = case["expect"], []
    tools = [t["tool"] for t in out["tool_trace"]]
    answer = out["answer"]
    for t in e.get("tools_include", []):
        results.append((f"calls {t}", t in tools))
    for t in e.get("tools_exclude", []):
        results.append((f"avoids {t}", t not in tools))
    if "min_citations" in e:
        results.append(("cites docs", len(out["sources"]) >= e["min_citations"]))
    if "source_titles_any" in e:
        results.append(("right doc cited", any(s["title"] in e["source_titles_any"] for s in out["sources"])))
    if "answer_contains_any" in e:
        results.append(("answer content", any(k in answer for k in e["answer_contains_any"])))
    for k in e.get("answer_not_contains", []):
        results.append((f"no leak '{k[:8]}'", k not in answer))
    if "min_scripts" in e:
        results.append(("offers scripts", len(out["scripts"]) >= e["min_scripts"]))
    if "notes_written_min" in e:
        results.append(("writes memory", len(out["remembered"]) >= e["notes_written_min"]))
    if "max_llm_calls" in e:
        results.append(("no wasted calls", out["usage"]["llm_calls"] <= e["max_llm_calls"]))
    # Always-on guards
    results.append(("no invalid citations", not out["invalid_citations"]))
    results.append(("answered", bool(answer.strip())))
    return results


def run_case(case: dict, kb: KnowledgeBase, library: list) -> dict:
    memory = Memory(":memory:")
    tid = f"t_eval_{case['id']}"
    seed(memory, tid, case)
    ctx = ToolContext(trainee_id=tid, kb=kb, memory=memory, library=library,
                      business_types=BUSINESS_TYPES)
    t0 = time.time()
    try:
        out = coach_reply(ctx, case["message"])
    except Exception as e:
        return {"id": case["id"], "type": case["type"], "error": f"{type(e).__name__}: {e}",
                "checks": [("ran", False)], "latency_ms": int((time.time() - t0) * 1000)}
    return {"id": case["id"], "type": case["type"], "answer": out["answer"],
            "tools": [t["tool"] for t in out["tool_trace"]], "sources": [s["title"] for s in out["sources"]],
            "usage": out["usage"], "checks": check(case, out), "latency_ms": int((time.time() - t0) * 1000)}


def summarize(results: list) -> dict:
    checks = [ok for r in results for _, ok in r["checks"]]
    cases_passed = sum(all(ok for _, ok in r["checks"]) for r in results)
    lat = sorted(r["latency_ms"] for r in results)
    tokens = sum(r.get("usage", {}).get("prompt_tokens", 0) + r.get("usage", {}).get("completion_tokens", 0)
                 for r in results)
    return {"cases": len(results), "cases_passed": cases_passed,
            "check_pass_rate": round(sum(checks) / max(1, len(checks)), 3),
            "errors": sum("error" in r for r in results),
            "p50_ms": lat[len(lat) // 2] if lat else None, "p95_ms": lat[int(0.95 * (len(lat) - 1))] if lat else None,
            "total_tokens": tokens,
            "metrics": llm.METRICS.snapshot()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=time.strftime("coach-%Y%m%d-%H%M%S"))
    ap.add_argument("--only", help="comma-separated case ids")
    ap.add_argument("--chaos", type=float, default=0.0, help="LLM_CHAOS_RATE for a failure drill")
    ap.add_argument("--model", help="override LLM_MODEL")
    args = ap.parse_args()
    if args.chaos:
        os.environ["LLM_CHAOS_RATE"] = str(args.chaos)
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    cases = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    kb = KnowledgeBase.load(embed_fn=llm.embed)
    library = json.loads((ROOT.parent / "data" / "library.json").read_text(encoding="utf-8"))

    out_dir = ROOT / "runs" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[coach-eval] {args.run_id}: {len(cases)} cases, chain={llm.model_chain()}, chaos={args.chaos}, "
          f"kb={'hybrid' if kb.vectors is not None else 'bm25-only'}\n")
    results = []
    with open(out_dir / "coach_results.jsonl", "w", encoding="utf-8") as f:
        for case in cases:
            r = run_case(case, kb, library)
            results.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            failed = [name for name, ok in r["checks"] if not ok]
            mark = "✓" if not failed else "✗"
            print(f"  {mark} {r['id']} {r['type']:12s} {r['latency_ms']:6d}ms  tools={r.get('tools')}"
                  + (f"  FAILED: {failed}" if failed else "") + (f"  ERROR: {r['error'][:120]}" if "error" in r else ""))

    s = summarize(results)
    (out_dir / "coach_summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\ncases {s['cases_passed']}/{s['cases']}  checks {s['check_pass_rate']:.1%}  errors {s['errors']}  "
          f"p50 {s['p50_ms']}ms  p95 {s['p95_ms']}ms  tokens {s['total_tokens']}  "
          f"fallbacks {s['metrics']['fallbacks']}  cost ¥{s['metrics']['total_cost_cny']}")


if __name__ == "__main__":
    main()
