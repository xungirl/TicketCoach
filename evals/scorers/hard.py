"""
Deterministic (hard-metric) scorer: schema/field/content checks that need no
LLM. Reads runs/<run_id>/results.jsonl, writes runs/<run_id>/scores_hard.jsonl.

Usage: python -m evals.scorers.hard <run_id>
"""

import json
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = EVALS_DIR / "runs"
DATASET_DIR = EVALS_DIR / "dataset"

TICKET_KEYS = ["ticket_id", "business_type", "customer_profile", "channel",
               "dialogue", "resolution", "tags"]
SCRIPT_KEYS = ["title", "customer_persona", "scenario", "emotion_arc",
               "challenge_points", "standard_response", "scoring_criteria", "actor_prompt"]


def _checks_for_ticket(ticket, generated: bool):
    """Shared structural checks for a ticket dict (generated or normalized)."""
    yield "ticket_schema", all(k in ticket for k in TICKET_KEYS), \
        f"missing: {[k for k in TICKET_KEYS if k not in ticket]}"
    dialogue = ticket.get("dialogue") or []
    roles = {t.get("role") for t in dialogue if isinstance(t, dict)}
    yield "dialogue_roles_valid", bool(dialogue) and roles <= {"customer", "agent"}, \
        f"roles={sorted(roles)}, turns={len(dialogue)}"
    if generated:
        # The generation prompt demands a real back-and-forth (8-12 turns).
        yield "dialogue_length", len(dialogue) >= 6, f"turns={len(dialogue)}"
        yield "dialogue_both_roles", roles == {"customer", "agent"}, f"roles={sorted(roles)}"


def _check_review_like(obj, n_dims=None):
    """Structural checks shared by review_script and evaluate_session outputs."""
    score = obj.get("overall_score")
    yield "overall_score_range", isinstance(score, (int, float)) and 0 <= score <= 100, f"score={score}"
    dims = obj.get("dimensions") or []
    ok = bool(dims) and all(
        isinstance(d, dict) and isinstance(d.get("score"), (int, float)) and 0 <= d["score"] <= 10
        for d in dims
    )
    yield "dimension_scores_range", ok, f"n_dims={len(dims)}"
    if n_dims is not None:
        yield "dimension_count", len(dims) == n_dims, f"expected {n_dims}, got {len(dims)}"


def check_case(case, output, scripts):
    """Yield (check_name, passed, detail) tuples for one case."""
    ctype = case["type"]
    hc = case.get("hard_checks", {})

    if ctype == "generate":
        ticket, script, review = output["ticket"], output["script"], output["review"]
        yield from _checks_for_ticket(ticket, generated=case.get("corrupt") is None)
        kw = hc.get("ticket_business_keyword")
        if kw:
            yield "business_keyword", kw in str(ticket.get("business_type", "")), \
                f"want '{kw}' in '{ticket.get('business_type')}'"
        yield "script_schema", all(k in script for k in SCRIPT_KEYS), \
            f"missing: {[k for k in SCRIPT_KEYS if k not in script]}"
        crit = script.get("scoring_criteria") or []
        yield "scoring_criteria_shape", 3 <= len(crit) <= 5 and all(
            isinstance(c, dict) and c.get("dimension") and c.get("description") for c in crit
        ), f"n={len(crit)}"
        cps = script.get("challenge_points") or []
        yield "challenge_points_count", 3 <= len(cps) <= 6, f"n={len(cps)}"
        yield from _check_review_like(review, n_dims=5)

    elif ctype == "normalize":
        ticket = output["ticket"]
        yield from _checks_for_ticket(ticket, generated=False)
        blob = json.dumps(ticket, ensure_ascii=False)
        for s in hc.get("must_contain", []):
            yield f"must_contain[{s}]", s in blob, ""
        for s in hc.get("must_not_contain", []):
            yield f"must_not_contain[{s}]", s not in blob, "fabrication suspected" if s in blob else ""
        for field in hc.get("expect_empty", []):
            v = ticket.get(field)
            yield f"expect_empty[{field}]", not v, f"got: {str(v)[:80]}"
        if hc.get("expect_no_agent_turns"):
            agent_turns = [t for t in ticket.get("dialogue", []) if t.get("role") == "agent"]
            yield "no_agent_turns", not agent_turns, f"fabricated {len(agent_turns)} agent turns"

    elif ctype == "evaluate":
        ev = output["evaluation"]
        yield from _check_review_like(ev)
        script = scripts[case["script_ref"]]
        want = {c["dimension"] for c in script["scoring_criteria"]}
        got = {d.get("name") for d in ev.get("dimensions", [])}
        yield "dimensions_cover_criteria", want <= got, f"missing: {sorted(want - got)}"
        band = case["expected_score"]
        score = ev.get("overall_score")
        in_band = isinstance(score, (int, float)) and band["min"] <= score <= band["max"]
        yield "score_in_expected_band", in_band, f"score={score}, band=[{band['min']},{band['max']}]"


def score_run(run_id: str) -> dict:
    run_dir = RUNS_DIR / run_id
    with open(DATASET_DIR / "scripts.json", encoding="utf-8") as f:
        scripts = json.load(f)
    cases = {}
    with open(DATASET_DIR / "cases.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                cases[c["id"]] = c

    rows, n_pass = [], 0
    with open(run_dir / "results.jsonl", encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    for r in results:
        case = cases[r["case_id"]]
        case = {**case, "corrupt": r.get("corrupt")}
        if "error" in r:
            rows.append({"case_id": r["case_id"], "scorer": "hard", "check": "run_ok",
                         "passed": False, "detail": r["error"]})
            continue
        rows.append({"case_id": r["case_id"], "scorer": "hard", "check": "run_ok",
                     "passed": True, "detail": ""})
        for name, passed, detail in check_case(case, r["output"], scripts):
            rows.append({"case_id": r["case_id"], "scorer": "hard", "check": name,
                         "passed": bool(passed), "detail": detail if not passed else ""})

    n_pass = sum(1 for x in rows if x["passed"])
    with open(run_dir / "scores_hard.jsonl", "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    summary = {"run_id": run_id, "checks": len(rows), "passed": n_pass,
               "pass_rate": round(n_pass / len(rows), 3) if rows else None}
    print(f"[hard] {summary['passed']}/{summary['checks']} checks passed "
          f"({summary['pass_rate']:.1%})" if rows else "[hard] no results")
    for x in rows:
        if not x["passed"]:
            print(f"[hard]   FAIL {x['case_id']} :: {x['check']} :: {x['detail']}")
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m evals.scorers.hard <run_id>")
    score_run(sys.argv[1])
