"""
Eval harness runner: executes dataset cases against core.pipeline and dumps
trace.jsonl + results.jsonl into evals/runs/<run_id>/.

Running and scoring are decoupled: this script only collects data (costs LLM
calls); scoring reads the dumped files offline (free to re-run).

Usage (from project root):
    python -m evals.run_eval                          # all cases
    python -m evals.run_eval --type normalize --limit 2
    python -m evals.run_eval --ids gen-01 eval-03
    python -m evals.run_eval --model qwen-turbo --run-id sabotage-weak-model
    python -m evals.run_eval --corrupt drop-dialogue --run-id sabotage-drop
    python -m evals.run_eval --corrupt lazy-transcript --run-id sabotage-lazy
"""

import argparse
import copy
import datetime
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVALS_DIR = Path(__file__).resolve().parent
DATASET_DIR = EVALS_DIR / "dataset"
RUNS_DIR = EVALS_DIR / "runs"

# Sabotage modes used to validate the harness itself (step 4 of the plan):
# a valid harness MUST show a clear score drop under each of these.
CORRUPT_MODES = {
    "drop-dialogue": "generate: delete ticket['dialogue'] before script extraction",
    "lazy-transcript": "evaluate: replace all agent turns with brush-off phrases",
}

LAZY_AGENT_PHRASES = ["您好。", "请稍等。", "这个不清楚，您再问问别人吧。", "好的。"]


def load_cases(types=None, ids=None, limit=None):
    cases = []
    with open(DATASET_DIR / "cases.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    if types:
        cases = [c for c in cases if c["type"] in types]
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    if limit:
        cases = cases[:limit]
    return cases


def load_scripts():
    with open(DATASET_DIR / "scripts.json", encoding="utf-8") as f:
        return json.load(f)


def make_lazy(transcript):
    """Sabotage: keep customer turns, replace agent turns with brush-offs."""
    out = []
    i = 0
    for turn in transcript:
        if turn["role"] == "agent":
            out.append({"role": "agent", "text": LAZY_AGENT_PHRASES[i % len(LAZY_AGENT_PHRASES)]})
            i += 1
        else:
            out.append(turn)
    return out


def run_case(case, scripts, pipeline, corrupt=None):
    """Execute one case, return its output dict (raises on hard failure)."""
    ctype = case["type"]

    if ctype == "generate":
        if corrupt == "drop-dialogue":
            ticket = pipeline.generate_ticket(case["input"])
            ticket = copy.deepcopy(ticket)
            ticket.pop("dialogue", None)
            script = pipeline.ticket_to_script(ticket)
            review = pipeline.review_script(ticket, script)
            return {"ticket": ticket, "script": script, "review": review}
        result = pipeline.run_pipeline(case["input"])
        return {"ticket": result["ticket"], "script": result["script"], "review": result["review"]}

    if ctype == "normalize":
        return {"ticket": pipeline.normalize_ticket_from_text(case["input"])}

    if ctype == "evaluate":
        script = scripts[case["script_ref"]]
        transcript = case["transcript"]
        if corrupt == "lazy-transcript":
            transcript = make_lazy(transcript)
        return {
            "evaluation": pipeline.evaluate_session(script, transcript),
            "transcript_used": transcript,
        }

    raise ValueError(f"unknown case type: {ctype}")


def main():
    parser = argparse.ArgumentParser(description="TicketCoach eval runner")
    parser.add_argument("--type", nargs="*", choices=["generate", "normalize", "evaluate"])
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", help="override LLM_MODEL for this run (sabotage: weaker model)")
    parser.add_argument("--corrupt", choices=list(CORRUPT_MODES))
    parser.add_argument("--run-id", help="default: timestamp")
    args = parser.parse_args()

    if args.model:
        os.environ["LLM_MODEL"] = args.model

    # Import after the env override so _get_model() picks it up.
    from core import pipeline

    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.type, args.ids, args.limit)
    scripts = load_scripts()
    print(f"[eval] run_id={run_id}  cases={len(cases)}  model={pipeline._get_model()}  corrupt={args.corrupt}")

    n_errors = 0
    started = datetime.datetime.now().isoformat(timespec="seconds")
    with open(run_dir / "trace.jsonl", "w", encoding="utf-8") as trace_f, \
         open(run_dir / "results.jsonl", "w", encoding="utf-8") as results_f:
        for i, case in enumerate(cases, 1):
            print(f"[eval] ({i}/{len(cases)}) {case['id']} ({case['type']}) ...")
            collector = []
            pipeline.set_trace_collector(collector)
            t0 = time.time()
            entry = {"case_id": case["id"], "type": case["type"], "corrupt": args.corrupt}
            try:
                entry["output"] = run_case(case, scripts, pipeline, corrupt=args.corrupt)
            except Exception as e:
                entry["error"] = f"{type(e).__name__}: {e}"
                n_errors += 1
                print(f"[eval]   ERROR: {entry['error']}")
            finally:
                pipeline.set_trace_collector(None)
            entry["elapsed_ms"] = int((time.time() - t0) * 1000)
            entry["n_llm_calls"] = sum(1 for ev in collector if ev.get("event") == "llm_call")
            for ev in collector:
                ev["case_id"] = case["id"]
                trace_f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            results_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            trace_f.flush()
            results_f.flush()

    meta = {
        "run_id": run_id,
        "model": pipeline._get_model(),
        "corrupt": args.corrupt,
        "started": started,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_cases": len(cases),
        "n_errors": n_errors,
    }
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[eval] done. {len(cases) - n_errors} ok / {n_errors} errors → {run_dir}")


if __name__ == "__main__":
    main()
