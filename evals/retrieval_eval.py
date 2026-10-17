"""
Retrieval eval: does the KB return the right help doc for a trainee's question?

    venv/bin/python -m evals.retrieval_eval                # all available modes
    venv/bin/python -m evals.retrieval_eval --mode bm25    # zero-cost, offline

Labels are doc-level (evals/dataset/retrieval.jsonl lists the doc titles that
answer each query). Metrics:
    Hit@1 / Hit@5  — a relevant doc is the top hit / among the top 5 chunks
    MRR            — mean of 1 / rank of the first relevant chunk
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import llm  # noqa: E402
from core.kb import KnowledgeBase  # noqa: E402

DATASET = Path(__file__).parent / "dataset" / "retrieval.jsonl"


def evaluate(kb: KnowledgeBase, cases: list, mode: str, k: int = 5) -> dict:
    hit1 = hitk = mrr = 0.0
    misses, latencies, degraded = [], [], 0
    for case in cases:
        t = time.time()
        hits = kb.search(case["query"], k=k, mode=mode)
        latencies.append((time.time() - t) * 1000)
        if hits and mode != "bm25" and "vector" not in hits[0]["retrieval"]:
            degraded += 1
        ranks = [i for i, h in enumerate(hits) if h["title"] in case["relevant"]]
        if ranks:
            hitk += 1
            mrr += 1 / (ranks[0] + 1)
            hit1 += ranks[0] == 0
        else:
            misses.append({"id": case["id"], "query": case["query"],
                           "got": [h["title"] for h in hits[:3]]})
    n = len(cases)
    latencies.sort()
    return {"mode": mode, "n": n, "hit@1": round(hit1 / n, 3), f"hit@{k}": round(hitk / n, 3),
            "mrr": round(mrr / n, 3), "p50_ms": round(latencies[n // 2], 1),
            "degraded_to_bm25": degraded, "misses": misses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bm25", "vector", "hybrid"], action="append")
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    cases = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    kb = KnowledgeBase.load(embed_fn=llm.embed)
    modes = args.mode or (["bm25", "vector", "hybrid"] if kb.vectors is not None else ["bm25"])
    if kb.vectors is None and set(modes) - {"bm25"}:
        print("⚠ no vectors.npy — vector/hybrid will fall back to BM25. Run scripts.build_kb --embed-only.")

    print(f"{len(kb.chunks)} chunks, {len(cases)} queries\n")
    print(f"{'mode':8s} {'hit@1':>6s} {'hit@5':>6s} {'mrr':>6s} {'p50ms':>7s}")
    for mode in modes:
        r = evaluate(kb, cases, mode)
        print(f"{mode:8s} {r['hit@1']:6.3f} {r['hit@5']:6.3f} {r['mrr']:6.3f} {r['p50_ms']:7.1f}"
              + (f"   (degraded {r['degraded_to_bm25']}/{r['n']})" if r["degraded_to_bm25"] else ""))
        if args.show_misses:
            for m in r["misses"]:
                print(f"   ✗ {m['id']} {m['query']}  ->  {m['got']}")


if __name__ == "__main__":
    main()
