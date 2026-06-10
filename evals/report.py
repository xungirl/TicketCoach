"""
Report generator: aggregates scores_hard.jsonl + scores_judge.jsonl.

Usage:
    python -m evals.report <run_id>              # single-run report → report.md
    python -m evals.report <baseline> <sabotage> # compare runs (harness validation)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVALS_DIR / "runs"
DATASET_DIR = EVALS_DIR / "dataset"


def _load_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_run(run_id):
    run_dir = RUNS_DIR / run_id
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    case_types = {}
    with open(DATASET_DIR / "cases.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                case_types[c["id"]] = c["type"]
    return {
        "meta": meta,
        "hard": _load_jsonl(run_dir / "scores_hard.jsonl"),
        "judge": _load_jsonl(run_dir / "scores_judge.jsonl"),
        "case_types": case_types,
    }


def summarize(run):
    """Per-type hard pass rate and judge mean."""
    ct = run["case_types"]
    out = defaultdict(lambda: {"hard_pass": 0, "hard_total": 0, "judge_scores": []})
    for x in run["hard"]:
        s = out[ct.get(x["case_id"], "?")]
        s["hard_total"] += 1
        s["hard_pass"] += bool(x["passed"])
    for x in run["judge"]:
        if x["score"] is not None:
            out[ct.get(x["case_id"], "?")]["judge_scores"].append(x["score"])
    summary = {}
    for t, s in out.items():
        summary[t] = {
            "hard_pass_rate": s["hard_pass"] / s["hard_total"] if s["hard_total"] else None,
            "judge_mean": round(sum(s["judge_scores"]) / len(s["judge_scores"]), 2)
            if s["judge_scores"] else None,
        }
    # overall
    total = sum(1 for x in run["hard"])
    passed = sum(1 for x in run["hard"] if x["passed"])
    judged = [x["score"] for x in run["judge"] if x["score"] is not None]
    summary["__overall__"] = {
        "hard_pass_rate": passed / total if total else None,
        "judge_mean": round(sum(judged) / len(judged), 2) if judged else None,
    }
    return summary


def _fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v:.1%}" if pct else f"{v}"


def single_report(run_id):
    run = load_run(run_id)
    s = summarize(run)
    m = run["meta"]
    lines = [
        f"# Eval Report — `{run_id}`",
        "",
        f"- 被测模型: `{m['model']}`  | corrupt: `{m.get('corrupt')}`  | cases: {m['n_cases']} (errors: {m['n_errors']})",
        f"- 时间: {m['started']} → {m['finished']}",
        "",
        "| 类型 | 硬指标通过率 | 裁判均分(1-5) |",
        "|---|---|---|",
    ]
    for t in ("generate", "normalize", "evaluate"):
        if t in s:
            lines.append(f"| {t} | {_fmt(s[t]['hard_pass_rate'], pct=True)} | {_fmt(s[t]['judge_mean'])} |")
    o = s["__overall__"]
    lines.append(f"| **总计** | **{_fmt(o['hard_pass_rate'], pct=True)}** | **{_fmt(o['judge_mean'])}** |")

    fails = [x for x in run["hard"] if not x["passed"]]
    lines += ["", f"## 硬指标失败项（{len(fails)}）", ""]
    for x in fails:
        lines.append(f"- `{x['case_id']}` :: {x['check']} — {x['detail']}")
    low = [x for x in run["judge"] if x["score"] is not None and x["score"] <= 2]
    if low:
        lines += ["", "## 裁判低分项（≤2）", ""]
        for x in low:
            lines.append(f"- `{x['case_id']}` :: {x['score']}/5 — {x['reasoning']}")

    report = "\n".join(lines) + "\n"
    out = RUNS_DIR / run_id / "report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"[report] written to {out}")


def compare_report(baseline_id, sabotage_id):
    a, b = load_run(baseline_id), load_run(sabotage_id)
    sa, sb = summarize(a), summarize(b)
    lines = [
        f"# Harness Validation — `{baseline_id}` (baseline) vs `{sabotage_id}` (sabotage: {b['meta'].get('corrupt') or b['meta']['model']})",
        "",
        "| 类型 | 指标 | baseline | sabotage | Δ |",
        "|---|---|---|---|---|",
    ]
    drops, flats = [], []
    for t in ("generate", "normalize", "evaluate", "__overall__"):
        if t not in sa or t not in sb:
            continue
        label = "总计" if t == "__overall__" else t
        for key, pct in (("hard_pass_rate", True), ("judge_mean", False)):
            va, vb = sa[t][key], sb[t][key]
            if va is None or vb is None:
                continue
            delta = vb - va
            lines.append(f"| {label} | {key} | {_fmt(va, pct)} | {_fmt(vb, pct)} | {delta:+.2f} |")
            if t == "__overall__":
                (drops if delta < -0.05 else flats).append(key)
    lines.append("")
    if drops and not flats:
        lines.append("**结论：坏化后分数显著下降 → 评估 harness 有效。**")
    elif drops:
        lines.append(f"**结论：部分指标下降（{drops}），但 {flats} 未动 → 对应指标可能不灵敏，需检查。**")
    else:
        lines.append("**结论：坏化后分数没有下降 → 评估是瞎的，rubric/检查项需要返工！**")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        single_report(sys.argv[1])
    elif len(sys.argv) == 3:
        compare_report(sys.argv[1], sys.argv[2])
    else:
        sys.exit("usage: python -m evals.report <run_id> [<sabotage_run_id>]")
