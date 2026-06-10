"""
LLM-as-judge scorer: grades each case's output against its rubric (1-5).
Judge model is deliberately different from the model under test:
  - GEMINI_API_KEY set  → Google Gemini (model: GEMINI_MODEL, default gemini-2.5-flash)
  - otherwise           → dashscope via JUDGE_MODEL (default qwen-plus)

Reads runs/<run_id>/results.jsonl, writes runs/<run_id>/scores_judge.jsonl.

Usage: python -m evals.scorers.judge <run_id>
"""

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

EVALS_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = EVALS_DIR / "runs"
DATASET_DIR = EVALS_DIR / "dataset"

JUDGE_INSTRUCTIONS = """你是一名严格、客观的评估裁判。请根据【评判标准】对【被评内容】打分。

打分标尺（1-5 整数）：
5 = 完全符合标准；4 = 基本符合，有小瑕疵；3 = 部分符合，有明显不足；
2 = 大部分不符合；1 = 严重失败或答非所问。

只输出合法 JSON，不要 markdown 代码块：{"score": <1-5整数>, "reasoning": "<两三句话说明扣分点或亮点>"}"""


def _strip_fences(text: str) -> str:
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", text.strip(), re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------------------
# Judge backends
# ---------------------------------------------------------------------------

def _judge_gemini(prompt: str) -> str:
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={os.environ['GEMINI_API_KEY']}")
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _judge_dashscope(prompt: str) -> str:
    import openai
    client = openai.OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        timeout=120,
    )
    resp = client.chat.completions.create(
        model=os.environ.get("JUDGE_MODEL", "qwen-plus"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


def get_judge():
    if os.environ.get("GEMINI_API_KEY"):
        return _judge_gemini, "gemini:" + os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return _judge_dashscope, "dashscope:" + os.environ.get("JUDGE_MODEL", "qwen-plus")


# ---------------------------------------------------------------------------
# Prompt assembly per case type
# ---------------------------------------------------------------------------

def build_prompt(case, output, scripts) -> str:
    ctype = case["type"]
    parts = [JUDGE_INSTRUCTIONS, "\n【评判标准】\n" + case["rubric"]]

    if ctype == "generate":
        parts.append("\n【任务输入】（生成参数）\n" + json.dumps(case["input"], ensure_ascii=False))
        parts.append("\n【被评内容 1：生成的工单】\n" + json.dumps(output["ticket"], ensure_ascii=False, indent=1))
        parts.append("\n【被评内容 2：提炼的培训剧本】\n" + json.dumps(output["script"], ensure_ascii=False, indent=1))
    elif ctype == "normalize":
        parts.append("\n【任务输入】（原始工单文本，整理时不得编造缺失信息）\n" + case["input"])
        parts.append("\n【被评内容：结构化整理结果】\n" + json.dumps(output["ticket"], ensure_ascii=False, indent=1))
    elif ctype == "evaluate":
        script = scripts[case["script_ref"]]
        parts.append("\n【任务输入 1：剧本评分维度】\n" + json.dumps(script["scoring_criteria"], ensure_ascii=False))
        transcript = output.get("transcript_used", case["transcript"])
        lines = [f"{t['role']}: {t['text']}" for t in transcript]
        parts.append("\n【任务输入 2：对练记录】\n" + "\n".join(lines))
        parts.append("\n【被评内容：考官给出的评估结果】\n"
                     "注意：考官按规定使用「维度 0-10 分、总分 0-100 分」的标尺，这是正确的，不要因此扣分。\n"
                     "你要判断的是：考官的评估是否公正、是否有对话原文依据、维度分与点评是否匹配客服的实际表现。\n"
                     "你自己的 1-5 分只用于表达「这份评估的质量」。\n"
                     + json.dumps(output["evaluation"], ensure_ascii=False, indent=1))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Scoring loop
# ---------------------------------------------------------------------------

def score_run(run_id: str) -> dict:
    run_dir = RUNS_DIR / run_id
    judge_fn, judge_name = get_judge()
    print(f"[judge] using {judge_name}")

    with open(DATASET_DIR / "scripts.json", encoding="utf-8") as f:
        scripts = json.load(f)
    cases = {}
    with open(DATASET_DIR / "cases.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                cases[c["id"]] = c
    with open(run_dir / "results.jsonl", encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    rows = []
    for i, r in enumerate(results, 1):
        if "error" in r:
            rows.append({"case_id": r["case_id"], "scorer": "judge", "score": None,
                         "reasoning": "run errored, skipped", "judge_model": judge_name})
            continue
        case = cases[r["case_id"]]
        prompt = build_prompt(case, r["output"], scripts)
        try:
            raw = judge_fn(prompt)
            verdict = json.loads(_strip_fences(raw))
            score = int(verdict["score"])
            assert 1 <= score <= 5
            rows.append({"case_id": r["case_id"], "scorer": "judge", "score": score,
                         "reasoning": verdict.get("reasoning", ""), "judge_model": judge_name})
            print(f"[judge] ({i}/{len(results)}) {r['case_id']}: {score}/5")
        except Exception as e:
            rows.append({"case_id": r["case_id"], "scorer": "judge", "score": None,
                         "reasoning": f"judge error: {e}", "judge_model": judge_name})
            print(f"[judge] ({i}/{len(results)}) {r['case_id']}: ERROR {e}")

    with open(run_dir / "scores_judge.jsonl", "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    scored = [x["score"] for x in rows if x["score"] is not None]
    mean = round(sum(scored) / len(scored), 2) if scored else None
    print(f"[judge] mean score: {mean} over {len(scored)} cases")
    return {"run_id": run_id, "judge": judge_name, "mean": mean, "n_scored": len(scored)}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m evals.scorers.judge <run_id>")
    score_run(sys.argv[1])
