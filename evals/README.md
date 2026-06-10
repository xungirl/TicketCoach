# TicketCoach Eval Harness

包在 agent 外面的评估层，业务代码零改动（仅 `core/pipeline.py` 多了一个默认关闭的 trace 钩子）。
链路：**插桩 → 测试集 → 跑批落盘 → 双轨打分 → 报告 → 坏化自检**。

## 目录

```
evals/
  dataset/cases.jsonl    # 24 条测试 case（generate 8 / normalize 10 / evaluate 6）
  dataset/scripts.json   # evaluate 类 case 共用的对练剧本
  run_eval.py            # 跑批：调 core.pipeline，落盘 trace + results（花 LLM 调用）
  scorers/hard.py        # 硬指标：schema/字段/编造检测/分数区间（零成本，可反复跑）
  scorers/judge.py       # LLM 裁判：按 rubric 打 1-5 分（Gemini 优先，无 key 退回 qwen-plus）
  report.py              # 单 run 报告 / 双 run 对比（坏化验证）
  runs/<run_id>/         # trace.jsonl, results.jsonl, scores_*.jsonl, report.md（gitignored）
```

## 用法（项目根目录）

```bash
# 1. 跑批（唯一花被测模型调用的步骤）
venv/bin/python -m evals.run_eval --run-id baseline             # 全量 24 条
venv/bin/python -m evals.run_eval --type normalize --limit 2    # 抽样

# 2. 打分（离线读文件，改 rubric/裁判后可随意重跑）
venv/bin/python -m evals.scorers.hard baseline
venv/bin/python -m evals.scorers.judge baseline

# 3. 报告
venv/bin/python -m evals.report baseline

# 4. 坏化自检：故意搞坏 agent，分数必须掉，否则评估无效
venv/bin/python -m evals.run_eval --type evaluate --corrupt lazy-transcript --run-id sab-lazy
venv/bin/python -m evals.run_eval --type generate --corrupt drop-dialogue --run-id sab-drop
venv/bin/python -m evals.run_eval --model qwen-turbo --run-id sab-weak
venv/bin/python -m evals.report baseline sab-lazy               # 对比 + 结论
```

## 裁判模型

在 `.env` 加 `GEMINI_API_KEY`（可选 `GEMINI_MODEL`，默认 gemini-2.5-flash）即切换为 Gemini 裁判；
否则用 dashscope 的 `JUDGE_MODEL`（默认 qwen-plus）。裁判与被测模型（qwen3-max）刻意分开，避免自评偏差。

## 已验证的结论（2026-06-10）

- 基线 evaluate 6/6 落在期望分数区间（93/20/20/93/20/63），硬指标 30/30 通过。
- 坏化实验（lazy-transcript，客服台词全替换为敷衍话术）：好对话从 93 → 30，
  硬指标通过率 100% → 90%，恰好挂掉 3 条 `score_in_expected_band` —— **harness 对该坏化有效**。
- 裁判分在此坏化下不掉是正确的（裁判评的是「考官评估是否公正」，考官给敷衍客服低分正是公正）；
  要验证裁判本身，用 `--corrupt drop-dialogue`（剧本质量下降 → 裁判分应掉）。
- 迭代记录：裁判曾把「考官的 0-100 分制」误判为违规标尺（qwen-plus 锚定了自己的 1-5 标尺），
  已在 judge prompt 中显式声明考官标尺修复——典型的「先怀疑裁判，再相信分数」。

## 注意

- 单条 case 的分会抖（LLM 非确定），结论只看整组通过率/均分的差距。
- trace 含完整 prompt，`evals/runs/` 已 gitignore，勿提交。
