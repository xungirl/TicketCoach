# TicketCoach × 「AI Engineer in 2026」清单

把那条 23 项清单逐条落到 TicketCoach 上。✅ 已完成并有测试/数据　🟡 部分完成　⏳ 需要你来做（要钱、要真人、要你本人学）

分支：`agent-upgrade`（本地，未推送）。每一步一个 commit，`git log --oneline main..agent-upgrade` 可以看全。

| # | 清单项 | 状态 | 落在哪 | 还差什么 / 面试怎么讲 |
|---|---|---|---|---|
| 1 | Learn Python for real | ⏳ | — | 这是你本人的事。建议：把 `core/kb.py` 的 BM25 和 `core/memory.py` 的 `profile()` 合上文件自己重写一遍 |
| 2 | Understand an API before an agent | ✅ | `core/llm.py` | 手写的网关：OpenAI 兼容接口、错误分类（404/403/429/5xx/Arrearage 各自怎么处理） |
| 3 | Learn how an LLM works | ⏳ | CS336 | 做 CS336 A1 才算数（简历上那条也要等它） |
| 4 | Build with tool calling | ✅ | `core/agent.py` | 5 个工具、pydantic 校验参数、工具出错回传给模型而不是崩 |
| 5 | Learn RAG | ✅ | `core/kb.py`, `scripts/build_kb.py` | 30 篇真实阿里云帮助文档 → 580 块；回答带 [D1] 引用，引用不存在的会被剥掉 |
| 6 | Embeddings and vector search | 🟡 | `core/kb.py` | 代码写好了（numpy 余弦 + RRF 融合），**向量还没生成**：账户欠费，embedding 接口也被封。充值后跑 `python -m scripts.build_kb --embed-only` 再跑 `python -m evals.retrieval_eval` 比较 bm25 / vector / hybrid |
| 7 | Build a real agent, just one | ✅ | `/api/coach` | AI 教练：有步数预算、会自己决定查文档 / 查档案 / 推剧本 |
| 8 | Give it memory | ✅ | `core/memory.py` | 长期记忆：每次评分自动入库；画像算出最弱维度、进步趋势、常漏挑战点、没练过的产品线。短期记忆：最近 12 轮对话 |
| 9 | Real tools, not a mock | ✅ | `read_help_doc` 等 | 实时抓 help.aliyun.com（域名白名单 + 必须先检索过才能读）、真 SQLite、真剧本库 |
| 10 | Retries and fallbacks | ✅ | `core/llm.py` | 同模型 SDK 重试 → 换模型 → 换厂商；熔断器 + 半开探测。**这是今晚真实事故逼出来的**（见 #15） |
| 11 | Force structured output | ✅ | `core/schemas.py` | pydantic 校验 + 把错误喂回模型让它自己修 |
| 12 | Add auth | ✅ | `core/auth.py` | 姓名 + 口令换 HMAC 签名 token；工具不收 trainee_id，防止 prompt 注入读别人数据 |
| 13 | Add observability | ✅ | `core/obs.py` | JSON 日志 + request_id 贯穿（含流式响应），`/api/metrics` 按模型/步骤给 tokens、p50/p95、成本 |
| 14 | Add evals | 🟡 | `evals/` | 检索评测已出数（BM25 Hit@5 0.967）；教练 15 条 case 的 harness 写好并自测过，**等能调用模型再跑**：`python -m evals.coach_eval --run-id coach-baseline` |
| 15 | Break it with real failures | ✅/⏳ | 事故记录在 README | **真实事故**：qwen3-max 下线 404 + 账户欠费 Arrearage，线上 demo 全挂。单测里有 chaos 测试；真模型下的演练：`python -m evals.coach_eval --chaos 0.5` |
| 16 | Deploy it publicly | 🟡 | Cloud Run | 已部署，但**现在是挂的**，新代码也还没部署（需要你确认，见下方） |
| 17 | See what it costs in prod | 🟡 | `/api/metrics`, `LLM_PRICES` | 统计已做；单价要你从百炼控制台抄进 `LLM_PRICES`（我没有硬编码，网上查到的价格互相矛盾） |
| 18 | Lower the latency | ✅ | `app/main.py`, `LLM_STEP_MODELS` | 批量并行（测试里 3 条各 0.4s：并行 <0.9s，串行要 ≥1.2s）、按步骤分档模型、共享 HTTP 连接池；同一个造工单 prompt 今晚实测：qwen3.8-max 31.9s / qwen-max 15.7s / qwen-plus 14.5s |
| 19 | Read the logs | ⏳ | README 有过滤语句 | 部署后用 Cloud Logging 按 `jsonPayload.message="coach_turn"` 看真实用户怎么用 |
| 20 | Fix what the user actually breaks | ⏳ | — | 需要真用户：把链接发给你朋友的阿里云团队，按日志和反馈修 |
| 21 | Open the app | ⏳ | — | 同上，让真人用起来 |
| 22 | Followable README | ✅ | `README.md` | 英文、5 行 quickstart、环境变量表、评测表（只放实测数字）、已知限制、事故记录 |
| 23 | Put the demo on the CV | 🟡 | 简历已有 Demo 链接 | 等 #6 #14 跑出真实数字后再更新简历要点（不能提前写） |

## 你醒来后要做的事（按顺序）

1. **给阿里云百炼账户充值**（或者在 `.env` 加 `GEMINI_API_KEY`，让 Gemini 兜底）。不做这一步，线上 demo 和所有评测都跑不起来。
2. 本地验证：
   ```bash
   cd ~/TicketCoach
   sed -i '' 's/^LLM_MODEL=qwen3-max/LLM_MODEL=qwen-plus/' .env   # qwen3-max 已下线
   venv/bin/python -m pytest -q
   venv/bin/python -m scripts.build_kb --embed-only                 # 生成向量
   venv/bin/python -m evals.retrieval_eval --show-misses            # bm25 vs vector vs hybrid
   venv/bin/python -m evals.coach_eval --run-id coach-baseline      # 教练 15 条
   venv/bin/python -m evals.coach_eval --chaos 0.5 --run-id coach-chaos
   ```
3. 看结果、修失败的 case（这就是 #20 的预演）。
4. 决定是否合并到 main、推送、重新部署。部署前要在 Cloud Run 加环境变量：`AUTH_SECRET`（随机长字符串）、`LLM_MODEL=qwen-plus`，有 Gemini key 的话加 `GEMINI_API_KEY`。
5. 有了真实数字后，再改简历上 TicketCoach 那几条。

## 面试 30 秒版本

> 我的 demo 有一天全挂了：先是主模型被下线返回 404，然后整个账户欠费，所有模型和 embedding 接口一起被封。我就把所有调用收进一个网关：同模型重试 → 换模型 → 跨厂商兜底，账户级错误直接跳过这个厂商所有模型，再加熔断器避免每个请求都去撞死掉的模型。检索那边 embedding 挂了会自动降级成 BM25，所以知识库照样能用。这些都有离线测试，也能用 chaos 模式在真模型上演练。
