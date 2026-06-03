# TicketCoach — 阿里云客服培训陪练系统

> 面向【阿里云（Alibaba Cloud）客服/技术支持】：把真实（脏）工单经多步 LLM 流水线转成培训剧本，再让你与 AI 扮演的客户实战对练并自动评分。

---

## 项目简介

TicketCoach 是一个基于大语言模型的客服培训辅助工具，构成「生成 → 对练 → 评分」完整闭环：

1. **Step 1 — 工单生成**：模拟真实脏数据客服工单（口语化、有错别字、情绪波动）
2. **Step 2 — 剧本提取**：将工单结构化为标准培训角色扮演剧本
3. **Step 3 — 质量审核**：自动评分并给出改进建议
4. **实战对练**：你扮演客服打字，AI 用剧本的 `actor_prompt` 实时扮演该难缠客户，多轮对话
5. **对练评分**：对练结束后，按剧本的 `scoring_criteria` 给你的表现逐维度打分 + 点评

### API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/generate` | 跑三步流水线，返回 `{ticket, script, review}` |
| POST | `/api/batch?n=N` | 批量生成 N 条 + 统计 |
| POST | `/api/chat` | 对练一轮：传 `actor_prompt` + 对话历史，返回客户下一句 |
| POST | `/api/evaluate` | 传 `script` + 对练记录，按评分维度给客服打分 |
| GET  | `/api/options` | 前端下拉框选项 |
| GET  | `/api/health` | 健康检查 |

## 本地运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

`.env` 文件内容示例：

```
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
```

支持任何兼容 OpenAI API 格式的模型服务，例如：
- 阿里云百炼（通义千问）：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- DeepSeek：`https://api.deepseek.com`

### 3. 启动服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

打开浏览器访问：[http://localhost:8000](http://localhost:8000)

### 4. 命令行模式（仅流水线，不启动服务器）

```bash
python -m core.pipeline
```

---

## Render 部署

1. Fork 本仓库到 GitHub
2. 在 [Render](https://render.com) 创建新的 **Web Service**
3. 配置：
   - **Build Command**：`pip install -r requirements.txt`
   - **Start Command**：`uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. 在 Render 环境变量中添加：
   - `LLM_API_KEY`
   - `LLM_BASE_URL`
   - `LLM_MODEL`

---

## 架构图

```
用户浏览器
    │
    ▼
┌─────────────────────────────────────────┐
│           FastAPI (app/main.py)          │
│  GET /          → 前端页面               │
│  POST /api/generate → 单次生成           │
│  POST /api/batch    → 批量生成           │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│         Pipeline (core/pipeline.py)      │
│                                          │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Step 1   │→ │ Step 2   │→ │Step 3  │ │
│  │工单生成  │  │剧本提取  │  │质量审核│ │
│  └──────────┘  └──────────┘  └────────┘ │
│         ↑ 质量分 < 60 时重新生成 ↑       │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│      LLM API (OpenAI-compatible)         │
│  通义千问 / DeepSeek / 其他              │
└─────────────────────────────────────────┘
```

---

## 截图

*（启动后截图占位）*

---

## 技术栈

- **后端**：Python 3.10+, FastAPI, openai SDK
- **前端**：原生 HTML/CSS/JS（无框架依赖）
- **LLM**：任意 OpenAI-compatible API
