"""
TicketCoach Core Pipeline
3-step LLM pipeline: ticket generation → script extraction → quality review
"""

import json
import os
import random
import re
import sys
import time
from typing import Optional

import openai
from dotenv import load_dotenv

# Allow running this file directly (`python core/pipeline.py`) by adding the
# project root to sys.path, so the `core` package can be imported either way.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.prompts import (
    TICKET_SYSTEM_PROMPT,
    TICKET_USER_TEMPLATE,
    SCRIPT_SYSTEM_PROMPT,
    SCRIPT_USER_TEMPLATE,
    REVIEW_SYSTEM_PROMPT,
    REVIEW_USER_TEMPLATE,
    CHAT_SYSTEM_WRAPPER,
    EVALUATE_SYSTEM_PROMPT,
    EVALUATE_USER_TEMPLATE,
    NORMALIZE_TICKET_SYSTEM_PROMPT,
    NORMALIZE_TICKET_USER_TEMPLATE,
    AGENT_SYSTEM_PROMPT,
)

# Load .env file from project root
load_dotenv()

# ---------------------------------------------------------------------------
# LLM client setup
# ---------------------------------------------------------------------------

def get_client() -> openai.OpenAI:
    """Read env vars LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, return openai.OpenAI client."""
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    if not api_key:
        raise ValueError("LLM_API_KEY environment variable is not set. Please check your .env file.")

    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        # Stronger models (e.g. qwen-max / reasoning models) need more headroom
        # for long generations; configurable via LLM_TIMEOUT env var.
        timeout=float(os.environ.get("LLM_TIMEOUT", "180")),
    )


def _get_model() -> str:
    """Return model name from env, default to qwen-plus."""
    return os.environ.get("LLM_MODEL", "qwen-plus")


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences if present."""
    text = text.strip()
    # Match ```json or ``` at start, ``` at end
    pattern = r'^```(?:json)?\s*\n?(.*?)\n?```$'
    match = re.match(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _raw_chat(messages: list, json_mode: bool = False, temperature: float = 0.8) -> str:
    """
    Low-level chat completion: takes a full messages list, returns raw text.
    Shared by both the JSON pipeline calls and the free-text roleplay engine.
    """
    client = get_client()
    model = _get_model()

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
    except openai.OpenAIError as e:
        raise RuntimeError(f"LLM API error: {e}")


def call_llm(system: str, user: str, json_mode: bool = True) -> dict:
    """
    Single system+user LLM call returning a parsed dict.
    - Uses response_format={"type": "json_object"} when json_mode=True
    - Strips markdown code blocks if present
    - On JSON parse failure, retries once
    - Returns parsed dict
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    raw = ""
    for attempt in range(2):  # try up to 2 times
        raw = _raw_chat(messages, json_mode=json_mode)
        try:
            return json.loads(_strip_markdown_fences(raw))
        except json.JSONDecodeError as e:
            if attempt == 0:
                print(f"[pipeline] JSON parse error on attempt 1, retrying... ({e})")
                time.sleep(1)
                continue
            raise ValueError(
                f"Failed to parse LLM JSON response after 2 attempts. "
                f"Last error: {e}\nRaw response: {raw[:500]}"
            )


# ---------------------------------------------------------------------------
# Parameter randomization
# ---------------------------------------------------------------------------

# Alibaba Cloud product lines (this tool trains Alibaba Cloud support agents)
BUSINESS_TYPES = [
    "云服务器ECS/实例连接",
    "云服务器ECS/性能与卡顿",
    "对象存储OSS/数据与权限",
    "云数据库RDS/连接与性能",
    "域名服务/解析与续费",
    "ICP备案/备案审核",
    "账单计费/异常扣费",
    "弹性公网IP/带宽流量",
    "CDN/加速与回源",
    "云安全/DDoS与攻击防护",
    "SSL证书/申请与部署",
    "短信服务/签名与模板",
    "工单服务/响应与升级",
    "退款与代金券/费用纠纷",
    "容器服务ACK/集群运维",
]

EMOTIONS = ["平静", "不满", "愤怒"]

ISSUE_CATEGORIES = [
    "实例无法远程连接（SSH/RDP）",
    "控制台操作报错",
    "莫名扣费/账单不透明",
    "欠费停机/资源被释放",
    "数据误删要求恢复",
    "域名解析不生效",
    "备案被驳回",
    "数据库连接超时/失败",
    "网站遭遇DDoS攻击",
    "带宽跑满/流量费暴涨",
    "续费/退款纠纷",
    "配额或限额申请被拒",
    "工单响应慢/要求升级技术专家",
    "证书部署后不生效",
]

DIFFICULTIES = ["低", "中", "高"]


def random_params() -> dict:
    """Return a random combination of pipeline generation parameters."""
    return {
        "business_type": random.choice(BUSINESS_TYPES),
        "emotion": random.choice(EMOTIONS),
        "issue_category": random.choice(ISSUE_CATEGORIES),
        "difficulty": random.choice(DIFFICULTIES),
    }


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def generate_ticket(params: dict) -> dict:
    """
    Step 1: Generate a realistic (dirty) customer service ticket.
    Calls LLM with ticket prompt, injecting params into user message.
    """
    user_msg = TICKET_USER_TEMPLATE.format(
        business_type=params.get("business_type", "云服务器ECS/实例连接"),
        emotion=params.get("emotion", "不满"),
        issue_category=params.get("issue_category", "实例无法远程连接（SSH/RDP）"),
        difficulty=params.get("difficulty", "中"),
    )
    print(f"[pipeline] Step 1: Generating ticket (type={params.get('business_type')}, emotion={params.get('emotion')})...")
    ticket = call_llm(TICKET_SYSTEM_PROMPT, user_msg, json_mode=True)
    print(f"[pipeline] Step 1 done. ticket_id={ticket.get('ticket_id', 'N/A')}")
    return ticket


def ticket_to_script(ticket: dict) -> dict:
    """
    Step 2: Extract a structured roleplay training script from a ticket.
    Injects ticket JSON as user message.
    """
    ticket_json_str = json.dumps(ticket, ensure_ascii=False, indent=2)
    user_msg = SCRIPT_USER_TEMPLATE.format(ticket_json=ticket_json_str)
    print(f"[pipeline] Step 2: Extracting training script from ticket...")
    script = call_llm(SCRIPT_SYSTEM_PROMPT, user_msg, json_mode=True)
    print(f"[pipeline] Step 2 done. title={script.get('title', 'N/A')[:40]}")
    return script


def review_script(ticket: dict, script: dict) -> dict:
    """
    Step 3: Quality check and score the training script.
    Injects both ticket and script JSON as user message.
    """
    ticket_json_str = json.dumps(ticket, ensure_ascii=False, indent=2)
    script_json_str = json.dumps(script, ensure_ascii=False, indent=2)
    user_msg = REVIEW_USER_TEMPLATE.format(
        ticket_json=ticket_json_str,
        script_json=script_json_str,
    )
    print(f"[pipeline] Step 3: Reviewing script quality...")
    review = call_llm(REVIEW_SYSTEM_PROMPT, user_msg, json_mode=True)
    score = review.get("overall_score", "N/A")
    print(f"[pipeline] Step 3 done. overall_score={score}")
    return review


# ---------------------------------------------------------------------------
# Real-ticket ingestion (for teams that already have real tickets)
# ---------------------------------------------------------------------------

TICKET_KEYS = ["ticket_id", "business_type", "customer_profile", "channel",
               "dialogue", "resolution", "tags"]


def _fill_ticket_defaults(ticket: dict) -> dict:
    """Ensure a ticket dict has all expected keys with sane default types."""
    out = dict(ticket)
    out.setdefault("ticket_id", "")
    out.setdefault("business_type", "")
    out.setdefault("customer_profile", "")
    out.setdefault("channel", "")
    out.setdefault("dialogue", [])
    out.setdefault("resolution", "")
    out.setdefault("tags", [])
    return out


def normalize_ticket_from_text(raw_ticket: str) -> dict:
    """
    Turn a real ticket's raw content (free text or messy fields) into our
    structured ticket JSON, without fabricating missing info.
    """
    user_msg = NORMALIZE_TICKET_USER_TEMPLATE.format(raw_ticket=raw_ticket)
    ticket = call_llm(NORMALIZE_TICKET_SYSTEM_PROMPT, user_msg, json_mode=True)
    return _fill_ticket_defaults(ticket)


def coerce_ticket(item) -> dict:
    """
    Accept a ticket in any form (dict already in our shape, dict with other
    fields, or raw string) and return a normalized ticket dict.
    """
    if isinstance(item, dict) and isinstance(item.get("dialogue"), list) and item.get("dialogue"):
        # Already looks like our structured ticket — use as-is.
        return _fill_ticket_defaults(item)
    # Otherwise stringify and let the LLM structure it.
    text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, indent=2)
    return normalize_ticket_from_text(text)


# ---------------------------------------------------------------------------
# Roleplay chat engine (live sparring) + session evaluation
# ---------------------------------------------------------------------------

def chat_reply(actor_prompt: str, history: list) -> str:
    """
    Produce the next CUSTOMER turn in a live roleplay.

    actor_prompt: the script's actor_prompt (defines the customer character).
    history: list of {"role": "customer"|"agent", "text": str}, in order.
             The trainee plays the agent; the LLM plays the customer.
    Returns the customer's next message as plain text.
    """
    system = CHAT_SYSTEM_WRAPPER.format(actor_prompt=actor_prompt)
    messages = [{"role": "system", "content": system}]

    for turn in history:
        # customer turns are the assistant's own past lines; agent turns are "user"
        role = "assistant" if turn.get("role") == "customer" else "user"
        messages.append({"role": role, "content": turn.get("text", "")})

    if not history:
        # Kick off the conversation: the customer speaks first.
        messages.append({
            "role": "user",
            "content": "（系统提示：对练现在开始，请你作为这位客户主动说出第一句话，开启对话。只说客户会说的话。）",
        })

    return _raw_chat(messages, json_mode=False, temperature=0.9).strip()


def chat_reply_stream(actor_prompt: str, history: list):
    """
    Streaming version of chat_reply: yields the customer's reply piece by piece
    as the model generates it (for a real-time, typing-style roleplay UX).
    """
    system = CHAT_SYSTEM_WRAPPER.format(actor_prompt=actor_prompt)
    messages = [{"role": "system", "content": system}]
    for turn in history:
        role = "assistant" if turn.get("role") == "customer" else "user"
        messages.append({"role": role, "content": turn.get("text", "")})
    if not history:
        messages.append({
            "role": "user",
            "content": "（系统提示：对练现在开始，请你作为这位客户主动说出第一句话，开启对话。只说客户会说的话。）",
        })

    client = get_client()
    model = _get_model()
    stream = client.chat.completions.create(
        model=model, messages=messages, temperature=0.9, stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            yield delta.content


def agent_reply_stream(script: dict, history: list):
    """
    Streaming MODEL-AGENT reply, for AI-vs-AI demo roleplay. Plays an exemplary
    support agent guided by the script's scenario + standard_response.
    (Agent turns map to assistant, customer turns to user.)
    """
    system = AGENT_SYSTEM_PROMPT.format(
        scenario=script.get("scenario", ""),
        standard_response=script.get("standard_response", ""),
    )
    messages = [{"role": "system", "content": system}]
    for turn in history:
        role = "assistant" if turn.get("role") == "agent" else "user"
        messages.append({"role": role, "content": turn.get("text", "")})

    client = get_client()
    model = _get_model()
    stream = client.chat.completions.create(
        model=model, messages=messages, temperature=0.7, stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            yield delta.content


def _format_transcript(transcript: list) -> str:
    """Render a roleplay transcript as readable text for the evaluator."""
    lines = []
    for turn in transcript:
        who = "客服(agent)" if turn.get("role") == "agent" else "客户(customer)"
        lines.append(f"{who}: {turn.get('text', '')}")
    return "\n".join(lines)


def evaluate_session(script: dict, transcript: list) -> dict:
    """
    Score the trainee's (agent's) performance after a roleplay session,
    using the script's scoring_criteria. Returns a review dict.
    """
    user_msg = EVALUATE_USER_TEMPLATE.format(
        script_json=json.dumps(script, ensure_ascii=False, indent=2),
        transcript=_format_transcript(transcript),
    )
    print(f"[pipeline] Evaluating roleplay session ({len(transcript)} turns)...")
    result = call_llm(EVALUATE_SYSTEM_PROMPT, user_msg, json_mode=True)
    print(f"[pipeline] Evaluation done. overall_score={result.get('overall_score', 'N/A')}")
    return result


# ---------------------------------------------------------------------------
# Full pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(params: Optional[dict] = None) -> dict:
    """
    Run all 3 steps of the TicketCoach pipeline.

    If params=None, uses random_params().
    If review overall_score < 60, regenerates the script once (quality loop).

    Returns: {"ticket": ..., "script": ..., "review": ..., "params": ...}
    """
    if params is None:
        params = random_params()

    # Normalize params: replace empty/None values with random choices
    effective_params = {
        "business_type": params.get("business_type") or random.choice(BUSINESS_TYPES),
        "emotion": params.get("emotion") or random.choice(EMOTIONS),
        "issue_category": params.get("issue_category") or random.choice(ISSUE_CATEGORIES),
        "difficulty": params.get("difficulty") or random.choice(DIFFICULTIES),
    }

    print(f"[pipeline] Starting pipeline with params: {effective_params}")

    # Step 1: Generate ticket
    ticket = generate_ticket(effective_params)

    # Step 2: Extract script
    script = ticket_to_script(ticket)

    # Step 3: Review quality
    review = review_script(ticket, script)

    # Quality loop: if score < 60, regenerate script once
    overall_score = review.get("overall_score", 100)
    if isinstance(overall_score, (int, float)) and overall_score < 60:
        print(f"[pipeline] Score {overall_score} < 60, regenerating script (quality loop)...")
        script = ticket_to_script(ticket)
        review = review_script(ticket, script)
        print(f"[pipeline] After regeneration, new score: {review.get('overall_score', 'N/A')}")

    return {
        "ticket": ticket,
        "script": script,
        "review": review,
        "params": effective_params,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    print("=" * 60)
    print("TicketCoach Pipeline — CLI Mode")
    print("=" * 60)

    result = run_pipeline()

    print("\n" + "=" * 60)
    print("TICKET (工单)")
    print("=" * 60)
    pprint.pprint(result["ticket"], width=100)

    print("\n" + "=" * 60)
    print("SCRIPT (培训剧本)")
    print("=" * 60)
    pprint.pprint(result["script"], width=100)

    print("\n" + "=" * 60)
    print("REVIEW (质量评审)")
    print("=" * 60)
    pprint.pprint(result["review"], width=100)

    print("\n[pipeline] All done!")
