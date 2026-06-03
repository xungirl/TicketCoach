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
        timeout=60.0,
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


def call_llm(system: str, user: str, json_mode: bool = True) -> dict:
    """
    Single LLM call with JSON parsing + retry logic.
    - Uses response_format={"type": "json_object"} when json_mode=True
    - Strips markdown code blocks if present
    - On JSON parse failure, retries once
    - Timeout set to 60s
    - Returns parsed dict
    """
    client = get_client()
    model = _get_model()

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.8,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(2):  # try up to 2 times
        try:
            response = client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            cleaned = _strip_markdown_fences(raw)
            result = json.loads(cleaned)
            return result
        except json.JSONDecodeError as e:
            if attempt == 0:
                print(f"[pipeline] JSON parse error on attempt 1, retrying... ({e})")
                time.sleep(1)
                continue
            else:
                raise ValueError(f"Failed to parse LLM JSON response after 2 attempts. Last error: {e}\nRaw response: {raw[:500]}")
        except openai.OpenAIError as e:
            raise RuntimeError(f"LLM API error: {e}")


# ---------------------------------------------------------------------------
# Parameter randomization
# ---------------------------------------------------------------------------

BUSINESS_TYPES = [
    "电商/退货",
    "电商/物流延误",
    "电商/商品质量",
    "金融/信用卡扣费",
    "金融/贷款申请",
    "电信/套餐变更",
    "电信/话费异常",
    "教育/课程退款",
    "教育/课程质量投诉",
    "餐饮/外卖订单问题",
    "出行/订单取消",
    "出行/司机投诉",
    "保险/理赔纠纷",
    "游戏/账号封禁申诉",
    "游戏/虚拟商品纠纷",
]

EMOTIONS = ["平静", "不满", "愤怒"]

ISSUE_CATEGORIES = [
    "色差/实物与描述不符",
    "物流延误/包裹丢失",
    "扣费异常/重复扣款",
    "服务态度差",
    "商品质量问题",
    "虚假宣传",
    "退款长时间未到账",
    "账号安全问题",
    "优惠券/促销活动纠纷",
    "发货错误/漏发",
    "售后政策不合理",
    "平台规则不透明",
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
        business_type=params.get("business_type", "电商/退货"),
        emotion=params.get("emotion", "不满"),
        issue_category=params.get("issue_category", "物流延误"),
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
