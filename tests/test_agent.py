"""Coach agent loop, tools, memory and auth — the LLM is a scripted fake."""

import json
import time

import pytest

from core import agent, auth
from core.agent import ToolContext, check_citations, coach_reply, run_tool
from core.kb import KnowledgeBase
from core.llm import ChatResult
from core.memory import Memory

DOC_URL = "https://help.aliyun.com/zh/ecs/port"


def make_kb():
    chunks = [
        {"text": "端口不通排查 > 场景一\n检查安全组入方向规则是否放行 22 端口", "title": "端口不通排查",
         "heading": "场景一", "url": DOC_URL, "business_type": "云服务器ECS/实例连接"},
        {"text": "退订规则\n五天无理由退款每年一次", "title": "退订规则", "heading": "",
         "url": "https://help.aliyun.com/zh/refund", "business_type": "退款与代金券/费用纠纷"},
    ]
    return KnowledgeBase(chunks)


LIBRARY = [
    {"script": {"title": "ECS 连不上的愤怒客户", "challenge_points": ["情绪激动"]},
     "ticket": {"business_type": "云服务器ECS/实例连接"}, "params": {"difficulty": "高"}},
    {"script": {"title": "退款纠纷", "challenge_points": []},
     "ticket": {"business_type": "退款与代金券/费用纠纷"}, "params": {"difficulty": "中"}},
]


@pytest.fixture
def ctx():
    return ToolContext(trainee_id="t_me", kb=make_kb(), memory=Memory(":memory:"),
                       library=LIBRARY, business_types=["云服务器ECS/实例连接"])


def tool_call(name, args, i=0):
    return {"id": f"call_{name}_{i}", "name": name, "arguments": json.dumps(args, ensure_ascii=False)}


class Script:
    """Fake chat_fn: returns queued responses; records the messages it saw."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.seen = []

    def __call__(self, messages, **kw):
        self.seen.append({"messages": [dict(m) for m in messages], "tools": kw.get("tools")})
        step = self.steps.pop(0)
        if isinstance(step, str):
            return ChatResult(content=step, model="fake", latency_ms=1, prompt_tokens=10, completion_tokens=5,
                              message={"role": "assistant", "content": step})
        calls = step
        return ChatResult(content="", model="fake", latency_ms=1, prompt_tokens=10, completion_tokens=5,
                          tool_calls=calls,
                          message={"role": "assistant", "content": "", "tool_calls": [
                              {"id": c["id"], "type": "function",
                               "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})


# -- loop -----------------------------------------------------------------------

def test_search_then_cited_answer(ctx):
    fake = Script([tool_call("search_help_docs", {"query": "SSH 端口 不通"})],
                  "先查安全组是否放行 22 端口 [D1]。")
    out = coach_reply(ctx, "客户 ssh 连不上怎么办", chat_fn=fake)
    assert out["answer"].endswith("[D1]。")
    assert out["sources"][0]["url"] == DOC_URL
    assert out["tool_trace"][0]["tool"] == "search_help_docs" and out["tool_trace"][0]["ok"]
    # the tool result was fed back as a tool message tied to the call id
    tool_msg = fake.seen[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_search_help_docs_0"
    assert out["usage"]["llm_calls"] == 2


def test_hallucinated_citation_is_stripped(ctx):
    fake = Script([tool_call("search_help_docs", {"query": "安全组"})], "放行端口 [D1]，另见 [D7]。")
    out = coach_reply(ctx, "安全组", chat_fn=fake)
    assert "[D7]" not in out["answer"] and out["invalid_citations"] == ["D7"]


def test_step_budget_forces_final_answer_without_tools(ctx, monkeypatch):
    monkeypatch.setattr(agent, "MAX_STEPS", 2)
    fake = Script([tool_call("search_help_docs", {"query": "a"}, 0)],
                  [tool_call("search_help_docs", {"query": "b"}, 1)],
                  "基于已有信息的回答")
    out = coach_reply(ctx, "问题", chat_fn=fake)
    assert out["stopped"] == "step_budget" and out["answer"] == "基于已有信息的回答"
    assert fake.seen[-1]["tools"] is None


def test_bad_tool_args_become_error_result(ctx):
    res = run_tool(ctx, "search_help_docs", '{"query": ""}')
    assert "error" in res
    assert "error" in run_tool(ctx, "nope", "{}")
    assert "error" in run_tool(ctx, "remember", "not json")


def test_tool_crash_is_contained(ctx):
    ctx.memory.add_note = lambda *a: (_ for _ in ()).throw(RuntimeError("db locked"))
    res = run_tool(ctx, "remember", json.dumps({"note": "下周考核"}))
    assert res["error"].startswith("工具执行失败")


# -- tools ----------------------------------------------------------------------

def test_read_doc_only_after_search_and_only_allowlisted(ctx):
    assert "error" in run_tool(ctx, "read_help_doc", json.dumps({"url": "https://evil.example.com/x"}))
    assert "error" in run_tool(ctx, "read_help_doc", json.dumps({"url": DOC_URL}))  # not searched yet
    run_tool(ctx, "search_help_docs", json.dumps({"query": "安全组 端口"}))
    ctx.fetch = lambda url: '<h1>端口不通排查</h1><div class="markdown-body"><h2>步骤</h2><p>检查安全组</p></div>'
    agent._doc_cache.clear()
    res = run_tool(ctx, "read_help_doc", json.dumps({"url": DOC_URL}))
    assert "检查安全组" in res["content"] and res["ref"].startswith("D")


def test_profile_is_scoped_to_token_identity(ctx):
    ctx.memory.record_session("t_other", {"title": "别人的剧本"}, {"overall_score": 99, "dimensions": []})
    # the model tries to pass someone else's id — the tool ignores arguments
    res = run_tool(ctx, "get_trainee_profile", json.dumps({"trainee_id": "t_other"}))
    assert res["sessions_count"] == 0


def test_find_scripts_filters_and_records_for_ui(ctx):
    res = run_tool(ctx, "find_practice_scripts", json.dumps({"business_type": "云服务器ECS/实例连接"}))
    assert [s["script_id"] for s in res["scripts"]] == [0]
    assert ctx.scripts_shown[0]["title"] == "ECS 连不上的愤怒客户"


def test_remembered_note_is_injected_next_turn(ctx):
    coach_reply(ctx, "我下周要考核账单类", chat_fn=Script([tool_call("remember", {"note": "下周考核账单类工单"})], "好的"))
    fake = Script("你好")
    coach_reply(ctx, "嗨", chat_fn=fake)
    assert "下周考核账单类工单" in fake.seen[0]["messages"][0]["content"]


def test_check_citations_dedupes():
    _, cited, invalid = check_citations("a [D1] b [D1] c [D2]", {"D1": {}, "D2": {}})
    assert cited == ["D1", "D2"] and invalid == []


# -- memory ---------------------------------------------------------------------

def ev(score, dims, missed=()):
    return {"overall_score": score, "dimensions": [{"name": n, "score": s} for n, s in dims],
            "missed_challenge_points": list(missed)}


def test_profile_aggregates():
    m = Memory(":memory:")
    m.record_session("t", {"title": "a"}, ev(40, [("共情", 3), ("专业", 7)], ["未安抚情绪"]), "ECS")
    m.record_session("t", {"title": "b"}, ev(50, [("共情", 4), ("专业", 8)], ["未安抚情绪"]), "ECS")
    m.record_session("t", {"title": "c"}, ev(80, [("共情", 7), ("专业", 9)]), "账单")
    m.record_session("t", {"title": "d"}, ev(90, [("共情", 8), ("专业", 9)]), "账单")
    p = m.profile("t", ["ECS", "账单", "CDN"])
    assert p["sessions_count"] == 4 and p["avg_score"] == 65.0
    assert p["weakest_dimensions"][0]["name"] == "共情"
    assert p["recurring_missed_points"] == ["未安抚情绪"]
    assert p["trend_vs_earlier"] == 40.0          # (90+80)/2 - (50+40)/2
    assert p["never_practiced"] == ["CDN"]


def test_notes_are_bounded_and_forget_wipes():
    m = Memory(":memory:")
    for i in range(60):
        m.add_note("t", f"note {i}")
    assert len(m.notes("t", limit=100)) == 50
    assert m.notes("t", limit=1)[0]["text"] == "note 59"
    m.forget("t")
    assert m.notes("t") == [] and m.profile("t")["sessions_count"] == 0


# -- auth -----------------------------------------------------------------------

def test_token_roundtrip_and_tamper(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET", "s3cret")
    tid = auth.trainee_id_for("  张三 ")
    assert tid == auth.trainee_id_for("张三")
    tok = auth.issue_token(tid)
    assert auth.verify_token(tok) == tid
    forged = tok.replace(tid, "t_0000000000")
    assert auth.verify_token(forged) is None
    assert auth.verify_token("garbage") is None
    monkeypatch.setenv("AUTH_SECRET", "rotated")
    assert auth.verify_token(tok) is None


def test_token_expiry(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET", "s")
    tok = auth.issue_token("t_x", ttl_s=-1)
    assert auth.verify_token(tok) is None
