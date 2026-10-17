"""Schema validation + self-repair in call_llm (LLM replaced by a script)."""

import json

import pytest

from core import pipeline
from core.schemas import Review, Ticket

GOOD_REVIEW = {"overall_score": 82, "dimensions": [{"name": "真实性", "score": 8, "comment": "ok"}],
               "issues": [], "suggestions": []}


def scripted(monkeypatch, *outputs):
    """Make _raw_chat return the given outputs in order; capture messages."""
    seen = []
    queue = list(outputs)

    def fake(messages, json_mode=False, temperature=0.8):
        seen.append(messages)
        return queue.pop(0)

    monkeypatch.setattr(pipeline, "_raw_chat", fake)
    return seen


def test_valid_output_passes_first_try(monkeypatch):
    seen = scripted(monkeypatch, json.dumps(GOOD_REVIEW))
    assert pipeline.call_llm("sys", "user", schema=Review)["overall_score"] == 82
    assert len(seen) == 1


def test_out_of_range_score_is_repaired_with_feedback(monkeypatch):
    bad = dict(GOOD_REVIEW, overall_score=180)
    seen = scripted(monkeypatch, json.dumps(bad), json.dumps(GOOD_REVIEW))
    assert pipeline.call_llm("sys", "user", schema=Review)["overall_score"] == 82
    repair_prompt = seen[1][-1]["content"]
    assert "overall_score" in repair_prompt and "100" in repair_prompt
    assert seen[1][-2] == {"role": "assistant", "content": json.dumps(bad)}


def test_markdown_fenced_json_is_accepted(monkeypatch):
    scripted(monkeypatch, "```json\n" + json.dumps(GOOD_REVIEW) + "\n```")
    assert pipeline.call_llm("sys", "user", schema=Review)["dimensions"][0]["score"] == 8


def test_gives_up_after_second_bad_output(monkeypatch):
    scripted(monkeypatch, "not json", '{"overall_score": 50}')
    with pytest.raises(ValueError, match="dimensions"):
        pipeline.call_llm("sys", "user", schema=Review)


def test_ticket_roles_are_normalized():
    t = Ticket.model_validate({"dialogue": [{"role": "客户", "text": "连不上"}, {"role": "Agent", "text": "您好"}]})
    assert [d.role for d in t.dialogue] == ["customer", "agent"]


def test_ticket_without_dialogue_is_rejected():
    with pytest.raises(Exception):
        Ticket.model_validate({"ticket_id": "TK-1", "dialogue": []})


def test_unknown_fields_are_kept():
    r = Review.model_validate(dict(GOOD_REVIEW, extra_note="kept"))
    assert r.model_dump()["extra_note"] == "kept"
