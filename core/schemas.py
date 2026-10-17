"""
Output contracts for every JSON-producing LLM step.

json_mode only guarantees *syntactically* valid JSON. These models guarantee
the *shape*: required fields, allowed enum values, score ranges. When the
model breaks the contract, call_llm feeds the validation errors back and asks
it to repair its own output (see core.pipeline.call_llm).

extra="allow": unknown fields are kept, so prompt tweaks that add a field
never break the API.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


def _round_score(v):
    # "85" / 85.4 -> 85; anything non-numeric is left for the validator to reject
    try:
        return round(float(v))
    except (TypeError, ValueError):
        return v


class Turn(_Loose):
    role: Literal["customer", "agent"]
    text: str

    @field_validator("role", mode="before")
    @classmethod
    def _norm_role(cls, v):
        # Models sometimes write "Customer" / "客户" / "客服"
        mapping = {"客户": "customer", "顾客": "customer", "客服": "agent"}
        v = str(v).strip()
        return mapping.get(v, v.lower())


class Ticket(_Loose):
    ticket_id: str = ""
    business_type: str = ""
    customer_profile: str = ""
    channel: str = ""
    dialogue: list[Turn] = Field(min_length=1)
    resolution: str = ""
    tags: list[str] = []


class Criterion(_Loose):
    dimension: str
    description: str


class Script(_Loose):
    title: str
    customer_persona: str
    scenario: str
    emotion_arc: str
    challenge_points: list[str] = Field(min_length=1)
    standard_response: str
    scoring_criteria: list[Criterion] = Field(min_length=1)
    actor_prompt: str = Field(min_length=20)


class DimensionScore(_Loose):
    name: str
    score: int = Field(ge=0, le=10)
    comment: str = ""

    _score = field_validator("score", mode="before")(_round_score)


class Review(_Loose):
    _score = field_validator("overall_score", mode="before")(_round_score)
    overall_score: int = Field(ge=0, le=100)
    dimensions: list[DimensionScore] = Field(min_length=1)
    issues: list[str] = []
    suggestions: list[str] = []


class SessionEvaluation(_Loose):
    _score = field_validator("overall_score", mode="before")(_round_score)
    overall_score: int = Field(ge=0, le=100)
    dimensions: list[DimensionScore] = Field(min_length=1)
    highlights: list[str] = []
    improvements: list[str] = []
    missed_challenge_points: list[str] = []


class NormalizedTicket(Ticket):
    # A real ticket may legitimately have no parsable dialogue; don't force
    # the model to invent one (the normalize prompt forbids fabrication).
    dialogue: list[Turn] = []


def format_errors(err) -> str:
    """Compact, model-readable list of validation errors."""
    lines = []
    for e in err.errors()[:10]:
        loc = ".".join(str(p) for p in e["loc"]) or "(root)"
        lines.append(f"- {loc}: {e['msg']}")
    return "\n".join(lines)
