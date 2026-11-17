"""The golden fixtures shared with llm-budget (Rust): both libraries must
produce and accept exactly the same JSON and baggage."""

import json
from pathlib import Path

import pytest

from agent_budget import (
    EXTENSION_URI,
    BudgetContext,
    ContextError,
    from_a2a_metadata,
    from_baggage,
    to_a2a_metadata,
    to_baggage,
)

FIXTURES = Path(__file__).parent / "fixtures" / "budget_context"


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


ROUNDTRIP = load("roundtrip.json")["cases"]
NORMALIZE = load("normalize.json")
INVALID = load("invalid.json")


@pytest.mark.parametrize("case", ROUNDTRIP, ids=[c["name"] for c in ROUNDTRIP])
def test_roundtrip(case):
    ctx = BudgetContext.from_dict(case["context"])
    assert ctx.to_dict() == case["context"]
    assert to_baggage(ctx) == case["baggage"]
    back = from_baggage(case["baggage"])
    assert (back.to_dict() if back else None) == case["baggage_context"]
    md = to_a2a_metadata(ctx)
    assert md[EXTENSION_URI] == case["context"]
    assert from_a2a_metadata(md) == ctx


@pytest.mark.parametrize("case", NORMALIZE["json"], ids=[c["name"] for c in NORMALIZE["json"]])
def test_normalize_json(case):
    assert BudgetContext.from_dict(case["input"]).to_dict() == case["context"]


@pytest.mark.parametrize("case", NORMALIZE["baggage"], ids=[c["name"] for c in NORMALIZE["baggage"]])
def test_normalize_baggage(case):
    got = from_baggage(case["input"])
    assert (got.to_dict() if got else None) == case["context"]


@pytest.mark.parametrize("payload", INVALID["json"], ids=[json.dumps(p)[:60] for p in INVALID["json"]])
def test_invalid_json(payload):
    with pytest.raises(ContextError):
        BudgetContext.from_dict(payload)
    with pytest.raises(ContextError):
        from_a2a_metadata({EXTENSION_URI: payload})


@pytest.mark.parametrize("header", INVALID["baggage"])
def test_invalid_baggage(header):
    with pytest.raises(ContextError):
        from_baggage(header)


def test_fixtures_match_the_rust_crate_copy():
    """When run inside a checkout next to llm-budget, the two copies of the
    fixtures must be byte-identical."""
    rust = Path(__file__).resolve().parents[2] / "llm-budget" / "tests" / "fixtures" / "budget_context"
    if not rust.is_dir():
        pytest.skip("llm-budget checkout not next to this one")
    for f in FIXTURES.iterdir():
        assert f.read_bytes() == (rust / f.name).read_bytes(), f.name
