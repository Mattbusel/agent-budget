"""``agent-budget audit``: the replay and the command line."""

import json
import subprocess
import sys

import llm_prices
import pytest

from agent_budget.audit import AuditInputError, main, parse_jsonl, replay
from agent_budget.pricing import Pricer

UNDER_BOOKED = json.dumps(
    {"model": "gpt-5.5", "usage": {"prompt_tokens": 100_000, "completion_tokens": 5_000}, "booked_usd": 0.115}
)


def test_soft_limit_lets_the_last_call_overshoot():
    line = json.dumps({"model": "gpt-4o", "usage": {"prompt_tokens": 100_000, "completion_tokens": 2_000}})
    r = replay(parse_jsonl("\n".join([line] * 3)), 0.50)
    assert r.soft.calls_through == 2
    assert r.soft.fired_at_call == 3
    assert r.soft.crossed_cap_at_call == 2
    assert r.soft.money_past_cap_usd == pytest.approx(0.04)
    assert r.hard.admitted == 1 and r.hard.refused_over_cap == 2
    assert r.leaked


def test_under_booking_tracker_runs_far_past_the_cap():
    # The PraisonAI pattern: gpt-5.5 booked at a $1/$3 default.
    r = replay(parse_jsonl("\n".join([UNDER_BOOKED] * 20)), 1.0)
    truth = llm_prices.lookup("gpt-5.5").cost(100_000, 5_000)
    assert truth > 4 * 0.115
    assert r.soft.fired_at_call == 10
    assert r.soft.money_past_cap_usd > 2.0
    assert r.drift[0].ratio < 0.25
    assert r.hard.spent_usd <= 1.0
    assert "MONEY PAST THE CAP" in str(r)


def test_unpriced_cache_and_stale_calls_are_named():
    no_cache = llm_prices.Price(provider="self", model="no-cache", input=1.0, output=1.0, checked="2026-11-19")
    old = llm_prices.Price(provider="self", model="old", input=1.0, output=1.0, checked="2020-01-01")
    pricer = Pricer(prices={"no-cache": no_cache, "old": old}, today="2026-11-19")
    calls = parse_jsonl("\n".join(json.dumps(x) for x in [
        {"model": "ghost-model", "input_tokens": 10, "output_tokens": 10},
        {"model": "no-cache", "input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 900},
        {"model": "no-cache", "input_tokens": 1_000, "output_tokens": 10},
        {"model": "old", "input_tokens": 1_000, "output_tokens": 10},
    ]))
    r = replay(calls, 10.0, pricer=pricer)
    assert (r.hard.refused_unpriced, r.hard.refused_cache_price, r.hard.refused_stale, r.hard.admitted) == (1, 1, 1, 1)
    assert r.soft.unpriced_calls_through == 2
    assert r.stale == {"old": "2020-01-01"}
    assert {u.model for u in r.unpriced} == {"ghost-model", "no-cache"}


def test_bad_input_is_named():
    with pytest.raises(AuditInputError, match="line 2"):
        parse_jsonl('{"model":"gpt-4o","usage":{"prompt_tokens":1,"completion_tokens":1}}\noops')
    with pytest.raises(AuditInputError, match='"model"'):
        parse_jsonl('{"usage":{}}')
    with pytest.raises(ValueError, match="cap"):
        replay([], 0)


def test_cli_exit_codes_and_json(tmp_path, capsys):
    log = tmp_path / "calls.jsonl"
    log.write_text("\n".join([UNDER_BOOKED] * 12), encoding="utf-8")
    assert main(["audit", str(log), "--cap", "1"]) == 2
    assert "stopped the run at call 10" in capsys.readouterr().out
    assert main(["audit", str(log), "--cap", "1", "--json"]) == 2
    v = json.loads(capsys.readouterr().out)
    assert v["soft"]["calls_through"] == 9 and v["hard"]["spent_usd"] <= 1.0
    clean = tmp_path / "clean.jsonl"
    clean.write_text('{"model":"gpt-4o","usage":{"prompt_tokens":1000,"completion_tokens":100}}\n', encoding="utf-8")
    assert main(["audit", str(clean), "--cap", "$5"]) == 0
    assert main(["audit", str(clean)]) == 1  # no cap, no audit
    assert main(["audit", str(clean), "--cap", "0"]) == 1
    assert main(["audit", str(tmp_path / "missing.jsonl"), "--cap", "1"]) == 1


def test_python_dash_m_runs_the_cli(tmp_path):
    log = tmp_path / "calls.jsonl"
    log.write_text(UNDER_BOOKED + "\n", encoding="utf-8")
    out = subprocess.run([sys.executable, "-m", "agent_budget", "audit", str(log), "--cap", "0.01"],
                         capture_output=True, text=True)
    assert out.returncode == 2, out.stderr
    assert "MONEY PAST THE CAP" in out.stdout
