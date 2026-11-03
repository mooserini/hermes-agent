"""Frozen parity table: the auxiliary recovery ladder's decisions on a corpus of real error shapes.

The corpus is every error object the auxiliary client's former private predicates and ``classify_api_error``
saw across the test suite (harvested live), a synthetic set covering each HTTP status and transport shape,
and one row per class-name or wording rule of the removed predicates (an SDK class NAME such as botocore's
``EndpointConnectionError`` outside the builtin hierarchy, or a needle in a realistic body, on each status the
rule was gated on). Each row is recorded as (class + MRO, status, message, body, route provider) and pins the
decisions the ladder takes from the shared classifier, so a classifier change that moves an aux decision fails
here and names the row. A predicate row's ``why`` says whether the decision is base's (``base``) or a named,
accepted change.

After an intended classifier change, rewrite the expectations from the current code and review the diff row
by row::

    python -m tests.agent.test_auxiliary_error_policy_parity --regenerate
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import httpx
import openai
import pytest

from agent import auxiliary_error_policy as aux_errors
from agent.auxiliary_client import _classify_failure
from agent.auxiliary_stream_watchdog import AuxNoProgressTimeout
from agent.errors import InvalidProviderResponseError

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "aux_error_policy_parity.json"

DECISIONS = (
    "same_provider_retry", "auth_refresh", "pool_rotation", "billing_unhealthy_mark",
    "provider_fallback", "nous_model_heal", "skip_retry_compression", "evict_client",
)
_CITATIONS = {"base", "approved", "ruling", "status decides", "by construction"}

_REQ = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
_TYPED = {cls.__name__: cls for cls in (AuxNoProgressTimeout, InvalidProviderResponseError)}
_SDK_BY_STATUS = {
    400: openai.BadRequestError, 401: openai.AuthenticationError, 403: openai.PermissionDeniedError,
    404: openai.NotFoundError, 409: openai.ConflictError, 422: openai.UnprocessableEntityError,
    429: openai.RateLimitError,
}
_BASES = (("TimeoutError", TimeoutError), ("ConnectionError", ConnectionError), ("RuntimeError", RuntimeError),
          ("OSError", OSError), ("ValueError", ValueError))


def _body(text):
    if text in (None, "None", ""):
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def reconstruct(spec):
    """Rebuild a corpus error from its recorded shape (class name + MRO, status, message, body)."""
    name, mro, status, msg, body = spec["cls"], spec["mro"], spec["status"], spec["msg"], _body(spec["body"])
    if name in _TYPED:
        return _TYPED[name](msg)
    if "APITimeoutError" in mro:
        return openai.APITimeoutError(request=_REQ)
    if "APIConnectionError" in mro:
        return openai.APIConnectionError(message=msg, request=_REQ)
    if "APIStatusError" in mro and isinstance(status, int):
        cls = _SDK_BY_STATUS.get(status, openai.InternalServerError if status >= 500 else openai.APIStatusError)
        return cls(msg, response=httpx.Response(status, request=_REQ), body=body)
    if "APIError" in mro and status is None:
        return openai.APIError(msg, _REQ, body=body)
    base = next((b for tag, b in _BASES if tag in mro), Exception)
    exc = type(name, (base,), {})(msg)
    if status is not None:
        exc.status_code = status
    if body is not None:
        exc.body = body
    if spec.get("resp_status") is not None:
        exc.response = type("Response", (), {"status_code": spec["resp_status"], "headers": {}})()
    return exc


def decisions(exc, provider):
    """Every aux recovery decision for ``exc`` as the ladder takes it on a ``provider`` route: the failure is
    classified once, through the same seam the ladder uses, and every decision reads that verdict."""
    failure = _classify_failure(exc, provider, None, None, None)
    verdict = failure.verdict
    label = aux_errors.fallback_label(failure)
    return {
        "same_provider_retry": aux_errors.same_provider_retry(verdict),
        "auth_refresh": aux_errors.auth_refresh(verdict),
        "pool_rotation": verdict.should_rotate_credential,
        "billing_unhealthy_mark": aux_errors.benches_provider(verdict),
        "provider_fallback": label is not None,
        "nous_model_heal": verdict.reason.value == "model_not_found",
        "skip_retry_compression": aux_errors.skip_same_provider_retry("compression", failure),
        "evict_client": bool(verdict.error_context.get("transport")),
        "label": label,
        "reason": verdict.reason.value,
        "transport": verdict.error_context.get("transport"),
    }


ROWS = json.loads(FIXTURE.read_text(encoding="utf-8"))


def compact(decided):
    """The fixture's form: the decisions that fire, plus the fallback label, transport kind and reason."""
    return {"on": [d for d in DECISIONS if decided[d]], "label": decided["label"],
            "transport": decided["transport"], "reason": decided["reason"]}


@pytest.fixture(autouse=True)
def _no_plugin_classifiers(monkeypatch):
    """The built-in classifier only: plugin discovery would otherwise rerun for every row."""
    from hermes_cli import plugins

    monkeypatch.setattr(plugins, "get_plugin_error_classification", lambda **_kw: None)


def test_corpus_covers_every_decision_both_ways_and_cites_every_change():
    for decision in DECISIONS:
        fired = sum(decision in row["expected"]["on"] for row in ROWS)
        assert 0 < fired < len(ROWS), decision
    assert all(row["why"].split(":")[0] in _CITATIONS for row in ROWS if "why" in row)
    shapes = [json.dumps([row[k] for k in ("cls", "mro", "status", "resp_status", "msg", "body", "provider")])
              for row in ROWS]
    assert len(set(shapes)) == len(shapes), "duplicate fixture rows (same exception twice)"


@pytest.mark.parametrize("row", ROWS, ids=[f"{i}-{r['cls']}-{r['status']}" for i, r in enumerate(ROWS)])
def test_policy_decision_parity(row):
    assert compact(decisions(reconstruct(row), row["provider"])) == row["expected"], row.get("why")


def _regenerate() -> None:
    from unittest.mock import patch

    with patch("hermes_cli.plugins.get_plugin_error_classification", lambda **_kw: None):
        rows = [{**row, "expected": compact(decisions(reconstruct(row), row["provider"]))} for row in ROWS]
    FIXTURE.write_text("[\n" + ",\n".join(json.dumps(row) for row in rows) + "\n]\n", encoding="utf-8")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--regenerate"]:
        _regenerate()
