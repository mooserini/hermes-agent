"""Phrasings and shape facts the auxiliary client used to match privately, now in the shared classifier.

Most rows classified differently (or not at all) before the auxiliary ladder's private predicates were
folded into ``classify_api_error``; the rest guard what must not change.
"""
import time

import httpx
import openai
import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.errors import InvalidProviderResponseError

_R = FailoverReason
_REQ = httpx.Request("POST", "https://example.invalid/v1/chat/completions")


class _Err(Exception):
    """A bare status-carrying error (``MockAPIError`` always sets a body, which the SSE fact reads)."""

    def __init__(self, msg, status=None, body=None):
        super().__init__(msg)
        if status is not None:
            self.status_code = status
        if body is not None:
            self.body = body


_ROWS = [  # (message, status, reason, rotates, error_context facts that must be set)
    # free tier, account walls and credit wording on the statuses that carry them
    ("This model isn't available on the free tier", 404, _R.billing, True, {}),
    ("Quota exceeded for this key", 403, _R.billing, True, {"quota_exhausted": True}),
    ("this model requires a subscription, upgrade for access: https://ollama.com/upgrade", 403, _R.billing, True, {}),
    ("RESOURCE_EXHAUSTED: project quota", 404, _R.billing, True, {}),
    ("Insufficient funds for this request", None, _R.billing, True, {}),
    ("This request requires more credits, but can only afford 7117", 402, _R.billing, True, {}),
    ("Budget limit exceeded (monthly limit)", None, _R.billing, True, {}),
    ("model_not_supported_on_free_tier", 400, _R.billing, True, {}),
    # a 429 that says "rate limit" and names a billing phrase stays a rate limit (main's rule)
    ("Rate limit exceeded: insufficient credits for burst", 429, _R.rate_limit, True, {}),
    # RESOURCE_EXHAUSTED: a rate limit unless a daily/weekly window is named (billing without a reset)
    ("RESOURCE_EXHAUSTED: Worker local total request limit reached (32/32)", 429, _R.rate_limit, True, {}),
    ("RESOURCE_EXHAUSTED: Worker local total request limit reached (32/32)", None, _R.rate_limit, True, {}),
    ("RESOURCE_EXHAUSTED: You have reached your daily limit.", 429, _R.billing, True, {"quota_exhausted": True}),
    ("You've hit your weekly usage limit.", None, _R.billing, True, {"quota_exhausted": True}),
    ("Too many tokens per day", 429, _R.billing, True, {"quota_exhausted": True}),
    ("Too many tokens per day, please wait before retrying", 429, _R.rate_limit, True, {"quota_exhausted": True}),
    # an exceeded quota with no named window stays a rate limit (DashScope's per-minute throttle)
    ("Quota exceeded for quota metric 'Generate Content API requests per day'", 429, _R.rate_limit, True,
     {"quota_exhausted": True}),
    ('event:error :HTTP_STATUS/429 data:{"code":"Throttling.AllocationQuota","message":"Allocated quota exceeded."}',
     429, _R.rate_limit, True, {"quota_exhausted": True}),
    # credentials
    ("unauthenticated:bad-credentials", 403, _R.auth, True, {}),
    ("Error code: 403 - unauthenticated:bad-credentials", None, _R.auth, True, {}),
    ("Error code: 401 - invalid key", None, _R.auth, True, {}),
    # models
    ("Model 'x' not found in our configuration or OpenRouter catalog.", 404, _R.model_not_found, False, {}),
    ("The model qwen3:8b is not supported when using this account", 400, _R.model_entitlement, True, {}),
    ("model kimi-k2-thinking is not supported when using this account", None, _R.model_entitlement, True, {}),
    ("This model is not supported for this account.", 400, _R.model_entitlement, True, {}),
    ("Error code: 400 - {'error': {'message': \"The 'kimi-k2' model is not supported for this account.\"}}",
     None, _R.model_entitlement, True, {}),
    # a feature or a parameter on the route: the main loop's verdict, never a rotation
    ("This model is not supported for image input", 400, _R.format_error, False, {"route_rejects_model": True}),
    ("Streaming is not supported for this account", 400, _R.format_error, False, {"route_rejects_model": True}),
    ("Streaming with this model is not supported for this account", None, _R.unknown, False,
     {"route_rejects_model": True}),
    ("Tool use is not supported with this model", 400, _R.format_error, False, {"route_rejects_model": True}),
    ("The model parameter temperature is not supported for this account", 400, _R.format_error, False, {}),
    ("Parameter reasoning_effort for model o3 is not supported for this account", None, _R.unknown, False, {}),
    ("The model tool_choice is not supported for this account", 400, _R.format_error, False, {}),
    ("Vision: model is not supported for this account", None, _R.unknown, False, {}),
    # transport
    ("RemoteProtocolError: response ended prematurely", None, _R.timeout, False, {"transport": "connection"}),
    ("[Errno 8] nodename nor servname provided, or not known", None, _R.timeout, False, {"transport": "connection"}),
]


@pytest.mark.parametrize("message,status,reason,rotates,facts", _ROWS, ids=[f"{r[1]}-{r[0][:48]}" for r in _ROWS])
def test_phrasing(message, status, reason, rotates, facts):
    verdict = classify_api_error(_Err(message, status), provider="openrouter")
    assert (verdict.reason, verdict.should_rotate_credential) == (reason, rotates)
    assert facts.items() <= verdict.error_context.items()
    if reason in (_R.billing, _R.rate_limit, _R.model_not_found, _R.model_entitlement):
        assert verdict.should_fallback is True


def test_a_hook_verdict_is_not_reinterpreted(monkeypatch):
    from hermes_cli import plugins

    monkeypatch.setattr(plugins, "get_plugin_error_classification",
                        lambda **_kw: {"reason": _R.rate_limit, "should_fallback": True})
    verdict = classify_api_error(_Err("You have reached your daily limit.", 429))
    assert verdict.reason is _R.rate_limit and "quota_exhausted" not in verdict.error_context


def test_transport_kind_and_statusless_structured_body():
    timed_out = classify_api_error(openai.APITimeoutError(request=_REQ))
    refused = classify_api_error(httpx.ConnectError("[Errno 61] Connection refused"))
    assert (timed_out.reason, timed_out.error_context["transport"]) == (_R.timeout, "timeout")
    assert (refused.reason, refused.error_context["transport"]) == (_R.timeout, "connection")
    assert "transport" not in classify_api_error(_Err("Bad request", 400)).error_context
    sse = openai.APIError("stream error", _REQ, body={"type": "server_error", "message": "upstream failed"})
    assert classify_api_error(sse).error_context["statusless_structured"] is True
    assert "statusless_structured" not in classify_api_error(_Err("plain failure")).error_context


class EndpointConnectionError(Exception):
    """botocore's class name, outside the builtin ConnectionError/OSError hierarchy."""


class AuthenticationError(Exception):
    """A provider SDK's auth class without a status attribute."""


def test_sdk_class_names_carry_the_status():
    assert classify_api_error(EndpointConnectionError("Could not connect to the endpoint URL")).reason is _R.timeout
    verdict = classify_api_error(AuthenticationError("token rejected"))
    assert (verdict.status_code, verdict.reason) == (401, _R.auth)


def test_invalid_provider_response_is_incomplete_response():
    verdict = classify_api_error(InvalidProviderResponseError("Auxiliary compression: LLM returned invalid response"))
    assert (verdict.reason, verdict.should_fallback) == (_R.incomplete_response, True)


@pytest.mark.parametrize("status,code,reason,rotate", [
    (503, "ServiceUnavailableException", _R.overloaded, False),
    (403, "AccessDeniedException", _R.auth, False),
    (429, "ThrottlingException", _R.rate_limit, True),
])
def test_botocore_status_is_read_from_the_response_metadata(status, code, reason, rotate):
    class ClientError(Exception):
        def __init__(self):
            super().__init__(f"An error occurred ({code}) when calling the Converse operation")
            self.response = {"ResponseMetadata": {"HTTPStatusCode": status}, "Error": {"Code": code}}

    verdict = classify_api_error(ClientError())
    assert (verdict.status_code, verdict.reason, verdict.should_rotate_credential) == (status, reason, rotate)


def test_status_carried_only_on_the_response_is_read():
    response = httpx.Response(429, request=_REQ)
    verdict = classify_api_error(httpx.HTTPStatusError("Too Many Requests", request=_REQ, response=response))
    assert (verdict.status_code, verdict.reason) == (429, _R.rate_limit)


def _repeat(unit, size=200_000):
    return (unit * (size // len(unit) + 1))[:size]


_HOSTILE_BODIES = {
    "gating-tails": _repeat("the model gpt-4.1 is not supported for this account "),
    "id-separators": _repeat("model " + "a-" * 64 + ": " + "/" * 129 + " is not supported for this account "),
    "route-wording": _repeat("tool use is not supported with this model "),
    "walls": _repeat("quota daily limit credits budget limit requires a subscription "),
}


@pytest.mark.parametrize("status", [400, None], ids=["400", "statusless"])
@pytest.mark.parametrize("body", list(_HOSTILE_BODIES.values()), ids=list(_HOSTILE_BODIES))
def test_hostile_200kb_bodies_classify_quickly(body, status):
    classify_api_error(_Err("warm up", 400))
    started = time.perf_counter()
    classify_api_error(_Err(body, status), provider="openrouter", model="gpt-4.1")
    assert time.perf_counter() - started < 0.5
