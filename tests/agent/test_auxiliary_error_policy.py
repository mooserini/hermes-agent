"""Auxiliary recovery decisions come from the shared classifier.

The aux ladder used to decide retry/fallback/refresh with private substring predicates and consulted
``classify_api_error`` only for telemetry, so a plugin or provider-profile classifier never steered an
aux decision and the two drifted. These tests drive ``call_llm`` end to end with a scripted client;
each ``behaviour change`` case fails on the predicate implementation it replaced.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from agent import auxiliary_error_policy as aux_errors
from agent.auxiliary_client import async_call_llm, call_llm
from agent.error_classifier import ClassifiedError, FailoverReason
from agent.errors import InvalidProviderResponseError
from tests.agent.conftest import sdk_status_error as sdk

_REQ = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
_MESSAGES = [{"role": "user", "content": "hi"}]


class _Plain(Exception):
    pass


def _ok(text="fallback"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _failure(error, provider="openrouter"):
    return aux_errors.Failure(error, aux_errors.classify_attempt(error, provider))


def run_ladder(error, *, provider="openrouter", task=None, retries=1, client=None, resolved_key=None,
               is_async=False, **call_kw):
    """``call_llm`` (``async_call_llm`` when ``is_async``) on a client whose every request raises ``error``
    (a list: one outcome per request); returns what the ladder did."""
    create = AsyncMock if is_async else MagicMock
    primary = MagicMock()
    primary.base_url = "https://openrouter.ai/api/v1"
    primary.api_key = "sk-primary"
    for name, value in (client or {}).items():
        setattr(primary, name, value)
    primary.chat.completions.create = create(side_effect=error)
    fallback = MagicMock()
    fallback.base_url = "https://api.example.com/v1"
    fallback.chat.completions.create = create(return_value=_ok())
    marks, rotations, refreshes = [], [], []

    def _recover(pool_provider, exc, **_kw):
        rotations.append(exc)
        return False

    resolved = (provider, "some-model", None, resolved_key, None)
    with (
        patch("agent.auxiliary_client._resolve_task_provider_model", return_value=resolved),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "some-model")),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp),
        patch("agent.auxiliary_client._transient_retry_count", return_value=retries),
        patch("agent.auxiliary_client._TRANSIENT_RETRY_BACKOFF_BASE", 0.0),
        patch("agent.auxiliary_client._recoverable_pool_provider", return_value="openrouter"),
        patch("agent.auxiliary_client._recover_provider_pool", side_effect=_recover),
        patch("agent.auxiliary_client._mark_provider_unhealthy", side_effect=lambda *a, **kw: marks.append(a)),
        patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_main_fallback_chain", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_payment_fallback", return_value=(fallback, "fb-model", "fallback")),
        patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(fallback, "fb-model", "fallback")),
        patch("agent.auxiliary_rate_limit_backoff.rate_limit_wait", return_value=0.0),
        patch("agent.auxiliary_client._refresh_provider_credentials",
              side_effect=lambda slug, **_kw: refreshes.append(slug) or True),
    ):
        try:
            if is_async:
                result = asyncio.run(async_call_llm(task=task, messages=_MESSAGES, **call_kw))
            else:
                result = call_llm(task=task, messages=_MESSAGES, **call_kw)
        except Exception as exc:  # the outcome under test
            result = exc
    return SimpleNamespace(
        attempts=primary.chat.completions.create.call_count,
        fell_back=fallback.chat.completions.create.call_count > 0,
        unhealthy=bool(marks), rotated=bool(rotations), refreshed=refreshes, result=result)


# ── accepted behaviour changes (each fails on the replaced predicates) ──────
BEHAVIOUR_CHANGES = [  # (case id, error, route provider, expected outcome)
    # DNS / refused connections are transient: same-provider retry, then fallback.
    ("dns-refused", _Plain("[Errno 8] nodename nor servname provided, or not known"), "openrouter",
     dict(attempts=2, fell_back=True)),
    # 5xx, 503/529 overload and 408 fall back once the retries are spent.
    ("http-500", sdk(500, "Internal Server Error"), "openrouter", dict(attempts=2, fell_back=True)),
    ("http-503", sdk(503, "Service Unavailable"), "openrouter", dict(attempts=2, fell_back=True)),
    ("http-408", sdk(408, "Request Timeout"), "openrouter", dict(attempts=2, fell_back=True)),
    # a 400-coded billing body benches the provider and falls back.
    ("400-billing-body", sdk(400, "Insufficient credits. Add more using https://openrouter.ai/credits"), "openrouter",
     dict(attempts=1, fell_back=True, unhealthy=True, rotated=True)),
    # a missing model falls back on every route, not only Nous; so does a free-tier 404.
    ("404-model-not-found", sdk(404, "model_not_found: The model 'x' does not exist"), "openrouter",
     dict(attempts=1, fell_back=True)),
    ("404-no-tool-endpoints", sdk(404, "No endpoints found that support tool use."), "openrouter",
     dict(attempts=1, fell_back=True)),
    ("404-free-tier-model", sdk(404, "model not found (free tier)"), "openrouter", dict(attempts=1, fell_back=True)),
    # a generic 403 on an auto route falls back (no credential refresh).
    ("403-generic-auto", sdk(403, "Forbidden"), "auto", dict(attempts=1, fell_back=True, rotated=False)),
    # a 403 upstream_unavailable is an overload: retried, then fallback.
    ("403-upstream-unavailable", sdk(403, "upstream down", code="upstream_unavailable"), "openrouter",
     dict(attempts=2, fell_back=True)),
    # a structured usage-limit code on a 429 is billing: the provider is benched.
    ("429-usage-limit-code", sdk(429, "usage limit reached", code="usage_limit_reached"), "openrouter",
     dict(fell_back=True, unhealthy=True)),
    # a 429 that says rate limit and names a billing phrase stays a rate limit: rotated, not benched.
    ("429-rate-limit-with-billing-words", sdk(429, "Rate limit exceeded: insufficient credits for burst"), "openrouter",
     dict(fell_back=True, rotated=True, unhealthy=False)),
    # a 400-coded rate limit rotates the pool and falls back.
    ("400-rate-limit", sdk(400, "Rate limit reached for requests"), "openrouter", dict(fell_back=True, rotated=True)),
    # a 429 'temporarily overloaded' backs off on the same provider; the key is never rotated.
    ("429-overloaded", sdk(429, "The service may be temporarily overloaded, please try again later"), "openrouter",
     dict(attempts=2, fell_back=True, rotated=False)),
    # safety, WAF, data-policy and certificate failures fall back without retrying.
    ("400-data-inspection", sdk(400, "Input data may contain inappropriate content.", type="data_inspection_failed"),
     "openrouter", dict(attempts=1, fell_back=True)),
    ("403-waf", sdk(403, "Attention Required! | Cloudflare — Sorry, you have been blocked"), "openrouter",
     dict(attempts=1, fell_back=True)),
    ("ssl-cert", _Plain("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"), "openrouter",
     dict(attempts=1, fell_back=True)),
    # the Codex account entitlement 400 and account gating of the model rotate before falling back.
    ("codex-entitlement", sdk(400, "The 'claude-sonnet' model is not supported when using Codex with a ChatGPT account."),
     "openrouter", dict(attempts=1, fell_back=True, rotated=True)),
    ("account-gated-model", sdk(400, "The model qwen3:8b is not supported when using this account"), "openrouter",
     dict(attempts=1, fell_back=True, rotated=True)),
    # a server-injected prompt_cache_retention 400 is a server flake: retried.
    ("400-server-injected-param", sdk(400, "prompt_cache_retention is not supported on this model"), "openrouter",
     dict(attempts=2, fell_back=True)),
    # a status-less in-stream rate limit is a rate limit: the pool rotates.
    ("statusless-rate-limit", openai.APIError("Rate limit exceeded", _REQ, body={"code": "rate_limit_exceeded",
                                                                                "message": "slow"}),
     "openrouter", dict(fell_back=True, rotated=True)),
    # a request-validation 5xx is the request's fault: no pointless same-provider retry.
    ("5xx-request-validation", sdk(502, "Unknown parameter: 'frequency_penalty'"), "openrouter",
     dict(attempts=1, fell_back=False)),
    # an OpenRouter-wrapped upstream 429 leaves a healthy key alone and falls back.
    ("openrouter-upstream-429", sdk(429, "Provider returned error", wrapped=True, code=429,
                                    metadata={"raw": "x is rate-limited upstream", "provider_name": "DeepInfra"}),
     "openrouter", dict(attempts=1, fell_back=True, rotated=False)),
    # Codex "Failed to extract accountId from token" is a credential failure: the pool acts on it.
    ("codex-account-id", _Plain("Failed to extract accountId from token"), "auto", dict(fell_back=True, rotated=True)),
]


@pytest.mark.parametrize("case,error,provider,expected", BEHAVIOUR_CHANGES, ids=[c[0] for c in BEHAVIOUR_CHANGES])
def test_behaviour_change(case, error, provider, expected):
    outcome = run_ladder(error, provider=provider)
    observed = {key: getattr(outcome, key) for key in expected}
    assert observed == expected, (case, outcome.result)


# ── hooks steer aux decisions ───────────────────────────────────────────


def test_plugin_classification_hook_steers_the_aux_ladder(monkeypatch):
    from hermes_cli import plugins

    def _hook(**kw):
        if kw["status_code"] == 403 and "org wallet empty" in kw["error_message"]:
            return {"reason": FailoverReason.billing, "retryable": False, "should_fallback": True,
                    "should_rotate_credential": True}
        return None

    monkeypatch.setattr(plugins, "get_plugin_error_classification", _hook)
    outcome = run_ladder(sdk(403, "org wallet empty"), provider="openrouter")
    assert (outcome.fell_back, outcome.unhealthy, outcome.rotated) == (True, True, True)


def test_provider_profile_classifier_steers_the_aux_ladder(monkeypatch):
    import providers

    class _Profile:
        @staticmethod
        def classify_api_error(error, **_kw):
            return {"reason": "overloaded"} if "custom gateway hiccup" in str(error) else None

    real = providers.get_provider_profile
    monkeypatch.setattr(providers, "get_provider_profile", lambda slug: _Profile() if slug == "openrouter" else real(slug))
    outcome = run_ladder(sdk(400, "custom gateway hiccup"), provider="openrouter")
    # A plain 400 aborts on the built-in pipeline; the profile's overload verdict retries, then falls back.
    assert (outcome.attempts, outcome.fell_back) == (2, True)


def test_a_not_retryable_verdict_skips_the_same_provider_retry():
    for retryable, expected in ((True, True), (False, False)):
        verdict = ClassifiedError(reason=FailoverReason.server_error, retryable=retryable)
        assert aux_errors.same_provider_retry(verdict) is expected


# ── decisions moved from the private predicates (unchanged behaviour) ───────


@pytest.mark.parametrize("error", [
    sdk(402, "Payment Required"),
    _Plain("Budget limit exceeded (monthly limit). Contact your org admin."),
    sdk(403, "this model requires a subscription, upgrade for access: https://ollama.com/upgrade"),
    sdk(404, "Model 'gpt-5' is not available on the free tier. Upgrade."),
    sdk(429, "Too many tokens per day, please wait"),
])
def test_billing_and_spent_quota_bench_the_provider(error):
    assert aux_errors.fallback_label(_failure(error)) == aux_errors.PAYMENT_ERROR


@pytest.mark.parametrize("spelling", ["RESOURCE_EXHAUSTED", "ResourceExhausted", "resource-exhausted"])
@pytest.mark.parametrize("status", [None, 429])
def test_resource_exhausted_without_a_quota_window_is_a_rate_limit(spelling, status):
    """NIM's concurrency cap names no window: it rotates and falls back but does not bench."""
    error = _Plain(f"{spelling}: Worker local total request limit reached (32/32)")
    if status is not None:
        error.status_code = status
    failure = _failure(error)
    assert not aux_errors.benches_provider(failure.verdict)
    assert aux_errors.fallback_label(failure) == aux_errors.RATE_LIMIT


@pytest.mark.parametrize("error,label", [
    (sdk(404, "Not Found"), None),
    (sdk(404, "Model 'gpt-5.4-mini' not found. The requested model does not exist in our configuration or "
              "OpenRouter catalog."), aux_errors.MODEL_NOT_FOUND),
    (sdk(400, "openrouter/foo/bar is not a valid model ID"), aux_errors.MODEL_NOT_FOUND),
    (sdk(404, "Your API key is blocked or out of funds. model_not_found"), aux_errors.PAYMENT_ERROR),
    (sdk(400, "insufficient credits: model is not supported on free tier"), aux_errors.PAYMENT_ERROR),
    (openai.APIError("stream error", _REQ, body={"type": "server_error", "message": "upstream failed"}),
     aux_errors.STRUCTURED_ERROR),
    (_Plain("stream_error: mid_stream_failure"), None),
    (_Plain("Request timed out."), aux_errors.TIMED_OUT),
    (_Plain("Connection refused"), aux_errors.CONNECTION_ERROR),
])
def test_fallback_labels(error, label):
    assert aux_errors.fallback_label(_failure(error)) == label


def test_critical_path_skip_reads_the_transport_kind():
    from agent.auxiliary_stream_watchdog import AuxNoProgressTimeout

    timed_out = _failure(_Plain("Request timed out."))
    assert aux_errors.skip_same_provider_retry("compression", timed_out)
    assert not aux_errors.skip_same_provider_retry("compression", _failure(_Plain("Connection refused")))
    assert not aux_errors.skip_same_provider_retry("title_generation", _failure(AuxNoProgressTimeout("no output")))
    assert not aux_errors.skip_same_provider_retry("session_search", timed_out)


# ── a recovery attempt's failure is its own, not the error it followed ─────


def _invalid_response(*_args, **_kwargs):
    try:
        raise AttributeError("missing choices[0].message")
    except AttributeError as exc:
        raise InvalidProviderResponseError("LLM returned invalid response") from exc


@pytest.mark.parametrize("retry_outcome", [
    ConnectionError("Connection refused"), TimeoutError("timed out"), RuntimeError("Connection reset by peer"),
    _invalid_response,
], ids=["refused", "timeout", "reset", "invalid-answer"])
def test_failure_after_a_401_refresh_falls_back_without_another_rotation(retry_outcome):
    """401 -> credential refresh -> the retry cannot connect or answers garbage: the retry's own failure
    decides (no second credential rotation, provider fallback), not the 401 before it."""
    outcomes = iter([sdk(401, "invalid api key"), retry_outcome])

    def create(*args, **kwargs):
        outcome = next(outcomes, retry_outcome)
        if callable(outcome) and not isinstance(outcome, BaseException):
            outcome()
        raise outcome

    outcome = run_ladder(create)
    assert outcome.refreshed == ["openrouter"]
    assert (outcome.rotated, outcome.fell_back) == (False, True)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_generic_403_after_a_rate_limit_backoff_reraises_on_an_explicit_provider(is_async):
    """429 then a generic 403 on an explicit provider with no task fallback chain is a 403: re-raised."""
    outcome = run_ladder([sdk(429, "Rate limit exceeded"), sdk(403, "Forbidden")], is_async=is_async)
    assert isinstance(outcome.result, openai.PermissionDeniedError)
    assert outcome.attempts == 2 and not outcome.fell_back


@pytest.mark.parametrize("message", ["Tool use is not supported with this model",
                                     "Streaming is not supported for this account. Set the stream parameter to false."])
def test_feature_restriction_falls_back_without_touching_the_key(message):
    """Unchanged from the predicates: no parameter rung names the feature, so it is a route rejection."""
    outcome = run_ladder(sdk(400, message))
    assert (outcome.attempts, outcome.fell_back, outcome.rotated) == (1, True, False)


# ── a parameter the request carries is the request's own failure ─────────

_PARAMETER_NAMED = [
    "Unsupported parameter: 'temperature' is not supported with this model.",
    "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
    "Parameter 'response_format' is not supported with this model",
    "temperature is not supported for this account",
    "The model parameter temperature is not supported for this account",
]


def _statusless(message):
    return openai.APIError(message, _REQ, body={"message": message, "type": "invalid_request_error"})


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("error", [sdk(400, m) for m in _PARAMETER_NAMED] + [_statusless(m) for m in _PARAMETER_NAMED[:3]]
                         + [_Plain(_PARAMETER_NAMED[3])],
                         ids=[f"400-{i}" for i in range(len(_PARAMETER_NAMED))] + [f"statusless-{i}" for i in range(3)]
                         + ["plain-3"])
def test_parameter_named_rejection_with_no_rung_reraises(error, is_async):
    """With no parameter rung to apply (the field is not on the wire) the rejection would follow the
    request to any provider: re-raised without a fallback."""
    outcome = run_ladder(error, is_async=is_async)
    assert outcome.result is error
    assert (outcome.attempts, outcome.fell_back, outcome.rotated) == (1, False, False)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("final", [sdk(400, "Unsupported parameter: 'max_tokens' is not supported with this model."),
                                   _statusless("Unsupported parameter: 'max_tokens' is not supported with this model."),
                                   sdk(400, "Bad Request")],
                         ids=["400", "statusless", "request-shape"])
def test_failure_after_a_spent_parameter_rung_falls_back(final, is_async):
    """The temperature rung ran and the repaired request still failed: fall back."""
    first = sdk(400, "Unsupported parameter: 'temperature' is not supported with this model.")
    outcome = run_ladder([first, final], is_async=is_async, temperature=0.3)
    assert (outcome.attempts, outcome.fell_back, outcome.rotated) == (2, True, False)


# ── the credential pool ────────────────────────────────────────────────


class _Pool:
    def __init__(self):
        self.rotations = []

    def has_credentials(self):
        return True

    def try_refresh_current(self):
        return None

    def mark_exhausted_and_rotate(self, **kwargs):
        self.rotations.append(kwargs)
        return object()


@pytest.fixture
def pool(monkeypatch):
    from agent import auxiliary_client as ac

    pool = _Pool()
    monkeypatch.setattr(ac, "load_pool", lambda _provider: pool)
    monkeypatch.setattr(ac, "_evict_cached_clients", lambda _provider: None)
    return pool


@pytest.mark.parametrize("error", [
    sdk(400, "This model is not supported for image input"), sdk(400, "model_not_supported: tools"),
    sdk(400, "Streaming with this model is not supported for this account"),
    _Plain("Parameter reasoning_effort for model o3 is not supported for this account"),
    sdk(400, "The model tool_choice is not supported for this account"),
])
def test_route_level_rejection_records_no_pool_cooldown(pool, error):
    """A model_entitlement rotation would bench key+model for a year, even a sole key."""
    from agent import auxiliary_client as ac

    assert ac._recover_provider_pool("openrouter", error, failed_api_key="sk-only") is False
    assert pool.rotations == []


def test_account_gated_model_rotates_for_that_model_only(pool):
    from agent import auxiliary_client as ac

    error = sdk(400, "The model kimi-k2 is not supported for this account")
    assert ac._recover_provider_pool("openrouter", error, failed_api_key="sk-a") is True
    assert [(r["status_code"], r["failure_reason"]) for r in pool.rotations] == [(400, "model_entitlement")]


def test_auto_routed_nous_classifies_with_the_credential_the_client_sent(monkeypatch):
    """The anonymous Nous JWT on the auto-routed client marks the 429 a welcome-tier gate: fall back,
    never rotate the tier's only credential."""
    from tests.hermes_cli.anon_portal import make_jwt

    monkeypatch.setattr("agent.auxiliary_client._refresh_nous_recommended_model", lambda **_kw: None)
    gate = sdk(429, "model not available on the free tier", reason="model_not_free")
    outcome = run_ladder(gate, provider="auto", resolved_key="sk-resolved", client={
        "api_key": make_jwt(), "base_url": "https://inference-api.nousresearch.com/v1",
        "_hermes_aux_effective_provider": "nous"})
    assert (outcome.rotated, outcome.fell_back) == (False, True)


# ── fallback candidates ─────────────────────────────────────────────────


def _candidate(outcomes, *, is_async):
    client = MagicMock(base_url="https://openrouter.ai/api/v1", api_key="sk-fallback")
    client.chat.completions.create = (AsyncMock if is_async else MagicMock)(side_effect=outcomes)
    return client


def _run_candidate(client, *, is_async, retry_client=None, label="openrouter", **overrides):
    from agent.auxiliary_client import _call_fallback_candidate_async, _call_fallback_candidate_sync

    marks = []
    kw = dict(task="title_generation", messages=_MESSAGES, temperature=0.3, max_tokens=16, tools=None,
              effective_timeout=30.0, effective_extra_body={}, reasoning_config=None)
    kw.update(overrides)
    with (
        patch("agent.auxiliary_client._mark_provider_unhealthy", side_effect=lambda *a, **k: marks.append((a, k))),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp),
        patch("agent.auxiliary_client._refresh_provider_credentials", return_value=retry_client is not None),
        patch("agent.auxiliary_client._get_cached_client", return_value=(retry_client, "fb-model")),
    ):
        if is_async:
            result = asyncio.run(_call_fallback_candidate_async(client, "fb-model", label, **kw))
        else:
            result = _call_fallback_candidate_sync(client, "fb-model", label, **kw)
    return result, marks


@pytest.mark.parametrize("is_async", [False, True])
def test_candidate_whose_parameter_repair_still_fails_is_skipped(is_async):
    client = _candidate([sdk(400, "Unsupported value: 'temperature' does not support 0.3 with this model."),
                         sdk(400, "Bad Request")], is_async=is_async)
    result, marks = _run_candidate(client, is_async=is_async)
    assert result is None and client.chat.completions.create.call_count == 2
    assert [k["reason"] for _a, k in marks] == [aux_errors.MODEL_INCOMPATIBLE]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("message,overrides", [
    ("Bad Request", {}),
    (_PARAMETER_NAMED[0], {"temperature": None}),  # temperature not on the wire: its rung cannot apply
], ids=["request-shape", "parameter"])
def test_candidate_request_failure_with_no_repair_still_raises(is_async, message, overrides):
    client = _candidate([sdk(400, message)], is_async=is_async)
    with pytest.raises(openai.BadRequestError):
        _run_candidate(client, is_async=is_async, **overrides)
    assert client.chat.completions.create.call_count == 1


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("final,reason,ttl", [
    (sdk(503, "Service Unavailable"), aux_errors.PROVIDER_OVERLOADED, 60.0),
    (sdk(401, "invalid api key"), "stale fallback credential", None),
    (_invalid_response, aux_errors.INVALID_RESPONSE, 60.0),
    (RuntimeError("Connection reset by peer"), aux_errors.CONNECTION_ERROR, 60.0),
], ids=["503", "401", "invalid-answer", "reset"])
def test_refreshed_candidate_is_quarantined_for_its_final_failure(is_async, final, reason, ttl):
    client = _candidate([sdk(401, "invalid api key")], is_async=is_async)
    retry_client = _candidate(final if callable(final) else [final], is_async=is_async)
    result, marks = _run_candidate(client, is_async=is_async, retry_client=retry_client)
    assert result is None and retry_client.chat.completions.create.call_count == 1
    assert [(k["reason"], k["ttl"]) for _a, k in marks] == [(reason, ttl)]


@pytest.mark.parametrize("is_async", [False, True])
def test_candidate_generic_403_is_quarantined_as_an_auth_error(is_async):
    client = _candidate([sdk(403, "Forbidden")], is_async=is_async)
    result, marks = _run_candidate(client, is_async=is_async)
    assert result is None and [k["reason"] for _a, k in marks] == [aux_errors.AUTH_ERROR]


# ── every request classifies against the backend it reached ─────────────


def _record_classifications(monkeypatch):
    seen, real = [], aux_errors.classify_api_error

    def record(exc, **kw):
        classified = real(exc, **kw)
        seen.append((kw["provider"], kw["api_key"], classified.reason))
        return classified

    monkeypatch.setattr(aux_errors, "classify_api_error", record)
    return seen


def _go_upstream_429():
    import model_tools  # registers the bundled provider profiles

    body = {"error": {"type": "server_error", "message": "Upstream request failed: Rate limit exceeded."}}
    response = httpx.Response(429, json=body, request=httpx.Request("POST", "https://opencode.ai/zen/go/v1/x"))
    return openai.RateLimitError(f"Error code: 429 - {body}", response=response, body=body)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_candidate_classifies_with_its_own_backend_and_credential(monkeypatch, is_async):
    """An auto candidate tagged opencode-go: the Go profile reads the upstream throttle on the key the
    candidate sent (not the primary's)."""
    seen = _record_classifications(monkeypatch)
    client = _candidate([_go_upstream_429()], is_async=is_async)
    client._hermes_aux_effective_provider = "opencode-go"
    result, marks = _run_candidate(client, is_async=is_async, label="auto")
    assert result is None and marks
    assert set(seen) == {("opencode-go", "sk-fallback", FailoverReason.upstream_rate_limit)}


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_unavailable_primary_classifies_its_chain_fallback_on_the_concrete_backend(monkeypatch, is_async):
    """An explicit provider with no credentials runs on its task fallback_chain entry; that entry's failure
    classifies on its provider, not on the ``fallback_chain[0](opencode-go)`` display label."""
    seen = _record_classifications(monkeypatch)
    go = MagicMock(base_url="https://opencode.ai/zen/go/v1", api_key="sk-go")
    go.chat.completions.create = (AsyncMock if is_async else MagicMock)(side_effect=_go_upstream_429())
    with (
        patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("anthropic", "m", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(None, None)),
        patch("agent.auxiliary_client._try_configured_fallback_for_unavailable_client",
              return_value=(go, "mock-model", "fallback_chain[0](opencode-go)")),
        patch("agent.auxiliary_client._to_async_client", return_value=(go, "mock-model")),
        patch("agent.auxiliary_client._transient_retry_count", return_value=0),
        patch("agent.auxiliary_client._recoverable_pool_provider", return_value=None),
        patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(None, None, "")),
        patch("agent.auxiliary_client._mark_provider_unhealthy"),
        pytest.raises(openai.RateLimitError),
    ):
        if is_async:
            asyncio.run(async_call_llm(task="title_generation", messages=_MESSAGES))
        else:
            call_llm(task="title_generation", messages=_MESSAGES)
    assert seen and {(p, r) for p, _k, r in seen} == {("opencode-go", FailoverReason.upstream_rate_limit)}


@pytest.mark.parametrize("tag", ["effective-provider", "fallback-destination"])
def test_async_candidate_classifies_on_its_backend_through_the_real_conversion(monkeypatch, tag):
    """``async_call_llm`` converts a sync fallback candidate to async before sending: its failure still
    classifies on the backend the sync original was tagged with."""
    import model_tools  # registers the bundled provider profiles
    from agent.auxiliary_client import _FallbackDestination

    seen = _record_classifications(monkeypatch)
    body = {"error": {"type": "server_error", "message": "Upstream request failed: Rate limit exceeded."}}
    transport = httpx.MockTransport(lambda request: httpx.Response(429, json=body, request=request))
    go = openai.OpenAI(api_key="sk-go", base_url="https://opencode.ai/zen/go/v1", max_retries=0,
                       http_client=httpx.Client(transport=transport))
    if tag == "effective-provider":
        go._hermes_aux_effective_provider = "opencode-go"
    else:
        go._hermes_fallback_destination = _FallbackDestination("opencode-go", str(go.base_url), None, "go-model")
    primary = MagicMock(base_url="https://openrouter.ai/api/v1", api_key="sk-primary")
    primary.chat.completions.create = AsyncMock(side_effect=sdk(503, "Service Unavailable"))
    with (
        patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("openrouter", "m", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "m")),
        patch("agent.auxiliary_client._openai_http_client_kwargs",
              return_value={"http_client": httpx.AsyncClient(transport=transport)}),
        patch("agent.auxiliary_client._transient_retry_count", return_value=0),
        patch("agent.auxiliary_client._recoverable_pool_provider", return_value=None),
        patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(go, "go-model", "auto")),
        patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(None, None, "")),
        patch("agent.auxiliary_client._mark_provider_unhealthy"),
        pytest.raises(openai.APIError),
    ):
        asyncio.run(async_call_llm(task="title_generation", messages=_MESSAGES))
    assert ("opencode-go", "sk-go", FailoverReason.upstream_rate_limit) in seen


# ── real SDK clients: a refreshed attempt's own answer ─────────────────


def _wire_client(handle, *, key, is_async, url="https://openrouter.ai/api/v1"):
    """A real SDK client over ``httpx.MockTransport``."""
    transport = httpx.MockTransport(handle)
    if is_async:
        return openai.AsyncOpenAI(api_key=key, base_url=url, max_retries=0,
                                  http_client=httpx.AsyncClient(transport=transport))
    return openai.OpenAI(api_key=key, base_url=url, max_retries=0, http_client=httpx.Client(transport=transport))


def _json(status, body):
    return lambda request: httpx.Response(status, json=body, request=request)


_401 = _json(401, {"error": {"message": "invalid api key"}})


def _nous_ladder(primary, *, is_async, refreshed=None):
    """``call_llm`` / ``async_call_llm`` on a Nous route; the 401 refresh hands back ``refreshed``."""
    fallback = MagicMock(base_url="https://api.example.com/v1")
    fallback.chat.completions.create = (AsyncMock if is_async else MagicMock)(return_value=_ok())
    with (
        patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("nous", "old-model", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "old-model")),
        patch("agent.auxiliary_client._refresh_nous_auxiliary_client", return_value=(refreshed, "fb-model-new")),
        patch("agent.auxiliary_client._refresh_nous_recommended_model", return_value=None),
        patch("agent.auxiliary_client._transient_retry_count", return_value=0),
        patch("agent.auxiliary_client._recoverable_pool_provider", return_value=None),
        patch("agent.auxiliary_client._mark_provider_unhealthy"),
        patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(fallback, "fb", "fallback")),
    ):
        if is_async:
            return asyncio.run(async_call_llm(task="title_generation", messages=_MESSAGES))
        return call_llm(task="title_generation", messages=_MESSAGES)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("error_json", ['"backend worker failed"', '{"message": "backend worker failed"}'],
                         ids=["string-body", "object-body"])
def test_stream_error_after_a_nous_credential_refresh_falls_back(is_async, error_json):
    """401, Nous refresh, then the refreshed key's stream carries an SSE ``error`` event: a status-less
    structured error of the refreshed attempt, so it falls back instead of re-raising as the 401."""
    sse = lambda request: httpx.Response(200, content=f'data: {{"error": {error_json}}}\n\n'.encode(),
                                         headers={"content-type": "text/event-stream"}, request=request)
    primary = _wire_client(_401, key="sk-stale", is_async=is_async)
    refreshed = _wire_client(sse, key="sk-fresh", is_async=is_async, url="https://inference-api.nousresearch.com/v1")
    with patch("agent.auxiliary_client._aux_progress_active", return_value=True):  # every attempt streams
        result = _nous_ladder(primary, is_async=is_async, refreshed=refreshed)
    assert result.choices[0].message.content == "fallback"


_SHIM_COMPLETION = {"id": "x", "object": "chat.completion", "created": 0, "model": "fb-model-new",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "Connect timeout, please try again later."}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}}
_BAD_ANSWERS = {  # (body, content type, streamed)
    "malformed-sse": (b"data: {bad json}\n\n", "text/event-stream", True),
    "json-null": (b"null", "application/json", False),
    "error-envelope": (b'{"success": false, "data": {"error": {"message": "upstream worker crashed"}}}',
                       "application/json", False),
    "timeout-shim": (json.dumps(_SHIM_COMPLETION).encode(), "application/json", False),
}


def _candidate_outcome(first, refreshed, *, is_async, streamed):
    """``(raised type or None, [(quarantine reason, ttl)])`` of one fallback candidate call."""
    from agent.auxiliary_client import _call_fallback_candidate_async, _call_fallback_candidate_sync

    marks, raised = [], None
    kw = dict(task="title_generation", messages=_MESSAGES, temperature=None, max_tokens=16, tools=None,
              effective_timeout=30.0, effective_extra_body={}, reasoning_config=None)
    with (
        patch("agent.auxiliary_client._aux_progress_active", return_value=streamed),
        patch("agent.auxiliary_client._mark_provider_unhealthy", side_effect=lambda *a, **k: marks.append(k)),
        patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True),
        patch("agent.auxiliary_client._get_cached_client", return_value=(refreshed, "fb-model-new")),
    ):
        try:
            if is_async:
                asyncio.run(_call_fallback_candidate_async(first, "fb-model", "openrouter", **kw))
            else:
                _call_fallback_candidate_sync(first, "fb-model", "openrouter", **kw)
        except Exception as exc:
            raised = type(exc)
    return raised, [(k["reason"], k["ttl"]) for k in marks]


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("answer", list(_BAD_ANSWERS))
def test_refreshed_candidates_own_bad_answer_is_not_a_stale_credential(is_async, answer):
    """401, credential refresh, then the refreshed key's answer is malformed, null, an error envelope or a
    timeout shim: it gets the outcome the same answer gets with no 401 before it."""
    body, content_type, streamed = _BAD_ANSWERS[answer]

    def bad():
        return _wire_client(lambda request: httpx.Response(200, content=body, headers={"content-type": content_type},
                                                           request=request), key="sk-fresh", is_async=is_async)
    after_refresh = _candidate_outcome(_wire_client(_401, key="sk-stale", is_async=is_async), bad(),
                                       is_async=is_async, streamed=streamed)
    alone = _candidate_outcome(bad(), None, is_async=is_async, streamed=streamed)
    assert ("stale fallback credential", None) not in after_refresh[1]
    assert after_refresh == alone


# ── network failures ────────────────────────────────────────────────────


class EndpointConnectionError(Exception):
    """botocore's class name, outside the builtin ConnectionError/OSError hierarchy like botocore's own."""


_BEDROCK_URL = "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude/converse"


def _endpoint_connection_error():
    return EndpointConnectionError(f'Could not connect to the endpoint URL: "{_BEDROCK_URL}"')


def _botocore_endpoint_connection_error():
    exceptions = pytest.importorskip("botocore.exceptions")
    return exceptions.EndpointConnectionError(endpoint_url=_BEDROCK_URL)


def _cert_verification_error():
    import ssl

    return ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")


@pytest.mark.parametrize("make_error", [_cert_verification_error, _endpoint_connection_error,
                                        _botocore_endpoint_connection_error],
                         ids=["ssl", "endpoint-connection", "botocore-endpoint-connection"])
def test_compression_preserves_the_messages_after_network_failures(make_error):
    """A TLS failure or an unreachable endpoint is a dead network for the compressor (as before): it
    aborts and keeps every message instead of committing a placeholder summary."""
    from agent.context_compressor import ContextCompressor

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2,
                                       abort_on_summary_failure=False)
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(12)]
    with patch("agent.context_compressor.call_llm", side_effect=make_error()):
        result = compressor.compress(msgs, current_tokens=999999, force=True)
    assert result == msgs
    assert compressor._last_summary_network_failure is True
    assert compressor._last_compress_aborted is True


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("make_error", [_endpoint_connection_error, _botocore_endpoint_connection_error],
                         ids=["named", "botocore"])
def test_unreachable_endpoint_retries_then_falls_back(make_error, is_async):
    outcome = run_ladder([make_error(), make_error()], provider="bedrock", retries=1, is_async=is_async)
    assert outcome.attempts == 2 and outcome.fell_back
    assert not outcome.rotated and not outcome.unhealthy
