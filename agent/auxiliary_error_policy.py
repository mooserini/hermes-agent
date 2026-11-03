"""What the auxiliary recovery ladder does with a failed provider request, read from the shared verdict.

``classify_api_error`` (agent/error_classifier.py) is the one reader of an API error's status, code and
wording; plugin ``transform_api_error_classification`` hooks and ``ProviderProfile.classify_api_error``
steer it. The ladder classifies each failure once, on the route of the request that raised it
(``classify_attempt``), and these pure functions of that verdict decide: retry the same provider,
refresh or rotate the credential, bench the provider, or fall back. The fallback labels are the ones
the logs, the unhealthy marks and the candidate quarantine TTLs (agent/auxiliary_health.py) key on.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import httpx

from agent.error_classifier import ClassifiedError, FailoverReason, classify_api_error

_R = FailoverReason

AUTH_ERROR = "auth error"
PAYMENT_ERROR = "payment error"
RATE_LIMIT = "rate limit"
MODEL_NOT_FOUND = "model not found"
MODEL_INCOMPATIBLE = "model incompatible with route"
INVALID_RESPONSE = "invalid provider response"
STRUCTURED_ERROR = "structured provider error"
TIMED_OUT = "request timed out"
CONNECTION_ERROR = "connection error"
PROVIDER_OVERLOADED = "provider overloaded"
SERVER_ERROR = "server error"

# Same provider, same request, a moment later: a blip the provider recovers from by itself.
_TRANSIENT = frozenset({_R.timeout, _R.overloaded, _R.server_error})
# The caller's request shape failed, not the provider: fall back only once a parameter rung changed the
# request and it still failed (main-loop parity: agent/turn_api_error.py's "spent" fallbacks).
_REQUEST_SHAPE = frozenset({
    _R.format_error, _R.role_alternation, _R.multimodal_tool_content_unsupported, _R.reasoning_mandatory,
    _R.llama_cpp_grammar_pattern, _R.image_too_large, _R.image_corrupt, _R.invalid_encrypted_content,
    _R.thinking_signature, _R.data_policy_denied, _R.context_overflow, _R.payload_too_large,
    _R.oauth_long_context_beta_forbidden,
})
_LABELS = {
    _R.auth: AUTH_ERROR, _R.auth_permanent: AUTH_ERROR, _R.billing: PAYMENT_ERROR,
    _R.rate_limit: RATE_LIMIT, _R.upstream_rate_limit: RATE_LIMIT,
    _R.model_not_found: MODEL_NOT_FOUND, _R.model_entitlement: MODEL_INCOMPATIBLE,
    _R.incomplete_response: INVALID_RESPONSE, _R.overloaded: PROVIDER_OVERLOADED, _R.server_error: SERVER_ERROR,
    # The main loop compresses into the standard tier; an auxiliary request cannot, so it moves on.
    _R.long_context_tier: "long context tier",
}
# A stream negotiation that failed with one of these is the provider's answer: re-raised, not re-sent
# non-streaming.
_PROVIDER_REFUSALS = _TRANSIENT | {_R.auth, _R.auth_permanent, _R.billing, _R.rate_limit, _R.upstream_rate_limit}

# Tasks on a user-visible critical path (compression blocks resuming an oversized session; vision stalls
# the serialised turn queue): a same-provider retry after a full-budget timeout costs another whole window,
# so they skip straight to fallback; fast blips still retry (#54465). Title generation joins them because
# its retries multiplied the user's ``auxiliary.title_generation.timeout`` on a slow local model (#89445).
TIMEOUT_NO_RETRY_TASKS = frozenset({"compression", "vision", "title_generation"})


class Failure(NamedTuple):
    """One failed provider request: the exception, its verdict on the route that raised it, and whether a
    parameter rung's detector names the rejected field (the request's own parameter, not the route)."""
    error: Exception
    verdict: ClassifiedError
    rejects_parameter: bool = False


def classify_attempt(error: Exception, provider: Optional[str], client: Any = None, model: Any = None,
                     base_url: Optional[str] = "") -> ClassifiedError:
    """``classify_api_error`` on the backend, model, endpoint and credential ``client`` was sent to. The
    routing label ``auto`` names no backend, so the provider-profile hook only sees a concrete slug."""
    slug = str(provider or "").strip().lower()
    url, key = getattr(client, "base_url", None), getattr(client, "api_key", None)
    return classify_api_error(
        error, provider="" if slug == "auto" else slug, model=model if isinstance(model, str) else "",
        base_url=str(url) if isinstance(url, (str, httpx.URL)) and str(url) else str(base_url or ""),
        api_key=key if isinstance(key, str) else None)


def same_provider_retry(verdict: ClassifiedError) -> bool:
    """A transient blip (timeout, connection, overload, 5xx), unless a hook marks it not retryable."""
    return verdict.reason in _TRANSIENT and verdict.retryable


def skip_same_provider_retry(task: Optional[str], failure: Failure) -> bool:
    """A critical-path task that spent a whole time budget; a zero-output first-token failure
    (``AuxNoProgressTimeout``) is cheap and keeps the retry."""
    from agent.auxiliary_stream_watchdog import AuxNoProgressTimeout

    return (task in TIMEOUT_NO_RETRY_TASKS and failure.verdict.error_context.get("transport") == "timeout"
            and not isinstance(failure.error, AuxNoProgressTimeout))


def auth_refresh(verdict: ClassifiedError) -> bool:
    """A credential refused as such (401-class): refresh, then rotate. A generic 403 falls back instead."""
    return verdict.reason is _R.auth and verdict.should_rotate_credential


def same_key_retry(verdict: ClassifiedError) -> bool:
    """A per-key throttle that clears in seconds; not a spent quota window."""
    return verdict.reason is _R.rate_limit and not verdict.error_context.get("quota_exhausted")


def benches_provider(verdict: ClassifiedError) -> bool:
    """Credit or a quota window is spent: mark the provider unhealthy so later aux calls skip it."""
    return verdict.reason is _R.billing or bool(verdict.error_context.get("quota_exhausted"))


def fallback_label(failure: Failure, *, rung_spent: bool = False) -> Optional[str]:
    """Why another provider should take this request, or None when the failure is the request's own and
    would follow it anywhere. Transient reasons count: the ladder runs once same-provider retries are
    spent. ``rung_spent``: a parameter rung already changed the request and it still failed."""
    verdict = failure.verdict
    reason, context = verdict.reason, verdict.error_context
    if benches_provider(verdict):
        return PAYMENT_ERROR
    if reason is _R.timeout:
        return CONNECTION_ERROR if context.get("transport") == "connection" else TIMED_OUT
    if reason in _LABELS:
        return _LABELS[reason]
    if failure.rejects_parameter or reason in _REQUEST_SHAPE and not context.get("route_rejects_model"):
        if rung_spent:
            return MODEL_INCOMPATIBLE
        return STRUCTURED_ERROR if context.get("statusless_structured") and not failure.rejects_parameter else None
    if context.get("route_rejects_model"):
        return MODEL_INCOMPATIBLE
    if verdict.should_fallback:
        return reason.value.replace("_", " ")
    return STRUCTURED_ERROR if context.get("statusless_structured") else None


def ladder_accepts(failure: Failure, *, rung_spent: bool = False) -> bool:
    """A later rung (credential rotation, provider fallback) acts on this failure; else it re-raises."""
    return failure.verdict.should_rotate_credential or fallback_label(failure, rung_spent=rung_spent) is not None


def stream_negotiation_reraises(verdict: ClassifiedError) -> bool:
    """A failed stream negotiation that is the provider's answer (any 429 and a welcome-tier refusal
    included): re-raised unchanged rather than re-sent non-streaming."""
    return (verdict.reason in _PROVIDER_REFUSALS or verdict.status_code == 429 or benches_provider(verdict)
            or "welcome_refusal" in verdict.error_context)
