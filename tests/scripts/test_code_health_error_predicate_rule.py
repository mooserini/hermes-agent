"""HX014: a new private API-error predicate on the aux/compressor/turn paths blocks; existing ones
are grandfathered (moved code keeps its debt) and deciding from the shared classifier passes."""

from __future__ import annotations

import ast

import pytest

from scripts.code_health.config import RULES_BY_ID, rule_applies
from scripts.code_health.py_rules import CHECKERS, Ctx, canonical_tree
from tests.scripts.test_code_health import _commit, _repo, _verdict


def _hits(src: str) -> list[int]:
    return sorted(set(CHECKERS["HX014"](canonical_tree(ast.parse(src)), Ctx())))


@pytest.mark.parametrize("src", [
    # red: status-code reads of an exception parameter
    "def _is_rate_limit(exc):\n    return exc.status_code == 429\n",
    "def f(err):\n    return getattr(err, 'status_code', None) in (502, 503)\n",
    "def f(exc):\n    return exc.response.status_code == 400\n",
    "def f(last_error):\n    code = last_error.status_code\n    return code == 500\n",
    # red: substring tests of the exception's text
    "def _is_ctx_overflow(exc):\n    return 'context length' in str(exc).lower()\n",
    "def f(ex):\n    msg = str(ex).lower()\n    return 'rate limit' in msg\n",
    "def f(ex):\n    msg = str(ex)\n    low = msg.casefold()\n    return 'x' in low\n",  # a chain
    "def f(exc):\n    return _contains_any(str(exc).lower(), PATTERNS)\n",
    "def f(exc):\n    return str(exc).startswith('Error code: 429')\n",
    "import re\ndef f(exc):\n    return bool(re.search(r'40[13]', str(exc)))\n",
    "def f(exc):\n    if (m := str(exc).lower()) and 'timeout' in m:\n        return True\n",
    # red: a lambda, and a parameter typed as an exception whatever its name
    "is_overflow = lambda exc: 'too long' in str(exc).lower()\n",
    "check = lambda failure_exc: failure_exc.status_code == 404\n",
    "def f(failure: Exception):\n    return 'x' in str(failure)\n",
    "import httpx\ndef f(failure: httpx.HTTPStatusError):\n    return failure.status_code == 1\n",
    "from typing import Optional\ndef f(failure: Optional[Exception]):\n    return 'x' in str(failure)\n",
    "def f(failure: RuntimeError | None):\n    return 'x' in str(failure)\n",
    # red: a method's exception parameter
    "class C:\n    def _should_fallback(self, exc):\n        return exc.status_code == 402\n",
])
def test_predicate_shapes_are_flagged(src):
    assert _hits(src), src


@pytest.mark.parametrize("src", [
    # green: deciding from a ClassifiedError / policy verdict
    "def f(classified):\n    return classified.status_code == 429\n",
    "def f(exc):\n    verdict = classify_api_error(exc)\n    return verdict.retryable\n",
    "def f(verdict: ClassifiedError):\n    return verdict.status_code == 429\n",
    "def f(verdict: 'ClassifiedError | None' = None):\n    return 'quota' in str(verdict)\n",
    "def f(err: ClassifiedError):\n    return err.status_code == 429\n",  # the annotation wins over the name
    # green: an exception parameter used but not probed
    "def f(exc):\n    logger.warning('failed: %s', exc)\n    raise RuntimeError(str(exc)) from exc\n",
    "def f(exc):\n    return str(exc) in SEEN\n",  # the exception text is the needle, not the haystack
    # green: a non-exception parameter, or a container of exceptions
    "def f(resp):\n    return resp.status_code == 200 and 'ok' in str(resp)\n",
    "def f(errors: list[Exception]):\n    return 'x' in str(errors)\n",
    # green: a nested def's probe of ITS OWN non-exception param, and an outer probe-free def
    "def outer(exc):\n    def inner(resp):\n        return resp.status_code\n    return inner\n",
    # green: an exception caught locally (not a parameter) is not this rule's shape
    "def f(client):\n    try:\n        client.call()\n    except Exception as err:\n        return err.status_code\n",
])
def test_classifier_reads_and_non_exceptions_pass(src):
    assert _hits(src) == [], src


def test_nested_def_is_judged_on_its_own_params():
    src = ("def outer(resp):\n    def inner(exc):\n        return exc.status_code == 1\n"
           "    return resp.status_code\n")
    assert _hits(src) == [3]


def test_one_hit_per_offending_expression():
    src = ("def f(exc):\n    if exc.status_code == 429:\n        return True\n"
           "    msg = str(exc).lower()\n    return 'quota' in msg or 'rate' in msg\n")
    assert _hits(src) == [2, 5]


@pytest.mark.parametrize("path, applies", [
    ("agent/auxiliary_client.py", True),
    ("agent/auxiliary_error_policy.py", True),  # reads ClassifiedError only, so it stays in scope
    ("agent/context_compressor.py", True),
    ("agent/turn_runner.py", True),
    ("agent/run_agent.py", False),
    ("agent/error_classifier.py", False),  # the shared classifier itself
    ("gateway/run.py", False),
    ("tests/agent/test_auxiliary_client.py", False),
])
def test_scope(path, applies):
    assert rule_applies(RULES_BY_ID["HX014"], path) is applies


def test_rule_entry():
    rule = RULES_BY_ID["HX014"]
    assert rule.source == "ast" and rule.blocking
    assert "classify_api_error" in rule.fix and "agent.auxiliary_error_policy" in rule.fix


_OLD = ("def _is_rate_limit(exc):\n    return exc.status_code == 429\n\n\n"
        "def _is_overflow(exc):\n    return 'context length' in str(exc).lower()\n")
_NEW = "def _is_quota(exc):\n    return 'quota' in str(exc).lower()\n"
_AUX = "agent/auxiliary_client.py"


def _judge(tmp_path, capsys, head: dict[str, str | None]):
    repo, base = _repo(tmp_path)
    base = _commit(repo, {_AUX: _OLD})
    return _verdict(repo, base, head, capsys)


@pytest.mark.parametrize("head, blocks", [
    # red: a new predicate, in an old in-scope file or a new one
    ({_AUX: _OLD + "\n\n" + _NEW}, True),
    ({"agent/turn_retry.py": _NEW}, True),
    # red: a second probe added to an old predicate
    ({_AUX: _OLD.replace("== 429\n", "== 429 or 'rate' in str(exc)\n")}, True),
    # green: the old predicates untouched while the file changes around them
    ({_AUX: _OLD + "\n\ndef helper():\n    return 3\n"}, False),
    # green: an old predicate moved to a sibling keeps its debt
    ({_AUX: _OLD.split("\n\n\n")[0] + "\n", "agent/auxiliary_overflow.py": _OLD.split("\n\n\n")[1]}, False),
    # green: the classifier, a waiver with a reason, and out-of-scope paths
    ({"agent/turn_retry.py": "from agent.error_classifier import classify_api_error\n\n\n"
      "def _is_quota(exc):\n    return classify_api_error(exc).reason == 'quota'\n"}, False),
    ({"agent/turn_retry.py": _NEW.replace("lower()\n", "lower()  # health: allow HX014 -- provider"
                                          " quirk the classifier must not generalise\n")}, False),
    ({"agent/run_agent.py": _NEW}, False),
])
def test_verdict(tmp_path, capsys, head, blocks):
    code, out = _judge(tmp_path, capsys, head)
    assert code == (1 if blocks else 0), out
    if blocks:
        assert "HX014" in out and "classify_api_error" in out, out
