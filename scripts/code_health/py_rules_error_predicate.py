"""HX014: a private API-error predicate (registered in ``py_rules.CHECKERS``).

The auxiliary-client, compressor and turn paths each grew their own "is this a <kind> error?" helpers
that read an exception's ``status_code`` or grep its text; each copy drifts from the shared classifier
(``classify_api_error``), so the same provider error is retried on one path and fatal on another. This
flags a function or lambda that takes an exception and decides from its status code or its text. A
parameter typed as the classifier's verdict (``ClassifiedError``) is not an exception, whatever its name.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable

from scripts.code_health.py_ast import _FUNCS, Ctx, _call_name
from scripts.code_health.py_rules_sleep import _own

_EXC_NAME = re.compile(r"(?:exc|err|error|e|ex|exception|\w+_(?:err|error|exc))$")
_EXC_ANNOTATION = re.compile(r"(?:(?:typing\.)?Optional\[)?[\w.]*(?:Exception|Error)\]?(?: \| None)?")
_TEXT_CALLS = frozenset({"str", "repr"})
_FOLDS = frozenset({"lower", "upper", "casefold", "strip"})
_PROBES = frozenset({"startswith", "endswith", "search", "match", "fullmatch", "_contains_any"})


def _exception_params(func: ast.AST) -> frozenset[str]:
    args = func.args  # type: ignore[attr-defined]
    names = set()
    for param in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        annotation = ast.unparse(param.annotation).strip("'\"") if param.annotation else ""
        if "ClassifiedError" not in annotation and (
                _EXC_ANNOTATION.fullmatch(annotation) or not annotation and _EXC_NAME.match(param.arg)):
            names.add(param.arg)
    return frozenset(names)


def _is_text(node: ast.AST, params: frozenset[str], texts: set[str]) -> bool:
    """``str(exc)`` / ``repr(exc)``, case-folded, or a local bound to one."""
    if isinstance(node, ast.Name):
        return node.id in texts
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _FOLDS:
        return _is_text(node.func.value, params, texts)
    return (isinstance(node, ast.Call) and _call_name(node) in _TEXT_CALLS and len(node.args) == 1
            and isinstance(node.args[0], ast.Name) and node.args[0].id in params)


def _status_owner(node: ast.AST) -> ast.AST | None:
    """``X`` for ``X.status_code`` / ``getattr(X, "status_code", ...)``, through one ``.response``."""
    owner = None
    if isinstance(node, ast.Attribute) and node.attr == "status_code":
        owner = node.value
    elif (isinstance(node, ast.Call) and _call_name(node) == "getattr" and len(node.args) >= 2
          and isinstance(node.args[1], ast.Constant) and node.args[1].value == "status_code"):
        owner = node.args[0]
    if isinstance(owner, ast.Attribute) and owner.attr == "response":
        return owner.value
    if isinstance(owner, ast.Call) and _call_name(owner) == "getattr" and len(owner.args) >= 2 \
            and isinstance(owner.args[1], ast.Constant) and owner.args[1].value == "response":
        return owner.args[0]
    return owner


def _offends(node: ast.AST, params: frozenset[str], texts: set[str]) -> bool:
    owner = _status_owner(node)
    if isinstance(owner, ast.Name) and owner.id in params:
        return True
    if isinstance(node, ast.Compare):
        return any(isinstance(op, (ast.In, ast.NotIn)) and _is_text(right, params, texts)
                   for op, right in zip(node.ops, node.comparators))
    if isinstance(node, ast.Call):
        name = _call_name(node).rsplit(".", 1)[-1]
        if name in ("startswith", "endswith") and isinstance(node.func, ast.Attribute):
            return _is_text(node.func.value, params, texts)
        return name in _PROBES and any(_is_text(arg, params, texts) for arg in node.args)
    return False


def private_error_predicate(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    """Each status-code read or text probe of an exception parameter in the def or lambda taking it (a
    nested def is judged on its own parameters); one hit per offending expression."""
    for func in ast.walk(tree):
        if not isinstance(func, (*_FUNCS, ast.Lambda)) or not (params := _exception_params(func)):
            continue
        nodes = list(_own(func))
        texts: set[str] = set()
        for _ in range(3):  # msg = str(exc); low = msg.lower()
            texts |= {target.id for node in nodes if isinstance(node, (ast.Assign, ast.NamedExpr))
                      for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                      if isinstance(target, ast.Name) and _is_text(node.value, params, texts)}
        yield from (node.lineno for node in nodes if isinstance(node, ast.expr) and _offends(node, params, texts))
