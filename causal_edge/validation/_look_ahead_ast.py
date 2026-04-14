"""AST helpers for the static look-ahead analyzer.

Extracted from look_ahead.py to keep each file under the 400-line structural
limit (tests/test_structure.py::TestFileSizeLimit). All helpers are private
(underscore prefix) and imported by look_ahead.py.

The def-use analysis here is intentionally scope-local (function-bounded). It
handles the patterns that routinely generate false positives in quant code:
windowed slices, list comprehensions, ML validation params.
"""
from __future__ import annotations

import ast


NUMPY_REDUCTIONS = {"std", "mean", "var"}

ML_PARAM_TOKENS = (
    "_hist", "hist_", "_slice", "_window", "window_",
    "x_fit", "y_fit", "x_val", "y_val", "x_test", "y_test",
    "x_train", "y_train", "prob", "prediction", "pred_",
)


def string_literal_spans(source: str) -> list[tuple[int, int]]:
    """Return (start, end) character offsets of every string literal."""
    spans: list[tuple[int, int]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return spans
    lines = source.split("\n")
    line_starts = [0]
    for line in lines[:-1]:
        line_starts.append(line_starts[-1] + len(line) + 1)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            start = line_starts[node.lineno - 1] + node.col_offset
            end_line = getattr(node, "end_lineno", node.lineno)
            end_col = getattr(node, "end_col_offset",
                              node.col_offset + len(node.value))
            end = line_starts[end_line - 1] + end_col
            spans.append((start, end))
    return spans


def in_string_literal(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in spans)


def is_numpy_reduction(call: ast.Call) -> bool:
    """True for np.std/mean/var(...)."""
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr not in NUMPY_REDUCTIONS:
        return False
    return isinstance(func.value, ast.Name) and func.value.id in ("np", "numpy")


def numpy_call_name(call: ast.Call) -> str:
    return call.func.attr if isinstance(call.func, ast.Attribute) else "?"


def node_offset(node: ast.AST, source: str) -> int | None:
    lineno = getattr(node, "lineno", None)
    col = getattr(node, "col_offset", None)
    if lineno is None or col is None:
        return None
    lines = source.split("\n")
    if lineno < 1 or lineno > len(lines):
        return None
    return sum(len(ln) + 1 for ln in lines[:lineno - 1]) + col


def safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)[:50]
    except Exception:
        return getattr(node, "id", "?")


def collect_scope_bindings(func: ast.AST) -> dict[str, list[ast.AST]]:
    """Return {var_name: [rhs_node, ...]} for every assignment in this scope.
    Function parameters map to their ast.arg sentinel. Does NOT descend into
    nested function defs.
    """
    bindings: dict[str, list[ast.AST]] = {}
    if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = func.args
        all_args = (
            args.args + args.kwonlyargs
            + ([args.vararg] if args.vararg else [])
            + ([args.kwarg] if args.kwarg else [])
        )
        for a in all_args:
            bindings.setdefault(a.arg, []).append(a)
    for node in _walk_scope(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for name in _assign_targets(target):
                    bindings.setdefault(name, []).append(node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            bindings.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            bindings.setdefault(node.target.id, []).append(node.iter)
    return bindings


def _walk_scope(func: ast.AST):
    """Walk body of a function/module but don't descend into nested defs."""
    stack = list(getattr(func, "body", []))
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            stack.append(child)


def _assign_targets(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        result = []
        for elt in node.elts:
            result.extend(_assign_targets(elt))
        return result
    return []


def is_bounded_expr(node: ast.AST, bindings: dict[str, list[ast.AST]]) -> bool:
    """True if node is clearly a windowed/bounded expression (not full array)."""
    if contains_slice(node):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict,
                         ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        return True
    if isinstance(node, ast.Name):
        if name_is_bounded(node.id, bindings):
            return True
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        if name_is_bounded(node.value.id, bindings):
            return True
    if isinstance(node, ast.Call):
        if any(contains_slice(a) for a in node.args):
            return True
        if isinstance(node.func, ast.Attribute):
            inner = node.func.value
            if isinstance(inner, ast.Name) and name_is_bounded(inner.id, bindings):
                return True
    if isinstance(node, (ast.BinOp, ast.Compare, ast.BoolOp, ast.UnaryOp)):
        for child in ast.iter_child_nodes(node):
            if is_bounded_expr(child, bindings):
                return True
    return False


def name_is_bounded(name: str, bindings: dict[str, list[ast.AST]]) -> bool:
    lower = name.lower()
    if any(tok in lower for tok in ML_PARAM_TOKENS):
        return True
    rhs_list = bindings.get(name)
    if not rhs_list:
        return False
    return all(rhs_is_bounded(rhs) for rhs in rhs_list)


def rhs_is_bounded(rhs: ast.AST) -> bool:
    if isinstance(rhs, ast.arg):
        lower = rhs.arg.lower()
        return any(tok in lower for tok in ML_PARAM_TOKENS)
    if isinstance(rhs, (ast.List, ast.Tuple, ast.Set, ast.Dict,
                        ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        return True
    if contains_slice(rhs):
        return True
    if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name):
        if rhs.func.id in ("range", "enumerate", "zip"):
            return True
    if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Attribute):
        for child in ast.walk(rhs):
            if isinstance(child, ast.Name) and any(
                tok in child.id.lower() for tok in ML_PARAM_TOKENS
            ):
                return True
    return False


def contains_slice(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript) and isinstance(child.slice, ast.Slice):
            return True
    return False
