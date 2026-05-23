"""V7-N03: tl.dot(trans_b=True) → tl.dot(tl.trans(b)) AST rewrite."""

from __future__ import annotations

import ast
import inspect
import textwrap

from cppmega_mlx.nn._triton_bridge import (
    rewrite_dot_trans_b_to_transpose,
)


class _FakeTL:
    @staticmethod
    def dot(a, b, **kw):
        if kw.get("trans_b"):
            b = b.T
        return a @ b

    @staticmethod
    def trans(b):
        return b.T


tl = _FakeTL  # exposed at module scope so __globals__ resolves it.


def kernel(a, b):
    return tl.dot(a, b, trans_b=True)


def kernel_no_trans(a, b):
    return a @ b


def test_rewrite_replaces_trans_b_with_tl_trans_in_ast():
    src = textwrap.dedent(inspect.getsource(kernel))
    tree = ast.parse(src)
    has_trans_b_kw = any(
        isinstance(node, ast.keyword) and node.arg == "trans_b"
        for node in ast.walk(tree)
    )
    assert has_trans_b_kw, "fixture sanity: original has trans_b kwarg"

    rewritten = rewrite_dot_trans_b_to_transpose(kernel)
    import numpy as np
    a = np.random.RandomState(0).randn(3, 4)
    b = np.random.RandomState(1).randn(5, 4)
    expected = a @ b.T
    got = rewritten(a, b)
    assert np.allclose(got, expected, atol=1e-6)


def test_rewrite_no_op_when_trans_b_absent():
    """A kernel without trans_b should round-trip unchanged."""
    rewritten = rewrite_dot_trans_b_to_transpose(kernel_no_trans)
    import numpy as np
    a = np.eye(3)
    b = np.eye(3)
    assert np.allclose(rewritten(a, b), a @ b)
