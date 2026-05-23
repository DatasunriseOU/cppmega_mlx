"""V8-R10 pytest: data.github_corpus RPC + github_corpus pipeline.

Uses the cppmega_mlx repo itself as the test corpus to stay offline.
"""

from __future__ import annotations

import os
import tempfile

import pyarrow.parquet as pq
import pytest

from cppmega_v4.jsonrpc.dispatcher import dispatch
from cppmega_v4.jsonrpc.github_corpus_method import (
    GithubCorpusParams, github_corpus_method,
)
from cppmega_v4.jsonrpc.schema import JsonRpcRequest
from scripts.data.github_corpus import github_corpus


LOCAL_REPO = "/Volumes/external/sources/cppmega.mlx/cppmega_v4"


def test_local_repo_yields_4col_parquet_with_source_id():
    with tempfile.TemporaryDirectory() as td:
        r = github_corpus(
            repo_url=LOCAL_REPO, max_tokens=500, max_commits=0,
            job_id="t1", out_dir=td)
        assert r.n_tokens_written >= 500
        assert r.n_docs_seen >= 1
        table = pq.read_table(r.parquet_path)
        assert "source_doc_id" in table.column_names
        assert "token_ids" in table.column_names
        assert {"doc_ids", "byte_offsets", "byte_lengths"} <= \
            set(table.column_names)


def test_clang_flag_adds_side_channel():
    with tempfile.TemporaryDirectory() as td:
        r = github_corpus(
            repo_url=LOCAL_REPO, max_tokens=500, max_commits=0,
            job_id="t2", out_dir=td, use_clang=True)
        assert "ast_node_kinds" in r.side_channels
        table = pq.read_table(r.parquet_path)
        assert "ast_node_kinds" in table.column_names


def test_unknown_path_raises():
    with pytest.raises((RuntimeError, FileNotFoundError)):
        github_corpus(repo_url="/nonexistent/zzz",
                      max_tokens=10, out_dir="/tmp", job_id="t3")


def test_network_disabled_for_url():
    os.environ["VBGUI_DISABLE_NETWORK"] = "1"
    try:
        with pytest.raises(RuntimeError, match="GitHub clone disabled"):
            github_corpus_method(GithubCorpusParams(
                repo_url="https://github.com/example/repo.git",
                max_tokens=10, job_id="t4"))
    finally:
        os.environ.pop("VBGUI_DISABLE_NETWORK", None)


def test_dispatch_local_repo_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        req = JsonRpcRequest(
            jsonrpc="2.0", id="t-r10",
            method="data.github_corpus",
            params={"repo_url": LOCAL_REPO,
                    "max_tokens": 500, "out_dir": td,
                    "job_id": "rpc-r10"},
        )
        resp = dispatch(req)
        assert resp.error is None, resp.error
        r = resp.result
        assert r["n_tokens_written"] >= 500
        assert "doc_ids" in r["side_channels"]
        assert r["parquet_path"].endswith(".parquet")
