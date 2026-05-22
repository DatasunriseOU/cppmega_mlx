"""V7-H06: job pause/resume primitive tests."""

from __future__ import annotations

import threading
import time

import pytest

from cppmega_v4.runtime import job_control as jc


@pytest.fixture(autouse=True)
def _reset():
    jc.reset()
    yield
    jc.reset()


def test_v7_h06_pause_sets_flag_resume_clears():
    assert jc.is_paused("t1") is False
    jc.pause("t1")
    assert jc.is_paused("t1") is True
    jc.resume("t1")
    assert jc.is_paused("t1") is False


def test_v7_h06_pause_is_idempotent():
    jc.pause("t1")
    jc.pause("t1")
    jc.resume("t1")
    assert jc.is_paused("t1") is False


def test_v7_h06_null_or_empty_token_never_paused():
    assert jc.is_paused(None) is False
    assert jc.is_paused("") is False


def test_v7_h06_wait_while_paused_returns_when_resumed():
    jc.pause("job-X")

    def _resumer():
        time.sleep(0.15)
        jc.resume("job-X")

    threading.Thread(target=_resumer, daemon=True).start()
    t0 = time.time()
    jc.wait_while_paused("job-X", poll_s=0.02, max_wait_s=2.0)
    elapsed = time.time() - t0
    assert 0.10 <= elapsed < 2.0


def test_v7_h06_wait_while_paused_bounded_by_max_wait():
    jc.pause("stuck")
    t0 = time.time()
    jc.wait_while_paused("stuck", poll_s=0.02, max_wait_s=0.1)
    elapsed = time.time() - t0
    assert 0.08 < elapsed < 0.5


def test_v7_h06_distinct_tokens_independent():
    jc.pause("a")
    assert jc.is_paused("a") is True
    assert jc.is_paused("b") is False
