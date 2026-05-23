"""V7-F06: gen_event_bus pub/sub + gen.run job_id wiring."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cppmega_v4.jsonrpc import create_app
from cppmega_v4.jsonrpc.gen_run_method import GenRunParams, gen_run
from cppmega_v4.runtime import gen_event_bus as bus


@pytest.fixture(autouse=True)
def _clean():
    bus.reset()
    yield
    bus.reset()


@pytest.fixture
def client():
    return TestClient(create_app(cache_capacity=2))


def test_gen_run_without_job_id_does_not_publish():
    res = gen_run(GenRunParams(prompt_tokens=[0], eos_token_id=-1,
                                 max_new_tokens=3))
    # No subscribers, no job_id — bus stays empty.
    assert bus.subscriber_count("anything") == 0
    assert len(res.events) == 3


def test_gen_run_with_job_id_publishes_per_token_and_finish():
    q = bus.subscribe("job-7")
    res = gen_run(GenRunParams(prompt_tokens=[0], eos_token_id=-1,
                                 max_new_tokens=4,
                                 job_id="job-7"))
    received: list = []
    while True:
        ev = q.get(timeout=1.0)
        if ev is None:
            break
        received.append(ev)
    assert len(received) == 4
    for ev in received:
        assert "token_id" in ev
        assert "step" in ev
    assert res.finish_reason == "length"


def test_gen_run_unsubscribe_isolates_jobs():
    q1 = bus.subscribe("job-A")
    q2 = bus.subscribe("job-B")
    gen_run(GenRunParams(prompt_tokens=[0], eos_token_id=-1,
                          max_new_tokens=2, job_id="job-A"))
    # job-A queue receives events + sentinel; job-B stays untouched.
    a_count = 0
    while not q1.empty():
        ev = q1.get_nowait()
        if ev is None:
            break
        a_count += 1
    assert a_count == 2
    assert q2.empty()
    bus.unsubscribe("job-A", q1)
    bus.unsubscribe("job-B", q2)
