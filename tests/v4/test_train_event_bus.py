"""V7-H05: train_event_bus pub/sub correctness."""

from __future__ import annotations

import queue
import threading

import pytest

from cppmega_v4.runtime import train_event_bus as bus


@pytest.fixture(autouse=True)
def _clean():
    bus.reset()
    yield
    bus.reset()


def test_publish_with_no_subscribers_is_noop():
    bus.publish("run-x", {"step": 0})
    assert bus.subscriber_count("run-x") == 0


def test_subscribe_then_publish_delivers_event():
    q = bus.subscribe("run-1")
    bus.publish("run-1", {"step": 0, "loss": 1.5})
    bus.publish("run-1", None)
    assert q.get_nowait() == {"step": 0, "loss": 1.5}
    assert q.get_nowait() is None


def test_publish_to_other_run_id_is_isolated():
    q1 = bus.subscribe("run-A")
    q2 = bus.subscribe("run-B")
    bus.publish("run-A", {"step": 0})
    assert q1.get_nowait()["step"] == 0
    with pytest.raises(queue.Empty):
        q2.get_nowait()


def test_unsubscribe_removes_queue_from_distribution():
    q = bus.subscribe("run-1")
    bus.unsubscribe("run-1", q)
    bus.publish("run-1", {"step": 0})
    assert bus.subscriber_count("run-1") == 0


def test_multiple_subscribers_all_receive():
    q1 = bus.subscribe("run-1")
    q2 = bus.subscribe("run-1")
    bus.publish("run-1", {"step": 0})
    assert q1.get_nowait()["step"] == 0
    assert q2.get_nowait()["step"] == 0


def test_cross_thread_publish_safe():
    q = bus.subscribe("run-thr")

    def _producer():
        for i in range(5):
            bus.publish("run-thr", {"step": i})
        bus.publish("run-thr", None)

    t = threading.Thread(target=_producer)
    t.start()
    t.join(timeout=2.0)
    received: list = []
    while True:
        ev = q.get(timeout=1.0)
        if ev is None:
            break
        received.append(ev["step"])
    assert received == [0, 1, 2, 3, 4]
