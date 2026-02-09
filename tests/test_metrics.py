from bias_stackelberg.eval.metrics import compute_metrics


def test_compute_metrics_basic() -> None:
    rows = [
        {"before": {"score": 0.4}, "after": {"score": 0.2}, "decision": {"action": "rewrite"}},
        {"before": {"score": 0.1}, "after": {"score": 0.1}, "decision": {"action": "accept"}},
    ]
    m = compute_metrics(rows)
    assert m.n == 2
    assert m.mean_before == (0.4 + 0.1) / 2
    assert m.mean_after == (0.2 + 0.1) / 2
    assert m.rewrite_rate == 0.5
    assert m.accept_rate == 0.5
