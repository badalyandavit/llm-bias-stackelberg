import pytest

from bias_stackelberg.core.types import BiasReport


def test_bias_report_roundtrip() -> None:
    r = BiasReport(
        score=0.25,
        categories=["stereotype", "gender"],
        rationale="Contains a gender stereotype.",
        evidence_spans=["women are bad at math"],
        trace={"rubric": {"node": "gender_stereotype", "confidence": 0.8}},
    )

    d = r.to_dict()
    r2 = BiasReport.from_dict(d)
    assert r2 == r


def test_bias_report_validate_dict_rejects_missing_keys() -> None:
    with pytest.raises(ValueError):
        BiasReport.validate_dict({"score": 0.1})
