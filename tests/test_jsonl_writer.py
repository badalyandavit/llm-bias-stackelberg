import json

from bias_stackelberg.core.jsonl import JsonlWriter
from bias_stackelberg.core.types import BiasReport


def test_jsonl_writer_writes_valid_json_lines(tmp_path) -> None:
    p = tmp_path / "logs" / "out.jsonl"

    r = BiasReport(
        score=0.5,
        categories=["toxicity"],
        rationale="Example rationale",
        evidence_spans=[],
        trace={"a": 1, "b": {"c": True}},
    )

    with JsonlWriter(p) as w:
        w.write({"k": "v"})
        w.write(r)

    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    obj0 = json.loads(lines[0])
    obj1 = json.loads(lines[1])

    assert obj0 == {"k": "v"}
    assert set(obj1.keys()) == {"categories", "evidence_spans", "rationale", "score", "trace"}
