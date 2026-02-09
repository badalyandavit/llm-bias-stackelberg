import json

from bias_stackelberg.data.io import load_examples_jsonl


def test_load_examples_jsonl(tmp_path) -> None:
    p = tmp_path / "prompts.jsonl"
    rows = [
        {"id": "a", "prompt": "hello", "meta": {"k": 1}},
        {"prompt": "world"},
    ]
    p.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")

    ex = load_examples_jsonl(str(p))
    assert len(ex) == 2
    assert ex[0].id == "a"
    assert ex[0].prompt == "hello"
    assert ex[0].meta == {"k": 1}
    assert ex[1].id == "line_2"
    assert ex[1].prompt == "world"
