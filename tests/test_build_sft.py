import json

from bias_stackelberg.core.prompts import detox_rewrite_prompt
from bias_stackelberg.data.sft import BuildSftConfig, build_sft_dataset


def test_build_sft_keeps_only_improving_rewrites(tmp_path) -> None:
    preds = tmp_path / "predictions.jsonl"
    out_dir = tmp_path / "sft_out"

    lines = [
        {
            "id": "a",
            "prompt": "p",
            "meta": {},
            "y0": {"text": "Women are bad at math.", "meta": {}},
            "y1": {"text": "People improve with practice.", "meta": {}},
            "before": {
                "score": 0.5,
                "categories": [],
                "rationale": "",
                "evidence_spans": [],
                "trace": {},
            },
            "after": {
                "score": 0.0,
                "categories": [],
                "rationale": "",
                "evidence_spans": [],
                "trace": {},
            },
            "decision": {
                "action": "rewrite",
                "reason": "ok",
                "before_score": 0.5,
                "after_score": 0.0,
            },
        },
        {
            "id": "b",
            "prompt": "p2",
            "meta": {},
            "y0": {"text": "Neutral.", "meta": {}},
            "y1": {"text": "Neutral.", "meta": {}},
            "before": {
                "score": 0.0,
                "categories": [],
                "rationale": "",
                "evidence_spans": [],
                "trace": {},
            },
            "after": {
                "score": 0.0,
                "categories": [],
                "rationale": "",
                "evidence_spans": [],
                "trace": {},
            },
            "decision": {
                "action": "abstain",
                "reason": "no",
                "before_score": 0.0,
                "after_score": None,
            },
        },
    ]
    preds.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    summary = build_sft_dataset(
        cfg=BuildSftConfig(in_predictions=str(preds), out_dir=str(out_dir), min_improvement=1e-9)
    )
    assert summary["kept"] == 1
    assert summary["dropped"] == 1

    sft_path = out_dir / "sft.jsonl"
    ref_path = out_dir / "reference.jsonl"
    drop_path = out_dir / "dropped.jsonl"
    sum_path = out_dir / "build_summary.json"

    assert sft_path.exists()
    assert ref_path.exists()
    assert drop_path.exists()
    assert sum_path.exists()

    sft_lines = sft_path.read_text(encoding="utf-8").splitlines()
    assert len(sft_lines) == 1
    obj = json.loads(sft_lines[0])
    assert obj["id"] == "a"
    assert obj["prompt"] == detox_rewrite_prompt("Women are bad at math.")
    assert "completion" in obj
