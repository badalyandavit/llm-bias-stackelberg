import json

from bias_stackelberg.data.toy import toy_examples
from bias_stackelberg.eval.runner import EvalAConfig, run_option_a
from bias_stackelberg.follower.option_a import OptionAConfig
from bias_stackelberg.models import GenConfig


def test_eval_runner_writes_artifacts(tmp_path) -> None:
    out_dir = tmp_path / "run"
    cfg = EvalAConfig(
        out_dir=str(out_dir),
        gen=GenConfig(seed=0, max_tokens=16, temperature=0.7, top_p=1.0),
        follower=OptionAConfig(trigger_threshold=0.0),
    )

    m = run_option_a(toy_examples(), cfg=cfg)
    assert m.n == 4

    preds = out_dir / "predictions.jsonl"
    mets = out_dir / "metrics.json"

    assert preds.exists()
    assert mets.exists()

    d = json.loads(mets.read_text(encoding="utf-8"))
    assert d["n"] == 4
