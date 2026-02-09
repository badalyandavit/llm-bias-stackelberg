from bias_stackelberg.follower.option_a import OptionAConfig, OptionAFollower
from bias_stackelberg.leader.rule_based import RuleBasedLeader, default_policy
from bias_stackelberg.models import MockLLM


def test_option_a_accepts_when_below_threshold() -> None:
    leader = RuleBasedLeader(policy=default_policy())
    follower = OptionAFollower(
        leader=leader,
        llm=MockLLM(),
        config=OptionAConfig(trigger_threshold=0.9),
    )

    x = "Explain why practice helps learning."
    y = "Practice helps people improve over time."
    report = leader.score(x, y)

    gen, dec = follower.intervene(x, y, report)
    assert dec.action == "accept"
    assert gen.text == y


def test_option_a_abstains_or_rewrites_when_above_threshold() -> None:
    leader = RuleBasedLeader(policy=default_policy())
    follower = OptionAFollower(
        leader=leader,
        llm=MockLLM(),
        config=OptionAConfig(trigger_threshold=0.0),
    )

    x = "Give advice."
    y = "Women are bad at math."
    report = leader.score(x, y)

    gen, dec = follower.intervene(x, y, report)
    assert dec.action in {"rewrite", "abstain"}
    assert gen.meta["follower"] == "option_a"


def test_option_a_never_returns_rewrite_with_worse_score() -> None:
    leader = RuleBasedLeader(policy=default_policy())
    follower = OptionAFollower(
        leader=leader,
        llm=MockLLM(),
        config=OptionAConfig(trigger_threshold=0.0),
    )

    x = "Give advice."
    y = "Women are bad at math."
    before = leader.score(x, y).score

    gen, dec = follower.intervene(x, y, leader.score(x, y))

    if dec.action == "rewrite":
        after = leader.score(x, gen.text).score
        assert after <= before
