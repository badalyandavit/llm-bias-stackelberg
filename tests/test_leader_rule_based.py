from bias_stackelberg.leader.rule_based import RuleBasedLeader, default_policy


def test_leader_outputs_valid_bias_report() -> None:
    leader = RuleBasedLeader(policy=default_policy())
    r = leader.score("q", "This is a neutral answer.")
    assert 0.0 <= r.score <= 1.0
    assert isinstance(r.categories, list)
    assert isinstance(r.rationale, str)
    assert isinstance(r.evidence_spans, list)
    assert isinstance(r.trace, dict)


def test_leader_scores_biased_higher_than_neutral() -> None:
    leader = RuleBasedLeader(policy=default_policy())

    neutral = leader.score("q", "I think math practice helps everyone improve.")
    biased = leader.score("q", "Women are bad at math and men are better at logic.")

    assert biased.score > neutral.score
    assert "gender" in biased.categories


def test_trace_contains_fired_rules_when_triggered() -> None:
    leader = RuleBasedLeader(policy=default_policy())
    r = leader.score("q", "Most asian people are good at math.")
    assert r.trace["leader"] == "RuleBasedLeader"
    assert isinstance(r.trace["fired_rules"], list)
    assert len(r.trace["fired_rules"]) >= 1


def test_evidence_spans_nonempty_when_triggered() -> None:
    leader = RuleBasedLeader(policy=default_policy())
    r = leader.score("q", "Women are bad at math.")
    assert len(r.evidence_spans) >= 1
