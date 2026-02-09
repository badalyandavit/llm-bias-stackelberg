from bias_stackelberg.models import GenConfig, MockLLM


def test_mock_llm_deterministic_same_seed() -> None:
    llm = MockLLM()
    cfg = GenConfig(seed=123, max_tokens=10, temperature=0.5, top_p=0.9)

    a = llm.generate("hello world", config=cfg).text
    b = llm.generate("hello world", config=cfg).text
    assert a == b


def test_mock_llm_changes_with_seed() -> None:
    llm = MockLLM()
    a = llm.generate("hello world", config=GenConfig(seed=1)).text
    b = llm.generate("hello world", config=GenConfig(seed=2)).text
    assert a != b
