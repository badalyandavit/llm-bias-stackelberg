from bias_stackelberg.models import GenConfig, MockLLM


def test_kwargs_propagated_to_meta() -> None:
    llm = MockLLM()
    gen = llm.generate("x", config=GenConfig(seed=7), foo="bar", n=2)
    assert gen.meta["extra_kwargs"] == {"foo": "bar", "n": 2}
