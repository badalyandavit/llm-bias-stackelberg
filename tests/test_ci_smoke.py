def test_import_smoke() -> None:
    import bias_stackelberg

    assert isinstance(bias_stackelberg.__version__, str)
