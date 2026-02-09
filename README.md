# bias-stackelberg

Repository scaffold (Milestone 0).

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
pytest
ruff check .
ruff format .
```

## CI

GitHub Actions runs `pytest` and `ruff` on every push and pull request.
