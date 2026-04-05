import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "eval: marks tests as eval regression tests (run with: pytest eval/ -m eval)",
    )
