import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast unit tests with no GCS dependency")
    config.addinivalue_line("markers", "integration: tests that require GCS credentials")