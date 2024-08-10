import os

import pytest


@pytest.fixture(autouse=True)
def create_pytest_environ():
    os.environ["RUNNING_PYTEST"] = "true"
    yield
    del os.environ["RUNNING_PYTEST"]
