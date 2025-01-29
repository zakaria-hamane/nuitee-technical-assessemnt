import pytest
import logging

@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for tests"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )