# Create src/utils/metrics.py
from time import time
from functools import wraps
from typing import Callable
import logging

logger = logging.getLogger(__name__)

def timing_decorator(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        logger.info(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper