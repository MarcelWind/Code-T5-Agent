from typing import List
import math


def compute_sum(n: int) -> int:
    """Return sum of numbers from 0 to n-1 (inclusive)."""
    return n * (n - 1) // 2


def divide(a: float, b: float) -> float:
    """Return a divided by b. Returns NaN if b is zero."""
    if math.isclose(b, 0.0):
        return math.nan
    return a / b


def fibonacci(n: int) -> List[int]:
    """Return first n Fibonacci numbers."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    result = [0, 1]
    for i in range(n - 2):
        result.append(result[-1] + result[-2])
    return result
