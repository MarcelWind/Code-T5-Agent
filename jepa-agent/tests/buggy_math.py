from typing import List


def compute_sum(n: int) -> int:
    """Return sum of numbers from 1 to n (inclusive)."""
    return sum(range(1, n + 1))


def divide(a: float, b: float) -> float:
    """Return a divided by b."""
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
