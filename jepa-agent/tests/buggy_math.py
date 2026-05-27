from typing import List, Generator, Union
import math


def compute_sum(n: int) -> int:
    """Return sum of numbers from 0 to n-1."""
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n <= 0:
        return 0
    return n * (n - 1) // 2


def divide(a: float, b: float) -> float:
    """Return a divided by b. Returns NaN if b is zero."""
    if math.isclose(b, 0.0):
        return math.nan
    return a / b


def fibonacci(n: int, as_list: bool = True) -> Union[List[int], Generator[int, None, None]]:
    """Return first n Fibonacci numbers. If as_list is True (default), return list; else return generator."""
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")
    if n > 1000000:
        raise OverflowError("n too large, may cause memory issues")
    def gen() -> Generator[int, None, None]:
        a, b = 0, 1
        for _ in range(n):
            yield a
            a, b = b, a + b
    if as_list:
        return list(gen())
    return gen()