from typing import List, Tuple
import random

def process_random_uint_vector(k: int, n: int) -> Tuple[List[int], List[int]]:
    """Generates a random uint vector with K highest possible element and N total elements,
    sorts it, removes duplicates, and returns both the sorted list with dupes and the deduped list."""
    vec = [random.randint(0, k-1) for _ in range(n)]
    sorted_vec = sorted(vec)
    deduped = []
    for x in sorted_vec:
        if not deduped or x != deduped[-1]:
            deduped.append(x)
    return sorted_vec, deduped