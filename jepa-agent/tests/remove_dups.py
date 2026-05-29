from typing import List

def remove_duplicates(arr: List[int]) -> List[int]:
    if not arr:
        return []
    i = 0
    for x in arr:
        if i == 0 or x != arr[i-1]:
            arr[i] = x
            i += 1
    return arr[:i]