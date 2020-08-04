from typing import List


def multiple_of(values: List[int]):
    def check(n: int):
        for _ in values:
            if n % _ == 0:
                return True
        return False

    return check