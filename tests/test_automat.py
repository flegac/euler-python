import json
import random

from euler.lib.automate import Automate


def build_data(states: int, edges: int):
    S = list(map(str, range(states)))
    A = list(map(str, range(5)))
    Q = Automate.builder()

    for i in range(edges):
        s1 = random.choice(S)
        a = random.choice(A)
        s2 = random.choice(S)
        Q[s1][a][s2] += 1

    return {
        "A": list(sorted(A)),
        "S": list(sorted(S)),
        "Q": Automate.builder_clean(Q),
        "I": random.choice(S),
        "T": list(sorted(random.sample(S, 10)))
    }


def test_automat():
    data = build_data(20, 30)
    a = Automate.from_json(data)
    data2 = a.to_json()
    assert data == data2


def test_from_scratch():
    states = 1000
    edges = 12
    S = list(map(str, range(states)))
    A = list(map(str, range(5)))

    Q = Automate.from_scratch()
    Q.T = set(sorted(random.sample(S, 5)))
    Q.I = S[0]

    for s in Q.T:
        Q.add(Q.I, A[0], s, 1)
    for i in range(edges):
        s1 = random.choice(S)
        a = random.choice(A)
        s2 = random.choice(S)
        Q.add(s1, a, s2, 1)

    Q.validate()
    print(json.dumps(Q.to_json(), indent=4, sort_keys=True))
