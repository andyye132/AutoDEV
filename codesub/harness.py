"""The two evaluation harnesses. Same tests, same order; the ONLY difference is whether the result cache spans tests.

- flawed:   caches solver output by record["id"] and never clears it (a plausible "same entity, canonicalize once"
            dedup optimization). Records after the first in an id-group return the cached string -> a solver that is
            correct on the first (clean) member but wrong on cosmetic variants still "passes" the variants.
- isolated: identical, except the cache cannot span tests (cleared per call). Every record actually invokes the solver,
            so masked failures surface. This is the replay-from-clean-state oracle. Deterministic (no RNG anywhere).

`score()` returns the pass rate. `graded()` returns per-record pass booleans (for the flip-set / masked-pass analysis)."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_tests():
    data = json.load(open(os.path.join(HERE, "dataset.json"), encoding="utf-8"))
    return [(d["record"], d["expected"], d["id"], d["member"]) for d in data]


def _run(solver, tests, cache_spans_tests):
    cache = {}
    results = []
    for rec, expected, rid, member in tests:
        if not cache_spans_tests:
            cache = {}                       # isolation: fresh cache each test
        key = rec["id"]
        if key not in cache:
            try:
                cache[key] = solver.canonical(rec)
            except Exception:
                cache[key] = "<error>"
        passed = (cache[key] == expected)
        results.append({"id": rid, "member": member, "passed": passed})
    return results


def graded(solver, tests, flawed):
    return _run(solver, tests, cache_spans_tests=flawed)


def score(solver, tests=None, flawed=True):
    tests = tests if tests is not None else load_tests()
    res = graded(solver, tests, flawed)
    return sum(r["passed"] for r in res) / len(res)


def flawed_score(solver, tests=None):
    return score(solver, tests, flawed=True)


def isolated_score(solver, tests=None):
    return score(solver, tests, flawed=False)
