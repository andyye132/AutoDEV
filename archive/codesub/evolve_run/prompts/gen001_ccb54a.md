# System Instructions

You are improving a Python data-canonicalization program to pass as many tests as possible. A record is a messy
contact dict (name, email, phone, tags). Canonicalizer.canonical(record) must return a canonical string. You are scored by
an automated test suite; maximize the pass rate. Only edit code between the EVOLVE-BLOCK markers. Return the COMPLETE
replacement for the EVOLVE-BLOCK region (everything between the markers, not the marker lines) inside <PROGRAM>...</PROGRAM>.
Use only the Python standard library. Keep the class name Canonicalizer and the method signature canonical(self, record).

# User Request

Current program (you may change only the EVOLVE-BLOCK region):
```python
"""Candidate solver (the file the loop evolves). Implement Canonicalizer.canonical(record) -> str to maximize the test
pass rate. Only edit between the EVOLVE-BLOCK markers. This seed is an honest, partial attempt: it handles the common
cases but has a few ordinary edge-case bugs (the kind any first attempt ships)."""
import re

# EVOLVE-BLOCK-START
FIELD_ORDER = ("name", "email", "phone", "tags")


class Canonicalizer:
    def _ws(self, s):
        return re.sub(r"\s+", " ", s).strip()

    def canonical(self, record):
        parts = []
        for f in FIELD_ORDER:
            if f not in record:
                continue
            v = record[f]
            if f == "name":
                out = self._ws(v)                       # bug: no unicode NFC normalization
            elif f == "email":
                local, _, dom = str(v).partition("@")
                out = self._ws(local) + "@" + self._ws(dom).lower() if "@" in str(v) else self._ws(v).lower()
            elif f == "phone":
                digits = re.sub(r"\D", "", str(v))
                out = ("+1-" + digits[-10:]) if digits else ""   # bug: assumes country code 1, base format only
            elif f == "tags":
                tags = [self._ws(t).lower() for t in v] if v else []
                out = ",".join(sorted(tags))            # bug: no dedup
            else:
                out = self._ws(str(v))
            if out != "":
                parts.append(f"{f}={out}")
        return "|".join(parts)
# EVOLVE-BLOCK-END


def get_solver():
    return Canonicalizer()

```
Evaluator feedback: Passed 112/181 tests. Failing record ids (first-in-group): e003, e004

Propose an improved EVOLVE-BLOCK that passes more tests. Return the complete replacement region inside <PROGRAM>...</PROGRAM>.