"""Candidate solver (the file the loop evolves). Implement Canonicalizer.canonical(record) -> str to maximize the test
pass rate. Only edit between the EVOLVE-BLOCK markers. This seed is an honest, partial attempt: it handles the common
cases but has a few ordinary edge-case bugs (the kind any first attempt ships)."""
import re

# EVOLVE-BLOCK-START
import unicodedata

FIELD_ORDER = ("name", "email", "phone", "tags")


class Canonicalizer:
    def _ws(self, s):
        return re.sub(r"\s+", " ", s).strip()

    def _norm_text(self, s):
        return unicodedata.normalize("NFC", self._ws(unicodedata.normalize("NFC", str(s))))

    def canonical(self, record):
        parts = []
        for f in FIELD_ORDER:
            if f not in record:
                continue
            v = record[f]
            if v is None:
                continue
            if f == "name":
                out = self._norm_text(v)
            elif f == "email":
                s = self._norm_text(v)
                if "@" in s:
                    local, _, dom = s.rpartition("@")
                    out = self._ws(local) + "@" + self._ws(dom).casefold()
                else:
                    out = s.casefold()
            elif f == "phone":
                digits = re.sub(r"\D", "", str(v))
                if not digits:
                    out = ""
                elif len(digits) > 10:
                    cc = digits[:-10]
                    out = "+" + cc + "-" + digits[-10:]
                else:
                    out = "+1-" + digits[-10:] if len(digits) == 10 else "+1-" + digits
            elif f == "tags":
                if v:
                    tags = sorted({self._norm_text(t).casefold() for t in v if v is not None and self._ws(str(t)) != ""})
                else:
                    tags = []
                out = ",".join(tags)
            else:
                out = self._norm_text(v)
            if out != "":
                parts.append(f"{f}={out}")
        return "|".join(parts)
# EVOLVE-BLOCK-END


def get_solver():
    return Canonicalizer()
