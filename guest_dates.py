"""Extract stay-related dates from guest text; gate PMS tool arguments."""

from __future__ import annotations

import json
import re
import time
from datetime import date, timedelta
from typing import Iterable

_MAX_RANGE_DAYS = 400

# #region agent log
_DBG_LOG_PATH = "/Users/musa/Documents/repo's/hotel/.cursor/debug-2b3b1c.log"


def _agent_dbg_log(
    location: str,
    message: str,
    data: dict,
    hypothesis_id: str = "",
) -> None:
    try:
        rec = {
            "sessionId": "2b3b1c",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "hypothesisId": hypothesis_id,
        }
        with open(_DBG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


# #endregion

_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DOT = re.compile(r"\b(\d{1,2})[.](\d{1,2})[.](\d{4})\b")
_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")

_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_MONTH_NAMES_RE = "|".join(
    re.escape(name) for name in sorted(set(_MONTHS.keys()), key=len, reverse=True)
)

_MONTH_WORD = re.compile(
    rf"\b("
    rf"(?P<m1>{_MONTH_NAMES_RE})\s+(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?(?:\s*(?:,)?\s*(?P<y1>\d{{4}}))?"
    rf"|"
    rf"(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<m2>{_MONTH_NAMES_RE})(?:\s*(?:,)?\s*(?P<y2>\d{{4}}))?"
    rf")\b",
    re.IGNORECASE,
)

_DASH = r"[\u2013\u2014\-]"
# Same-month ranges: dash, "to", "through", "until" (e.g. "May 23rd to 25th", "23 through 25 May").
_RANGE_SEP = rf"(?:\s*{_DASH}\s*|\s+to\s+|\s+through\s+|\s+until\s+)"
_MONTH_THEN_DAY_RANGE = re.compile(
    rf"\b(?P<mr>{_MONTH_NAMES_RE})\s+"
    rf"(?P<da>\d{{1,2}})(?:st|nd|rd|th)?{_RANGE_SEP}"
    rf"(?P<db>\d{{1,2}})(?:st|nd|rd|th)?"
    rf"(?:\s*(?:,)?\s*(?P<yr>\d{{4}}))?\b",
    re.IGNORECASE,
)
_DAY_RANGE_THEN_MONTH = re.compile(
    rf"\b(?P<da>\d{{1,2}})(?:st|nd|rd|th)?{_RANGE_SEP}"
    rf"(?P<db>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<mr>{_MONTH_NAMES_RE})"
    rf"(?:\s*(?:,)?\s*(?P<yr>\d{{4}}))?\b",
    re.IGNORECASE,
)
_DAY_BETWEEN_AND_MONTH = re.compile(
    rf"\bbetween\s+(?P<da>\d{{1,2}})(?:st|nd|rd|th)?\s+and\s+"
    rf"(?P<db>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<mr>{_MONTH_NAMES_RE})"
    rf"(?:\s*(?:,)?\s*(?P<yr>\d{{4}}))?\b",
    re.IGNORECASE,
)
_MONTH_BETWEEN_AND_DAY = re.compile(
    rf"\bbetween\s+(?P<mr>{_MONTH_NAMES_RE})\s+"
    rf"(?P<da>\d{{1,2}})(?:st|nd|rd|th)?\s+and\s+"
    rf"(?P<db>\d{{1,2}})(?:st|nd|rd|th)?"
    rf"(?:\s*(?:,)?\s*(?P<yr>\d{{4}}))?\b",
    re.IGNORECASE,
)


def _ymd(y: int, m: int, d: int) -> str | None:
    try:
        date(y, m, d)
    except ValueError:
        return None
    return f"{y:04d}-{m:02d}-{d:02d}"


def _from_slash(a: str, b: str, y: str) -> str | None:
    ia, ib, iy = int(a), int(b), int(y)
    if ia > 12:
        return _ymd(iy, ib, ia)
    if ib > 12:
        return _ymd(iy, ia, ib)
    return _ymd(iy, ia, ib)


def _add_inclusive_day_span_to_month(y: int, mon: int, da: int, db: int, out: set[str]) -> None:
    lo, hi = (da, db) if da <= db else (db, da)
    if hi - lo > _MAX_RANGE_DAYS:
        return
    for d in range(lo, hi + 1):
        s = _ymd(y, mon, d)
        if s:
            out.add(s)


def _extract_guest_stay_date_literals_inner(
    text: str,
    *,
    default_year: int | None = None,
) -> frozenset[str]:
    if not text or not text.strip():
        return frozenset()
    yr = default_year if default_year is not None else date.today().year
    out: set[str] = set()
    t = text

    for m in _ISO.finditer(t):
        s = _ymd(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if s:
            out.add(s)

    for m in _DOT.finditer(t):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        s = _ymd(y, mo, d)
        if s:
            out.add(s)

    for m in _SLASH.finditer(t):
        s = _from_slash(m.group(1), m.group(2), m.group(3))
        if s:
            out.add(s)

    for m in _MONTH_THEN_DAY_RANGE.finditer(t):
        mon = _MONTHS.get(m.group("mr").lower())
        if not mon:
            continue
        y = int(m.group("yr")) if m.group("yr") else yr
        _add_inclusive_day_span_to_month(y, mon, int(m.group("da")), int(m.group("db")), out)

    for m in _DAY_RANGE_THEN_MONTH.finditer(t):
        mon = _MONTHS.get(m.group("mr").lower())
        if not mon:
            continue
        y = int(m.group("yr")) if m.group("yr") else yr
        _add_inclusive_day_span_to_month(y, mon, int(m.group("da")), int(m.group("db")), out)

    for m in _MONTH_WORD.finditer(t):
        if m.group("m1"):
            mon = _MONTHS.get(m.group("m1").lower())
            if not mon:
                continue
            d = int(m.group("d1"))
            y = int(m.group("y1")) if m.group("y1") else yr
            s = _ymd(y, mon, d)
            if s:
                out.add(s)
        elif m.group("m2"):
            mon = _MONTHS.get(m.group("m2").lower())
            if not mon:
                continue
            d = int(m.group("d2"))
            y = int(m.group("y2")) if m.group("y2") else yr
            s = _ymd(y, mon, d)
            if s:
                out.add(s)

    return frozenset(out)


def extract_guest_stay_date_literals(
    text: str,
    *,
    default_year: int | None = None,
) -> frozenset[str]:
    out = _extract_guest_stay_date_literals_inner(text, default_year=default_year)
    # #region agent log
    _agent_dbg_log(
        "guest_dates:extract_guest_stay_date_literals",
        "parsed",
        {
            "thread_sample": ((text or "")[:200]),
            "parsed_count": len(out),
            "parsed_sample": sorted(out)[:16],
        },
        "H1,H2",
    )
    # #endregion
    return out


def thread_text_for_stay_date_gate(
    user_lines: list[str],
    *,
    default_year: int | None = None,
) -> str:
    """Shortest suffix of guest lines that still parses at least one date."""
    if not user_lines:
        return ""
    n = len(user_lines)
    for start in range(n - 1, -1, -1):
        chunk = "\n".join(user_lines[start:])
        if _extract_guest_stay_date_literals_inner(chunk, default_year=default_year):
            return chunk
    return "\n".join(user_lines)


def literals_from_config(config_literals: Iterable[str] | None) -> frozenset[str] | None:
    if config_literals is None:
        return None
    return frozenset(str(x).strip() for x in config_literals if str(x).strip())


def stay_dates_fail_gate(
    check_in: str,
    check_out: str,
    allowed: frozenset[str] | None,
) -> str | None:
    if allowed is None:
        return None
    ci, co = check_in.strip(), check_out.strip()
    if not allowed:
        return (
            '{"ok":false,"blocked":true,"reason":"No explicit stay dates were found in the guest '
            'message. Ask them for check-in and check-out dates (e.g. YYYY-MM-DD) before calling '
            'availability, quote, or booking tools."}'
        )
    if ci not in allowed or co not in allowed:
        allowed_list = sorted(allowed)
        return (
            f'{{"ok":false,"blocked":true,"reason":"Stay dates must come from the guest text only. '
            f"check_in and check_out must be among {allowed_list!r} (normalized YYYY-MM-DD). "
            f'Got check_in={ci!r}, check_out={co!r}."}}'
        )
    return None


_MAX_READ_IMPLIED_NIGHTS = 60


def stay_dates_read_fail_gate(
    check_in: str,
    check_out: str,
    allowed: frozenset[str] | None,
) -> str | None:
    if allowed is None:
        return None
    strict = stay_dates_fail_gate(check_in, check_out, allowed)
    if strict is None:
        return None
    if not allowed:
        return strict
    ci_s, co_s = check_in.strip(), check_out.strip()
    try:
        di = date.fromisoformat(ci_s)
        do = date.fromisoformat(co_s)
    except ValueError:
        return strict
    if do <= di:
        return strict
    nights = (do - di).days
    if nights > _MAX_READ_IMPLIED_NIGHTS:
        return strict
    if ci_s not in allowed:
        return strict
    cur = di
    while cur < do:
        if cur.isoformat() not in allowed:
            return strict
        cur += timedelta(days=1)
    return None


def modify_dates_fail_gate(
    check_in: str,
    check_out: str,
    allowed: frozenset[str] | None,
) -> str | None:
    if allowed is None:
        return None
    parts = []
    if check_in.strip():
        parts.append(("check_in", check_in.strip()))
    if check_out.strip():
        parts.append(("check_out", check_out.strip()))
    if not parts:
        return None
    if not allowed:
        return (
            '{"ok":false,"blocked":true,"reason":"Changing stay dates requires explicit dates in '
            'the guest message; none parsed."}'
        )
    for name, val in parts:
        if val not in allowed:
            return (
                f'{{"ok":false,"blocked":true,"reason":"{name}={val!r} is not among guest-stated '
                f'dates {sorted(allowed)!r}."}}'
            )
    return None