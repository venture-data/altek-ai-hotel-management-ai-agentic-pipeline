"""Heuristics: when chat implies PMS booking identity; light email validation."""

from __future__ import annotations

import re

# Booking / PMS-action phrasing: attach guest before tools (mirrors former main.py regex).
BOOKING_CONTEXT_RE = re.compile(
    r"\b("
    r"book|bookings?|booked|reserve|reservations?|reserving|"
    r"stay(?:ing)?|check-?in|check-?out|"
    r"availability|available|vacanc|"
    r"room\s+(?:for|types?|only)|double\s+room|single\s+room|suite|"
    r"nightly|rate(?:s)?\s+for|quote|hold\s+(?:a\s+)?room|"
    r"cancel(?:lation|ing)?|modify(?:ing)?\s+(?:my\s+)?(?:booking|reservation|stay)"
    r")\b",
    re.IGNORECASE,
)


def message_needs_guest_for_booking(text: str) -> bool:
    return bool(BOOKING_CONTEXT_RE.search(text))


def looks_like_email(s: str) -> bool:
    if "@" not in s or s.startswith("@") or s.endswith("@"):
        return False
    local, _, domain = s.partition("@")
    return bool(local) and "." in domain
