"""Heuristic risk / policy gates for autonomous execution (human review when triggered)."""

from __future__ import annotations

import json
import re
from typing import Any


def analyze_email_risk(email_body: str) -> dict[str, Any]:
    """
    Flag messages that must not run autonomous PMS writes or queue guest correspondence.

    Scenario 3: refund on non-refundable booking → manual review.
    """
    t = email_body.lower()
    reasons: list[str] = []

    if "refund" in t and (
        "non-refundable" in t
        or "non refundable" in t
        or "nonrefundable" in t
        or re.search(r"non[-\s]?refund", t)
    ):
        reasons.append(
            "Guest is asking for a refund in connection with a non-refundable product; "
            "staff must review policy and payment records before any action."
        )

    if "refund" in t and any(
        x in t for x in ("chargeback", "dispute", "visa charge", "card charge reversed")
    ):
        reasons.append("Payment dispute / chargeback language requires manual review.")

    if any(x in t for x in ("lawyer", "attorney", "sue", "litigation", "legal action")):
        reasons.append("Legal threat; escalate to management.")

    if "wire" in t and "account" in t and any(x in t for x in ("transfer", "iban", "swift")):
        reasons.append("Possible payment-instructions fraud pattern; manual review.")

    if any(x in t for x in ("special exception", "override policy", "can you bend", "off the record")):
        reasons.append("Ambiguous policy-exception request; escalate to specialist review.")

    return {
        "manual_review_required": bool(reasons),
        "reasons": reasons,
        "summary_json": json.dumps({"manual_review_required": bool(reasons), "reasons": reasons}),
    }
