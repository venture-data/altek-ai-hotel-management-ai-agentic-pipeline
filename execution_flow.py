"""Shared execution-mode configuration for chat and inbound-email pipelines.

Two independent modes (config flag only):

- **autonomous**: read/write agent runs end-to-end. High-risk content still sets
  ``manual_review_only`` (read tools only) via ``risk.analyze_email_risk``.
- **approval**: read-only planner first; guest drafts under **DRAFT GUEST REPLY** are saved to
  **Email drafts** when the plan says **Requires PMS writes: no**. Chat **Approve** still appears
  whenever there is a draft or the plan says **yes**; the executor runs only when **yes**.
- **Booking commits**: ``pms_create_reservation`` / cancel / modify are blocked unless
  ``booking_pms_commit_allowed`` is true (staff release — approval execute step, or explicit
  autonomous toggle / env). Queue drafts first; guest-facing wording stays non-final until release.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from guest_dates import extract_guest_stay_date_literals, thread_text_for_stay_date_gate
from risk import analyze_email_risk

ExecutionMode = Literal["autonomous", "approval"]
HumanExecutorIntent = Literal["booking", "cancellation", "modification", "generic"]


def classify_human_executor_intent(approved_plan_text: str) -> HumanExecutorIntent:
    """
    Infer whether the approved planner output is mainly a **new booking**, a **cancellation/modify**, or mixed.
    Used so the executor follow-up does not steer cancellation confirmations toward availability/create.
    """
    p = (approved_plan_text or "").lower()
    has_cancel_tool = "pms_cancel_reservation" in p
    has_modify_tool = "pms_modify_reservation" in p
    has_create_tool = "pms_create_reservation" in p
    if has_modify_tool and not has_cancel_tool and not has_create_tool:
        return "modification"
    if has_cancel_tool and not has_create_tool:
        return "cancellation"
    if has_create_tool and not has_cancel_tool:
        return "booking"
    if has_cancel_tool and has_create_tool:
        return "generic"
    if re.search(r"\b(modify|change|reschedule)\b", p) and re.search(
        r"\b(reservation|booking|stay|dates?)\b", p
    ):
        if not re.search(r"\bcancel", p):
            return "modification"
    if re.search(r"\bcancel", p) and re.search(r"\b(reservation|booking)\b", p):
        return "cancellation"
    if re.search(r"\b(book|booking|reserve|availability|check[-_]?in)\b", p) and not re.search(
        r"\bcancel", p
    ):
        return "booking"
    return "generic"


def human_mode_desk_footer_note(kind: HumanExecutorIntent) -> str:
    """Streamlit suffix after human-mode executor turn; booking-specific wording only when relevant."""
    common = (
        "\n\n---\n*If the executor queued correspondence, check **Pending guest email** on the right; "
        "use **Approve** / **Reject** to file under **Approved correspondence** or rejected.*"
    )
    if kind == "booking":
        return (
            common
            + " *New stays: the reservation row is added to the mock PMS **only** after **Approve** (from **BOOKING_COMMIT**), not during chat.*"
        )
    if kind == "modification":
        return (
            common
            + " *Modifications: the mock PMS is updated **only** after **Approve** (from **BOOKING_MODIFY**), not during chat.*"
        )
    if kind == "generic":
        return (
            common
            + " *Desk **Approve** applies **BOOKING_COMMIT** (new stay), **BOOKING_MODIFY** (changes), or cancellation when applicable.*"
        )
    return common

_WRITES_LINE = re.compile(
    r"(?i)requires\s+pms\s+writes[^:\n]{0,32}:\s*`?\s*(yes|no)\b",
)


def planner_requires_pms_writes(plan_text: str) -> bool:
    """
    Parse planner output for **Requires PMS writes:** ``yes`` / ``no`` (executor / queue tools).

    When ``no``, the executor is not needed for PMS or queue tools; a **DRAFT GUEST REPLY** may
    still be filed to disk separately. When the marker is missing or ambiguous, returns ``True``.
    """
    if not (plan_text or "").strip():
        return True
    last: bool | None = None
    for m in _WRITES_LINE.finditer(plan_text):
        val = m.group(1).lower()
        tail = plan_text[m.end() : m.end() + 24].lstrip().lower()
        if tail.startswith("or "):
            continue
        last = val == "yes"
    if last is not None:
        return last
    return True


_DRAFT_SECTION = re.compile(
    r"(?is)(?:^|\n)\s*(?:#{1,3}\s*)?DRAFT\s+GUEST\s+REPLY\s*\n+(.*)\Z",
)


def extract_draft_guest_reply(plan_text: str) -> str | None:
    """Return the body under **DRAFT GUEST REPLY** (with or without ``###``), or None."""
    if not (plan_text or "").strip():
        return None
    m = _DRAFT_SECTION.search(plan_text.strip())
    if not m:
        return None
    body = m.group(1).strip()
    if len(body) < 2:
        return None
    return body


def human_mode_chat_approval_needed(plan_text: str) -> bool:
    """
    Show Human mode **Approve / Skip** under the planner message when PMS/queue tools may run
    (**Requires PMS writes: yes**) or when there is a **DRAFT GUEST REPLY** to acknowledge
    (draft may already be filed under pending ``drafts/``).
    """
    if planner_requires_pms_writes(plan_text):
        return True
    return extract_draft_guest_reply(plan_text) is not None


def save_human_mode_planner_draft(
    *,
    draft_body: str,
    to_email: str,
    subject: str = "Re: your message",
    queue_root: Path | None = None,
    notes_for_staff: str = "",
) -> Path:
    """Write pending ``drafts/*.md`` (not under ``approved/``)."""
    from review_queue import review_queue_root, save_queued_correspondence

    root = queue_root if queue_root is not None else review_queue_root(None)
    em = (to_email or "").strip() or "guest@unknown.local"
    return save_queued_correspondence(
        "drafts",
        em,
        (subject or "").strip() or "Re: your message",
        draft_body.strip(),
        notes_for_staff=notes_for_staff.strip(),
        direct_to_approved=False,
        queue_root=root,
    )


def prepare_turn_configurable(
    *,
    accumulated_user_lines: list[str],
    text_for_risk: str,
    execution_mode: ExecutionMode,
    review_queue_dir: str | None = None,
    booking_pms_commit_allowed: bool = False,
    guest_default_year: int | None = None,
    session_guest_email: str | None = None,
    session_guest_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Build the ``configurable`` fragment for ``stream_agent_turn`` / ``agent_reply``.

    ``guest_default_year``: when the guest omits a year, assume this calendar year so literals
    align with mock PMS availability (see ``MockHotelPMS.default_guest_stay_year``).

    Returns ``(configurable, risk)`` where ``risk`` is the dict from ``analyze_email_risk``.
    """
    risk = analyze_email_risk(text_for_risk)
    joined = "\n".join(accumulated_user_lines)
    date_gate_text = thread_text_for_stay_date_gate(
        accumulated_user_lines, default_year=guest_default_year
    )
    literals = sorted(
        extract_guest_stay_date_literals(date_gate_text, default_year=guest_default_year)
    )
    cfg: dict[str, Any] = {
        "guest_thread_text": joined,
        "guest_thread_text_for_date_gate": date_gate_text,
        "guest_date_literals": literals,
        "execution_mode": execution_mode,
        "booking_pms_commit_allowed": booking_pms_commit_allowed,
    }
    if guest_default_year is not None:
        cfg["guest_default_year"] = int(guest_default_year)
    if risk["manual_review_required"]:
        cfg["manual_review_only"] = True
        cfg["manual_review_reason"] = "; ".join(risk["reasons"])
    rq = (review_queue_dir or "").strip()
    if rq:
        cfg["review_queue_dir"] = rq
    em = (session_guest_email or "").strip()
    gid = (session_guest_id or "").strip()
    if em:
        cfg["session_guest_email"] = em
    if gid:
        cfg["session_guest_id"] = gid
    return cfg, risk


def executor_followup_user_message(
    *,
    approved_plan_text: str,
    original_user_message: str,
    session_guest_email: str = "",
    session_guest_id: str = "",
    streamlit_human_executor_intent: HumanExecutorIntent | None = None,
    streamlit_defer_new_stay_to_desk: bool = False,
) -> str:
    """User message for the write-capable executor phase after operator approval."""
    kind: HumanExecutorIntent = streamlit_human_executor_intent or classify_human_executor_intent(
        approved_plan_text
    )
    em = (session_guest_email or "").strip()
    gid = (session_guest_id or "").strip()
    sess = ""
    if em or gid:
        if kind == "cancellation":
            sess = (
                "**Authoritative session guest (do not invent placeholder emails or new guests):**\n"
                f"- Email: `{em or '(set session email)'}`\n"
                f"- PMS guest_id: `{gid or '(resolve with pms_find_guest_by_email)'}`\n"
                "Use **pms_list_guest_reservations** and **pms_cancel_reservation** (or modify tools) as in the plan. "
                "Do **not** call **pms_check_availability** or **pms_create_reservation** unless the plan explicitly "
                "adds a **new** stay after a cancel/change.\n"
                "Use **this email** for **pms_queue_correspondence_for_review** `to_email`.\n\n"
            )
        elif streamlit_defer_new_stay_to_desk:
            sess = (
                "**Authoritative session guest (Streamlit Human mode — desk creates new stays):**\n"
                f"- Email: `{em or '(set session email)'}`\n"
                f"- PMS guest_id: `{gid or '(resolve with pms_find_guest_by_email)'}`\n"
                "Do **not** call **pms_create_reservation** (blocked). Put **guest_id** and stay fields in "
                "**booking_commit_json** when you queue **booking_confirmations** or **drafts**. "
                "Do **not** call **pms_create_guest** if this guest already exists. Use **this email** for `to_email`.\n\n"
            )
        else:
            sess = (
                "**Authoritative session guest (Streamlit / host — do not invent placeholder emails or new guests):**\n"
                f"- Email: `{em or '(set session email)'}`\n"
                f"- PMS guest_id: `{gid or '(resolve with pms_find_guest_by_email using email above)'}`\n"
                "Use **this guest_id** for **pms_create_reservation**. Do **not** call **pms_create_guest** if this guest "
                "already exists in the PMS. Use **this email** for **pms_queue_correspondence_for_review** `to_email`.\n\n"
            )

    if kind == "cancellation":
        head = (
            "Operator approved execution of the **cancellation or reservation change** plan. "
            "Follow the **STRUCTURED ACTION PLAN** and use **pms_cancel_reservation** / modify tools as written. "
            "**Do not** run **pms_check_availability** or **pms_create_reservation** for this turn unless the plan "
            "explicitly books a **replacement** stay after a cancel.\n\n"
        )
        tail = (
            "The guest may have replied with only **yes** / **confirm** — interpret that as assent to the **cancellation "
            "or change** described in the plan, not as a new booking request.\n\n"
        )
    elif kind == "booking":
        if streamlit_defer_new_stay_to_desk:
            head = (
                "Operator approved execution (Streamlit **Human** mode). Use read tools as needed "
                "(**pms_check_availability** / **pms_quote_stay**) to verify the offer, then queue correspondence with "
                "**booking_commit_json** — **do not** call **pms_create_reservation**.\n\n"
            )
            tail = (
                "The guest confirmed the stay. Queue **booking_confirmations** (preferred) or **drafts** with a full "
                "**booking_commit_json** object. The PMS reservation is created **only** when staff click **Approve** on "
                "**Pending guest email**. Wording to the guest should reflect **pending staff release** / not yet in the "
                "PMS until released, if appropriate.\n\n"
            )
        else:
            head = (
                "Operator approved execution. Use read tools as needed (especially **pms_check_availability** before "
                "**pms_create_reservation**), then write tools to carry out the plan.\n\n"
            )
            tail = (
                "The guest may have replied with only **yes** / **confirm** — the stay is still the one from the thread "
                "and plan. Prefer **Stay parameters** from the approved **STRUCTURED ACTION PLAN** (check_in/check_out as "
                "**YYYY-MM-DD**, half-open nights). If those lines are missing, infer ISO dates from the **DRAFT GUEST REPLY** "
                "and guest lines, then verify with **pms_check_availability** before claiming no inventory.\n\n"
            )
    else:
        if streamlit_defer_new_stay_to_desk:
            head = (
                "Operator approved execution (Streamlit **Human** mode). For **new** stays, **do not** call "
                "**pms_create_reservation** — queue with **booking_commit_json**; staff **Approve** applies the PMS row. "
                "For cancel/modify, use the appropriate tools as in the plan.\n\n"
            )
        else:
            head = (
                "Operator approved execution. Carry out the **STRUCTURED ACTION PLAN** with appropriate read then write "
                "tools. If the plan mixes cancel/modify and new booking steps, follow the **order** in the plan; do not "
                "skip straight to availability for a new stay if the guest only confirmed a **cancellation**.\n\n"
            )
        tail = (
            "The guest may have replied with only **yes** / **confirm** — apply it to the **primary** action in the plan "
            "(cancel/modify vs new booking) using the plan text.\n\n"
        )

    return (
        head
        + sess
        + tail
        + f"Approved planning output (plan + draft for reference):\n{approved_plan_text}\n\n"
        + f"Original guest message (full thread text supplied to the date gate):\n{original_user_message.strip()}"
    )
