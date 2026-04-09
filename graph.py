"""LangGraph ReAct agent for inbound guest email → reasoning + PMS tools + reply."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt import create_react_agent

from agent_tools import (
    HOTEL_READ_TOOLS,
    HOTEL_TOOLS_ALL,
    HOTEL_WRITE_TOOLS,
    bind_pms_for_agent_turn,
)
from hotel_pms import MockHotelPMS


def prompt_example_year(guest_default_year: int | None = None) -> int:
    """
    Calendar year embedded in prompt ISO examples (matches ``guest_default_year`` / mock data when passed).
    """
    if guest_default_year is not None:
        try:
            y = int(guest_default_year)
            if 1990 <= y <= 2100:
                return y
        except (TypeError, ValueError):
            pass
    return date.today().year


def _prompt_iso(y: int, month: int, day: int) -> str:
    return date(y, month, day).isoformat()


def _ordinal_day(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _guest_day_and_month(d: date) -> str:
    return f"{_ordinal_day(d.day)} {d.strftime('%B')}"


def illustrative_stay_example_dates(year: int) -> tuple[date, date, date, date, date]:
    """
    Example stay anchors for prompts: **month shifts with year** (May–Oct cycle) so illustrations are not
    fixed on May. Returns (one_night_ci, one_night_co, two_endpoint_ci, two_endpoint_co, wrong_extra_night_co).
    """
    month = 5 + (year % 6)
    one_ci = date(year, month, 10)
    one_co = one_ci + timedelta(days=1)
    r_ci = date(year, month, 14)
    r_co = date(year, month, 16)
    r_bad = date(year, month, 17)
    return (one_ci, one_co, r_ci, r_co, r_bad)


DATE_POLICY = """\
Strict date handling: Do not invent check-in or check-out dates. Only use dates that appear \
explicitly in the guest message (or earlier user messages in this thread). If they ask for \
availability, a quote, or to book but give no calendar dates, ask for specific dates before calling \
pms_check_availability, pms_quote_stay, or pms_create_reservation. Vague phrases like "next weekend" \
or "for two nights" are not enough until they confirm actual dates. For pms_modify_reservation, any \
new dates you pass must also appear in the guest text. The PMS tools reject tool calls whose dates \
are not grounded in the guest wording. \
**Dynamic thread parsing:** the runtime supplies the full guest thread; stay-related dates are **re-parsed** from that text on each tool call (ranges with **– / to / through / until / between … and**, ISO, dotted and slash forms, etc.). \
**Recency for the stay-date gate:** when `guest_thread_text_for_date_gate` is in use, the gate uses a **suffix** of guest lines (from the last line that still yields parsable dates through the end). **Newer date questions override older ones** in that gate (e.g. April then May → May applies for tools). The full thread remains in chat history for you to read. \
When the guest says **yes** after you offered a concrete stay (room + check_in/check_out), book **that** stay — not an earlier date ask that is no longer in the active gate window."""


def date_policy_with_year_note(*, year: int) -> str:
    return (
        DATE_POLICY
        + f"\n\n**Illustrative ISO year in this deployment:** Example dates elsewhere in these instructions use **{year}** "
        + "when showing `YYYY-MM-DD` (aligned with `guest_default_year` / mock availability keys when configured)."
    )


DATE_SPAN_POLICY = """\
**Two-endpoint stays (default):** When the guest gives **two** day numbers in the same month — dash, **to**, \
**through**, **until**, etc. (e.g. **"May 23rd to 25th"**, "23–25 May", "May 15 through 28", "from 15 May to 28 May") — \
treat the **first day as check-in** and the **second day as check-out** (departure morning, half-open nights). \
Example: **May 23 → May 25** means two occupied nights (nights of the 23rd and 24th), **not** three nights and **not** \
check-out on May 26. **Do not** tell the guest their check-out must be the calendar day **after** the second number \
when they clearly named that second number as their **leave** / **checkout** day. \
\n\n**Listed nights (different rule):** If they **list** separate calendar days as nights to stay (e.g. **"21st, 22nd and 23rd May"**, \
"the nights of May 21, 22, 23"), the last day is the **last occupied night**; then check_in = first listed day and \
check_out = the **morning after** that last night (one day past the last listed date). \
\n\n**Span for the date gate:** The parser still expands **every calendar day** from the first through the last \
day number in a range (inclusive) so tools can validate nights inside that window; that does **not** mean check_out \
must always be the day after the last expanded day — use the **two-endpoint vs listed-nights** rule above for what to pass \
to **pms_check_availability** / **pms_quote_stay** / **pms_create_reservation**. \
\n\nTwo-day spans (e.g. "May 21st–22nd", "21st to 22nd May"): **earlier** = check_in, **later** = check_out (one night). \
**"May 21st and 22nd"** (two consecutive days named): same — one night, check_out on the **second** day. \
If they say **one night** and only **one** calendar date, use check_in that day and check_out the **next** day. \
Only ask for clarification when months/years are missing or contradictory."""

TOOL_ERROR_POLICY = """\
Tool failures: When any tool returns JSON or text with blocked, ok: false, or an error, read it carefully. \
In the **same** reply, your **first** explanation to the guest should name **what kind** of problem it is \
(stay-date guard vs wrong tool args, inventory, write blocked, etc.) using the tool payload — not a vague "restriction" \
they must ask you to unpack. Summarize using the tool's "reason" and structured fields (e.g. parsed_dates_from_guest_text, \
hint, tool_check_in/tool_check_out) — do not call it a "parse failure" when the guest gave clear dates but the tool used \
the wrong year or endpoints. If parsed_dates_from_guest_text is present, retry with check_in and check_out from that set \
that match the guest's intent and the half-open night rule. Never apologize that you "can only check" arbitrary isolated days \
unless the tool error literally says that."""


def pms_availability_policy(*, year: int) -> str:
    one_ci, one_co, r_ci, r_co, r_bad = illustrative_stay_example_dates(year)
    iso1, iso2 = one_ci.isoformat(), one_co.isoformat()
    isoa, isob, isoc = r_ci.isoformat(), r_co.isoformat(), r_bad.isoformat()
    one_label = _guest_day_and_month(one_ci)
    range_phrase = (
        f"{_ordinal_day(r_ci.day)} to {_ordinal_day(r_co.day)} {r_ci.strftime('%B')}"
        if r_ci.month == r_co.month
        else f"{_guest_day_and_month(r_ci)} to {_guest_day_and_month(r_co)}"
    )
    last_day_word = _ordinal_day(r_co.day)
    return f"""\
Mock PMS inventory: each date key in mock_hotel_data.json is **available rooms for this guest-night**. A stay is \
(check_in, check_out) with **each occupied night = a date d where check_in <= d < check_out** (departure morning = \
check_out). **One night** (illustration: **{one_label}**) → check_in **{iso1}**, check_out **{iso2}** — not the same date twice. \
**Two-endpoint range** like **"{range_phrase}"** (first day = arrival, second = departure morning) → check_in **{isoa}**, \
check_out **{isob}** (two occupied nights); **not** check_out **{isoc}** unless they asked to include the **night of the \
{last_day_word}** as an extra night. \
Same check_in and check_out yields **zero nights** and empty availability even if inventory shows rooms on that date. \
**Never** describe a valid multi-night or one-night stay as requiring check_in **equal to** check_out — that is **false**; \
check_out is always the **next** calendar day (or later) for any stay with nights > 0. \
When pms_check_availability returns a non-empty JSON list with available_rooms_for_stay > 0, **state those room types \
accurately**; do not say there are no rooms. Each row may include **quote_standard_rate_rp001** — that total is from the \
**same** mock inventory as the row; **use it** for Standard Rate (RP001) instead of a second quote call that might use \
wrong dates and contradict availability."""


def inventory_tool_grounding_policy(*, year: int) -> str:
    one_ci, one_co, _, _, _ = illustrative_stay_example_dates(year)
    iso1, iso2 = one_ci.isoformat(), one_co.isoformat()
    one_label = _guest_day_and_month(one_ci)
    return f"""\
Tool-grounded inventory (Human + Autonomous): Do **not** tell the guest a room is unavailable, or that availability \
**changed**, without a **fresh** tool result in **this** turn (**pms_check_availability**, **pms_quote_stay**, or the \
**error** field from **pms_create_reservation**). If you already showed availability for specific nights, the guest's \
**yes / proceed / confirm** locks the **same** stay: reuse the **same** check_in and check_out as **YYYY-MM-DD** \
(half-open nights — illustration one night **{one_label}** → check_in **{iso1}**, check_out **{iso2}**), same **room_type_id** and \
**rate_plan_id** you offered. If you are unsure, call **pms_check_availability** again before answering. **Never** use the \
same calendar date for check_in and check_out — that is **zero nights** and will look like “no rooms” even when inventory \
exists. If **pms_create_reservation** returns **ok: false**, report that tool's **error**; if it mentions availability or \
quote failure, run **pms_check_availability** once with the **intended** ISO dates before claiming inventory is gone."""

BOOKING_CONSISTENCY_POLICY = """\
Booking truth depends on `booking_pms_commit_allowed`:

**When it is false** (planner-only / human mode before execute): Treat reservations as **not** final until staff approve \
a queued draft and commits are released. Use **pending staff confirmation** / **subject to final approval**. Do not claim \
the PMS already holds the booking. Queue **pms_queue_correspondence_for_review** as needed.

**When it is true** (Streamlit **Autonomous** chat, or the approved executor phase): Guest assent in this thread \
(e.g. **yes**, **proceed**, **book it** after you offered concrete dates, room, and price) is enough to call \
**pms_create_reservation** when policy allows. **Do not** ask for a separate "staff confirmation" step or extra \
round-trip—complete the booking, then queue the confirmation. Category **booking_confirmations** is **only** for the \
on-disk guest email **after** **pms_create_reservation** returns **ok: true** (real confirmed stay). Until then, ask \
whether to proceed **in your chat reply only**; do not queue **booking_confirmations** or use confirmation-style \
subjects. If you must record a quote on disk before create, use category **drafts** only. In autonomous Streamlit, \
**drafts** / **booking_confirmations** file under **approved/** automatically. Until **ok: true**, do not tell the guest \
the stay is finally confirmed. \
**Streamlit Human mode (defer flag on):** **Do not** call **pms_create_reservation** in chat — queue **booking_commit_json**; \
staff **Approve** on **Pending guest email** creates the PMS row. Until then, tell the guest the stay is **subject to staff release** \
as appropriate.

After **ok: true** (autonomous or non-defer executor), match check_in / check_out / room. If the guest confirmed without restating dates, use **exactly** \
what you last proposed. Never contradict your own prior availability unless a new tool run shows different inventory."""

REPLY_COHERENCE_POLICY = """\
Reply style: Give **one** clear answer per turn. Do **not** paste the same paragraph twice or argue both sides \
("I can check…" then "I cannot check…") for the same request. \
**allowed_dates_from_guest_text** (and similar fields) mean the PMS gate is enforcing dates parsed from the thread — \
they are **not** permission to tell the guest you "may only ever" discuss certain days. If they ask for a new night, \
call availability for that intent (with correct check_in/check_out per half-open nights); if a tool blocks you, \
use that error's **reason** and **hint** — do not invent a vague "system restriction". \
If you already reported **no inventory** for a date after a successful tool call, do not re-hash that night unless the \
guest asks again or you are re-running the tool."""

# Only used in SYSTEM_PROMPT (default autonomous full agent). Human-mode planner/executor omit this block.
AUTONOMOUS_MODE_EXTRA_POLICY = """\
**Autonomous agent only (not Human-mode planner/executor):**

**Confirmation emails on disk:** Category **booking_confirmations** is **only** after **pms_create_reservation** returns \
**ok: true**. Before that, ask the guest to proceed **in chat**; do **not** queue a "booking confirmation" email or \
confirmation-style **Subject:**. Optional pre-create records belong in **drafts** only, without wording that implies the \
stay is already confirmed.

**Tone after a tool error:** If a write failed (e.g. date guard) but the guest's dates are valid and availability is \
fine, **retry with correct check_in/check_out** (half-open nights, **parsed_dates_from_guest_text**) in the same turn \
when possible. Do **not** open with "there was an error / issue with the booking attempt" and then repeat the same \
offer and "would you like to proceed?" — that sounds contradictory. Prefer one coherent message: either fix silently \
and confirm, or one short factual line plus a single clear next step — never both "failure" and "all set, proceed?" \
for the same dates in one reply."""


def transparency_and_reasons_policy(*, year: int) -> str:
    one_ci, one_co, _, _, _ = illustrative_stay_example_dates(year)
    iso1, iso2 = one_ci.isoformat(), one_co.isoformat()
    one_label = _guest_day_and_month(one_ci)
    return f"""\
Causes vs policies: Keep three buckets separate and tell the guest which applies **without** making them ask "what restriction?" \
**(A) Stay dates / tool guard** — Could not parse dates from the thread, or check_in/check_out failed the stay-date guard. \
Say so plainly (e.g. ask for YYYY-MM-DD or clearer phrasing); do not call this "hotel policy". \
**(B) Inventory** — After **pms_check_availability** (or quote), no suitable rooms for those nights. Say no availability for the requested stay; do not invent a separate "date restriction". \
**(C) Published hotel rules** — Cancellation tiers, pets, breakfast, parking, etc. **Only** after **pms_hotel_reference** (or text already in that tool output). Never invent rules. \
**Half-open nights are not a restriction:** For one night (illustration **{one_label}**), check_in **{iso1}** and check_out **{iso2}** (departure morning) is **correct** and **not** a conflict with hotel policy. \
Do **not** tell the guest that "the system does not permit check-out on **{iso2}**" in that scenario — that misstates how nights are counted. \
If they ask what policy says, call **pms_hotel_reference** and summarise only what it returns. \
If a tool JSON says **zero nights** / same calendar day for check_in and check_out, explain that **check_out must be later** (departure morning), not that the guest must make them **equal**."""


def _concat_standard_policy_blocks(*, year: int) -> str:
    return (
        date_policy_with_year_note(year=year)
        + "\n\n"
        + DATE_SPAN_POLICY
        + "\n\n"
        + pms_availability_policy(year=year)
        + "\n\n"
        + inventory_tool_grounding_policy(year=year)
        + "\n\n"
        + TOOL_ERROR_POLICY
        + "\n\n"
        + BOOKING_CONSISTENCY_POLICY
        + "\n\n"
        + REPLY_COHERENCE_POLICY
        + "\n\n"
        + transparency_and_reasons_policy(year=year)
    )


_SYSTEM_INTRO = """You are the reservations AI for a hotel. You handle inbound guest emails.

You have tools that talk to the property management system (PMS). For requests that change data
(bookings, new guest profiles, cancellations, modifications), use tools in a sensible order, for example:
- Resolve the guest: find by email from the message; if not found and you have their name, create a guest.
- For new stays: check availability, quote. When `booking_pms_commit_allowed` is true (autonomous or approved execute), \
and the guest confirms in thread, call **pms_create_reservation**; **after** **ok: true**, queue **one** \
**booking_confirmations** file via **pms_queue_correspondence_for_review** (not before). When commits are off, \
queue proposals and use pending-staff wording until release. Never tell the guest the stay is **finally** confirmed \
until **pms_create_reservation** returns **ok: true**.
- For changes: list or get reservations, then cancel or modify as policy allows. Non-refundable bookings cannot be modified on dates/room/party per hotel policy.
- For questions about policies or the hotel: use the hotel reference tool.
- When you record outbound copy: **pms_queue_correspondence_for_review** — use **booking_confirmations** only **after** \
**pms_create_reservation** returns **ok: true**; for quotes or "please confirm" steps use **drafts** (or **general**). \
Nothing is emailed; the tool saves markdown under review_queue/<category>/ and logs to stderr.

Rate plan IDs: RP001 Standard, RP002 Breakfast included, RP003 Non-refundable saver, RP004 Flexible with breakfast.
Room type IDs: RT001 Standard Single, RT002 Standard Double, RT003 Superior Double, RT004 Junior Suite.

You may get follow-up messages in the same thread: answer naturally and use prior context (dates, names, choices).

When the user wants copy for an outbound email, end with a concise, polite draft (optional subject on the first line).
For quick terminal Q&A, short direct answers are fine; still use tools when PMS or policy facts are needed.
If the guest first and last name are known from PMS or the message, address them as "Dear <First> <Last>," in the draft; only use "Dear Guest," when no name is available.

"""


def system_prompt_for_full_agent(*, guest_default_year: int | None = None) -> str:
    y = prompt_example_year(guest_default_year)
    return (
        _SYSTEM_INTRO
        + _concat_standard_policy_blocks(year=y)
        + "\n\n"
        + AUTONOMOUS_MODE_EXTRA_POLICY
    )


# Back-compat: default prompts use ``prompt_example_year()`` at import (today's year if no mock path).
SYSTEM_PROMPT = system_prompt_for_full_agent()

_PLANNER_HEAD = """You are a hotel reservations analyst. You have READ PMS tools (lookups, availability, quotes, policies) \
and **pms_queue_correspondence_for_review** — use that tool **only** with category **escalations** when a case must \
appear on the staff **Escalations** desk (it saves markdown; nothing is emailed automatically).

"""

_PLANNER_TAIL = """

Analyze the inbound guest message. Use tools to gather PMS context as needed.

Before any write could happen in production, operators require a clear plan. End your response with exactly two sections:

### STRUCTURED ACTION PLAN
Plain **markdown bullet points only** — do **not** use JSON or code fences for this section. Structure it like this:
- **Steps** — a bullet list (use `-` or `*`) of concrete actions in order; one action per line, clear and short (e.g. verify availability; quote RT002 + RP001; after guest confirms, executor calls **pms_create_reservation** then queues correspondence — in **this** read-only phase you only look up and plan). For **cancellation** / **modification** guest emails, note executor should queue **cancellations** / **modifications** (shown under **Pending guest email**). For **specialist escalation** (policy, refunds, disputes, legal, fraud patterns, etc.), you **must** call **pms_queue_correspondence_for_review** with category **escalations** **in this planner turn** (you have that tool) so staff see the case on the Streamlit **Escalations** desk for Approve/Reject — same visibility as autonomous risk escalations. Do **not** defer escalation filing to the executor unless the executor phase will run immediately; the executor turn often runs only after a booking-style confirmation.
- **Stay parameters (exact, for executor):** when you made a **concrete** offer (dates + room + rate), one line with backticks: check_in `YYYY-MM-DD`, check_out `YYYY-MM-DD`, room_type_id `RTxxxx`, rate_plan_id `RPxxxx`, adults N, children N. Use **half-open** nights (checkout = morning **after** the last night). Omit only if no specific stay was offered yet.
- **Requires PMS writes:** reply with exactly **`yes`** or **`no`** only (not the phrase "yes or no"). Use **`yes`** if the plan would change PMS data, call **pms_queue_correspondence_for_review**, or the guest should receive any logged outbound email (escalations, refunds, specialist follow-up, apologies — even if you only draft text in this phase). Use **`no`** only when there is **no** guest-facing reply to record (rare; internal/ops-only).
- **Risk notes** — bullet list of caveats, policy edge cases, or follow-ups ops should know; use `- None` or `None` if none.

### DRAFT GUEST REPLY
The full polite draft for the guest (never sent automatically by this system). If **Requires PMS writes:** is **no**, this is the complete suggested reply for this phase. If **yes**, the executor queues it with **pms_queue_correspondence_for_review** after operator approval.
Use "Dear <First> <Last>," when the guest name is known; otherwise "Dear Guest,".

**Planner / read-only — forbidden fiction:** In this phase **pms_create_reservation has not run**. Do **not** tell the guest their booking is **confirmed**, **final**, or **successfully completed** — not even to answer "why did it fail?" or to compare with another mode. Explain using **tool JSON only** (e.g. quote fields inside **pms_check_availability**, or **pms_quote_stay** errors). Never invent a confirmation to make the guest feel better.

If the guest needs no outbound message at all, set **Requires PMS writes:** **no** and omit or minimize **DRAFT GUEST REPLY**; otherwise use **yes**."""


def planner_system_prompt(*, guest_default_year: int | None = None) -> str:
    y = prompt_example_year(guest_default_year)
    return _PLANNER_HEAD + _concat_standard_policy_blocks(year=y) + _PLANNER_TAIL


_EXECUTOR_HEAD = """You are executing an operator-APPROVED plan on the mock PMS. You have **the same read tools as the planner** \
(**pms_check_availability**, **pms_quote_stay**, lookups, **pms_hotel_reference**) **plus** write tools (**pms_create_reservation**, **pms_create_guest**, cancel/modify, queue correspondence, etc.). The old wording said \
"write only" — **ignore that**; you **must** use read tools whenever \
you need to verify inventory or align ISO dates with the approved plan.

**Before pms_create_reservation** after the guest confirmed a prior offer: call **pms_check_availability** (or \
**pms_quote_stay**) with the **exact** **YYYY-MM-DD** check_in/check_out and party size you will book — then create. \
Use **Stay parameters** from the approved plan when present; they override fuzzy wording in the draft.

**Streamlit Human mode (configurable defer):** When **defer_new_reservation_until_desk_approve** is set, **pms_create_reservation** \
is **blocked**. Queue **booking_confirmations** (or **drafts**) with **booking_commit_json**; the mock PMS reservation appears \
**only** after staff **Approve** on **Pending guest email**. Guest-facing copy should not claim the PMS already holds the booking \
until that release. **Cancel** / **modify** tools may still run when **booking_pms_commit_allowed** allows.

**Otherwise (autonomous / email pipeline executor):** Call **pms_create_reservation** and get **ok: true** before final-confirmation \
wording when commits are allowed; then queue correspondence with **BOOKING_COMMIT** as needed.

"""

_EXECUTOR_TAIL = """

Follow the approved plan and inbound context. This phase has **staff approval**: when `booking_pms_commit_allowed` is on, you may call **pms_cancel_reservation** / **pms_modify_reservation** as the plan requires. **pms_create_reservation** runs only when **not** blocked by **defer_new_reservation_until_desk_approve** (Streamlit Human defers new stays to desk **Approve**). Always queue **pms_queue_correspondence_for_review** for guest-facing text as appropriate.
When composing guest-facing correspondence, use "Dear <First> <Last>," if the guest name is known; otherwise "Dear Guest,".

**Streamlit Human mode — queue categories:** Use **cancellations** for guest-facing cancellation acknowledgement letters; \
**modifications** for change confirmations; **booking_confirmations** with **booking_commit_json** for new stays (PMS row on desk **Approve**); \
**drafts** for quotes and pre-commit offers; **escalations** for specialist / policy cases (those appear on the **Escalations** desk, not under Pending guest email).

If a tool returns blocked or ok: false, stop and explain using the tool's reason field (see policy above)."""


def executor_system_prompt(*, guest_default_year: int | None = None) -> str:
    y = prompt_example_year(guest_default_year)
    return _EXECUTOR_HEAD + _concat_standard_policy_blocks(year=y) + _EXECUTOR_TAIL


_REVIEW_HEAD = """The inbound message was flagged for HUMAN REVIEW (policy-sensitive or financially risky).

"""

_REVIEW_TAIL = """

You have **read** PMS tools plus **pms_queue_correspondence_for_review** — use **category `escalations` only** \
(no other categories). Do not imply that refunds, credits, or contract changes have been processed in the PMS.

**Escalation filing (required before you finish):** Call **pms_queue_correspondence_for_review** with category \
**escalations** so staff see the case on the Streamlit **Escalations** desk (pending `review_queue/escalations/`):
- **to_email:** guest address from context / session; never a placeholder if a real address is known.
- **subject:** e.g. "Re: your request — specialist follow-up".
- **body:** substance only (2–5 short plain paragraphs) — **no** "Dear…" / formal sign-off in body (the template adds framing). Same style as autonomous escalation.
- **notes_for_staff:** what was flagged and what the guest was told.

Use read tools only if they help you acknowledge facts. Output the same two sections as the planning agent:

### STRUCTURED ACTION PLAN
Markdown bullet points only (no JSON). Include **Steps** (include **file escalation** via tool above), **Requires PMS writes:** **yes** (escalation queue only), and **Risk notes** as relevant.

### DRAFT GUEST REPLY
A polite holding email. For risky/ambiguous requests, explicitly say the case was escalated and a specialist will contact them shortly.
Default sentence (you may lightly tailor tone): "I have raised your request to a specialist, and they will contact you shortly."
Do not promise outcomes you cannot authorize.
Use "Dear <First> <Last>," when the guest name is known; otherwise "Dear Guest,"."""


def review_only_system_prompt(*, guest_default_year: int | None = None) -> str:
    y = prompt_example_year(guest_default_year)
    return _REVIEW_HEAD + _concat_standard_policy_blocks(year=y) + _REVIEW_TAIL


def _concat_risk_escalation_policy_blocks(*, year: int) -> str:
    return (
        date_policy_with_year_note(year=year)
        + "\n\n"
        + DATE_SPAN_POLICY
        + "\n\n"
        + pms_availability_policy(year=year)
        + "\n\n"
        + inventory_tool_grounding_policy(year=year)
        + "\n\n"
        + TOOL_ERROR_POLICY
        + "\n\n"
        + REPLY_COHERENCE_POLICY
        + "\n\n"
        + transparency_and_reasons_policy(year=year)
    )


_AUTONOMOUS_RISK_HEAD = """The guest message was flagged as **sensitive, risky, or ambiguous** (e.g. refunds on non-refundable \
rates, disputes, legal language, policy exceptions). You are in **autonomous** mode but **must not** change the PMS \
(no new reservations, cancellations, modifications, or guest creates). `booking_pms_commit_allowed` is off for this turn.

You have **read** PMS tools plus **pms_queue_correspondence_for_review** only.

**Reply style:** Write **one** natural, professional message to the guest — **no** section headers such as \
`### STRUCTURED ACTION PLAN` or `### DRAFT GUEST REPLY`, and no internal bullet plan visible to the guest. \
Be concise and human.

**Escalation filing:** Call **pms_queue_correspondence_for_review** with category **escalations** (required before you finish):
- **to_email:** the guest address from context or a sensible placeholder if unknown.
- **subject:** clear, e.g. "Re: your request — specialist follow-up".
- **body:** **Substance only** — 2–5 short **plain paragraphs** (acknowledgment, what you understand, limits of this channel, that a specialist will review policy/payment/booking records). **No** "Dear…" line and **no** formal sign-off or signature — the system adds salutation, a standard **What happens next** block, and a Reservations signature when saving. **No** markdown headings inside **body**. Your chat reply may still read as a full email; the **body** argument should be the middle part only so the on-disk draft is one coherent letter.
- **notes_for_staff:** summarize what was flagged, what you told the guest, and what a human should do next.

Do not promise refunds, credits, or outcomes you cannot authorize. Offer escalation to a specialist when appropriate.

"""


def autonomous_risk_escalation_prompt(*, guest_default_year: int | None = None) -> str:
    y = prompt_example_year(guest_default_year)
    return _AUTONOMOUS_RISK_HEAD + _concat_risk_escalation_policy_blocks(year=y) + "\n"


PLANNER_SYSTEM_PROMPT = planner_system_prompt()
EXECUTOR_SYSTEM_PROMPT = executor_system_prompt()
REVIEW_ONLY_SYSTEM_PROMPT = review_only_system_prompt()
AUTONOMOUS_RISK_ESCALATION_PROMPT = autonomous_risk_escalation_prompt()

def build_model() -> ChatOpenAI:
    """Chat model for the agent (set OPENAI_API_KEY in the environment)."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required to build the agent. Export it before invoking the graph."
        )
    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.1")),
    )


def build_email_agent(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    tools: Sequence[Any] | None = None,
    system_prompt: str | None = None,
    name: str = "hotel_guest_email_agent",
) -> Any:
    """Compiled LangGraph: pass `configurable` with execution_mode, writes_approved, manual_review_only, etc.

    Default tools: full read+write agent with standard system prompt.
    """
    model = build_model()
    t = list(tools) if tools is not None else list(HOTEL_TOOLS_ALL)
    p = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    return create_react_agent(
        model=model,
        tools=t,
        prompt=p,
        name=name,
        checkpointer=checkpointer,
    )


def _config(pms: MockHotelPMS, thread_id: str, configurable: dict[str, Any] | None = None) -> dict[str, Any]:
    c: dict[str, Any] = {
        "pms": pms,
        "thread_id": thread_id,
        "execution_mode": "autonomous",
        "writes_approved": False,
        "booking_pms_commit_allowed": False,
        "defer_new_reservation_until_desk_approve": False,
        "manual_review_only": False,
        "manual_review_reason": "",
    }
    if configurable is not None:
        c.update(configurable)
    # LangGraph may merge other configurable fragments; keep PMS and thread stable for tools.
    c["pms"] = pms
    c["thread_id"] = thread_id
    return {"configurable": c}


def _text_from_content_block(content: Any) -> str:
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _stream_chunk_text(chunk: Any) -> str:
    return _text_from_content_block(getattr(chunk, "content", None))


def _parse_stream_event(item: Any) -> tuple[str | None, Any]:
    """Normalize LangGraph stream items (single or multi mode)."""
    if not isinstance(item, tuple):
        return None, item
    if len(item) == 2 and item[0] in (
        "messages",
        "updates",
        "values",
        "custom",
        "debug",
        "tasks",
        "checkpoints",
    ):
        return item[0], item[1]
    if len(item) == 3:
        return item[1], item[2]
    return None, item


def stream_agent_turn(
    agent: Any,
    user_text: str,
    *,
    pms: MockHotelPMS,
    thread_id: str,
    on_token: Callable[[str], None] | None = None,
    on_tool_start: Callable[[str], None] | None = None,
    on_tool_done: Callable[[str], None] | None = None,
    configurable: dict[str, Any] | None = None,
) -> list[BaseMessage]:
    """
    Run one turn with LangGraph stream: optional token + tool callbacks for terminal UX.

    Falls back to invoke if streaming yields an unexpected shape.
    """
    config = _config(pms, thread_id, configurable)
    input_state = {"messages": [HumanMessage(content=user_text.strip())]}

    announced: set[str] = set()
    last_messages: list[BaseMessage] | None = None

    with bind_pms_for_agent_turn(thread_id, pms):
        try:
            stream_iter = agent.stream(
                input_state,
                config=config,
                stream_mode=["messages", "updates", "values"],
            )
        except Exception:
            return agent_reply(agent, user_text, pms=pms, thread_id=thread_id, configurable=configurable)

        for item in stream_iter:
            mode, payload = _parse_stream_event(item)
            if mode is None:
                continue

            if mode == "values" and isinstance(payload, dict):
                m = payload.get("messages")
                if isinstance(m, list) and m:
                    last_messages = m

            if mode == "messages" and on_token:
                if isinstance(payload, tuple) and len(payload) >= 1:
                    chunk = payload[0]
                    meta = payload[1] if len(payload) > 1 else {}
                    if isinstance(meta, dict):
                        node = meta.get("langgraph_node")
                        if node is not None and node != "agent":
                            continue
                    piece = _stream_chunk_text(chunk)
                    if piece:
                        on_token(piece)

            if mode == "updates" and isinstance(payload, dict):
                agent_upd = payload.get("agent")
                if isinstance(agent_upd, dict) and on_tool_start:
                    for m in agent_upd.get("messages") or []:
                        if isinstance(m, AIMessage) and m.tool_calls:
                            for tc in m.tool_calls:
                                if isinstance(tc, dict):
                                    name = str(tc.get("name", ""))
                                    tid = str(tc.get("id", name))
                                else:
                                    name = str(getattr(tc, "name", "") or "")
                                    tid = str(getattr(tc, "id", None) or name)
                                if name and tid not in announced:
                                    announced.add(tid)
                                    on_tool_start(name)

                tools_upd = payload.get("tools")
                if isinstance(tools_upd, dict) and on_tool_done:
                    for m in tools_upd.get("messages") or []:
                        if isinstance(m, ToolMessage):
                            nm = getattr(m, "name", None) or "tool"
                            on_tool_done(str(nm))

        if isinstance(last_messages, list) and last_messages:
            return last_messages

        try:
            snap = agent.get_state(config)
            values = getattr(snap, "values", None) or {}
            msgs = values.get("messages")
            if isinstance(msgs, list) and msgs:
                return msgs
        except Exception:
            pass

        return agent_reply(agent, user_text, pms=pms, thread_id=thread_id, configurable=configurable)


def agent_reply(
    agent: Any,
    user_text: str,
    *,
    pms: MockHotelPMS,
    thread_id: str,
    configurable: dict[str, Any] | None = None,
) -> list[BaseMessage]:
    """One conversational turn; with a checkpointer, history is keyed by thread_id."""
    with bind_pms_for_agent_turn(thread_id, pms):
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_text.strip())]},
            config=_config(pms, thread_id, configurable),
        )
    return result["messages"]


def run_on_email(
    inbound: str,
    *,
    guest_from_email: str | None = None,
    pms: MockHotelPMS | None = None,
    thread_id: str = "default",
    agent: Any | None = None,
    configurable: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run the agent on one inbound email body (single shot; no prior thread unless agent has a checkpointer).

    guest_from_email: optional envelope sender to include in context for the model.
    """
    pms = pms or MockHotelPMS()
    use_agent = agent or build_email_agent(
        system_prompt=system_prompt_for_full_agent(guest_default_year=pms.default_guest_stay_year()),
    )
    header = (
        f"(Guest email address on file for lookup: {guest_from_email})\n\n"
        if guest_from_email
        else ""
    )
    text = header + inbound.strip()
    messages = agent_reply(use_agent, text, pms=pms, thread_id=thread_id, configurable=configurable)
    return {"messages": messages, "pms": pms}


def last_ai_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            if isinstance(m.content, str):
                return m.content
            parts = []
            for block in m.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
    return ""
