"""LangChain tools for the mock PMS; READ vs WRITE split + execution guards."""

from __future__ import annotations

import contextvars
import json
import re
from contextlib import contextmanager
from datetime import datetime
from typing import Annotated, Iterator

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import InjectedToolArg, tool

from guest_booking_context import looks_like_email
from guest_dates import (
    _agent_dbg_log,
    extract_guest_stay_date_literals,
    literals_from_config,
    modify_dates_fail_gate,
    stay_dates_fail_gate,
    stay_dates_read_fail_gate,
)
from hotel_pms import MockHotelPMS
from review_queue import (
    normalize_category,
    save_queued_correspondence,
    try_commit_booking_from_correspondence_file,
)

# Injected RunnableConfig: LangGraph supplies configurable (pms, thread, …); not exposed to the LLM as tool args.
PmsToolConfig = Annotated[RunnableConfig, InjectedToolArg]

# LangChain's tool runner only auto-injects RunnableConfig when the parameter type is exactly
# `RunnableConfig`, not `Annotated[..., InjectedToolArg]`, so `config` is often missing. Use
# ``bind_pms_for_agent_turn`` from graph (contextvar + per-thread_id registry).
_tool_pms_ctx: contextvars.ContextVar[MockHotelPMS | None] = contextvars.ContextVar(
    "_tool_pms_ctx", default=None
)

# When LangGraph runs tools, RunnableConfig injection often fails unless the parameter is exactly
# RunnableConfig (see PmsToolConfig note below). Contextvars can also be lost across async/tool
# boundaries during streaming. Active turns register here by thread_id (and optional single-fallback).
_pms_by_thread: dict[str, MockHotelPMS] = {}
# Nested ``bind_pms_for_agent_turn`` (e.g. stream fallback → ``agent_reply``) must not pop keys
# while an outer bind still holds the same thread_id — ToolNode may still run tools in a pool.
_pms_registry_depth: dict[str, int] = {}

# Last line of defence: tools sometimes run with empty config on a code path where neither
# configurable['pms'], the contextvar, nor thread_id registry match (e.g. subgraph / streaming).
# While ``bind_pms_for_agent_turn`` is open, the stack holds the PMS for the innermost active turn.
_pms_active_stack: list[MockHotelPMS] = []


@contextmanager
def tool_pms_scope(pms: MockHotelPMS) -> Iterator[None]:
    token = _tool_pms_ctx.set(pms)
    try:
        yield
    finally:
        _tool_pms_ctx.reset(token)


def _pms_registry_keys(thread_id: str) -> frozenset[str]:
    """Keys used for ``_pms_by_thread`` so tools still resolve when LangGraph passes only a checkpoint id."""
    tid = (thread_id or "").strip() or "__default__"
    keys = {tid}
    if "::" in tid:
        base = tid.split("::", 1)[0].strip()
        if base:
            keys.add(base)
    return frozenset(keys)


@contextmanager
def bind_pms_for_agent_turn(thread_id: str, pms: MockHotelPMS) -> Iterator[None]:
    """
    Use around agent.invoke/stream so PMS tools resolve even when config injection is empty.
    Registers ``pms`` under ``thread_id`` and, when it contains ``::``, under the prefix before
    ``::`` (LangGraph tool nodes may only see the checkpoint thread id, not the composite id).
    """
    keys = _pms_registry_keys(thread_id)
    token = _tool_pms_ctx.set(pms)
    for k in keys:
        _pms_by_thread[k] = pms
        _pms_registry_depth[k] = _pms_registry_depth.get(k, 0) + 1
    _pms_active_stack.append(pms)
    try:
        yield
    finally:
        if _pms_active_stack and _pms_active_stack[-1] is pms:
            _pms_active_stack.pop()
        for k in keys:
            d = _pms_registry_depth.get(k, 0) - 1
            if d <= 0:
                _pms_registry_depth.pop(k, None)
                _pms_by_thread.pop(k, None)
            else:
                _pms_registry_depth[k] = d
        _tool_pms_ctx.reset(token)


def _pms(config: RunnableConfig | None) -> MockHotelPMS:
    raw = config or {}
    c = raw.get("configurable") or {}
    if not isinstance(c, dict):
        c = {}
    pms = c.get("pms")
    if isinstance(pms, MockHotelPMS):
        return pms
    parent = var_child_runnable_config.get()
    if isinstance(parent, dict):
        pc = parent.get("configurable") or {}
        if isinstance(pc, dict):
            pp = pc.get("pms")
            if isinstance(pp, MockHotelPMS):
                return pp
            ptid = pc.get("thread_id")
            if isinstance(ptid, str) and ptid.strip():
                t0 = ptid.strip()
                reg = _pms_by_thread.get(t0)
                if isinstance(reg, MockHotelPMS):
                    return reg
                if "::" in t0:
                    reg = _pms_by_thread.get(t0.split("::", 1)[0].strip())
                    if isinstance(reg, MockHotelPMS):
                        return reg
    fb = _tool_pms_ctx.get()
    if isinstance(fb, MockHotelPMS):
        return fb
    tid = c.get("thread_id") or raw.get("thread_id")
    if isinstance(tid, str) and tid.strip():
        t = tid.strip()
        reg = _pms_by_thread.get(t)
        if isinstance(reg, MockHotelPMS):
            return reg
        if "::" in t:
            reg = _pms_by_thread.get(t.split("::", 1)[0].strip())
            if isinstance(reg, MockHotelPMS):
                return reg
    if _pms_by_thread:
        vals = list(_pms_by_thread.values())
        first = vals[0]
        if all(isinstance(v, MockHotelPMS) and v is first for v in vals):
            return first
    if _pms_active_stack:
        top = _pms_active_stack[-1]
        if isinstance(top, MockHotelPMS):
            return top
    raise RuntimeError("Missing configurable['pms']: MockHotelPMS instance")


def _cfg(config: RunnableConfig) -> dict:
    return (config or {}).get("configurable") or {}


def _session_guest_from_cfg(config: RunnableConfig | None) -> tuple[str, str]:
    c = _cfg(config)
    return str(c.get("session_guest_email") or "").strip(), str(c.get("session_guest_id") or "").strip()


_BAD_EMAIL_MARKERS = (
    "placeholder",
    "unknown.local",
    "guest@guest",
    "your_email",
    "your.email",
    "@example.com",
    "noreply",
    "changeme",
)


def _is_placeholder_guest_email(email: str) -> bool:
    e = (email or "").strip().lower()
    if not e:
        return True
    return any(m in e for m in _BAD_EMAIL_MARKERS)


def _scrub_placeholder_emails_in_body(body: str, real_email: str) -> str:
    """Replace common LLM placeholder recipient strings in queued correspondence body."""
    r = (real_email or "").strip()
    if not r:
        return body
    out = body or ""
    for pat in (
        r"guest_email_placeholder\S*",
        r"guest_user_email",
        r"guest\s+user\s+email",
        r"guest@unknown\.local",
        r"guest@guest\S*",
    ):
        out = re.sub(pat, r, out, flags=re.IGNORECASE)
    return out


def _pms_guest_by_id(pms: MockHotelPMS, guest_id: str) -> dict | None:
    gid = (guest_id or "").strip()
    if not gid:
        return None
    for g in pms.list_guests():
        if str(g.get("id", "")).strip() == gid:
            return dict(g)
    return None


def _review_queue_path(config: RunnableConfig) -> str | None:
    v = _cfg(config).get("review_queue_dir")
    return str(v).strip() if v else None


def _guest_date_literals(config: RunnableConfig):
    """frozenset from configurable, or None if key absent (no enforcement)."""
    if "guest_date_literals" not in _cfg(config):
        return None
    raw = _cfg(config).get("guest_date_literals")
    if raw is None:
        return frozenset()
    if isinstance(raw, (str, bytes)):
        return literals_from_config([raw])
    return literals_from_config(list(raw))


def _allowed_stay_dates(config: RunnableConfig):
    """
    Allowed YYYY-MM-DD set for gating: prefer live parsing of the full guest thread so follow-ups
    and implied nights stay in sync; fall back to guest_date_literals if thread text is absent.
    """
    c = _cfg(config)
    thread = (c.get("guest_thread_text_for_date_gate") or c.get("guest_thread_text") or "").strip()
    if thread:
        dy = c.get("guest_default_year")
        if dy is not None and str(dy).strip() != "":
            try:
                y = int(dy)
                out = extract_guest_stay_date_literals(thread, default_year=y)
            except (TypeError, ValueError):
                out = extract_guest_stay_date_literals(thread)
        else:
            out = extract_guest_stay_date_literals(thread)
    else:
        out = _guest_date_literals(config)
    # #region agent log
    _agent_dbg_log(
        "agent_tools:_allowed_stay_dates",
        "resolved",
        {
            "has_guest_thread_text_key": "guest_thread_text" in c,
            "thread_len": len(thread),
            "thread_sample": thread[:200],
            "allowed_count": len(out or frozenset()) if out is not None else None,
            "allowed_sample": sorted(out or [])[:16] if out is not None else None,
        },
        "H1,H2,H3",
    )
    # #endregion
    return out


def _personalize_salutation(body: str, to_email: str, config: RunnableConfig) -> str:
    """
    Replace leading "Dear Guest," with "Dear <First> <Last>," when PMS has a matching guest.
    Keeps behavior unchanged when no guest is found or salutation is already personalized.
    """
    guest = _pms(config).find_guest_by_email(to_email)
    if not guest:
        return body
    first = str(guest.get("first_name", "")).strip()
    last = str(guest.get("last_name", "")).strip()
    full = " ".join(x for x in (first, last) if x).strip()
    if not full:
        return body
    return re.sub(
        r"(?im)^(\s*dear\s+)guest(\s*,)",
        rf"\1{full}\2",
        body,
        count=1,
    )


def _write_allowed(config: RunnableConfig) -> tuple[bool, str]:
    """Block PMS mutations and queued correspondence unless policy allows."""
    c = _cfg(config)
    if c.get("manual_review_only"):
        r = c.get("manual_review_reason") or "This request requires manual review before any system action."
        return False, r
    mode = c.get("execution_mode", "autonomous")
    if mode == "approval" and not c.get("writes_approved"):
        return (
            False,
            "Human approval is required (approval mode): propose a plan first; "
            "PMS writes and queuing guest correspondence run only after explicit operator approval.",
        )
    return True, ""


def _block_write(config: RunnableConfig) -> str | None:
    ok, msg = _write_allowed(config)
    if ok:
        return None
    return json.dumps({"ok": False, "blocked": True, "reason": msg})


def _pms_booking_mutation_block(config: RunnableConfig) -> str | None:
    """
    Reservations are not written to the PMS until staff release (draft approval / operator toggle).
    Applies in autonomous and approval modes until booking_pms_commit_allowed is set.
    """
    c = _cfg(config)
    if c.get("booking_pms_commit_allowed"):
        return None
    return json.dumps(
        {
            "ok": False,
            "blocked": True,
            "reason": (
                "PMS booking mutations are disabled until staff release. Queue the proposal with "
                "pms_queue_correspondence_for_review (e.g. category booking_confirmations). Do not "
                "tell the guest the reservation is finally confirmed until staff approve the draft "
                "and booking commits are enabled (human approval execute step, or operator release in "
                "autonomous mode / HOTEL_BOOKING_COMMIT)."
            ),
        },
        indent=2,
    )


def _stay_window_tool_error(check_in: str, check_out: str) -> str | None:
    """
    Mock PMS uses half-open nights [check_in, check_out). Equal or reversed dates → no valid nights.
    """
    try:
        di = datetime.strptime(check_in.strip(), "%Y-%m-%d")
        do = datetime.strptime(check_out.strip(), "%Y-%m-%d")
    except ValueError:
        return None
    if do > di:
        return None
    if do == di:
        y = di.year
        ex_ci = f"{y:04d}-05-21"
        ex_co = f"{y:04d}-05-22"
        return json.dumps(
            {
                "ok": False,
                "error": (
                    "Zero nights: check_in and check_out are the same calendar day. For stays with "
                    "one or more nights, check_out must be **after** check_in (departure morning). "
                    f"Example — one night on {ex_ci}: check_in={ex_ci}, check_out={ex_co}. "
                    "Never tell the guest that check_out must **equal** check_in — that is backwards."
                ),
                "check_in": check_in.strip(),
                "check_out": check_out.strip(),
            },
            indent=2,
        )
    return json.dumps(
        {
            "ok": False,
            "error": "check_out must be strictly after check_in (YYYY-MM-DD).",
            "check_in": check_in.strip(),
            "check_out": check_out.strip(),
        },
        indent=2,
    )


# --- Read path (safe for planning / lookups) ---


@tool
def pms_find_guest_by_email(email: str, config: PmsToolConfig) -> str:
    """Look up a guest profile in the PMS by email address (case-insensitive)."""
    g = _pms(config).find_guest_by_email(email)
    return json.dumps(g) if g else json.dumps({"found": False, "email": email})


@tool
def pms_check_availability(
    config: PmsToolConfig,
    check_in: str,
    check_out: str,
    adults: int = 1,
    children: int = 0,
) -> str:
    """List room types available for the stay (YYYY-MM-DD), fitting party size.

    Nights booked are every date d where check_in <= d < check_out (check_out is departure morning).
    One night starting 21 May → check_in and check_out are that date and the next calendar day in ISO form
    (year from the guest thread / guest_default_year). Same day for both = zero nights.

    Each listed room includes **quote_standard_rate_rp001** (total nights, NOK total, currency) computed from the
    same mock data as inventory — use it for Standard Rate pricing so you do not contradict yourself with a separate
    **pms_quote_stay** call that used different arguments. Call **pms_quote_stay** only for other rate plans (RP002–RP004).

    Dates must be grounded in the guest thread (re-parsed each call when guest_thread_text is set).
    Read path uses a night-aware rule so implied departure morning does not block availability."""
    allowed_here = _allowed_stay_dates(config)
    g = stay_dates_read_fail_gate(check_in, check_out, allowed_here)
    # #region agent log
    _agent_dbg_log(
        "agent_tools:pms_check_availability",
        "gate_result",
        {
            "check_in": check_in.strip(),
            "check_out": check_out.strip(),
            "adults": adults,
            "children": children,
            "blocked": g is not None,
            "blocked_preview": (g[:120] if g else None),
        },
        "H4,H5",
    )
    # #endregion
    if g:
        return g
    sw = _stay_window_tool_error(check_in, check_out)
    if sw:
        return sw
    rows = _pms(config).check_availability(check_in, check_out, adults, children)
    return json.dumps(rows, indent=2)


@tool
def pms_quote_stay(
    config: PmsToolConfig,
    check_in: str,
    check_out: str,
    room_type_id: str,
    rate_plan_id: str,
    adults: int = 1,
    children: int = 0,
) -> str:
    """Price a stay: room_type_id (e.g. RT002), rate_plan_id (e.g. RP001). Read-only quote.

    Same date grounding as pms_check_availability (thread-based, read-friendly gate).
    check_out must be after check_in; see pms_check_availability doc."""
    g = stay_dates_read_fail_gate(check_in, check_out, _allowed_stay_dates(config))
    if g:
        return g
    sw = _stay_window_tool_error(check_in, check_out)
    if sw:
        return sw
    q = _pms(config).quote_stay(
        check_in, check_out, room_type_id, rate_plan_id, adults, children
    )
    return json.dumps(q, indent=2)


@tool
def pms_get_reservation(reservation_id: str, config: PmsToolConfig) -> str:
    """Fetch one reservation by id (e.g. RES001)."""
    r = _pms(config).get_reservation(reservation_id)
    return json.dumps(r) if r else json.dumps({"found": False, "id": reservation_id})


@tool
def pms_list_guest_reservations(
    config: PmsToolConfig,
    guest_id: str,
    include_cancelled: bool = False,
) -> str:
    """List reservations for a guest_id."""
    rows = _pms(config).list_guest_reservations(guest_id, include_cancelled)
    return json.dumps(rows, indent=2)


@tool
def pms_hotel_reference(config: PmsToolConfig) -> str:
    """Hotel name, contact, check-in/out times, currency, and policy text for replies."""
    pms = _pms(config)
    h = pms.hotel
    block = {
        "hotel": h,
        "policies_summary": pms.get_policies_summary(),
        "rate_plans": [{"id": x["id"], "name": x["name"]} for x in pms.get_rate_plans()],
    }
    return json.dumps(block, indent=2)


HOTEL_READ_TOOLS = [
    pms_find_guest_by_email,
    pms_check_availability,
    pms_quote_stay,
    pms_get_reservation,
    pms_list_guest_reservations,
    pms_hotel_reference,
]


# --- Write path (guarded) ---


@tool
def pms_create_guest(
    config: PmsToolConfig,
    email: str,
    first_name: str,
    last_name: str,
    phone: str = "",
    nationality: str = "",
) -> str:
    """Create a new guest profile. Fails if email already exists."""
    b = _block_write(config)
    if b:
        return b
    pms = _pms(config)
    sess_em, sess_gid = _session_guest_from_cfg(config)
    email_norm = (email or "").strip()
    if sess_gid and _pms_guest_by_id(pms, sess_gid):
        return json.dumps(
            {
                "ok": False,
                "error": (
                    f"Session guest {sess_gid!r} is already in the PMS — do not call pms_create_guest. "
                    f"Use pms_create_reservation with that guest_id and email {sess_em!r} for correspondence."
                ),
            },
            indent=2,
        )
    if sess_em and pms.find_guest_by_email(sess_em):
        ex = pms.find_guest_by_email(sess_em)
        return json.dumps(
            {
                "ok": False,
                "error": (
                    f"Guest already exists for session email ({ex.get('id')!r}). "
                    "Do not create again; use that guest_id for reservations."
                ),
            },
            indent=2,
        )
    if sess_em and _is_placeholder_guest_email(email_norm):
        return json.dumps(
            {
                "ok": False,
                "error": (
                    f"Do not use a placeholder email. This session guest is {sess_em!r} — "
                    "use pms_find_guest_by_email or skip create if they already exist."
                ),
            },
            indent=2,
        )
    try:
        g = pms.create_guest(email, first_name, last_name, phone, nationality)
        return json.dumps({"ok": True, "guest": g})
    except ValueError as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool
def pms_create_reservation(
    config: PmsToolConfig,
    guest_id: str,
    room_type_id: str,
    rate_plan_id: str,
    check_in: str,
    check_out: str,
    adults: int = 1,
    children: int = 0,
    notes: str = "",
) -> str:
    """Write a reservation to the mock PMS (requires booking_pms_commit_allowed — staff release).

    check_in/check_out must match guest-stated dates: same rule as availability (occupied nights must
    fall on guest-parsed calendar days; check_out may be the morning after the last night).
    Until release, queue a draft instead; do not imply the guest is finally confirmed."""
    b = _block_write(config)
    if b:
        return b
    c0 = _cfg(config)
    if c0.get("defer_new_reservation_until_desk_approve"):
        return json.dumps(
            {
                "ok": False,
                "blocked": True,
                "reason": (
                    "Streamlit Human mode: new stays are **not** written to the PMS in chat. Queue "
                    "**booking_confirmations** or **drafts** with **booking_commit_json** (BOOKING_COMMIT). "
                    "Staff **Approve** on **Pending guest email** applies the reservation to the mock PMS."
                ),
            },
            indent=2,
        )
    bb = _pms_booking_mutation_block(config)
    if bb:
        return bb
    g = stay_dates_read_fail_gate(check_in, check_out, _allowed_stay_dates(config))
    if g:
        return g
    sw = _stay_window_tool_error(check_in, check_out)
    if sw:
        return sw
    pms = _pms(config)
    sess_em, sess_gid = _session_guest_from_cfg(config)
    gid_arg = (guest_id or "").strip()
    if sess_gid and gid_arg != sess_gid:
        return json.dumps(
            {
                "ok": False,
                "error": (
                    f"This session is for guest_id {sess_gid!r}; pass it to pms_create_reservation "
                    f"(tool was called with {gid_arg!r})."
                ),
            },
            indent=2,
        )
    if not sess_gid and sess_em:
        gf = pms.find_guest_by_email(sess_em)
        if gf and str(gf.get("id", "")).strip() != gid_arg:
            return json.dumps(
                {
                    "ok": False,
                    "error": (
                        f"For session email {sess_em!r} use guest_id {gf.get('id')!r} "
                        f"(tool was called with {gid_arg!r})."
                    ),
                },
                indent=2,
            )
    r = pms.create_reservation(
        guest_id,
        room_type_id,
        rate_plan_id,
        check_in,
        check_out,
        adults,
        children,
        notes,
    )
    return json.dumps(r, indent=2)


@tool
def pms_cancel_reservation(
    config: PmsToolConfig,
    reservation_id: str,
    reason: str = "",
) -> str:
    """Cancel a reservation and restore inventory in the mock PMS."""
    b = _block_write(config)
    if b:
        return b
    bb = _pms_booking_mutation_block(config)
    if bb:
        return bb
    return json.dumps(_pms(config).cancel_reservation(reservation_id, reason), indent=2)


@tool
def pms_modify_reservation(
    config: PmsToolConfig,
    reservation_id: str,
    check_in: str = "",
    check_out: str = "",
    room_type_id: str = "",
    rate_plan_id: str = "",
    adults: int = -1,
    children: int = -1,
    notes: str = "",
) -> str:
    """Modify a reservation. Pass empty strings / -1 to leave a field unchanged. Non-refundable stays cannot change dates/room/party."""
    b = _block_write(config)
    if b:
        return b
    bb = _pms_booking_mutation_block(config)
    if bb:
        return bb
    p = _pms(config)
    kwargs = {}
    if check_in:
        kwargs["check_in"] = check_in
    if check_out:
        kwargs["check_out"] = check_out
    if room_type_id:
        kwargs["room_type_id"] = room_type_id
    if rate_plan_id:
        kwargs["rate_plan_id"] = rate_plan_id
    if adults >= 0:
        kwargs["adults"] = adults
    if children >= 0:
        kwargs["children"] = children
    if notes:
        kwargs["notes"] = notes
    mg = modify_dates_fail_gate(
        check_in if check_in else "",
        check_out if check_out else "",
        _allowed_stay_dates(config),
    )
    if mg:
        return mg
    return json.dumps(p.modify_reservation(reservation_id, **kwargs), indent=2)


@tool
def pms_queue_correspondence_for_review(
    config: PmsToolConfig,
    category: str,
    to_email: str,
    subject: str,
    body: str,
    related_reservation_id: str = "",
    notes_for_staff: str = "",
    booking_commit_json: str = "",
    booking_modify_json: str = "",
) -> str:
    """
    Does NOT email anyone. Saves markdown under review_queue/<category>/ (**pending**) for most
    categories. Under **autonomous** mode, **drafts** and **booking_confirmations** may be filed
    directly under review_queue/approved/<category>/. Use **booking_confirmations** only after a
    successful **pms_create_reservation** (real confirmed stay); before that, use **drafts** for
    quotes or in-chat-style asks — not confirmation emails. **Escalations** always use pending
    ``review_queue/escalations/`` (listed on the Escalations desk).

    Optional ``booking_commit_json``: JSON object with keys guest_id, room_type_id, rate_plan_id,
    check_in, check_out, plus optional adults, children, notes. Embedded in the file; in **autonomous**
    mode only, with commits allowed and without ``defer_new_reservation_until_desk_approve``, saving
    under approved/ may run ``create_reservation`` immediately after save.

    Optional ``booking_modify_json`` (for category **modifications**, Streamlit Human mode): JSON object
    with **reservation_id** (or rely on **related_reservation_id**), plus optional check_in, check_out,
    room_type_id, rate_plan_id, adults, children, notes. Embedded as **BOOKING_MODIFY**; staff **Approve**
    on Pending guest email applies the change in the mock PMS.

    category: drafts | cancellations | modifications | booking_confirmations | escalations | general
    (aliases: draft, cancellation, modification, booking, etc.)
    """
    c = _cfg(config)
    cat_norm = normalize_category(category)
    # Risk-flagged autonomous turns: still allow filing **escalations** so staff get the trail.
    if c.get("manual_review_only") and cat_norm != "escalations":
        b = _block_write(config)
        if b:
            return b
    elif not c.get("manual_review_only"):
        # Human-mode planner (approval, writes_approved false): allow **escalations** only so the
        # Escalations desk gets files without waiting for a booking-confirm executor turn.
        escalation_planner_bypass = (
            cat_norm == "escalations"
            and c.get("execution_mode") == "approval"
            and not c.get("writes_approved")
        )
        if not escalation_planner_bypass:
            b = _block_write(config)
            if b:
                return b
    booking_commit: dict | None = None
    booking_modify: dict | None = None
    raw_bc = (booking_commit_json or "").strip()
    raw_bm = (booking_modify_json or "").strip()
    if raw_bc:
        try:
            booking_commit = json.loads(raw_bc)
            if not isinstance(booking_commit, dict):
                return json.dumps({"ok": False, "error": "booking_commit_json must be a JSON object."})
        except json.JSONDecodeError as e:
            return json.dumps({"ok": False, "error": f"Invalid booking_commit_json: {e}"})
    if raw_bm:
        try:
            booking_modify = json.loads(raw_bm)
            if not isinstance(booking_modify, dict):
                return json.dumps({"ok": False, "error": "booking_modify_json must be a JSON object."})
        except json.JSONDecodeError as e:
            return json.dumps({"ok": False, "error": f"Invalid booking_modify_json: {e}"})
    if booking_commit is not None and booking_modify is not None:
        return json.dumps(
            {"ok": False, "error": "Use either booking_commit_json or booking_modify_json, not both."}
        )
    sess_em, sess_gid = _session_guest_from_cfg(config)
    if booking_commit is not None and sess_gid:
        booking_commit = dict(booking_commit)
        booking_commit["guest_id"] = sess_gid
    to_norm = (to_email or "").strip()
    if sess_em and (
        not to_norm
        or _is_placeholder_guest_email(to_norm)
        or not looks_like_email(to_norm)
    ):
        to_email = sess_em
    body_scrubbed = _scrub_placeholder_emails_in_body(body, sess_em)
    body_fixed = _personalize_salutation(body_scrubbed, to_email, config)
    # Never default missing execution_mode to autonomous: that filed drafts under approved/ and ran
    # BOOKING_COMMIT immediately, creating PMS rows before Streamlit Human desk approval.
    direct = cat_norm in ("drafts", "booking_confirmations") and c.get("execution_mode") == "autonomous"
    hotel_display_name = ""
    try:
        h = _pms(config).hotel
        if isinstance(h, dict):
            hotel_display_name = str(h.get("name") or "").strip()
    except Exception:
        hotel_display_name = ""
    path = save_queued_correspondence(
        category,
        to_email,
        subject,
        body_fixed,
        related_reservation_id=related_reservation_id,
        notes_for_staff=notes_for_staff,
        booking_commit=booking_commit,
        booking_modify=booking_modify,
        direct_to_approved=direct,
        queue_root=_review_queue_path(config),
        hotel_display_name=hotel_display_name,
    )
    out: dict = {
        "ok": True,
        "queued": not direct,
        "filed_under_approved": direct,
        "email_not_sent": True,
        "path": str(path),
        "category": cat_norm,
    }
    if (
        direct
        and booking_commit
        and c.get("booking_pms_commit_allowed")
        and not c.get("defer_new_reservation_until_desk_approve")
    ):
        ok, msg = try_commit_booking_from_correspondence_file(path, _pms(config))
        out["booking_commit"] = {"ok": ok, "detail": msg}
    return json.dumps(out, indent=2)


HOTEL_WRITE_TOOLS = [
    pms_create_guest,
    pms_create_reservation,
    pms_cancel_reservation,
    pms_modify_reservation,
    pms_queue_correspondence_for_review,
]

# Read tools + queue only (e.g. autonomous risk / escalation path; no PMS mutations).
HOTEL_READ_AND_ESCALATION_QUEUE_TOOLS = list(HOTEL_READ_TOOLS) + [
    pms_queue_correspondence_for_review,
]

HOTEL_TOOLS_ALL = HOTEL_READ_TOOLS + HOTEL_WRITE_TOOLS
