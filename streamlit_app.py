"""
Hotel guest agent — Streamlit UI (single-page split layout: chat | operations desk).

Run: streamlit run streamlit_app.py
Or: python main.py  (defaults to launching this app on an interactive TTY)
"""

from __future__ import annotations

import html
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

_APP_ROOT = Path(__file__).resolve().parent
_LOGO_PATH = _APP_ROOT / "logo.png"

load_dotenv(_APP_ROOT / ".env")

from agent_tools import (
    HOTEL_READ_AND_ESCALATION_QUEUE_TOOLS,
    HOTEL_READ_TOOLS,
    HOTEL_WRITE_TOOLS,
)
from execution_flow import (
    ExecutionMode,
    classify_human_executor_intent,
    executor_followup_user_message,
    human_mode_desk_footer_note,
    prepare_turn_configurable,
)
from graph import (
    autonomous_risk_escalation_prompt,
    build_email_agent,
    executor_system_prompt,
    last_ai_text,
    planner_system_prompt,
    review_only_system_prompt,
    stream_agent_turn,
    system_prompt_for_full_agent,
)
from guest_booking_context import looks_like_email, message_needs_guest_for_booking
from hotel_pms import MockHotelPMS
from pms_guest import resolve_or_create_guest, session_guest_hint
from review_queue import (
    apply_pending_guest_email_on_approval,
    collect_approved_correspondence_paths,
    collect_escalation_desk_entries,
    collect_pending_paths,
    guest_correspondence_body_from_markdown_file,
    normalize_category,
    resolve_pending_draft,
    review_queue_root,
)

_GUEST_FLOW_KEY = "guest_registration_flow"
_HUMAN_MODE_LAST_PLAN_KEY = "human_mode_last_planner_reply"


def _is_guest_booking_confirmation(text: str) -> bool:
    """True on short affirmatives that should run the human-mode executor (new booking **or** cancel/modify confirm)."""
    t = (text or "").strip().lower()
    if not t:
        return False
    if re.match(
        r"^(y(es)?|yeah|yep|sure|ok(ay)?|please|go ahead|proceed|confirm(\s+it)?|book(\s+it)?)\b",
        t,
    ):
        return True
    if re.search(
        r"\b(proceed with (the )?booking|please book|book (it|them)|confirm (the|my) booking|"
        r"i'?d like to (book|proceed))\b",
        t,
    ):
        return True
    return False


def _run_human_mode_turn(
    *,
    raw_user: str,
    model_input: str,
    turn_cfg: dict,
    pms: MockHotelPMS,
    tid: str,
) -> tuple[str, list]:
    """
    Human mode: read-only planner while the guest is still deciding; after they confirm, run the
    write-capable executor once (queues correspondence; **new** reservations apply to the PMS only
    when staff **Approve** on **Pending guest email**). Cancel/modify may still run in the executor.
    """
    _ensure_streamlit_agents("approval", pms)
    last_plan = (st.session_state.get(_HUMAN_MODE_LAST_PLAN_KEY) or "").strip()
    if _is_guest_booking_confirmation(raw_user) and last_plan:
        st.session_state.approval_seq = int(st.session_state.get("approval_seq", 0)) + 1
        seq = st.session_state.approval_seq
        exec_cfg = {
            **turn_cfg,
            "execution_mode": "approval",
            "writes_approved": True,
            # Human mode defers booking/cancellation side-effects to desk Approve on pending email.
            "booking_pms_commit_allowed": False,
            "defer_new_reservation_until_desk_approve": True,
        }
        joined = "\n".join(st.session_state.accumulated_user_lines)
        ex_intent = classify_human_executor_intent(last_plan)
        follow = executor_followup_user_message(
            approved_plan_text=last_plan,
            original_user_message=joined,
            session_guest_email=st.session_state.get("guest_email") or "",
            session_guest_id=st.session_state.get("guest_id") or "",
            streamlit_human_executor_intent=ex_intent,
            streamlit_defer_new_stay_to_desk=True,
        )
        rep, msgs = _run_chat_turn(
            model_input_text=follow,
            pms=pms,
            thread_id=f"{tid}::exec::{seq}",
            turn_cfg=exec_cfg,
            agent=st.session_state.executor_agent,
        )
        st.session_state.pop(_HUMAN_MODE_LAST_PLAN_KEY, None)
        desk = human_mode_desk_footer_note(classify_human_executor_intent(last_plan))
        return rep + desk, msgs
    st.session_state.approval_seq = int(st.session_state.get("approval_seq", 0)) + 1
    seq = st.session_state.approval_seq
    plan_cfg = {**turn_cfg, "execution_mode": "approval", "writes_approved": False}
    reply, msgs = _run_chat_turn(
        model_input_text=model_input,
        pms=pms,
        thread_id=f"{tid}::plan::{seq}",
        turn_cfg=plan_cfg,
        agent=st.session_state.planner_agent,
    )
    st.session_state[_HUMAN_MODE_LAST_PLAN_KEY] = reply
    return reply, msgs


def _risk_flags_hint(risk: dict) -> str:
    """Prefix model input with heuristic flags (autonomous escalation path)."""
    rs = risk.get("reasons") or []
    if not rs:
        return ""
    return "[Internal flags for this turn (do not quote verbatim to the guest): " + " | ".join(rs) + "]\n\n"


def _env_path(key: str) -> str | None:
    v = os.environ.get(key, "").strip()
    return v or None


def _reset_chat_session() -> None:
    for k in (
        "thread_id",
        "guest_email",
        "guest_id",
        "accumulated_user_lines",
        "memory",
        "agent",
        "chat_agent",
        "planner_agent",
        "executor_agent",
        "review_agent",
        "autonomous_escalation_agent",
        "agent_mode_key",
        "approval_seq",
        _HUMAN_MODE_LAST_PLAN_KEY,
        "messages",
        _GUEST_FLOW_KEY,
    ):
        st.session_state.pop(k, None)


def _clear_guest_flow() -> None:
    st.session_state.pop(_GUEST_FLOW_KEY, None)


def _ensure_memory() -> None:
    if "memory" not in st.session_state:
        st.session_state.memory = MemorySaver()


def _ensure_streamlit_agents(mode: ExecutionMode, pms: MockHotelPMS) -> None:
    """Build or rebuild planner/executor/autonomous/review agents when mode or mock-data year changes."""
    _ensure_memory()
    mem = st.session_state.memory
    gy = pms.default_guest_stay_year()
    key = (mode, id(mem), gy)
    agents_ready = "review_agent" in st.session_state and (
        (
            mode == "autonomous"
            and "chat_agent" in st.session_state
            and "autonomous_escalation_agent" in st.session_state
        )
        or (
            mode == "approval"
            and "planner_agent" in st.session_state
            and "executor_agent" in st.session_state
        )
    )
    if st.session_state.get("agent_mode_key") == key and agents_ready:
        return
    st.session_state.agent_mode_key = key
    st.session_state.review_agent = build_email_agent(
        tools=list(HOTEL_READ_AND_ESCALATION_QUEUE_TOOLS),
        system_prompt=review_only_system_prompt(guest_default_year=gy),
        checkpointer=mem,
        name="hotel_review_streamlit",
    )
    if mode == "autonomous":
        st.session_state.chat_agent = build_email_agent(
            checkpointer=mem,
            system_prompt=system_prompt_for_full_agent(guest_default_year=gy),
            name="hotel_autonomous_streamlit",
        )
        st.session_state.autonomous_escalation_agent = build_email_agent(
            tools=list(HOTEL_READ_AND_ESCALATION_QUEUE_TOOLS),
            system_prompt=autonomous_risk_escalation_prompt(guest_default_year=gy),
            checkpointer=mem,
            name="hotel_autonomous_escalation_streamlit",
        )
        st.session_state.pop("planner_agent", None)
        st.session_state.pop("executor_agent", None)
    else:
        st.session_state.planner_agent = build_email_agent(
            tools=list(HOTEL_READ_AND_ESCALATION_QUEUE_TOOLS),
            system_prompt=planner_system_prompt(guest_default_year=gy),
            checkpointer=mem,
            name="hotel_planner_streamlit",
        )
        st.session_state.executor_agent = build_email_agent(
            tools=list(HOTEL_READ_TOOLS) + list(HOTEL_WRITE_TOOLS),
            system_prompt=executor_system_prompt(guest_default_year=gy),
            checkpointer=mem,
            name="hotel_executor_streamlit",
        )
        st.session_state.pop("chat_agent", None)
        st.session_state.pop("autonomous_escalation_agent", None)
    st.session_state.agent = st.session_state.get("chat_agent") or st.session_state.get(
        "planner_agent"
    )


def _pms_from_path(data_path: str | None) -> MockHotelPMS:
    if data_path and data_path.strip():
        return MockHotelPMS(data_path.strip())
    return MockHotelPMS()


def sync_session_pms(mock_data_path: str) -> MockHotelPMS:
    """One MockHotelPMS per data path for the whole Streamlit session (state survives reruns)."""
    key = mock_data_path.strip() or "__default__"
    if st.session_state.get("pms_data_key") != key:
        st.session_state.pms = _pms_from_path(mock_data_path or None)
        st.session_state.pms_data_key = key
    return st.session_state.pms


def _streamlit_booking_pms_commit_allowed(execution_mode: ExecutionMode) -> bool:
    """Autonomous chat may commit booking mutations; Human mode releases commits only via **Approve PMS writes**."""
    return execution_mode == "autonomous"


def _build_turn_cfg(
    accumulated_user_lines: list[str],
    text_for_risk: str,
    review_queue_dir: str | None,
    execution_mode: ExecutionMode,
    pms: MockHotelPMS,
    *,
    booking_pms_commit_allowed: bool = False,
) -> tuple[dict, dict]:
    em = (st.session_state.get("guest_email") or "").strip() or None
    gid = (st.session_state.get("guest_id") or "").strip() or None
    cfg, risk = prepare_turn_configurable(
        accumulated_user_lines=accumulated_user_lines,
        text_for_risk=text_for_risk,
        execution_mode=execution_mode,
        review_queue_dir=review_queue_dir,
        booking_pms_commit_allowed=booking_pms_commit_allowed,
        guest_default_year=pms.default_guest_stay_year(),
        session_guest_email=em,
        session_guest_id=gid,
    )
    # Human chat: new stays must not hit the PMS until staff Approve on Pending guest email
    # (BOOKING_COMMIT). Also blocks mistaken immediate commit if execution_mode were missing on a tool call.
    if execution_mode == "approval":
        cfg = {**cfg, "defer_new_reservation_until_desk_approve": True}
    return cfg, risk


def _run_chat_turn(
    *,
    model_input_text: str,
    pms: MockHotelPMS,
    thread_id: str,
    turn_cfg: dict,
    agent: object | None = None,
) -> tuple[str, list]:
    use_agent = agent if agent is not None else st.session_state.agent
    buf: list[str] = []
    tool_lines: list[str] = []

    def on_token(t: str) -> None:
        buf.append(t)

    with st.chat_message("assistant"):
        box = st.empty()
        tool_exp = st.expander("Tool activity", expanded=False)
        try:
            msgs = stream_agent_turn(
                use_agent,
                model_input_text,
                pms=pms,
                thread_id=thread_id,
                on_token=lambda t: (on_token(t), box.markdown("".join(buf))),
                on_tool_start=lambda n: tool_lines.append(f"→ **{n}** …"),
                on_tool_done=lambda n: tool_lines.append(f"✓ `{n}`"),
                configurable=turn_cfg,
            )
        except Exception as e:
            err = f"**Error:** {e}"
            box.markdown(err)
            return err, []
        if tool_lines:
            tool_exp.markdown("\n".join(tool_lines))
        full = "".join(buf).strip() or (last_ai_text(msgs) or "").strip() or "(no text)"
        box.markdown(full)
    return full, msgs


def _append_and_run_booking_turn(
    *,
    pending_booking_message: str,
    pms: MockHotelPMS,
    review_queue_dir: str | None,
) -> None:
    st.session_state.accumulated_user_lines.append(pending_booking_message)
    hint = session_guest_hint(
        st.session_state.guest_email,
        st.session_state.get("guest_id") or "",
    )
    model_input = hint + pending_booking_message
    chat_mode: ExecutionMode = st.session_state.get("chat_execution_mode", "autonomous")  # type: ignore[assignment]
    turn_cfg, risk = _build_turn_cfg(
        st.session_state.accumulated_user_lines,
        model_input,
        review_queue_dir,
        chat_mode,
        pms,
        booking_pms_commit_allowed=_streamlit_booking_pms_commit_allowed(chat_mode),
    )
    _ensure_streamlit_agents(chat_mode, pms)
    tid = st.session_state.thread_id
    st.session_state.messages.append({"role": "user", "content": pending_booking_message})
    if risk["manual_review_required"]:
        if chat_mode == "autonomous":
            esc_cfg = {**turn_cfg, "booking_pms_commit_allowed": False}
            reply, _ = _run_chat_turn(
                model_input_text=_risk_flags_hint(risk) + model_input,
                pms=pms,
                thread_id=f"{tid}::escalation",
                turn_cfg=esc_cfg,
                agent=st.session_state.autonomous_escalation_agent,
            )
        else:
            st.session_state.pop(_HUMAN_MODE_LAST_PLAN_KEY, None)
            reply, _ = _run_chat_turn(
                model_input_text=model_input,
                pms=pms,
                thread_id=f"{tid}::review::defer",
                turn_cfg=turn_cfg,
                agent=st.session_state.review_agent,
            )
        st.session_state.messages.append({"role": "assistant", "content": reply})
        return
    if chat_mode == "autonomous":
        reply, _ = _run_chat_turn(
            model_input_text=model_input,
            pms=pms,
            thread_id=tid,
            turn_cfg=turn_cfg,
            agent=st.session_state.chat_agent,
        )
        st.session_state.messages.append({"role": "assistant", "content": reply})
        return
    reply, _ = _run_human_mode_turn(
        raw_user=pending_booking_message,
        model_input=model_input,
        turn_cfg=turn_cfg,
        pms=pms,
        tid=tid,
    )
    st.session_state.messages.append({"role": "assistant", "content": reply})


def _init_mode_toggles() -> None:
    st.session_state.setdefault("chat_execution_mode", "autonomous")
    if "human_mode_toggle" not in st.session_state:
        st.session_state.human_mode_toggle = (
            st.session_state.chat_execution_mode == "approval"
        )
    if "autonomous_mode_toggle" not in st.session_state:
        st.session_state.autonomous_mode_toggle = (
            st.session_state.chat_execution_mode == "autonomous"
        )


def _on_human_mode_change() -> None:
    """Exactly one mode on: turning Human on turns Autonomous off; turning Human off enables Autonomous."""
    if st.session_state.human_mode_toggle:
        st.session_state.autonomous_mode_toggle = False
    else:
        st.session_state.autonomous_mode_toggle = True


def _on_autonomous_mode_change() -> None:
    """Exactly one mode on: turning Autonomous on turns Human off; turning Autonomous off enables Human."""
    if st.session_state.autonomous_mode_toggle:
        st.session_state.human_mode_toggle = False
    else:
        st.session_state.human_mode_toggle = True


def _sync_mode_toggles() -> ExecutionMode:
    """Keep Human vs Autonomous mutually exclusive and mirror into chat_execution_mode."""
    h = bool(st.session_state.human_mode_toggle)
    a = bool(st.session_state.autonomous_mode_toggle)
    if h == a:
        if h and a:
            st.session_state.autonomous_mode_toggle = False
        else:
            st.session_state.autonomous_mode_toggle = True
        st.rerun()
    mode: ExecutionMode = "approval" if st.session_state.human_mode_toggle else "autonomous"
    st.session_state.chat_execution_mode = mode
    return mode


def _parse_draft_header(path: Path) -> dict[str, str]:
    to_, subj, rel_res = "", "", ""
    try:
        text = path.read_text(encoding="utf-8")[:2000]
    except OSError:
        return {"to": "", "subject": "", "reservation": ""}
    for line in text.splitlines():
        if line.startswith("- **To:**"):
            to_ = line.split(":", 1)[-1].strip()
        elif line.startswith("- **Subject:**"):
            subj = line.split(":", 1)[-1].strip()
        elif line.startswith("- **Related reservation:**"):
            rel_res = line.split("`", 2)[1] if "`" in line else line.split(":", 1)[-1].strip()
    return {"to": to_, "subject": subj, "reservation": rel_res}


@st.dialog("Email preview")
def _correspondence_preview_dialog(markdown_body: str) -> None:
    """Modal preview of the on-disk draft (markdown)."""
    theme_type = st.context.theme.get("type")
    if theme_type == "dark":
        fg, bg, border = "#e8edf5", "rgba(30, 41, 59, 0.55)", "#64748b"
    else:
        # Light theme: high-contrast body (disabled text_area is too faint on Windows).
        fg, bg, border = "#0a0f1a", "#e8ecf4", "#7c8aa0"
    body_esc = html.escape(markdown_body)
    st.markdown(
        f"""<style>
        .hotel-email-preview-pre {{
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.875rem;
            line-height: 1.55;
            padding: 0.85rem 1rem;
            border-radius: 0.5rem;
            max-height: 28rem;
            overflow: auto;
            color: {fg} !important;
            -webkit-text-fill-color: {fg};
            background: {bg};
            border: 1px solid {border};
        }}
        </style>
        <pre class="hotel-email-preview-pre">{body_esc}</pre>""",
        unsafe_allow_html=True,
    )


def _dataframe_selection_rows(ev: Any) -> list[int]:
    """Row indices from ``st.dataframe(..., on_select='rerun')`` return value."""
    if ev is None:
        return []
    sel = getattr(ev, "selection", None)
    if sel is None and isinstance(ev, dict):
        sel = ev.get("selection")
    if sel is None:
        return []
    rows = getattr(sel, "rows", None)
    if rows is None and isinstance(sel, dict):
        rows = sel.get("rows") or []
    return [r for r in rows if isinstance(r, int)]


def _render_desk_rows_with_view(
    paths: list[Path],
    *,
    key_prefix: str,
    status_labels: list[str] | None = None,
    category_labels: list[str] | None = None,
) -> None:
    """
    Wide, scrollable ``st.dataframe`` (roomy row height / column widths). One row is always
    selected; **Preview** opens the Streamlit modal on the same page (no link navigation).
    """
    n = len(paths)
    if n == 0:
        st.caption("No items.")
        return
    has_st = bool(status_labels) and len(status_labels) == n
    has_cat = bool(category_labels) and len(category_labels) == n
    row_h = 44
    table_h = min(480, 72 + row_h * max(1, n))

    rows_data: list[dict[str, str]] = []
    for i, path in enumerate(paths):
        meta = _parse_draft_header(path)
        mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        row: dict[str, str] = {
            "file": path.name,
            "modified": mtime,
            "to": meta["to"] or "—",
            "subject": meta["subject"] or "—",
        }
        if has_st:
            row = {"status": status_labels[i], **row}  # type: ignore[index]
        if has_cat:
            row = {"category": category_labels[i], **row}  # type: ignore[index]
        rows_data.append(row)

    col_order: list[str] = []
    if has_st:
        col_order.append("status")
    if has_cat:
        col_order.append("category")
    col_order.extend(["file", "modified", "to", "subject"])
    df = pd.DataFrame(rows_data)[col_order]

    cfg: dict[str, Any] = {}
    if "status" in df.columns:
        cfg["status"] = st.column_config.TextColumn("Status", width=100)
    if "category" in df.columns:
        cfg["category"] = st.column_config.TextColumn("Category", width=110)
    cfg["file"] = st.column_config.TextColumn("File", width=240)
    cfg["modified"] = st.column_config.TextColumn("Modified", width=130)
    cfg["to"] = st.column_config.TextColumn("To", width=220)
    cfg["subject"] = st.column_config.TextColumn("Subject", width=400)

    st.caption(
        "Scroll **horizontally** and **vertically** in the table. Click a row to select it "
        "(one row is always selected), then **Preview** to open the draft in a modal on this page."
    )
    ev = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=table_h,
        row_height=row_h,
        column_config=cfg,
        on_select="rerun",
        selection_mode="single-row-required",
        key=f"{key_prefix}_grid",
    )
    rows_sel = _dataframe_selection_rows(ev)
    ix = rows_sel[0] if rows_sel else 0
    if ix >= n:
        ix = n - 1

    if st.button(
        "Preview",
        type="primary",
        use_container_width=True,
        key=f"{key_prefix}_open_preview",
    ):
        try:
            body = paths[ix].read_text(encoding="utf-8")
        except OSError as e:
            body = f"(Could not read file: {e})"
        _correspondence_preview_dialog(body)


def _human_mode_cancellation_approved_chat_message(ok: bool, detail: str) -> str:
    """Guest-facing chat text after staff **Approve** on a cancellation draft (do not echo pre-approval body)."""
    if not ok:
        return (
            "Staff approved sending this cancellation update, but we could not change the reservation "
            f"in our system: **{detail}**"
        )
    d = (detail or "").strip()
    low = d.lower()
    if low.startswith("cancelled "):
        rid = d[len("Cancelled ") :].strip()
        return (
            "Your cancellation request has been **approved**. "
            f"Reservation **{rid}** is now **cancelled** in our system."
        )
    if "already cancelled" in low:
        return (
            "Your cancellation request has been **approved**. "
            f"That stay was **already cancelled** in our records ({d})."
        )
    if "not found" in low and "nothing to cancel" in low:
        return (
            "Your cancellation request has been **approved**. "
            "We did not find an active reservation to cancel under that reference — "
            "your booking may already be removed or the ID may need a manual check."
        )
    if "no related reservation" in low:
        return (
            "Your cancellation request has been **approved**. "
            "There was no reservation linked to this message, so no booking status was changed."
        )
    if d and d != "No BOOKING_COMMIT in file":
        return (
            "Your cancellation request has been **approved**. "
            f"Booking status: {d}"
        )
    return (
        "Your cancellation request has been **approved**. "
        "The booking has been updated in our system."
    )


def _render_draft_panel(
    title: str,
    paths: list[Path],
    rq: Path,
    key_prefix: str,
    *,
    pms: MockHotelPMS | None = None,
    apply_booking_on_approve: bool = False,
    category_labels: list[str] | None = None,
) -> None:
    st.markdown(f"**{title}**")
    if not paths:
        st.caption("No pending items.")
        return
    cat = category_labels if category_labels and len(category_labels) == len(paths) else None
    _render_desk_rows_with_view(
        paths, key_prefix=f"{key_prefix}_tbl", category_labels=cat
    )
    st.divider()
    sel = st.selectbox(
        "Select file",
        range(len(paths)),
        format_func=lambda i: paths[i].name,
        key=f"{key_prefix}_sel",
    )
    a1, a2 = st.columns(2)
    if a1.button("Approve", type="primary", key=f"{key_prefix}_ok", use_container_width=True):
        try:
            dest = resolve_pending_draft(paths[sel], rq, "approved")
            base = f"Approved: `{dest.relative_to(rq)}`"
            ok, detail = True, ""
            if apply_booking_on_approve and pms is not None:
                ok, detail = apply_pending_guest_email_on_approval(dest, pms)
                if not ok:
                    st.warning(f"{base} — {detail}")
                elif detail == "No BOOKING_COMMIT in file":
                    st.success(base)
                else:
                    st.success(f"{base} — {detail}")
            else:
                st.success(base)
            if apply_booking_on_approve:
                desk_cat = normalize_category(dest.parent.name if dest.parent else "")
                if desk_cat == "cancellations" and pms is not None:
                    st.session_state.setdefault("messages", []).append(
                        {
                            "role": "assistant",
                            "content": _human_mode_cancellation_approved_chat_message(ok, detail),
                        }
                    )
                else:
                    guest_body = guest_correspondence_body_from_markdown_file(dest)
                    parts: list[str] = []
                    if guest_body:
                        parts.append(guest_body)
                    else:
                        parts.append(
                            f"Staff approved your pending message. It is filed under `{dest.relative_to(rq)}`."
                        )
                    if pms is not None:
                        if not ok:
                            parts.append(
                                f"\n\n*(PMS action was not applied: {detail})*"
                            )
                        elif detail and detail != "No BOOKING_COMMIT in file":
                            parts.append(f"\n\n*{detail}*")
                    st.session_state.setdefault("messages", []).append(
                        {"role": "assistant", "content": "\n".join(parts).strip()}
                    )
        except Exception as e:
            st.error(str(e))
        st.rerun()
    if a2.button("Reject", key=f"{key_prefix}_no", use_container_width=True):
        try:
            dest = resolve_pending_draft(paths[sel], rq, "rejected")
            st.info(f"Moved to rejected: {dest.relative_to(rq)}")
            if apply_booking_on_approve:
                cat = normalize_category(dest.parent.name if dest.parent else "")
                if cat == "booking_confirmations":
                    msg = (
                        "Your booking confirmation message was reviewed but not approved by staff yet. "
                        "No booking action was applied."
                    )
                elif cat == "cancellations":
                    msg = (
                        "Your cancellation message was reviewed but not approved by staff yet. "
                        "Your booking remains unchanged."
                    )
                else:
                    msg = (
                        "Your pending guest email was reviewed but not approved by staff yet. "
                        "No booking changes were applied."
                    )
                st.session_state.setdefault("messages", []).append(
                    {"role": "assistant", "content": msg}
                )
        except Exception as e:
            st.error(str(e))
        st.rerun()


def _render_escalations_panel(rq: Path, *, human_mode: bool = False) -> None:
    """``escalations/`` (pending) and ``approved/escalations/`` (after staff approve) with row **View**."""
    entries = collect_escalation_desk_entries(rq, max_items=120)
    st.markdown("**Escalations**")
    if human_mode:
        st.caption(
            "**Human mode:** Risk-flagged turns and specialist follow-ups should file **category `escalations`** "
            "via **pms_queue_correspondence_for_review** — they appear here for **Approve** / **Reject** (not under "
            "**Pending guest email**). **Pending** = `escalations/`; **Approved** = `approved/escalations/`."
        )
    else:
        st.caption(
            "**Pending** = in `escalations/` (including autonomous risk filings). **Approved** = moved to "
            "`approved/escalations/` after staff Approve. Use **View** on each row to preview. Escalations "
            "are not listed under Approved correspondence."
        )
    if not entries:
        st.caption("No escalation items.")
        return
    paths = [p for p, _ in entries]
    statuses = [s for _, s in entries]
    _render_desk_rows_with_view(
        paths, key_prefix="esc_desk", status_labels=statuses
    )
    st.divider()
    sel = st.selectbox(
        "Select file for Approve / Reject",
        range(len(entries)),
        format_func=lambda i: f"[{entries[i][1]}] {entries[i][0].name}",
        key="esc_desk_sel",
    )
    path, status = entries[sel]
    ev2, ev3 = st.columns(2)
    with ev2:
        if status == "pending" and st.button(
            "Approve", type="primary", key="esc_desk_ok", use_container_width=True
        ):
            try:
                dest = resolve_pending_draft(path, rq, "approved")
                st.success(f"Approved: `{dest.relative_to(rq)}`")
            except Exception as e:
                st.error(str(e))
            st.rerun()
    with ev3:
        if status == "pending" and st.button("Reject", key="esc_desk_no", use_container_width=True):
            try:
                dest = resolve_pending_draft(path, rq, "rejected")
                st.info(f"Moved to rejected: {dest.relative_to(rq)}")
            except Exception as e:
                st.error(str(e))
            st.rerun()
    if status != "pending":
        st.caption(
            "This selection is already under **approved/escalations/** — Approve/Reject apply only to **pending** rows."
        )


def _pms_snapshot_guests(pms: MockHotelPMS) -> list[dict[str, Any]]:
    """Prefer MockHotelPMS.list_guests(); fall back for older modules / instances."""
    fn = getattr(pms, "list_guests", None)
    if callable(fn):
        return fn()
    data = getattr(pms, "_data", None)
    if isinstance(data, dict):
        return [dict(g) for g in data.get("guests", [])]
    return []


def _pms_snapshot_reservations(pms: MockHotelPMS) -> list[dict[str, Any]]:
    """Prefer MockHotelPMS.list_reservations(); fall back for older modules / instances."""
    fn = getattr(pms, "list_reservations", None)
    if callable(fn):
        return fn()
    data = getattr(pms, "_data", None)
    if isinstance(data, dict):
        return [dict(r) for r in data.get("reservations", [])]
    return []


def _guests_dataframe(pms: MockHotelPMS) -> pd.DataFrame:
    guests = _pms_snapshot_guests(pms)
    if not guests:
        return pd.DataFrame(columns=["id", "name", "email", "phone", "nationality", "created_at"])
    rows = []
    for g in guests:
        rows.append(
            {
                "id": g.get("id", ""),
                "name": f"{g.get('first_name', '')} {g.get('last_name', '')}".strip(),
                "email": g.get("email", ""),
                "phone": g.get("phone", ""),
                "nationality": g.get("nationality", ""),
                "created_at": g.get("created_at", ""),
            }
        )
    return pd.DataFrame(rows)


def _bookings_dataframe(pms: MockHotelPMS) -> pd.DataFrame:
    by_id = {g["id"]: g for g in _pms_snapshot_guests(pms)}
    res = _pms_snapshot_reservations(pms)
    if not res:
        return pd.DataFrame(
            columns=[
                "id",
                "guest",
                "room_type_id",
                "rate_plan_id",
                "check_in",
                "check_out",
                "status",
                "total",
            ]
        )
    rows = []
    for r in res:
        gid = r.get("guest_id", "")
        g = by_id.get(gid, {})
        guest_label = (
            f"{g.get('first_name', '')} {g.get('last_name', '')} ({gid})".strip() or gid
        )
        rows.append(
            {
                "id": r.get("id", ""),
                "guest": guest_label,
                "room_type_id": r.get("room_type_id", ""),
                "rate_plan_id": r.get("rate_plan_id", ""),
                "check_in": r.get("check_in", ""),
                "check_out": r.get("check_out", ""),
                "status": r.get("status", ""),
                "total": r.get("total_amount", ""),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    page_icon: str = "🏨"
    if _LOGO_PATH.is_file():
        page_icon = str(_LOGO_PATH)
    st.set_page_config(
        page_title="Hotel guest agent",
        page_icon=page_icon,
        layout="wide",
    )

    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Set `OPENAI_API_KEY` (e.g. in `.env`) to run the LLM.")
        st.stop()

    mock_data_path = _env_path("HOTEL_MOCK_DATA") or ""
    review_queue_dir = _env_path("HOTEL_REVIEW_QUEUE_DIR") or ""

    if _LOGO_PATH.is_file():
        head_logo, head_text = st.columns([0.14, 0.86], gap="small")
        with head_logo:
            st.image(str(_LOGO_PATH), width=88)
        with head_text:
            st.title("Hotel guest agent")
            st.caption("Mock PMS · LangGraph · correspondence files are not emailed")
    else:
        st.title("Hotel guest agent")
        st.caption("Mock PMS · LangGraph · correspondence files are not emailed")

    _init_mode_toggles()

    top_a, top_b, top_c, top_d = st.columns([1.1, 1.1, 1.2, 1.4])
    with top_a:
        st.toggle(
            "Human mode",
            key="human_mode_toggle",
            on_change=_on_human_mode_change,
            help="Planner while the guest decides; after they confirm a booking, the executor runs once and files a pending email on the desk. **Approve** on the right applies the booking and moves the message to approved — no chat buttons.",
        )
    with top_b:
        st.toggle(
            "Autonomous mode",
            key="autonomous_mode_toggle",
            on_change=_on_autonomous_mode_change,
            help="Single agent runs read/write tools with booking commits on. Guest **yes / proceed** in chat is enough to save the reservation and file the outbound email under Approved—no Approve PMS writes step. Turning this **on** switches Human **off**.",
        )
    with top_c:
        if st.button("Reset conversation", type="secondary", use_container_width=True):
            _reset_chat_session()
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.accumulated_user_lines = []
            st.session_state.messages = []
            st.session_state.approval_seq = 0
            st.rerun()
    with top_d:
        if st.session_state.get("guest_email"):
            st.caption(f"Chat guest: {st.session_state.guest_email}")

    chat_mode = _sync_mode_toggles()
    pms = sync_session_pms(mock_data_path)
    rq = review_queue_root(review_queue_dir.strip() if review_queue_dir.strip() else None)

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.subheader("Chat")
        st.session_state.setdefault("thread_id", str(uuid.uuid4()))
        st.session_state.setdefault("accumulated_user_lines", [])
        st.session_state.setdefault("messages", [])
        st.session_state.setdefault("approval_seq", 0)

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        prompt = st.chat_input("Message… (type **reset** to clear chat & guest session state)")
        if prompt:
            raw = prompt.strip()
            if not raw:
                st.stop()
            low = raw.lower()
            if low == "reset":
                _reset_chat_session()
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.accumulated_user_lines = []
                st.session_state.messages = []
                st.rerun()

            gf = st.session_state.get(_GUEST_FLOW_KEY)

            if gf and gf.get("stage") == "need_email":
                st.session_state.messages.append({"role": "user", "content": raw})
                if not looks_like_email(raw):
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "That does not look like an email. Please send your address as `name@example.com`.",
                        }
                    )
                    st.rerun()
                em = raw.strip()
                existing = pms.find_guest_by_email(em)
                pending = gf["pending"]
                if existing:
                    st.session_state.guest_email = em
                    st.session_state.guest_id = str(existing["id"])
                    _clear_guest_flow()
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                f"Thanks — linked to **{existing.get('first_name', '')} "
                                f"{existing.get('last_name', '')}** (`{existing['id']}`)."
                            ),
                        }
                    )
                    _append_and_run_booking_turn(
                        pending_booking_message=pending,
                        pms=pms,
                        review_queue_dir=review_queue_dir or None,
                    )
                    st.rerun()
                st.session_state[_GUEST_FLOW_KEY] = {
                    "stage": "need_names",
                    "pending": pending,
                    "email": em,
                }
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "I do not have that email on file yet. Send your **first and last name** "
                            "in one line (e.g. `Alex Nordmann`)."
                        ),
                    }
                )
                st.rerun()

            if gf and gf.get("stage") == "need_names":
                parts = raw.split(None, 1)
                st.session_state.messages.append({"role": "user", "content": raw})
                if len(parts) < 2:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "Please include both first and last name, e.g. `Jane Doe`.",
                        }
                    )
                    st.rerun()
                first, last = parts[0], parts[1]
                em = gf["email"]
                pending = gf["pending"]
                st.session_state[_GUEST_FLOW_KEY] = {
                    "stage": "need_phone",
                    "pending": pending,
                    "email": em,
                    "first_name": first,
                    "last_name": last,
                }
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "What is your **phone number** (with country code if you like)? "
                            "Type **skip** to leave it blank for the mock PMS."
                        ),
                    }
                )
                st.rerun()

            if gf and gf.get("stage") == "need_phone":
                st.session_state.messages.append({"role": "user", "content": raw})
                lowp = raw.strip().lower()
                phone = "" if lowp in ("skip", "none", "n/a", "-") else raw.strip()
                st.session_state[_GUEST_FLOW_KEY] = {
                    "stage": "need_nationality",
                    "pending": gf["pending"],
                    "email": gf["email"],
                    "first_name": gf["first_name"],
                    "last_name": gf["last_name"],
                    "phone": phone,
                }
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "Nationality as a **2-letter country code** (e.g. NO, US, GB). "
                            "Type **skip** if unknown (stored as XX)."
                        ),
                    }
                )
                st.rerun()

            if gf and gf.get("stage") == "need_nationality":
                st.session_state.messages.append({"role": "user", "content": raw})
                lown = raw.strip().lower()
                nationality = "XX"
                if lown not in ("skip", "none", "n/a", "-", "no"):
                    code = raw.strip().upper().replace(" ", "")
                    if len(code) < 2:
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": "Please send two letters (e.g. `NO`) or **skip**.",
                            }
                        )
                        st.rerun()
                    nationality = code[:2]
                em = gf["email"]
                pending = gf["pending"]
                try:
                    _e, gid, row = resolve_or_create_guest(
                        pms,
                        em,
                        first_name=gf["first_name"],
                        last_name=gf["last_name"],
                        phone=gf.get("phone", ""),
                        nationality=nationality,
                    )
                except ValueError as e:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Could not create the profile: {e}"}
                    )
                    st.rerun()
                st.session_state.guest_email = em
                st.session_state.guest_id = gid
                _clear_guest_flow()
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"Profile created for **{row.get('first_name', '')} {row.get('last_name', '')}**."
                        ),
                    }
                )
                _append_and_run_booking_turn(
                    pending_booking_message=pending,
                    pms=pms,
                    review_queue_dir=review_queue_dir or None,
                )
                st.rerun()

            if message_needs_guest_for_booking(raw) and not st.session_state.get("guest_email"):
                st.session_state[_GUEST_FLOW_KEY] = {
                    "stage": "need_email",
                    "pending": raw,
                }
                st.session_state.messages.append({"role": "user", "content": raw})
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "To use booking and reservation tools I need your **guest email** "
                            "for the mock PMS. Please reply with your email address."
                        ),
                    }
                )
                st.rerun()

            st.session_state.accumulated_user_lines.append(raw)

            # Always inject session guest when we have an email so list/cancel/“active bookings?” turns
            # still see guest_id (regex “booking” does not match “bookings”, etc.).
            pending_guest_hint = ""
            if st.session_state.get("guest_email"):
                pending_guest_hint = session_guest_hint(
                    st.session_state.guest_email,
                    st.session_state.get("guest_id") or "",
                )

            model_input = pending_guest_hint + raw if pending_guest_hint else raw
            turn_cfg, risk = _build_turn_cfg(
                st.session_state.accumulated_user_lines,
                model_input,
                review_queue_dir or None,
                chat_mode,
                pms,
                booking_pms_commit_allowed=_streamlit_booking_pms_commit_allowed(chat_mode),
            )
            _ensure_streamlit_agents(chat_mode, pms)
            tid = st.session_state.thread_id

            with st.chat_message("user"):
                st.markdown(raw)

            if risk["manual_review_required"]:
                if chat_mode == "autonomous":
                    esc_cfg = {**turn_cfg, "booking_pms_commit_allowed": False}
                    reply, _msgs = _run_chat_turn(
                        model_input_text=_risk_flags_hint(risk) + model_input,
                        pms=pms,
                        thread_id=f"{tid}::escalation",
                        turn_cfg=esc_cfg,
                        agent=st.session_state.autonomous_escalation_agent,
                    )
                else:
                    st.session_state.pop(_HUMAN_MODE_LAST_PLAN_KEY, None)
                    reply, _msgs = _run_chat_turn(
                        model_input_text=model_input,
                        pms=pms,
                        thread_id=f"{tid}::review::{st.session_state.get('approval_seq', 0)}",
                        turn_cfg=turn_cfg,
                        agent=st.session_state.review_agent,
                    )
                st.session_state.messages.extend(
                    [
                        {"role": "user", "content": raw},
                        {"role": "assistant", "content": reply},
                    ]
                )
                st.rerun()

            if chat_mode == "autonomous":
                reply, _msgs = _run_chat_turn(
                    model_input_text=model_input,
                    pms=pms,
                    thread_id=tid,
                    turn_cfg=turn_cfg,
                    agent=st.session_state.chat_agent,
                )
                st.session_state.messages.extend(
                    [
                        {"role": "user", "content": raw},
                        {"role": "assistant", "content": reply},
                    ]
                )
                st.rerun()

            reply, _msgs = _run_human_mode_turn(
                raw_user=raw,
                model_input=model_input,
                turn_cfg=turn_cfg,
                pms=pms,
                tid=tid,
            )
            st.session_state.messages.extend(
                [
                    {"role": "user", "content": raw},
                    {"role": "assistant", "content": reply},
                ]
            )
            st.rerun()

    with right:
        st.subheader("Operations desk")

        st.markdown("**Guests**")
        gdf = _guests_dataframe(pms)
        st.dataframe(gdf, use_container_width=True, hide_index=True, height=min(280, 56 + 36 * max(1, len(gdf))))

        _render_escalations_panel(rq, human_mode=(chat_mode == "approval"))

        # Human mode: pending guest-facing correspondence (drafts, confirmations, cancellations, modifications).
        if chat_mode == "approval":
            pending_items: list[tuple[Path, str]] = []
            for p in collect_pending_paths(rq, category="drafts")[:100]:
                pending_items.append((p, "Email draft"))
            for p in collect_pending_paths(rq, category="booking_confirmations")[:100]:
                pending_items.append((p, "Booking confirmation"))
            for p in collect_pending_paths(rq, category="cancellations")[:100]:
                pending_items.append((p, "Cancellation email"))
            for p in collect_pending_paths(rq, category="modifications")[:100]:
                pending_items.append((p, "Modification email"))
            pending_items.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
            seen: set[Path] = set()
            merged_paths: list[Path] = []
            merged_labels: list[str] = []
            for p, lab in pending_items:
                if p in seen:
                    continue
                seen.add(p)
                merged_paths.append(p)
                merged_labels.append(lab)
            _render_draft_panel(
                "Pending guest email",
                merged_paths,
                rq,
                "pending_guest_email",
                pms=pms,
                apply_booking_on_approve=True,
                category_labels=merged_labels,
            )
            st.caption(
                "Includes **drafts**, **booking confirmations**, **cancellations**, and **modifications**. "
                "**Escalations** use the **Escalations** section above."
            )

        st.markdown("**Approved correspondence**")
        st.caption(
            "Markdown under `review_queue/approved/` (drafts, booking confirmations, etc.). "
            "**Escalations** are only on the Escalations desk. Nothing is emailed automatically."
        )
        appr_items = collect_approved_correspondence_paths(rq, max_files=120)
        if not appr_items:
            st.caption("No approved files yet.")
        else:
            appr_paths = [p for p, _ in appr_items]
            appr_cats = [c for _, c in appr_items]
            _render_desk_rows_with_view(
                appr_paths,
                key_prefix="appr_desk",
                category_labels=appr_cats,
            )

        st.markdown("**Bookings**")
        bdf = _bookings_dataframe(pms)
        st.dataframe(bdf, use_container_width=True, hide_index=True, height=min(320, 56 + 36 * max(1, len(bdf))))


if __name__ == "__main__":
    main()
