"""Inbound guest email: risk gate, planning vs execution (approval / autonomous)."""

from __future__ import annotations

import uuid
from typing import Any

from agent_tools import (
    HOTEL_READ_AND_ESCALATION_QUEUE_TOOLS,
    HOTEL_READ_TOOLS,
    HOTEL_WRITE_TOOLS,
)
from execution_flow import (
    ExecutionMode,
    executor_followup_user_message,
    extract_draft_guest_reply,
    planner_requires_pms_writes,
    prepare_turn_configurable,
    save_human_mode_planner_draft,
)
from graph import (
    agent_reply,
    autonomous_risk_escalation_prompt,
    build_email_agent,
    executor_system_prompt,
    last_ai_text,
    planner_system_prompt,
    review_only_system_prompt,
    system_prompt_for_full_agent,
)
from hotel_pms import MockHotelPMS


def run_inbound_email(
    email_body: str,
    *,
    guest_from_email: str | None,
    pms: MockHotelPMS,
    execution_mode: ExecutionMode = "autonomous",
    approve_callback: Any | None = None,
    extra_configurable: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Process one inbound email body.

    **Autonomous**
        Low risk: full read/write agent, end-to-end. **PMS booking mutations** (create/cancel/modify
        reservation) are blocked unless ``extra_configurable`` sets ``booking_pms_commit_allowed: true``
        (or equivalent from the host app).

    **Human approval**
        Low risk: read-only planner → if **Requires PMS writes: yes**, operator approval → executor;
        if **no**, **DRAFT GUEST REPLY** is saved under pending ``drafts/`` when a review queue path
        is configured. High risk: same as autonomous high risk (no execute phase).

    ``approve_callback``: called as ``approve_callback(plan_text, "")``; return True to run writes.
    """
    text = (
        (f"(Guest email address on file for lookup: {guest_from_email})\n\n" if guest_from_email else "")
        + email_body.strip()
    )
    thread_plan = str(uuid.uuid4())
    thread_exec = str(uuid.uuid4())
    rq = (extra_configurable or {}).get("review_queue_dir") if extra_configurable else None
    rq = str(rq).strip() if rq else None

    sess_id = ""
    if guest_from_email:
        gf = pms.find_guest_by_email(guest_from_email.strip())
        if gf:
            sess_id = str(gf.get("id") or "")
    base_cfg, risk = prepare_turn_configurable(
        accumulated_user_lines=[email_body.strip()],
        text_for_risk=email_body.strip(),
        execution_mode=execution_mode,
        review_queue_dir=rq,
        guest_default_year=pms.default_guest_stay_year(),
        session_guest_email=(guest_from_email or "").strip() or None,
        session_guest_id=sess_id or None,
    )

    out: dict[str, Any] = {
        "risk": risk,
        "execution_mode": execution_mode,
        "phases": [],
    }

    def merged_cfg(local: dict[str, Any]) -> dict[str, Any]:
        merged = {**base_cfg, **local}
        if extra_configurable:
            merged = {**merged, **extra_configurable}
        return merged

    if risk["manual_review_required"]:
        if execution_mode == "autonomous":
            agent = build_email_agent(
                tools=list(HOTEL_READ_AND_ESCALATION_QUEUE_TOOLS),
                system_prompt=autonomous_risk_escalation_prompt(
                    guest_default_year=pms.default_guest_stay_year()
                ),
                name="hotel_autonomous_escalation",
            )
            rs = risk.get("reasons") or []
            risk_prefix = (
                "[Internal flags (do not quote verbatim): " + " | ".join(rs) + "]\n\n"
                if rs
                else ""
            )
            cfg = merged_cfg({})
            cfg["booking_pms_commit_allowed"] = False
            msgs = agent_reply(
                agent,
                risk_prefix + text,
                pms=pms,
                thread_id=thread_plan,
                configurable=cfg,
            )
            out["phases"].append({"name": "autonomous_escalation", "messages": msgs})
        else:
            agent = build_email_agent(
                tools=list(HOTEL_READ_TOOLS),
                system_prompt=review_only_system_prompt(guest_default_year=pms.default_guest_stay_year()),
                name="hotel_review_only",
            )
            cfg = merged_cfg({})
            msgs = agent_reply(agent, text, pms=pms, thread_id=thread_plan, configurable=cfg)
            out["phases"].append({"name": "manual_review", "messages": msgs})
        out["final_reply_draft"] = last_ai_text(msgs)
        out["structured_plan_text"] = last_ai_text(msgs)
        out["executed"] = execution_mode == "autonomous"
        return out

    if execution_mode == "autonomous":
        agent = build_email_agent(
            name="hotel_autonomous",
            system_prompt=system_prompt_for_full_agent(guest_default_year=pms.default_guest_stay_year()),
        )
        cfg_auto = merged_cfg({})
        msgs = agent_reply(agent, text, pms=pms, thread_id=thread_plan, configurable=cfg_auto)
        out["phases"].append({"name": "autonomous", "messages": msgs})
        out["final_reply_draft"] = last_ai_text(msgs)
        out["structured_plan_text"] = last_ai_text(msgs)
        out["executed"] = True
        return out

    planner = build_email_agent(
        tools=list(HOTEL_READ_TOOLS),
        system_prompt=planner_system_prompt(guest_default_year=pms.default_guest_stay_year()),
        name="hotel_planner",
    )
    plan_cfg = merged_cfg({"execution_mode": "approval", "writes_approved": False})
    plan_msgs = agent_reply(planner, text, pms=pms, thread_id=thread_plan, configurable=plan_cfg)
    plan_blob = last_ai_text(plan_msgs)
    out["phases"].append({"name": "plan", "messages": plan_msgs})
    out["structured_plan_text"] = plan_blob

    if not planner_requires_pms_writes(plan_blob):
        out["executed"] = False
        out["final_reply_draft"] = plan_blob
        out["writes_required"] = False
        out["planner_draft_queued_path"] = None
        body = extract_draft_guest_reply(plan_blob)
        if body and rq:
            from review_queue import review_queue_root

            root = review_queue_root(rq)
            em = (guest_from_email or "").strip() or "guest@unknown.local"
            path = save_human_mode_planner_draft(
                draft_body=body,
                to_email=em,
                queue_root=root,
                notes_for_staff="Approval pipeline: planner marked Requires PMS writes: no.",
            )
            out["planner_draft_queued_path"] = str(path)
        return out
    out["writes_required"] = True

    approved = True
    if approve_callback is not None:
        approved = bool(approve_callback(plan_blob, ""))

    if not approved:
        out["executed"] = False
        out["final_reply_draft"] = plan_blob
        return out

    executor = build_email_agent(
        tools=list(HOTEL_READ_TOOLS) + list(HOTEL_WRITE_TOOLS),
        system_prompt=executor_system_prompt(guest_default_year=pms.default_guest_stay_year()),
        name="hotel_executor",
    )
    exec_cfg = merged_cfg(
        {
            "execution_mode": "approval",
            "writes_approved": True,
            "booking_pms_commit_allowed": True,
        }
    )
    follow = executor_followup_user_message(
        approved_plan_text=plan_blob or "",
        original_user_message=email_body.strip(),
        session_guest_email=(guest_from_email or "").strip(),
        session_guest_id=sess_id,
    )
    exec_msgs = agent_reply(executor, follow, pms=pms, thread_id=thread_exec, configurable=exec_cfg)
    out["phases"].append({"name": "execute", "messages": exec_msgs})
    out["executed"] = True
    out["final_reply_draft"] = last_ai_text(exec_msgs)
    return out
