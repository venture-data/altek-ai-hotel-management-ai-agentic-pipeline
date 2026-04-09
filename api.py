"""Minimal HTTP API for inbound guest email (mock PMS). Run: uvicorn api:app --reload

Interactive UI: streamlit run streamlit_app.py  (or `python main.py` on a TTY)
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

from email_pipeline import run_inbound_email
from hotel_pms import MockHotelPMS

app = FastAPI(title="Hotel guest email agent (mock PMS)", version="0.1.0")
_pms_singleton: MockHotelPMS | None = None


def _pms() -> MockHotelPMS:
    global _pms_singleton
    if _pms_singleton is None:
        _pms_singleton = MockHotelPMS()
    return _pms_singleton


class InboundEmailRequest(BaseModel):
    body: str = Field(..., description="Raw guest email text")
    guest_from_email: str | None = Field(None, description="Envelope / From for PMS lookup")
    execution_mode: Literal["autonomous", "approval"] = "autonomous"
    approve_writes: bool = Field(
        False,
        description="When execution_mode=approval, set true to execute writes without a second HTTP round-trip",
    )
    review_queue_dir: str | None = Field(
        None, description="Optional directory for queued correspondence markdown (default: app review_queue/)"
    )
    booking_pms_commit_allowed: bool = Field(
        False,
        description="When true, allow pms_create_reservation / cancel / modify in autonomous mode; default off until staff-style release",
    )


@app.post("/v1/inbound-email")
def inbound_email(req: InboundEmailRequest) -> dict:
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")

    def approve(_plan: str, __: str) -> bool:
        return req.approve_writes

    extra: dict = {}
    if req.review_queue_dir:
        extra["review_queue_dir"] = req.review_queue_dir
    if req.booking_pms_commit_allowed:
        extra["booking_pms_commit_allowed"] = True
    extra_kw = extra if extra else None
    out = run_inbound_email(
        req.body,
        guest_from_email=req.guest_from_email,
        pms=_pms(),
        execution_mode=req.execution_mode,
        approve_callback=approve if req.execution_mode == "approval" else None,
        extra_configurable=extra_kw,
    )
    return {
        "risk": out["risk"],
        "execution_mode": out["execution_mode"],
        "phases": [p["name"] for p in out["phases"]],
        "executed": out.get("executed"),
        "final_reply_draft": out.get("final_reply_draft"),
    }
