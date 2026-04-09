"""CLI: interactive terminal chat (default on TTY) or one-shot email / stdin."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Load before other local imports so submodule import order cannot skip the key.
load_dotenv(Path(__file__).resolve().parent / ".env")

import argparse
import os
import re
import sys
import threading
import uuid

from langgraph.checkpoint.memory import MemorySaver

from email_pipeline import run_inbound_email
from graph import build_email_agent, last_ai_text, stream_agent_turn
from guest_dates import extract_guest_stay_date_literals
from hotel_pms import MockHotelPMS
from review_queue import (
    REVIEW_CATEGORIES,
    print_review_queue_listing,
    run_interactive_review_session,
)
from risk import analyze_email_risk


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hotel guest agent — dynamic terminal chat or one-shot email (LangGraph + mock PMS)"
    )
    parser.add_argument(
        "--email-file",
        type=str,
        default=None,
        help="Read one inbound email from this file, then exit (no chat)",
    )
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="Process stdin as a single message and exit (useful when piping)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the built-in sample booking email once and exit (TTY quick test)",
    )
    parser.add_argument(
        "--from-address",
        type=str,
        default=None,
        metavar="EMAIL",
        help="Non-interactive: From address hint. Interactive: used as guest email when a booking-related turn needs PMS identity (skips typing that prompt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to mock_hotel_data.json (default: beside hotel_pms.py)",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["autonomous", "approval"],
        default="autonomous",
        help="Batch/email runs: autonomous (default) runs writes when safe; approval plans read-only then prompts before writes. Interactive chat uses autonomous execution with per-message risk blocking.",
    )
    parser.add_argument(
        "--show-review-queue",
        action="store_true",
        help="Open interactive review UI for pending drafts (TTY), or print a list if stdin/stdout is not a TTY. No LLM; no API key.",
    )
    parser.add_argument(
        "--review-queue-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Override directory for queued drafts (default: hotel/review_queue next to this package).",
    )
    parser.add_argument(
        "--review-queue-category",
        choices=list(REVIEW_CATEGORIES),
        default=None,
        metavar="CATEGORY",
        help="With --show-review-queue, only list this subfolder (drafts, cancellations, …).",
    )
    args = parser.parse_args()

    if args.show_review_queue:
        if sys.stdin.isatty() and sys.stdout.isatty():
            run_interactive_review_session(
                args.review_queue_dir,
                category=args.review_queue_category,
            )
        else:
            print_review_queue_listing(
                args.review_queue_dir,
                category=args.review_queue_category,
            )
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY (e.g. in .env) to run the LLM.", file=sys.stderr)
        sys.exit(1)

    pms = MockHotelPMS(args.data) if args.data else MockHotelPMS()

    if args.email_file:
        with open(args.email_file, encoding="utf-8") as f:
            body = f.read()
        _run_batch_pipeline(
            body,
            guest_from_email=args.from_address,
            pms=pms,
            execution_mode=args.execution_mode,
            review_queue_dir=args.review_queue_dir,
        )
        return

    if args.demo:
        _run_batch_pipeline(
            _demo_email_body(),
            guest_from_email=args.from_address,
            pms=pms,
            execution_mode=args.execution_mode,
            review_queue_dir=args.review_queue_dir,
        )
        return

    if args.one_shot or not sys.stdin.isatty():
        body = sys.stdin.read()
        if not body.strip():
            print("No input.", file=sys.stderr)
            sys.exit(1)
        _run_batch_pipeline(
            body,
            guest_from_email=args.from_address,
            pms=pms,
            execution_mode=args.execution_mode,
            review_queue_dir=args.review_queue_dir,
        )
        return

    _run_interactive_chat(
        pms, prefill_email=args.from_address, review_queue_dir=args.review_queue_dir
    )


# Booking / PMS-action phrasing: when we should attach a guest identity before calling the agent.
_BOOKING_CONTEXT_RE = re.compile(
    r"\b("
    r"book|booking|booked|reserve|reservation|reserving|"
    r"stay(?:ing)?|check-?in|check-?out|"
    r"availability|available|vacanc|"
    r"room\s+(?:for|types?|only)|double\s+room|single\s+room|suite|"
    r"nightly|rate(?:s)?\s+for|quote|hold\s+(?:a\s+)?room|"
    r"cancel(?:lation|ing)?|modify(?:ing)?\s+(?:my\s+)?(?:booking|reservation|stay)"
    r")\b",
    re.IGNORECASE,
)


def _message_needs_guest_for_booking(text: str) -> bool:
    return bool(_BOOKING_CONTEXT_RE.search(text))


def _run_interactive_chat(
    pms: MockHotelPMS, *, prefill_email: str | None, review_queue_dir: str | None
) -> None:
    memory = MemorySaver()
    agent = build_email_agent(checkpointer=memory)
    thread_id = str(uuid.uuid4())
    guest_email: str | None = None
    guest_id: str | None = None
    booking_email_prefill = prefill_email
    accumulated_user_lines: list[str] = []

    print(
        "Hotel guest agent — ask about the hotel, policies, or anything else.\n"
        "If your message is about booking, availability, or reservations, we'll ask for your "
        "guest email then (to match the PMS).\n"
        "Commands:  quit | exit  — leave   |   reset — clear chat + guest context\n"
        "Guest replies are never emailed: use --show-review-queue to list drafts saved under review_queue/.\n"
    )

    while True:
        try:
            line = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        raw = line.strip()
        if not raw:
            continue
        low = raw.lower()
        if low in ("quit", "exit", "q"):
            break
        if low == "reset":
            thread_id = str(uuid.uuid4())
            guest_email, guest_id = None, None
            accumulated_user_lines.clear()
            print("(New conversation — chat and guest context cleared for the model.)\n")
            continue

        accumulated_user_lines.append(raw)

        pending_guest_hint = ""
        if _message_needs_guest_for_booking(raw) and guest_email is None:
            guest_email, guest_id = _prompt_and_ensure_guest(
                pms,
                prefill_email=booking_email_prefill,
                for_booking=True,
            )
            pending_guest_hint = _session_guest_hint(guest_email, guest_id)

        text = pending_guest_hint + raw if pending_guest_hint else raw

        risk = analyze_email_risk(text)
        lit = sorted(
            extract_guest_stay_date_literals("\n".join(accumulated_user_lines))
        )
        turn_cfg: dict = {"guest_date_literals": lit}
        if risk["manual_review_required"]:
            turn_cfg = {
                **turn_cfg,
                "manual_review_only": True,
                "manual_review_reason": "; ".join(risk["reasons"]),
            }
        if review_queue_dir:
            turn_cfg = {**turn_cfg, "review_queue_dir": review_queue_dir}

        try:
            _interactive_agent_turn(
                agent, text, pms=pms, thread_id=thread_id, configurable=turn_cfg
            )
        except Exception as e:
            print(f"Error: {e}\n", file=sys.stderr)
            continue


def _session_guest_hint(email: str, guest_id: str) -> str:
    return (
        f"(Session guest email: {email}; PMS guest_id: {guest_id}. "
        "Use tools with this guest when taking actions for this person.)\n\n"
    )


def _prompt_email(*, prefill: str | None) -> str:
    if prefill and prefill.strip():
        return prefill.strip()
    while True:
        try:
            raw = input("Guest email: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0) from None
        if _looks_like_email(raw):
            return raw
        print("Enter a valid email address (e.g. name@example.com).", file=sys.stderr)


def _looks_like_email(s: str) -> bool:
    if "@" not in s or s.startswith("@") or s.endswith("@"):
        return False
    local, _, domain = s.partition("@")
    return bool(local) and "." in domain


def _prompt_and_ensure_guest(
    pms: MockHotelPMS,
    *,
    prefill_email: str | None,
    for_booking: bool = False,
) -> tuple[str, str]:
    if for_booking and not (prefill_email and prefill_email.strip()):
        print(
            "\nThat sounds like a booking or reservation topic — "
            "we need your guest email for the mock PMS.\n"
        )
    email = _prompt_email(prefill=prefill_email)
    existing = pms.find_guest_by_email(email)
    if existing:
        gid = str(existing["id"])
        print(
            f"Found existing profile: {existing['first_name']} {existing['last_name']} ({gid}).\n"
        )
        return email, gid

    print("No guest profile for this email — add one to the mock PMS.\n")
    while True:
        try:
            first = input("First name: ").strip()
            last = input("Last name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0) from None
        if first and last:
            break
        print("First and last name are required.", file=sys.stderr)
    try:
        phone = input("Phone (optional): ").strip()
        nationality = input("Nationality code, e.g. NO (optional): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0) from None

    guest = pms.create_guest(email, first, last, phone, nationality)
    gid = str(guest["id"])
    print(f"Created guest {gid} for {email}.\n")
    return email, gid


def _stderr_color_ok() -> bool:
    return sys.stderr.isatty() and os.environ.get("NO_COLOR") is None


class TerminalSpinner:
    """Polished stderr loader: moon glyph, soft palette, scrolling shimmer bar."""

    _moons = ("◐", "◓", "◑", "◒")
    _ramp = "░▒▓██▓▒░"
    _accent_orbit = 152
    _accent_label = 246
    _bar_lo = 60
    _bar_hi = 73

    def __init__(self, label: str = "Agent running") -> None:
        self.label = label
        self._stop = threading.Event()
        self._stopped = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stopped = False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=4)
            self._thread = None
        if sys.stderr.isatty():
            sys.stderr.write("\r\x1b[K\033[0m")
            sys.stderr.flush()

    def _shimmer_bar(self, frame: int, width: int = 16) -> str:
        parts: list[str] = []
        for col in range(width):
            ch = self._ramp[(frame + col) % len(self._ramp)]
            if _stderr_color_ok():
                span = self._bar_hi - self._bar_lo + 1
                wave = ((frame * 2 + col * 3) % span) + self._bar_lo
                parts.append(f"\033[38;5;{wave}m{ch}\033[0m")
            else:
                parts.append(ch)
        return "".join(parts)

    def _render(self, frame: int) -> str:
        moon = self._moons[frame % len(self._moons)]
        dots = "." * (1 + (frame // 2) % 3)
        tail = f"{self.label}{dots}"
        bar = self._shimmer_bar(frame)

        if _stderr_color_ok():
            return (
                f"\r\x1b[K "
                f"\033[38;5;{self._accent_orbit}m{moon}\033[0m  "
                f"\033[38;5;{self._accent_label}m{tail}\033[0m"
                f"  {bar} "
            )
        return f"\r\x1b[K {moon}  {tail}  {bar} "

    def _run(self) -> None:
        i = 0
        tty = sys.stderr.isatty()
        ansi_re = re.compile(r"\x1b\[[0-9;]*m")
        while not self._stop.is_set():
            if not tty and i % 3 != 0:
                if self._stop.wait(0.075):
                    break
                i += 1
                continue
            line = self._render(i)
            if tty:
                sys.stderr.write(line)
            else:
                plain = ansi_re.sub("", line.replace("\r\x1b[K", "", 1)).strip()
                sys.stderr.write(plain + "\n")
            sys.stderr.flush()
            if self._stop.wait(0.075):
                break
            i += 1


def _stderr_activity(line: str) -> None:
    if _stderr_color_ok():
        line = f"\033[38;5;245m{line}\033[0m"
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _interactive_agent_turn(
    agent: object,
    text: str,
    *,
    pms: MockHotelPMS,
    thread_id: str,
    configurable: dict | None = None,
) -> None:
    sys.stdout.write("\nAgent: ")
    sys.stdout.flush()
    spinner = TerminalSpinner("Composing your reply")
    spinner.start()
    got_token = False

    def on_token(t: str) -> None:
        nonlocal got_token
        if not got_token:
            got_token = True
            spinner.stop()
        sys.stdout.write(t)
        sys.stdout.flush()

    try:
        msgs = stream_agent_turn(
            agent,
            text,
            pms=pms,
            thread_id=thread_id,
            on_token=on_token,
            on_tool_start=lambda n: _stderr_activity(f"  → {n} …"),
            on_tool_done=lambda n: _stderr_activity(f"  ✓ {n}"),
            configurable=configurable,
        )
    finally:
        spinner.stop()
    print()
    if not got_token:
        print(last_ai_text(msgs) or "(no text)")
    print()


def _run_batch_pipeline(
    body: str,
    *,
    guest_from_email: str | None,
    pms: MockHotelPMS,
    execution_mode: str,
    review_queue_dir: str | None,
) -> None:
    def _approve(plan_blob: str, _: str) -> bool:
        print("\n--- Planner output (structured plan + draft) ---\n")
        print(plan_blob)
        if not sys.stdin.isatty():
            print(
                "\n(Non-interactive stdin: declining writes. Use a TTY or autonomous mode.)",
                file=sys.stderr,
            )
            return False
        ans = input(
            "\nApprove PMS writes and saving queued guest correspondence (no email sent)? [y/N]: "
        ).strip().lower()
        return ans in ("y", "yes")

    extra: dict = {}
    if review_queue_dir:
        extra["review_queue_dir"] = review_queue_dir
    if allow_booking_commit or os.environ.get("HOTEL_BOOKING_COMMIT", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        extra["booking_pms_commit_allowed"] = True
    extra_kw = extra if extra else None
    out = run_inbound_email(
        body,
        guest_from_email=guest_from_email,
        pms=pms,
        execution_mode=execution_mode,  # type: ignore[arg-type]
        approve_callback=_approve if execution_mode == "approval" else None,
        extra_configurable=extra_kw,
    )
    print("\n=== Pipeline summary ===")
    r = out["risk"]
    print(f"manual_review_required: {r['manual_review_required']}")
    for reason in r.get("reasons") or []:
        print(f"  · {reason}")
    for ph in out["phases"]:
        print(f"Phase: {ph['name']}")
    if "executed" in out:
        print(f"Writes executed: {out['executed']}")
    print("\n--- Final output (last phase) ---\n")
    print(out.get("final_reply_draft") or "(empty)")


def _demo_email_body() -> str:
    return """Subject: Booking request

Hi,

I'd like to book a double room from 2025-04-24 to 2025-04-26 for 2 adults.
Please use the standard flexible options if available.

Thanks,
Alex Nordmann
"""


if __name__ == "__main__":
    main()