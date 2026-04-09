"""On-disk folders for guest correspondence drafts — nothing is emailed."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

REVIEW_CATEGORIES = (
    "drafts",
    "cancellations",
    "modifications",
    "booking_confirmations",
    "escalations",
    "general",
)

# Categories that may be written straight under approved/ in autonomous mode (see agent_tools).
# Escalations always stay under pending ``escalations/`` so the desk lists them in one place.
_AUTONOMOUS_DIRECT_APPROVED_CATEGORIES = frozenset({"drafts", "booking_confirmations"})

_CATEGORY_ALIASES: dict[str, str] = {
    "draft": "drafts",
    "draft_reply": "drafts",
    "cancellation": "cancellations",
    "cancel": "cancellations",
    "modification": "modifications",
    "modify": "modifications",
    "booking": "booking_confirmations",
    "booking_confirmation": "booking_confirmations",
    "confirm": "booking_confirmations",
    "escalation": "escalations",
    "general": "general",
}


def review_queue_root(override: str | Path | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent / "review_queue"


def guest_correspondence_body_from_markdown_file(path: Path) -> str:
    """
    Return the guest-facing markdown body from a queued/approved correspondence file
    (content after the first ``---`` separator; staff header block above).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    for sep in ("\n---\n\n", "\n---\n", "\r\n---\r\n\r\n", "\r\n---\r\n"):
        if sep in text:
            body = text.split(sep, 1)[1].strip()
            break
    else:
        body = text.strip()
    low = body.lower()
    cut = low.find("\n### staff only")
    if cut != -1:
        body = body[:cut].rstrip()
    return body


def parse_booking_modify_from_text(text: str) -> dict | None:
    """Parse `- **BOOKING_MODIFY:** {...}` header line."""
    marker = "- **BOOKING_MODIFY:**"
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(marker):
            raw = s[len(marker) :].strip()
            if not raw:
                continue
            try:
                out = json.loads(raw)
                return out if isinstance(out, dict) else None
            except json.JSONDecodeError:
                pass
    return None


def parse_booking_commit_from_text(text: str) -> dict | None:
    """Parse `- **BOOKING_COMMIT:** {...}` header line, or legacy `<!-- BOOKING_COMMIT:{...} -->`."""
    marker = "- **BOOKING_COMMIT:**"
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(marker):
            raw = s[len(marker) :].strip()
            if not raw:
                continue
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
    m = re.search(
        r"<!--\s*BOOKING_COMMIT:({.+?})\s*-->",
        text,
        flags=re.DOTALL,
    )
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _related_reservation_id_from_header(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("- **Related reservation:**"):
            if "`" in line:
                parts = line.split("`")
                if len(parts) >= 2:
                    return parts[1].strip()
            return line.split(":", 1)[-1].strip().strip("`")
    return ""


def try_commit_booking_from_correspondence_file(path: Path, pms: object) -> tuple[bool, str]:
    """
    If the markdown file embeds BOOKING_COMMIT JSON, create a PMS reservation (idempotent when
    Related reservation already exists and is found in the PMS).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return False, str(e)
    payload = parse_booking_commit_from_text(text)
    if not payload:
        # Must not short-circuit on **Related reservation** here: cancellation/modify drafts cite
        # an existing reservation in the header but have no BOOKING_COMMIT; they need desk follow-up.
        return True, "No BOOKING_COMMIT in file"
    rel_id = _related_reservation_id_from_header(text)
    get_res = getattr(pms, "get_reservation", None)
    if rel_id and callable(get_res) and get_res(rel_id):
        return True, f"Already have reservation {rel_id}"
    req_k = ("guest_id", "room_type_id", "rate_plan_id", "check_in", "check_out")
    for k in req_k:
        if k not in payload or not str(payload[k]).strip():
            return False, f"BOOKING_COMMIT missing or empty: {k}"
    create = getattr(pms, "create_reservation", None)
    if not callable(create):
        return False, "PMS has no create_reservation"
    res = create(
        guest_id=str(payload["guest_id"]).strip(),
        room_type_id=str(payload["room_type_id"]).strip(),
        rate_plan_id=str(payload["rate_plan_id"]).strip(),
        check_in=str(payload["check_in"]).strip(),
        check_out=str(payload["check_out"]).strip(),
        adults=int(payload.get("adults", 1)),
        children=int(payload.get("children", 0)),
        notes=str(payload.get("notes", "")),
    )
    if res.get("ok"):
        rid = (res.get("reservation") or {}).get("id", "")
        return True, f"Created {rid}" if rid else "Created reservation"
    return False, str(res.get("error", res))


def try_apply_booking_modify_from_correspondence_file(path: Path, pms: object) -> tuple[bool, str]:
    """
    If the markdown file embeds BOOKING_MODIFY JSON, apply ``modify_reservation`` (Human desk Approve).
    ``reservation_id`` may be omitted when **Related reservation** is set in the header.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return False, str(e)
    payload = parse_booking_modify_from_text(text)
    if not payload:
        return True, "No BOOKING_MODIFY in file"
    rid = str(payload.get("reservation_id") or "").strip()
    if not rid:
        rid = _related_reservation_id_from_header(text).strip()
    if not rid:
        return False, "BOOKING_MODIFY missing reservation_id (and no Related reservation header)"
    kwargs: dict = {}
    for k in ("check_in", "check_out", "room_type_id", "rate_plan_id", "notes"):
        if k in payload and str(payload[k]).strip():
            kwargs[k] = str(payload[k]).strip()
    if "adults" in payload and payload["adults"] is not None:
        kwargs["adults"] = int(payload["adults"])
    if "children" in payload and payload["children"] is not None:
        kwargs["children"] = int(payload["children"])
    if not kwargs:
        return True, "No fields to change in BOOKING_MODIFY"
    modify = getattr(pms, "modify_reservation", None)
    if not callable(modify):
        return False, "PMS has no modify_reservation"
    out = modify(rid, **kwargs)
    if isinstance(out, dict) and out.get("ok"):
        return True, f"Modified {rid}"
    return False, str(out.get("error", out) if isinstance(out, dict) else out)


def apply_pending_guest_email_on_approval(path: Path, pms: object) -> tuple[bool, str]:
    """
    Apply PMS-side effects for a Human-mode desk approval:
    - BOOKING_COMMIT (new reservation) when present.
    - ``cancellations``: cancel **Related reservation** when no BOOKING_COMMIT.
    - ``modifications``: apply **BOOKING_MODIFY** when no BOOKING_COMMIT.
    """
    ok, detail = try_commit_booking_from_correspondence_file(path, pms)
    if not ok:
        return ok, detail
    if detail != "No BOOKING_COMMIT in file":
        return ok, detail
    cat = normalize_category(path.parent.name if path.parent else "")
    if cat == "cancellations":
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            return False, str(e)
        rel_id = _related_reservation_id_from_header(text)
        if not rel_id:
            return True, "No related reservation to cancel"
        get_res = getattr(pms, "get_reservation", None)
        if callable(get_res):
            cur = get_res(rel_id)
            if not cur:
                return True, f"Reservation {rel_id} not found (nothing to cancel)"
            if str(cur.get("status", "")).lower() == "cancelled":
                return True, f"Already cancelled {rel_id}"
        cancel = getattr(pms, "cancel_reservation", None)
        if not callable(cancel):
            return False, "PMS has no cancel_reservation"
        out = cancel(rel_id, "Approved from pending guest email")
        if isinstance(out, dict) and out.get("ok"):
            return True, f"Cancelled {rel_id}"
        return False, str(out.get("error", out) if isinstance(out, dict) else out)
    if cat == "modifications":
        return try_apply_booking_modify_from_correspondence_file(path, pms)
    return True, detail


def normalize_category(category: str) -> str:
    c = category.strip().lower().replace(" ", "_").replace("-", "_")
    if c in REVIEW_CATEGORIES:
        return c
    return _CATEGORY_ALIASES.get(c, "general")


def _safe_slug(s: str, max_len: int = 40) -> str:
    s = re.sub(r"[^\w.@+-]+", "_", s.strip())[:max_len].strip("_")
    return s or "guest"


def format_escalation_guest_email_template(
    guest_message_body: str,
    *,
    to_email: str,
    subject: str,
    hotel_display_name: str = "",
) -> str:
    """
    Professional escalation letter layout (markdown) for on-disk drafts.
    The model should pass **core paragraphs only** (no salutation/sign-off); this adds framing.
    """
    to_e = (to_email or "").strip() or "—"
    subj = (subject or "").strip() or "—"
    hotel = (hotel_display_name or "").strip() or "our hotel"
    core = (guest_message_body or "").strip()
    # Avoid double salutation if legacy drafts already start with "Dear …"
    lower = core.lstrip().lower()
    if lower.startswith("dear "):
        opening = ""
    else:
        opening = "**Dear Guest,**\n\n"

    return (
        "## Guest email — escalation draft\n\n"
        "| | |\n"
        "|:--|:--|\n"
        f"| **To** | `{to_e}` |\n"
        f"| **Subject** | {subj} |\n\n"
        "---\n\n"
        f"{opening}"
        "### Summary for the guest\n\n"
        f"{core if core else '*[No body was supplied — replace with reviewed text before sending.]*'}\n\n"
        "### What happens next\n\n"
        "Your enquiry has been **escalated** to a specialist who can review applicable policies, "
        "your booking or payment records, and any options that may apply. Someone from our team "
        "will follow up using the **email address above**; we cannot confirm final outcomes "
        "(including refunds, credits, or exceptions) from this automated draft until that review is complete.\n\n"
        f"Thank you for your patience and for contacting **{hotel}**.\n\n"
        "**Kind regards,**  \n\n"
        "**Reservations**  \n"
        f"*{hotel}*\n\n"
        "---\n\n"
        "### Staff only — do not send without review\n\n"
        "- Match **To** / **Subject** to the file header; read **Notes for staff** before release.\n"
        "- Verify all facts in the PMS and published policies; personalise the greeting if the guest name is known.\n"
        "- Remove or edit this section before any outbound send.\n\n"
        "— *Escalation queue — system draft*\n"
    )


def save_queued_correspondence(
    category: str,
    to_email: str,
    subject: str,
    body: str,
    *,
    related_reservation_id: str = "",
    notes_for_staff: str = "",
    booking_commit: dict | None = None,
    booking_modify: dict | None = None,
    direct_to_approved: bool = False,
    queue_root: Path | None = None,
    hotel_display_name: str = "",
) -> Path:
    """
    Write a markdown draft. Tells stderr up front that no email is sent, then saves the file.

    When ``direct_to_approved`` is True and category is **drafts** or **booking_confirmations**, the file
    is written under ``<root>/approved/<category>/``. **Escalations** always use ``<root>/escalations/``.
    Optional ``booking_commit`` is embedded for drafts when applicable.
    """
    sys.stderr.write(
        "\n[Correspondence] This action only creates an on-disk draft for staff review.\n"
    )
    sys.stderr.flush()

    root = review_queue_root(queue_root)
    cat = normalize_category(category)
    if direct_to_approved and cat not in _AUTONOMOUS_DIRECT_APPROVED_CATEGORIES:
        direct_to_approved = False
    if direct_to_approved:
        cat_dir = root / "approved" / cat
    else:
        cat_dir = root / cat
    cat_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = _safe_slug(to_email or "unknown")
    fname = f"{ts}_{slug}.md"
    path = cat_dir / fname

    title = (
        "# Approved guest correspondence (not sent)\n"
        if direct_to_approved
        else "# Queued guest correspondence (not sent)\n"
    )
    header_lines = [
        title,
        f"- **Category:** `{cat}`\n",
        f"- **To:** {to_email.strip()}\n",
        f"- **Subject:** {subject.strip()}\n",
    ]
    if related_reservation_id.strip():
        header_lines.append(f"- **Related reservation:** `{related_reservation_id.strip()}`\n")
    if notes_for_staff.strip():
        header_lines.append(f"- **Notes for staff:** {notes_for_staff.strip()}\n")
    if booking_commit:
        header_lines.append(
            "- **BOOKING_COMMIT:** "
            + json.dumps(booking_commit, separators=(",", ":"))
            + "\n"
        )
    if booking_modify:
        header_lines.append(
            "- **BOOKING_MODIFY:** "
            + json.dumps(booking_modify, separators=(",", ":"))
            + "\n"
        )
    header_lines.append("\n---\n\n")

    body_out = body.strip()
    if cat == "escalations":
        body_out = format_escalation_guest_email_template(
            body_out,
            to_email=to_email.strip(),
            subject=subject.strip(),
            hotel_display_name=hotel_display_name.strip(),
        )

    path.write_text("".join(header_lines) + body_out + "\n", encoding="utf-8")

    try:
        display = path.relative_to(Path.cwd())
    except ValueError:
        display = path
    where = f"approved/{cat}" if direct_to_approved else cat
    sys.stderr.write(f"[Correspondence] Saved under “{where}”: {display}\n")
    sys.stderr.flush()
    return path


def iter_review_category_dirs(root: Path) -> list[Path]:
    """Category folders (drafts, cancellations, …), not approved/rejected."""
    if not root.is_dir():
        return []
    return sorted(root / c for c in REVIEW_CATEGORIES if (root / c).is_dir())


def collect_pending_paths(root: Path, category: str | None = None) -> list[Path]:
    """Pending *.md files (exclude approved/ and rejected/ trees)."""
    paths: list[Path] = []
    if category:
        d = root / normalize_category(category)
        if d.is_dir():
            paths.extend(d.glob("*.md"))
    else:
        for d in iter_review_category_dirs(root):
            paths.extend(d.glob("*.md"))
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def collect_escalation_desk_entries(
    root: Path, *, max_items: int = 150
) -> list[tuple[Path, Literal["pending", "approved"]]]:
    """
    Escalation correspondence for the operations desk: **pending** under ``<root>/escalations/``
    (including autonomous risk filings) plus **staff-approved** under ``<root>/approved/escalations/``,
    merged and sorted by modification time (newest first).
    """
    items: list[tuple[Path, float, Literal["pending", "approved"]]] = []
    pend = root / "escalations"
    if pend.is_dir():
        for p in pend.glob("*.md"):
            items.append((p, p.stat().st_mtime, "pending"))
    appr = root / "approved" / "escalations"
    if appr.is_dir():
        for p in appr.glob("*.md"):
            items.append((p, p.stat().st_mtime, "approved"))
    items.sort(key=lambda x: -x[1])
    return [(p, s) for p, _, s in items[:max_items]]


def collect_approved_correspondence_paths(
    root: Path,
    *,
    categories: tuple[str, ...] | None = None,
    exclude_categories: tuple[str, ...] = ("escalations",),
    max_files: int = 200,
) -> list[tuple[Path, str]]:
    """
    Guest correspondence already under ``approved/<category>/`` (newest first).

    ``categories`` defaults to all ``REVIEW_CATEGORIES``. ``exclude_categories`` removes types that
    have their own desk (e.g. **escalations** live under the Escalations table only).
    """
    approved = root / "approved"
    if not approved.is_dir():
        return []
    cats = categories if categories is not None else REVIEW_CATEGORIES
    skip = {normalize_category(x) for x in exclude_categories}
    items: list[tuple[Path, float, str]] = []
    for cat in cats:
        cn = normalize_category(cat)
        if cn in skip:
            continue
        d = approved / cn
        if not d.is_dir():
            continue
        for p in d.glob("*.md"):
            items.append((p, p.stat().st_mtime, cn))
    items.sort(key=lambda x: -x[1])
    return [(p, c) for p, _, c in items[:max_files]]


def collect_approved_email_paths(root: Path, *, max_files: int = 200) -> list[Path]:
    """Correspondence under ``approved/drafts`` only (backward compatible)."""
    return [p for p, _ in collect_approved_correspondence_paths(root, categories=("drafts",), max_files=max_files)]


def _move_to_resolution(
    path: Path,
    root: Path,
    resolution: Literal["approved", "rejected"],
) -> Path:
    rel = path.relative_to(root)
    dest = root / resolution / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dest))
    return dest




def resolve_pending_draft(
    path: Path,
    root: Path,
    resolution: Literal["approved", "rejected"],
) -> Path:
    """Move a pending draft into approved/ or rejected/ under the same category path."""
    rp = path.resolve()
    rr = root.resolve()
    if not rp.exists():
        raise FileNotFoundError(f"Draft not found: {path}")
    try:
        rel = rp.relative_to(rr)
    except ValueError as e:
        raise ValueError("Draft must be inside review queue root") from e
    if rel.parts and rel.parts[0] in ("approved", "rejected"):
        raise ValueError("Draft is already resolved")
    return _move_to_resolution(rp, rr, resolution)


def print_review_queue_listing(
    queue_root: Path | None = None,
    *,
    category: str | None = None,
    max_files: int = 200,
) -> None:
    root = review_queue_root(queue_root)
    print(f"Review queue root: {root}\n")
    if not root.is_dir():
        print("(No review_queue directory yet — run the agent to create drafts.)\n")
        return

    paths = collect_pending_paths(root, category=category)[:max_files]
    if not paths:
        scope = f" ({category})" if category else ""
        print(f"(No pending drafts{scope}.)\n")
        return

    by_cat: dict[str, list[Path]] = {}
    for p in paths:
        by_cat.setdefault(p.parent.name, []).append(p)

    for cat in sorted(by_cat.keys()):
        files = by_cat[cat]
        print(f"— {cat}/ ({len(files)} pending *.md)")
        for f in files:
            rel = f
            try:
                rel = f.relative_to(Path.cwd())
            except ValueError:
                pass
            m = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"    [{m}] {rel}")
        print()


def run_interactive_review_session(
    queue_root: str | Path | None = None,
    *,
    category: str | None = None,
) -> None:
    """
    List pending drafts; open one to view / edit / approve / reject, or quit.
    Approved → review_queue/approved/<category>/…  Rejected → review_queue/rejected/…
    """
    root = review_queue_root(queue_root)
    print(
        "\n=== Review queue (pending drafts only) ===\n"
        "Nothing is emailed. Approve/reject only moves files under approved/ or rejected/.\n"
        "Main menu: number = open draft, r = refresh, q = quit\n"
        f"Root: {root}\n"
    )

    while True:
        paths = collect_pending_paths(root, category=category)
        print("--- Pending ---")
        if not paths:
            print("  (none)")
        else:
            for i, p in enumerate(paths, start=1):
                try:
                    rel = p.relative_to(root)
                except ValueError:
                    rel = p
                m = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                print(f"  {i}. [{m}] {rel}")

        print("\nEnter number to open, [r] refresh, [q] quit:", end=" ")
        try:
            choice = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.\n")
            return

        if choice in ("q", "quit", "exit"):
            print("Bye.\n")
            return
        if choice in ("r", "refresh"):
            print()
            continue
        if not choice.isdigit():
            print("Unknown command.\n")
            continue
        idx = int(choice)
        if idx < 1 or idx > len(paths):
            print("Invalid number.\n")
            continue

        _draft_detail_loop(paths[idx - 1], root)


def _draft_detail_loop(path: Path, root: Path) -> None:
    while True:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        print(f"\n>>> {rel}")
        print("  [v] view   [e] edit ($EDITOR or nano)   [a] approve → approved/")
        print("  [x] reject → rejected/   [b] back to list")
        try:
            cmd = input("Choice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            return

        if cmd in ("b", "back"):
            return
        if cmd in ("v", "view", ""):
            print("--- file content ---\n")
            try:
                print(path.read_text(encoding="utf-8"))
            except OSError as e:
                print(f"(read error: {e})")
            print("--- end ---\n")
            continue
        if cmd in ("e", "edit"):
            editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
            if not editor:
                editor = "notepad" if sys.platform == "win32" else "nano"
            try:
                subprocess.run([editor, str(path)], check=False)
                print("(Editor closed.)\n")
            except OSError as e:
                print(f"Could not run editor '{editor}': {e}\n")
            continue
        if cmd in ("a", "approve"):
            try:
                dest = _move_to_resolution(path, root, "approved")
                print(f"Approved → {dest}\n")
            except Exception as e:
                print(f"Error: {e}\n")
            return
        if cmd in ("x", "reject"):
            try:
                ans = input("Reject (move to rejected/)? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if ans not in ("y", "yes"):
                print("Cancelled.\n")
                continue
            try:
                dest = _move_to_resolution(path, root, "rejected")
                print(f"Rejected → {dest}\n")
            except Exception as e:
                print(f"Error: {e}\n")
            return
        print("Unknown choice.\n")
