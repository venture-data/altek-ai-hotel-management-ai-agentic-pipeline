"""Create or resolve mock PMS guest by email (no stdin)."""

from __future__ import annotations

from hotel_pms import MockHotelPMS


def resolve_or_create_guest(
    pms: MockHotelPMS,
    email: str,
    *,
    first_name: str = "",
    last_name: str = "",
    phone: str = "",
    nationality: str = "",
) -> tuple[str, str, dict]:
    """
    Return (email, guest_id, guest_row).
    Creates a profile if missing; first_name and last_name are required when creating.
    """
    e = email.strip()
    existing = pms.find_guest_by_email(e)
    if existing:
        return e, str(existing["id"]), dict(existing)
    fn, ln = first_name.strip(), last_name.strip()
    if not fn or not ln:
        raise ValueError("First and last name are required to create a new guest profile.")
    guest = pms.create_guest(e, fn, ln, phone.strip(), nationality.strip())
    return e, str(guest["id"]), dict(guest)


def session_guest_hint(email: str, guest_id: str) -> str:
    return (
        f"(Session guest email: {email}; PMS guest_id: {guest_id}. "
        "Use tools with this guest when taking actions for this person.)\n\n"
    )
