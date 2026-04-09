"""In-memory mock PMS backed by mock_hotel_data.json (mutations persist for process lifetime)."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _date_range_nights(check_in: str, check_out: str) -> list[str]:
    """Nights are [check_in, check_out) as date strings."""
    start = _parse_date(check_in)
    end = _parse_date(check_out)
    if end <= start:
        return []
    out = []
    cur = start
    while cur < end:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


class MockHotelPMS:
    def __init__(self, data_path: str | Path | None = None) -> None:
        root = Path(__file__).resolve().parent
        path = Path(data_path) if data_path else root / "mock_hotel_data.json"
        with open(path, encoding="utf-8") as f:
            self._data: dict[str, Any] = json.load(f)
        # Deep copy mutable sections we update at runtime
        self._data["guests"] = deepcopy(self._data.get("guests", []))
        self._data["reservations"] = deepcopy(self._data.get("reservations", []))
        avail = self._data.get("availability", {})
        self._data["availability"] = {
            k: dict(v) if isinstance(v, dict) else v for k, v in avail.items() if k != "_comment"
        }

    @property
    def hotel(self) -> dict[str, Any]:
        return self._data["hotel"]

    @property
    def policies(self) -> dict[str, Any]:
        return self._data["policies"]

    def get_room_types(self) -> list[dict[str, Any]]:
        return list(self._data.get("room_types", []))

    def get_rate_plans(self) -> list[dict[str, Any]]:
        return list(self._data.get("rate_plans", []))

    def default_guest_stay_year(self) -> int:
        """
        Calendar year to assume for guest phrases without a year (e.g. "May 21–22").
        Uses the latest year present in availability keys so gates match mock inventory.
        """
        years: list[int] = []
        for k in self._data.get("availability", {}):
            if not isinstance(k, str) or len(k) < 10:
                continue
            try:
                datetime.strptime(k[:10], "%Y-%m-%d")
                years.append(int(k[:4]))
            except ValueError:
                continue
        return max(years) if years else date.today().year

    def list_guests(self) -> list[dict[str, Any]]:
        """All guest profiles in the current mock session (including any created at runtime)."""
        return [dict(g) for g in self._data.get("guests", [])]

    def list_reservations(self) -> list[dict[str, Any]]:
        """All reservations in the current mock session."""
        return [dict(r) for r in self._data.get("reservations", [])]

    def find_guest_by_email(self, email: str) -> dict[str, Any] | None:
        e = email.strip().lower()
        for g in self._data["guests"]:
            if str(g.get("email", "")).strip().lower() == e:
                return dict(g)
        return None

    def create_guest(
        self,
        email: str,
        first_name: str,
        last_name: str,
        phone: str = "",
        nationality: str = "",
    ) -> dict[str, Any]:
        if self.find_guest_by_email(email):
            raise ValueError(f"Guest already exists: {email}")
        ids = [g["id"] for g in self._data["guests"] if g.get("id", "").startswith("G")]
        n = max((int(x[1:]) for x in ids if len(x) > 1 and x[1:].isdigit()), default=0) + 1
        guest = {
            "id": f"G{n:03d}",
            "first_name": first_name.strip(),
            "last_name": last_name.strip(),
            "email": email.strip(),
            "phone": phone.strip(),
            "nationality": nationality.strip() or "XX",
            "created_at": datetime.now().strftime("%Y-%m-%d"),
        }
        self._data["guests"].append(guest)
        return dict(guest)

    def _min_availability_for_stay(self, check_in: str, check_out: str, room_type_id: str) -> int:
        nights = _date_range_nights(check_in, check_out)
        if not nights:
            return 0
        mins = []
        for d in nights:
            day = self._data["availability"].get(d, {})
            mins.append(int(day.get(room_type_id, 0)))
        return min(mins) if mins else 0

    def check_availability(
        self, check_in: str, check_out: str, adults: int = 1, children: int = 0
    ) -> list[dict[str, Any]]:
        """Return room types that fit party size with min inventory across stay nights."""
        need = adults + children
        results = []
        for rt in self._data["room_types"]:
            if int(rt.get("max_occupancy", 0)) < need:
                continue
            rid = rt["id"]
            inv = self._min_availability_for_stay(check_in, check_out, rid)
            if inv <= 0:
                continue
            row: dict[str, Any] = {
                "room_type_id": rid,
                "name": rt["name"],
                "max_occupancy": rt["max_occupancy"],
                "base_rate_per_night": rt["base_rate_per_night"],
                "available_rooms_for_stay": inv,
            }
            # Same mock math as quote_stay — avoids LLM calling quote with different args and contradicting this row.
            q = self.quote_stay(
                check_in, check_out, rid, "RP001", adults, children
            )
            if q.get("ok"):
                row["quote_standard_rate_rp001"] = {
                    "rate_plan_id": "RP001",
                    "rate_plan_name": q.get("rate_plan", ""),
                    "nights": q.get("nights"),
                    "total_amount_nok": q.get("total_amount_nok"),
                    "currency": q.get("currency"),
                }
            results.append(row)
        return results

    def quote_stay(
        self,
        check_in: str,
        check_out: str,
        room_type_id: str,
        rate_plan_id: str,
        adults: int = 1,
        children: int = 0,
    ) -> dict[str, Any]:
        nights = _date_range_nights(check_in, check_out)
        n_nights = len(nights)
        if n_nights == 0:
            return {"ok": False, "error": "Invalid dates: need check_out after check_in."}

        rt = next((r for r in self._data["room_types"] if r["id"] == room_type_id), None)
        rp = next((r for r in self._data["rate_plans"] if r["id"] == rate_plan_id), None)
        if not rt or not rp:
            return {"ok": False, "error": "Unknown room_type_id or rate_plan_id."}

        if self._min_availability_for_stay(check_in, check_out, room_type_id) <= 0:
            listing = self.check_availability(check_in, check_out, adults, children)
            ids = [x["room_type_id"] for x in listing]
            return {
                "ok": False,
                "error": "No availability for this room type on those dates.",
                "check_in": check_in,
                "check_out": check_out,
                "adults": adults,
                "children": children,
                "room_type_id": room_type_id,
                "available_room_type_ids_for_same_party_and_dates": ids,
                "hint": (
                    "If a room appeared under pms_check_availability for this stay but this quote failed, "
                    "you used different check_in, check_out, adults, or children between the two calls — "
                    "reuse the exact arguments from the availability call. Prefer quote_standard_rate_rp001 "
                    "inside the availability JSON and skip a separate quote for RP001 unless you need another rate plan."
                ),
            }

        base = float(rt["base_rate_per_night"]) * n_nights * float(rp["rate_modifier"])
        breakfast_extra = 0.0
        if rp.get("includes_breakfast"):
            per = float(rp.get("breakfast_supplement_per_person") or 0)
            # When included in rate plan, supplement may be 0 (bundled in modifier) or explicit
            guests = adults + children
            breakfast_extra = per * guests * n_nights
        total = int(round(base + breakfast_extra))
        policy_key = rp.get("cancellation_policy", "standard")
        policy_text = (
            self._data.get("policies", {})
            .get("cancellation", {})
            .get(policy_key, "")
        )
        return {
            "ok": True,
            "nights": n_nights,
            "room_type": rt["name"],
            "rate_plan": rp["name"],
            "total_amount_nok": total,
            "currency": self._data["hotel"].get("currency", "NOK"),
            "cancellation_policy": policy_text,
        }

    def create_reservation(
        self,
        guest_id: str,
        room_type_id: str,
        rate_plan_id: str,
        check_in: str,
        check_out: str,
        adults: int = 1,
        children: int = 0,
        notes: str = "",
    ) -> dict[str, Any]:
        quote = self.quote_stay(
            check_in, check_out, room_type_id, rate_plan_id, adults, children
        )
        if not quote.get("ok"):
            return {"ok": False, "error": quote.get("error", "Quote failed")}

        if not any(g["id"] == guest_id for g in self._data["guests"]):
            return {"ok": False, "error": f"Unknown guest_id: {guest_id}"}

        inv = self._min_availability_for_stay(check_in, check_out, room_type_id)
        if inv <= 0:
            return {"ok": False, "error": "No inventory left for this stay."}

        nights = _date_range_nights(check_in, check_out)
        for d in nights:
            self._data["availability"].setdefault(d, {})
            cur = int(self._data["availability"][d].get(room_type_id, 0))
            self._data["availability"][d][room_type_id] = max(0, cur - 1)

        res_ids = [
            r["id"]
            for r in self._data["reservations"]
            if str(r.get("id", "")).startswith("RES")
        ]
        n = max((int(x[3:]) for x in res_ids if len(x) > 3 and x[3:].isdigit()), default=0) + 1
        res = {
            "id": f"RES{n:03d}",
            "guest_id": guest_id,
            "room_type_id": room_type_id,
            "rate_plan_id": rate_plan_id,
            "check_in": check_in,
            "check_out": check_out,
            "adults": adults,
            "children": children,
            "status": "confirmed",
            "total_amount": quote["total_amount_nok"],
            "notes": notes.strip(),
            "created_at": datetime.now().strftime("%Y-%m-%d"),
        }
        self._data["reservations"].append(res)
        return {"ok": True, "reservation": dict(res)}

    def list_guest_reservations(self, guest_id: str, include_cancelled: bool = False) -> list[dict[str, Any]]:
        out = []
        for r in self._data["reservations"]:
            if r.get("guest_id") != guest_id:
                continue
            if not include_cancelled and r.get("status") == "cancelled":
                continue
            out.append(dict(r))
        return out

    def get_reservation(self, reservation_id: str) -> dict[str, Any] | None:
        for r in self._data["reservations"]:
            if r.get("id") == reservation_id:
                return dict(r)
        return None

    def _rate_plan_for(self, rate_plan_id_v: str) -> dict[str, Any] | None:
        return next((r for r in self._data["rate_plans"] if r["id"] == rate_plan_id_v), None)

    def _release_reservation_inventory(self, reservation_row: dict[str, Any]) -> None:
        if reservation_row.get("status") == "cancelled":
            return
        nights = _date_range_nights(reservation_row["check_in"], reservation_row["check_out"])
        rt = reservation_row["room_type_id"]
        for d in nights:
            day = self._data["availability"].setdefault(d, {})
            cur = int(day.get(rt, 0))
            day[rt] = cur + 1

    def _consume_reservation_inventory(
        self, check_in: str, check_out: str, room_type_id: str, count: int = 1
    ) -> bool:
        """Return True if at least `count` rooms available for each night."""
        if self._min_availability_for_stay(check_in, check_out, room_type_id) < count:
            return False
        nights = _date_range_nights(check_in, check_out)
        for d in nights:
            day = self._data["availability"].setdefault(d, {})
            cur = int(day.get(room_type_id, 0))
            self._data["availability"][d][room_type_id] = max(0, cur - count)
        return True

    def cancel_reservation(self, reservation_id: str, reason: str = "") -> dict[str, Any]:
        row = next((r for r in self._data["reservations"] if r.get("id") == reservation_id), None)
        if not row:
            return {"ok": False, "error": f"Unknown reservation: {reservation_id}"}
        if row.get("status") == "cancelled":
            return {"ok": True, "reservation": dict(row), "note": "Already cancelled."}
        self._release_reservation_inventory(row)
        row["status"] = "cancelled"
        note = (row.get("notes") or "").strip()
        suffix = f"\nCancelled: {reason}" if reason else "\nCancelled."
        row["notes"] = (note + suffix).strip()
        return {"ok": True, "reservation": dict(row)}

    def modify_reservation(
        self,
        reservation_id: str,
        check_in: str | None = None,
        check_out: str | None = None,
        room_type_id: str | None = None,
        rate_plan_id: str | None = None,
        adults: int | None = None,
        children: int | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        row = next((r for r in self._data["reservations"] if r.get("id") == reservation_id), None)
        if not row:
            return {"ok": False, "error": f"Unknown reservation: {reservation_id}"}
        if row.get("status") == "cancelled":
            return {"ok": False, "error": "Cannot modify a cancelled reservation."}

        rp_cur = self._rate_plan_for(row["rate_plan_id"])
        if not rp_cur:
            return {"ok": False, "error": "Invalid rate plan on reservation."}

        new_ci = check_in if check_in is not None else row["check_in"]
        new_co = check_out if check_out is not None else row["check_out"]
        new_rt = room_type_id if room_type_id is not None else row["room_type_id"]
        new_rp = rate_plan_id if rate_plan_id is not None else row["rate_plan_id"]
        new_adults = adults if adults is not None else int(row.get("adults", 1))
        new_children = children if children is not None else int(row.get("children", 0))

        substantive_change = (
            new_ci != row["check_in"]
            or new_co != row["check_out"]
            or new_rt != row["room_type_id"]
            or new_rp != row["rate_plan_id"]
            or new_adults != int(row.get("adults", 1))
            or new_children != int(row.get("children", 0))
        )

        if substantive_change and rp_cur.get("cancellation_policy") == "non_refundable":
            return {
                "ok": False,
                "error": "Non-refundable bookings cannot be modified (dates, room, or party) per hotel policy.",
            }

        if notes is not None and not substantive_change:
            row["notes"] = notes.strip()
            return {"ok": True, "reservation": dict(row), "note": "Notes updated only."}

        if not substantive_change and notes is None:
            return {"ok": True, "reservation": dict(row), "note": "No changes supplied."}

        rt_def = next((r for r in self._data["room_types"] if r["id"] == new_rt), None)
        rp_def = self._rate_plan_for(new_rp)
        if not rt_def or not rp_def:
            return {"ok": False, "error": "Invalid room type or rate plan."}
        if int(rt_def.get("max_occupancy", 0)) < new_adults + new_children:
            return {"ok": False, "error": "Party size exceeds room max occupancy."}

        q = self.quote_stay(new_ci, new_co, new_rt, new_rp, new_adults, new_children)
        if not q.get("ok"):
            return {"ok": False, "error": q.get("error", "Quote failed for modified stay.")}

        old_ci, old_co, old_rt = row["check_in"], row["check_out"], row["room_type_id"]
        self._release_reservation_inventory(row)
        if not self._consume_reservation_inventory(new_ci, new_co, new_rt, 1):
            self._consume_reservation_inventory(old_ci, old_co, old_rt, 1)
            return {"ok": False, "error": "No availability for modified dates or room type."}

        row["check_in"] = new_ci
        row["check_out"] = new_co
        row["room_type_id"] = new_rt
        row["rate_plan_id"] = new_rp
        row["adults"] = new_adults
        row["children"] = new_children
        row["total_amount"] = q["total_amount_nok"]
        if notes is not None:
            row["notes"] = notes.strip()
        return {"ok": True, "reservation": dict(row)}

    def get_policies_summary(self) -> str:
        p = self._data.get("policies", {})
        parts = [json.dumps(p.get("cancellation", {}), indent=2)]
        for key in ("pets", "breakfast", "parking", "extra_bed", "children"):
            if key in p:
                parts.append(f"{key}: {p[key]}")
        return "\n".join(parts)
