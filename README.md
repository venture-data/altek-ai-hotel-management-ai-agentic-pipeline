# Hotel Reservations AI (Mock PMS)

A production-style hotel reservations assistant built around a **LangGraph agent**, a **mock PMS**, and a **human-in-the-loop review workflow**.

It supports:
- conversational booking flows
- policy/risk-aware escalation
- draft correspondence queueing and approval
- optional PMS write execution (guarded by approval/release flags)

---

## Project Goals

- Provide a realistic agentic reservation experience without touching real systems.
- Keep guest communication polished while enforcing operational safety.
- Separate **planning / drafting** from **final PMS mutation**.
- Make date handling, tool calls, and escalation behavior deterministic and auditable.

---

## Core Features

- **Mock PMS tools** for guest lookup, availability, quote, reservation create/modify/cancel.
- **Strict date grounding** from guest thread text (with recency-aware parsing).
- **Risk gate** for high-risk/ambiguous requests (manual review path).
- **Draft-first confirmation flow**: bookings are not finally confirmed until release/approval gates permit PMS mutation.
- **Review queue** persisted on disk with pending/approved/rejected lifecycle.
- **Streamlit chat UI** with queue review and approve/reject buttons for drafts.
- **CLI** and **FastAPI** entry points for batch/integration usage.

---

## Repository Structure

- `streamlit_app.py` — primary interactive UI (chat + review queue)
- `main.py` — CLI entrypoint (interactive chat / one-shot / batch)
- `api.py` — FastAPI endpoint for inbound-email style runs
- `graph.py` — LangGraph agent builder + system prompts
- `agent_tools.py` — tool layer and write guards
- `hotel_pms.py` — mock PMS domain and inventory/reservation logic
- `guest_dates.py` — date extraction + gating helpers
- `guest_booking_context.py` — lightweight helpers for when a turn needs a guest email
- `execution_flow.py` — shared turn configuration and execution-mode behavior
- `email_pipeline.py` — planner/executor pipeline for inbound style processing
- `risk.py` — heuristic risk detection and manual-review triggers
- `review_queue.py` — on-disk draft queue and approval/rejection utilities
- `mock_hotel_data.json` — mock hotel, policies, inventory, and seed reservations

---

## Prerequisites

- Python 3.10+
- OpenAI API key

Install dependencies:

```bash
pip install -r requirements.txt
```

Set environment (either form works; importing `graph` / running Streamlit or `api` loads a repo-root `.env` via python-dotenv):

```bash
cp .env.sample .env   # then put your real key in .env
# or:
export OPENAI_API_KEY=your_key_here
```

Optional env values:
- `OPENAI_MODEL` (default from code)
- `HOTEL_MOCK_DATA` (alternate mock JSON path)
- `HOTEL_REVIEW_QUEUE_DIR` (alternate queue root)
- `HOTEL_BOOKING_COMMIT=1` (for **`main.py` batch paths** — `--demo`, `--email-file`, stdin / `--one-shot` — sets `booking_pms_commit_allowed` when combined with autonomous execution; interactive TTY chat and Streamlit use their own commit toggles)

---

## How to Run

### 1) Streamlit UI (recommended)

```bash
streamlit run streamlit_app.py
```

Current UI includes:
- **Chat** tab — **Autonomous** vs **Human** mode:
  - **Autonomous**: single pass with writes allowed when policy gates permit (including booking commits when configured).
  - **Human**: planner-style read-only pass, then staff-style approval before writes execute (mirrors CLI/API `approval` execution mode).
- **Review queue** tab
  - open pending draft
  - **Approve draft** (moves to `approved/`)
  - **Reject draft** (moves to `rejected/`)

### 2) CLI

```bash
python main.py
```

Useful options:

```bash
python main.py --help
python main.py --demo
python main.py --email-file sample_email.txt --execution-mode approval
python main.py --data /path/to/mock_hotel_data.json
python main.py --show-review-queue
```

`--execution-mode` applies to **batch** paths (`--email-file`, `--demo`, stdin / `--one-shot`). Interactive TTY chat uses autonomous execution with per-turn risk blocking (see `main.py` help text).

### 3) API

```bash
uvicorn api:app --reload
```

Endpoint: `POST /v1/inbound-email`

Key request fields:
- `body`
- `guest_from_email`
- `execution_mode` (`autonomous` or `approval`)
- `approve_writes` (approval mode helper)
- `booking_pms_commit_allowed` (release gate for booking mutations)

---

## Operating Model

### Draft-First Reservation Rule

The system is intentionally configured so reservation mutations are blocked unless release is granted:
- `pms_create_reservation`
- `pms_cancel_reservation`
- `pms_modify_reservation`

The gate is controlled by `booking_pms_commit_allowed`.

Implication:
- the assistant can check availability/quote and queue correspondence drafts,
- but should not claim final confirmation until PMS mutation succeeds with release enabled.

---

## 1. Agent Architecture

### High-level flow

1. **User input** enters Streamlit/CLI/API.
2. `execution_flow.prepare_turn_configurable()` builds runtime config:
   - parsed date literals
   - execution mode flags
   - risk/manual-review flags
   - booking commit release flag
3. `graph.build_email_agent()` creates the LangGraph ReAct agent with proper tools/prompts.
4. Agent executes tools from `agent_tools.py` against `hotel_pms.py`.
5. Guest-facing output is drafted and queued via `review_queue.py`.

### Modes

- **Autonomous mode**
  - single pass read/write-capable agent
  - still constrained by risk/manual-review and booking-commit gate

- **Approval mode**
  - planner phase (read-only)
  - operator approval
  - executor phase (write-capable + release flag enabled)

### Tooling boundary

- **Read tools**: lookup, availability, quote, policies
- **Write tools**: guest create, reservation create/cancel/modify, queue correspondence
- Guardrails are centralized in tool layer (not only prompt text).

---

## 2. Prompt Design

Prompts in `graph.py` are policy-composed (modular blocks) and then injected into:
- main system prompt
- planner prompt
- executor prompt
- review-only prompt

### Key prompt blocks

- **Date policy**: no invented dates; thread-grounded dates only.
- **Date span policy**: half-open stay semantics and range interpretation (examples in the text still use generic “May …” phrasing for the rules; ISO-heavy blocks below follow the configured year/month logic).
- **Tool error policy**: explain concrete causes from tool JSON, avoid vague narratives.
- **Availability policy**: enforce correct night-count logic (`check_out > check_in` for non-zero nights).
- **Booking consistency policy**: draft-first truthfulness; no premature confirmation.
- **Reply coherence policy**: no contradictory/duplicated answers.
- **Transparency policy**: separate date guard issues vs inventory vs hotel policy.

### Illustrative dates in prompts

Runtime wiring passes **`guest_default_year`** (from `MockHotelPMS.default_guest_stay_year()`, derived from availability keys in `mock_hotel_data.json`) into `graph.prompt_example_year()`. That keeps **`YYYY-MM-DD` examples** in the composed prompts aligned with the mock inventory year when the app constructs the agent.

Availability, inventory-grounding, and transparency policy snippets use **`illustrative_stay_example_dates(year)`** so concrete check-in/check-out illustrations are **not hardcoded to a single calendar month**; the example month cycles with the year (May–October) while staying consistent for a given deployment year.

### Review-mode response behavior

For risky/ambiguous requests, review prompt explicitly requires escalation language, e.g.:
> "I have raised your request to a specialist, and they will contact you shortly."

This ensures safe, clear handoff messaging.

---

## 3. Engineering Quality

### Guardrails in code (not prompt-only)

- `risk.py` blocks sensitive classes of requests from autonomous execution.
- `agent_tools.py` enforces:
  - execution-mode write restrictions
  - booking mutation release gate
  - date gate validation
  - stay-window correctness

### Deterministic date handling

`guest_dates.py` normalizes date literals and supports recency-aware gating so latest user date intent is prioritized in active booking context.

### Auditability and operations

- Every outbound guest message is queued to disk rather than sent.
- Review queue has explicit pending/approved/rejected states.
- Streamlit includes direct approve/reject actions.

### Separation of concerns

- UI: `streamlit_app.py`
- orchestration: `execution_flow.py`, `email_pipeline.py`
- prompting/agent graph: `graph.py`
- domain logic: `hotel_pms.py`
- policy/risk: `risk.py`

This keeps behavior maintainable and testable.

---

## Safety Notes

- This repository is a **mock environment**; no real email dispatch occurs.
- Drafts are persisted under review queue directories.
- Keep `booking_pms_commit_allowed` off in environments where staff release is mandatory.
- Always validate policy-sensitive outcomes in human review mode.

---

## Quick Start Checklist

1. Install deps: `pip install -r requirements.txt`
2. Set `OPENAI_API_KEY`
3. Run UI: `streamlit run streamlit_app.py`
4. Test chat booking flow
5. Review and approve/reject drafts in Review queue tab
6. Enable booking commit release only when operationally appropriate

---

## Future Improvements (recommended)

- Add automated unit tests for date parsing edge cases and gate behavior.
- Add integration tests for autonomous vs approval mode transitions.
- Add metrics/logging hooks for tool failures and escalation frequency.
- Add role-based auth around approval/release actions for multi-user deployment.
