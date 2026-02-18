"""Shared test utilities and constants."""

import sys
from pathlib import Path

# Ensure the agent module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import BlackKeyXAdvisor  # noqa: E402, F401

# ---------------------------------------------------------------------------
# Tool mock definitions (strip self/ctx, keep user-facing params only)
# ---------------------------------------------------------------------------

TOOL_MOCKS = {
    "end_call": lambda: "Call ended successfully",
    # "transfer_to_human": lambda: "Transfer requested - flagged for human callback",
    "request_callback": lambda callback_datetime, callback_notes="": (
        f"Callback scheduled for {callback_datetime}"
    ),
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def assert_assistant_message(result, llm, intent: str):
    """Assert the next event is an assistant message matching the given intent."""
    await (
        result.expect.next_event(type="message")
        .judge(llm, intent=intent)
    )
