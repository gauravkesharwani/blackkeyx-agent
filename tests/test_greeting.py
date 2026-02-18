"""Tests for inbound and outbound greeting behavior."""

import pytest

from helpers import assert_assistant_message


# ---------------------------------------------------------------------------
# Inbound call greetings
# ---------------------------------------------------------------------------


async def test_inbound_greeting(inbound_session, llm):
    """Inbound caller gets warm greeting with Alex from Black Key Exchange intro."""
    result = await inbound_session.run(
        user_input="Hi, I'm interested in CRE investments."
    )
    await assert_assistant_message(
        result,
        llm,
        "Introduces themselves as Alex from Black Key Exchange. "
        "Warm and professional greeting. Asks how they can help with "
        "commercial real estate investment goals.",
    )


# ---------------------------------------------------------------------------
# Outbound call greetings
# ---------------------------------------------------------------------------


async def test_outbound_greeting_identity_confirmation(outbound_session, llm):
    """Outbound call: agent introduces as Alex, asks to confirm the caller's identity."""
    result = await outbound_session.run(user_input="Hello?")
    await assert_assistant_message(
        result,
        llm,
        "Introduces themselves as Alex from Black Key Exchange and asks to "
        "confirm if they are speaking with Sarah Johnson. "
        "Does NOT greet by name before introducing self.",
    )


async def test_outbound_wrong_person(outbound_session, llm):
    """When caller says wrong person, agent apologizes and handles gracefully."""
    await outbound_session.run(user_input="Hello?")
    result = await outbound_session.run(
        user_input="No, Sarah doesn't live here anymore."
    )
    await assert_assistant_message(
        result,
        llm,
        "Apologizes politely. May ask if there's a way to reach the intended "
        "person or a better number. Does NOT proceed with qualification questions.",
    )


async def test_outbound_confirmed_then_timing(outbound_session, llm):
    """After identity confirmed, agent asks about timing before qualification."""
    await outbound_session.run(user_input="Hello?")
    result = await outbound_session.run(user_input="Yes, this is Sarah.")
    await assert_assistant_message(
        result,
        llm,
        "Asks if this is a good time to talk. "
        "Does NOT jump directly into qualification questions.",
    )


async def test_outbound_bad_timing_asks_for_callback(outbound_session, llm):
    """When investor is busy, agent asks when to call back."""
    await outbound_session.run(user_input="Hello?")
    await outbound_session.run(user_input="Yes, this is Sarah.")
    result = await outbound_session.run(
        user_input="Actually I'm really busy right now, can you call back later?"
    )
    await assert_assistant_message(
        result,
        llm,
        "Acknowledges the busy schedule politely and asks when would be "
        "a better time to call back.",
    )
