"""Tests for CRE investor qualification conversation flow."""

import pytest

from helpers import assert_assistant_message


async def test_asks_one_question_at_a_time(inbound_session, llm):
    """Agent should ask ONE question at a time, not multiple."""
    result = await inbound_session.run(
        user_input="I'm interested in commercial real estate investing."
    )
    await assert_assistant_message(
        result,
        llm,
        "Asks a single qualification question. Does NOT ask multiple questions "
        "in one response. Response is concise (2-3 sentences max).",
    )


async def test_acknowledges_market_preference(inbound_session, llm):
    """Agent acknowledges market preference and asks a follow-up."""
    await inbound_session.run(
        user_input="I want to invest in commercial real estate."
    )
    result = await inbound_session.run(
        user_input="I'm mainly interested in Dallas and Austin, Texas."
    )
    await assert_assistant_message(
        result,
        llm,
        "Acknowledges the Texas/Dallas/Austin market preference and asks "
        "a follow-up qualification question about a different topic.",
    )


async def test_full_qualification_flow(inbound_session, llm):
    """Multi-turn test: investor provides several data points in sequence."""
    await inbound_session.run(
        user_input="Hi, I'm interested in CRE investments."
    )

    # Geographic preference
    r2 = await inbound_session.run(
        user_input="I'm mainly interested in Dallas and Austin, Texas."
    )
    await assert_assistant_message(
        r2, llm,
        "Acknowledges Texas market interest. Asks another qualification question.",
    )

    # Property type
    r3 = await inbound_session.run(
        user_input="I like industrial properties and self-storage."
    )
    await assert_assistant_message(
        r3, llm,
        "Acknowledges industrial and self-storage interest. "
        "Asks another qualification question.",
    )

    # Investment strategy
    r4 = await inbound_session.run(user_input="I prefer value-add deals.")
    await assert_assistant_message(
        r4, llm,
        "Acknowledges value-add strategy preference. Asks another qualification question.",
    )


async def test_conversational_tone(inbound_session, llm):
    """Responses should be conversational, not bullet points or lists."""
    result = await inbound_session.run(
        user_input="I'm looking to invest about 2 million in multifamily properties."
    )
    await assert_assistant_message(
        result,
        llm,
        "Responds conversationally without bullet points or numbered lists. "
        "Shows genuine interest in the investor's goals.",
    )


async def test_follow_up_on_interesting_points(inbound_session, llm):
    """Agent should follow up with empathy on investor experiences."""
    await inbound_session.run(user_input="I want to invest in real estate.")
    result = await inbound_session.run(
        user_input="I had a bad experience with an office building in Detroit "
        "that went bankrupt."
    )
    await assert_assistant_message(
        result,
        llm,
        "Shows empathy about the bad experience. May ask a follow-up about it "
        "or naturally transition to understanding risk tolerance or markets to avoid.",
    )
