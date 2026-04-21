"""Tests for voicemail detection and handling via agent tool calls — Test IDs O-4, O-5, O-8."""

import pytest

from livekit.agents import AgentSession, mock_tools
from livekit.plugins import openai

from helpers import TOOL_MOCKS, BlackKeyXAdvisor, assert_assistant_message


async def test_voicemail_greeting_triggers_handle_voicemail(outbound_session, llm):
    """O-4: Classic voicemail greeting → agent calls handle_voicemail tool."""
    result = await outbound_session.run(
        user_input=(
            "Hi, you've reached the voicemail of Sarah Johnson. "
            "I'm not available right now. Please leave a message after the beep."
        )
    )
    result.expect.contains_function_call(name="handle_voicemail")


async def test_carrier_voicemail_triggers_detection(outbound_session, llm):
    """O-4 variant: Carrier-style VM greeting triggers handle_voicemail."""
    result = await outbound_session.run(
        user_input=(
            "The person you are trying to reach is not available. "
            "Please leave your message after the tone. "
            "When you are finished, you may hang up or press pound for more options."
        )
    )
    result.expect.contains_function_call(name="handle_voicemail")


async def test_mailbox_full_triggers_voicemail(outbound_session, llm):
    """O-6: 'Mailbox is full' triggers handle_voicemail tool."""
    result = await outbound_session.run(
        user_input=(
            "The mailbox is full and cannot accept any messages at this time. Goodbye."
        )
    )
    result.expect.contains_function_call(name="handle_voicemail")


async def test_mid_conversation_voicemail_transfer(llm):
    """O-8: Human answers, then transfers to VM → agent detects and calls handle_voicemail."""
    async with AgentSession(llm=llm) as session:
        with mock_tools(BlackKeyXAdvisor, TOOL_MOCKS):
            ctx = {
                "name": "Sarah Johnson",
                "outbound": True,
                "capital_available": "$5M",
                "timeline": "Immediately",
            }
            await session.start(BlackKeyXAdvisor(investor_context=ctx))

            # Human answers initially
            await session.run(user_input="Hello?")
            await session.run(user_input="Yes, this is Sarah.")
            await session.run(user_input="Let me transfer you, hold on.")

            # Then voicemail picks up
            result = await session.run(
                user_input=(
                    "Hi you've reached Sarah's voicemail. "
                    "Please leave a message after the beep."
                )
            )
            result.expect.contains_function_call(name="handle_voicemail")


async def test_normal_greeting_does_not_trigger_voicemail(outbound_session, llm):
    """Negative case: Normal human greeting does NOT trigger handle_voicemail."""
    result = await outbound_session.run(user_input="Hello? Who is this?")
    assert not any(
        ev.type == "function_call" and ev.item.name == "handle_voicemail"
        for ev in result.events
    ), "handle_voicemail tool was called on a normal human greeting"


async def test_inbound_save_caller_name(llm):
    """I-1: Inbound caller tells name → save_caller_name tool is called."""
    async with AgentSession(llm=llm) as session:
        with mock_tools(BlackKeyXAdvisor, TOOL_MOCKS):
            # New inbound caller (no prior context)
            await session.start(BlackKeyXAdvisor(investor_context={}))

            result = await session.run(
                user_input="Hi, my name is Michael Chen."
            )
            result.expect.contains_function_call(name="save_caller_name")


async def test_returning_caller_no_name_ask(llm):
    """I-2: Returning caller → agent greets by name, does NOT ask for name."""
    async with AgentSession(llm=llm) as session:
        with mock_tools(BlackKeyXAdvisor, TOOL_MOCKS):
            ctx = {
                "name": "John Smith",
                "capital_available": "$2M",
                "timeline": "3-6 months",
            }
            await session.start(BlackKeyXAdvisor(investor_context=ctx))

            result = await session.run(
                user_input="Hey, I'm calling back about our previous conversation."
            )
            await assert_assistant_message(
                result,
                llm,
                "Greets the caller by name (John). Does NOT ask for their name. "
                "Acknowledges the previous conversation or asks how they can help.",
            )
