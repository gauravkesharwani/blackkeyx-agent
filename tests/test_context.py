"""Tests for investor context personalization."""

import pytest

from livekit.agents import AgentSession, mock_tools
from livekit.plugins import openai

from helpers import TOOL_MOCKS, BlackKeyXAdvisor, assert_assistant_message


async def test_uses_investor_name(outbound_session, llm):
    """Outbound call mentions the investor's name from context."""
    result = await outbound_session.run(user_input="Hello?")
    await assert_assistant_message(
        result,
        llm,
        "Introduces as Alex from Black Key Exchange and mentions "
        "'Sarah' or 'Sarah Johnson' when asking to confirm identity.",
    )


async def test_knows_capital_available(inbound_session, llm):
    """Agent references known capital amount from investor context."""
    await inbound_session.run(
        user_input="Hi, I'm ready to discuss investments."
    )
    result = await inbound_session.run(
        user_input="So what do you already know about me?"
    )
    await assert_assistant_message(
        result,
        llm,
        "References that the investor mentioned $2M in capital "
        "and/or a 3-6 month timeline from the chatbot interaction.",
    )


async def test_minimal_context_defaults(llm):
    """Agent handles empty context without saying 'Unknown' literally or
    fabricating an investor name."""
    async with AgentSession(llm=llm) as session:
        with mock_tools(BlackKeyXAdvisor, TOOL_MOCKS):
            await session.start(BlackKeyXAdvisor(investor_context={}))
            result = await session.run(
                user_input="Hello, I'm interested in CRE investments."
            )

            # Grab the first assistant message text
            assistant_text = next(
                "".join(ev.item.content) if isinstance(ev.item.content, list) else ev.item.content
                for ev in result.events
                if ev.type == "message" and ev.item.role == "assistant"
            )

            # Core assertions: literal "Unknown" must not appear, and no
            # fabricated investor name (common names from other test contexts)
            # should leak in when no name was provided.
            lower = assistant_text.lower()
            assert "unknown" not in lower, (
                f"Agent said 'Unknown' literally: {assistant_text!r}"
            )
            for fabricated in ("john", "sarah", "roberto", "wei", "priya"):
                assert fabricated not in lower, (
                    f"Agent fabricated a caller name {fabricated!r}: {assistant_text!r}"
                )


@pytest.mark.parametrize(
    "investor_name",
    ["Roberto Garcia", "Wei Zhang", "Priya Patel"],
)
async def test_various_investor_names(llm, investor_name):
    """Agent correctly personalizes outbound call for different investor names."""
    ctx = {
        "name": investor_name,
        "capital_available": "$1M",
        "timeline": "6 months",
        "outbound": True,
    }
    async with AgentSession(llm=llm) as session:
        with mock_tools(BlackKeyXAdvisor, TOOL_MOCKS):
            await session.start(BlackKeyXAdvisor(investor_context=ctx))
            result = await session.run(user_input="Hello?")
            await assert_assistant_message(
                result,
                llm,
                f"Introduces as Alex from Black Key Exchange and asks to "
                f"confirm if speaking with {investor_name}.",
            )
