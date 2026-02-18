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
    """Agent handles empty context without saying 'Unknown' literally."""
    async with AgentSession(llm=llm) as session:
        with mock_tools(BlackKeyXAdvisor, TOOL_MOCKS):
            await session.start(BlackKeyXAdvisor(investor_context={}))
            result = await session.run(
                user_input="Hello, I'm interested in CRE investments."
            )
            await assert_assistant_message(
                result,
                llm,
                "Greets warmly without using a specific investor name. "
                "Does NOT say the word 'Unknown' for capital or timeline.",
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
