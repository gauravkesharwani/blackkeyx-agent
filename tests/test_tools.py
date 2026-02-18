"""Tests for function tool calling: end_call, request_callback."""

import pytest

from livekit.agents import AgentSession, mock_tools
from livekit.plugins import openai

from helpers import TOOL_MOCKS, BlackKeyXAdvisor, assert_assistant_message


# async def test_transfer_to_human_on_request(inbound_session, llm):
#     """Agent calls transfer_to_human when investor asks for a real person."""
#     result = await inbound_session.run(
#         user_input="Hi, can I speak to a real person please?"
#     )
#     result.expect.contains_function_call(name="transfer_to_human")


async def test_end_call_when_wrapping_up(inbound_session, llm):
    """Agent calls end_call when the conversation is wrapping up."""
    await inbound_session.run(
        user_input="Hi, I'm interested in CRE investments."
    )
    await inbound_session.run(
        user_input="I like Dallas, industrial, value-add, moderate risk, 5 year hold."
    )
    await inbound_session.run(
        user_input="Target IRR 12-15%, LP preferred, avoid Midwest, deploy in 3 months."
    )
    result = await inbound_session.run(
        user_input="I think that covers everything. Thanks, let's wrap up!"
    )
    result.expect.contains_function_call(name="end_call")


async def test_request_callback_with_datetime(outbound_session, llm):
    """request_callback tool is called with callback_datetime argument."""
    await outbound_session.run(user_input="Hello?")
    await outbound_session.run(user_input="Yes, this is Sarah.")
    await outbound_session.run(
        user_input="I'm really busy, can you call me back?"
    )
    result = await outbound_session.run(
        user_input="Tuesday at 2pm would work."
    )
    result.expect.contains_function_call(name="request_callback")


async def test_end_call_error_handling(llm):
    """Agent handles end_call tool failure gracefully."""

    def failing_end_call():
        raise RuntimeError("Room deletion failed")

    error_mocks = {
        "end_call": failing_end_call,
        # "transfer_to_human": TOOL_MOCKS["transfer_to_human"],
        "request_callback": TOOL_MOCKS["request_callback"],
    }

    async with AgentSession(llm=llm) as session:
        with mock_tools(BlackKeyXAdvisor, error_mocks):
            ctx = {"name": "Test User", "capital_available": "$1M", "timeline": "3 months"}
            await session.start(BlackKeyXAdvisor(investor_context=ctx))
            result = await session.run(
                user_input="OK goodbye, end the call now please."
            )
            # Agent should call end_call (which errors), then still respond
            result.expect.contains_function_call(name="end_call")
            await assert_assistant_message(
                result,
                llm,
                "Acknowledges the request to end the call or provides a farewell, "
                "even if an error occurred internally.",
            )
