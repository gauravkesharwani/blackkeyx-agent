"""Shared fixtures for BlackKeyX voice agent tests."""

import sys
from pathlib import Path

# Add tests/ dir (for helpers) and agent root (for agent module) to sys.path
_tests_dir = str(Path(__file__).resolve().parent)
_agent_dir = str(Path(__file__).resolve().parent.parent)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

import pytest
import pytest_asyncio

from livekit.agents import AgentSession, mock_tools
from livekit.plugins import openai

from helpers import BlackKeyXAdvisor, TOOL_MOCKS


# ---------------------------------------------------------------------------
# Reusable investor contexts
# ---------------------------------------------------------------------------

SAMPLE_INBOUND_CONTEXT = {
    "name": "John Smith",
    "capital_available": "$2M",
    "timeline": "3-6 months",
}

SAMPLE_OUTBOUND_CONTEXT = {
    "name": "Sarah Johnson",
    "capital_available": "$5M",
    "timeline": "Immediately",
    "outbound": True,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def inbound_context():
    return SAMPLE_INBOUND_CONTEXT.copy()


@pytest.fixture
def outbound_context():
    return SAMPLE_OUTBOUND_CONTEXT.copy()


@pytest_asyncio.fixture
async def llm():
    """OpenAI text LLM for agent session and judge evaluations."""
    async with openai.LLM(model="gpt-4o-mini") as llm_instance:
        yield llm_instance


@pytest_asyncio.fixture
async def inbound_session(llm, inbound_context):
    """Agent session for inbound call scenario with all tools mocked."""
    async with AgentSession(llm=llm) as session:
        with mock_tools(BlackKeyXAdvisor, TOOL_MOCKS):
            await session.start(BlackKeyXAdvisor(investor_context=inbound_context))
            yield session


@pytest_asyncio.fixture
async def outbound_session(llm, outbound_context):
    """Agent session for outbound call scenario with all tools mocked."""
    async with AgentSession(llm=llm) as session:
        with mock_tools(BlackKeyXAdvisor, TOOL_MOCKS):
            await session.start(BlackKeyXAdvisor(investor_context=outbound_context))
            yield session
