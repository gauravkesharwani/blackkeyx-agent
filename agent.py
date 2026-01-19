"""
BlackKeyX Voice AI Agent

Automated investor qualification agent using LiveKit Agents v1.3.x.
Handles both inbound and outbound calls to qualify CRE investors.
"""

from dotenv import load_dotenv
import httpx
import json
import os

from livekit import agents, api, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io, get_job_context, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Module-level dict to store callback requests by room name
_callback_requests: dict[str, dict] = {}


class BlackKeyXAdvisor(Agent):
    """BlackKeyX AI Investment Advisor for investor qualification."""

    def __init__(self, investor_context: dict | None = None) -> None:
        self.investor_context = investor_context or {}

        # Build dynamic instructions based on investor context
        name = self.investor_context.get("name", "there")
        capital = self.investor_context.get("capital_available", "Unknown")
        timeline = self.investor_context.get("timeline", "Unknown")

        instructions = f"""You are Sarah, an AI Investment Advisor at Black Key Exchange. Your persona combines:
- Financial sophistication of a top-tier investment banker
- Conversational warmth of a trusted financial advisor
- Professional yet personable tone

You're calling {name} who expressed interest in CRE investments through our platform.

What you already know from the chatbot:
- Capital available: {capital}
- Investment timeline: {timeline}

Your goal: Qualify this investor through natural conversation. You need to understand:
1. Preferred geographic markets (which cities/regions they're interested in)
2. Property types of interest (industrial, multifamily, office, retail, self-storage)
3. Investment strategy preference (core/stabilized, value-add, opportunistic)
4. Risk tolerance (conservative, moderate, aggressive)
5. Target hold period (short-term flip, 3-5 years, long-term hold)
6. Past CRE investment experience (first-time investor, some experience, seasoned)

Conversation guidelines:
- Keep responses conversational and concise (2-3 sentences max)
- Ask ONE question at a time - never multiple questions in one response
- Listen actively and ask follow-up questions on interesting points
- Show genuine interest in their investment goals
- Aim for a 5-7 minute natural conversation
- When you've gathered enough information, summarize what you learned
- End by explaining that a team member will follow up with matching deals

Handling outbound call scenarios:
- If they say it's the WRONG PERSON: Apologize politely, ask if they can connect you with {name} or if there's a better number to reach them. If not possible, thank them and end the call gracefully.
- If they say it's a BAD TIME or they're BUSY:
  1. Acknowledge their busy schedule politely (e.g., "I completely understand, I know you're busy")
  2. Ask when would be a good time to call back (e.g., "When would be a better time for us to reconnect?")
  3. Wait for them to give you a specific time (e.g., "Tuesday at 2pm", "tomorrow morning", "next week")
  4. Once they give you a time, use the request_callback tool with their preferred time
  5. The tool will confirm the callback and end the call automatically
- If they seem hesitant: Briefly explain that they expressed interest through our platform. Don't be pushy - respect their decision if they decline.
- If they want to proceed: Transition naturally into the qualification conversation.

Remember: You're having a phone conversation, so be natural and avoid overly formal language.
Don't use bullet points or lists in your responses - speak conversationally.
"""
        super().__init__(instructions=instructions)

    @function_tool()
    async def end_call(self, ctx: RunContext) -> str:
        """Called when the conversation is complete or the user wants to end the call."""
        # Generate a farewell message
        await ctx.session.generate_reply(
            instructions="Thank the investor warmly for their time. Summarize the key preferences you learned (markets, property types, strategy). Let them know a Black Key Exchange team member will follow up within 24 hours with deals that match their criteria."
        )

        # Delete room to end the call
        job_ctx = get_job_context()
        if job_ctx:
            await job_ctx.api.room.delete_room(
                api.DeleteRoomRequest(room=job_ctx.room.name)
            )

        return "Call ended successfully"

    @function_tool()
    async def transfer_to_human(self, ctx: RunContext) -> str:
        """Called when the investor requests to speak with a human representative."""
        await ctx.session.generate_reply(
            instructions="Let the investor know you'll connect them with a   representative. Thank them for their patience."
        )

        # In production, this would trigger a warm transfer
        # For now, just end the call and flag for human follow-up
        return "Transfer requested - flagged for human callback"

    @function_tool()
    async def request_callback(
        self,
        ctx: RunContext,
        callback_datetime: str,
        callback_notes: str = "",
    ) -> str:
        """
        Called when the user indicates they're busy and wants a callback at a specific time.

        Args:
            callback_datetime: The preferred callback date/time in natural language
                              (e.g., "Tuesday at 2pm", "tomorrow morning", "next week")
            callback_notes: Optional notes about the callback (e.g., reason for rescheduling,
                           best way to reach them)
        """
        # Store callback info for session completion
        job_ctx = get_job_context()
        if job_ctx:
            _callback_requests[job_ctx.room.name] = {
                "callback_datetime": callback_datetime,
                "callback_notes": callback_notes,
            }

        # Generate confirmation message
        await ctx.session.generate_reply(
            instructions=f"""Confirm the callback time with the investor.
They requested: {callback_datetime}.
Thank them warmly for their time and let them know the Black Key Exchange team will
reach out at their preferred time. Keep it brief and friendly."""
        )

        # End the call after confirmation
        if job_ctx:
            await job_ctx.api.room.delete_room(
                api.DeleteRoomRequest(room=job_ctx.room.name)
            )

        return f"Callback scheduled for {callback_datetime}"


server = AgentServer()


@server.rtc_session(agent_name="blackkeyx-advisor")
async def blackkeyx_agent(ctx: agents.JobContext):
    """Main entrypoint for the BlackKeyX voice agent."""

    # Parse investor context from job metadata (passed from backend)
    investor_context = {}
    if ctx.job.metadata:
        try:
            investor_context = json.loads(ctx.job.metadata)
        except json.JSONDecodeError:
            pass

    is_outbound = investor_context.get("outbound", False)

    # Create session with STT-LLM-TTS pipeline via LiveKit Inference
    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    def build_transcript(items: list) -> str:
        """Extract text transcript from chat history items."""
        parts = []
        for item in items:
            if item.get("type") != "message":
                continue
            role = item.get("role", "unknown")
            content_list = item.get("content", [])
            text_parts = [c for c in content_list if isinstance(c, str)]
            if text_parts:
                parts.append(f"{role}: {' '.join(text_parts)}")
        return "\n".join(parts)

    async def send_transcript():
        """Save transcript when session ends."""
        try:
            history = session.history.to_dict()
            items = history.get("items", [])
            transcript = build_transcript(items)

            payload = {
                "room_name": ctx.room.name,
                "transcript": transcript,
                "history": items,
            }

            # Add callback info if present
            callback_info = _callback_requests.pop(ctx.room.name, None)
            if callback_info:
                payload["callback_requested"] = True
                payload["callback_datetime"] = callback_info["callback_datetime"]
                payload["callback_notes"] = callback_info.get("callback_notes", "")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BACKEND_URL}/api/v1/voice/session-complete",
                    json=payload,
                    timeout=10.0,
                )
                if response.status_code == 200:
                    print(f"Transcript saved for room: {ctx.room.name}")
                else:
                    print(f"Failed to save transcript: {response.status_code}")
                    print(f"Response body: {response.text}")
        except Exception as e:
            print(f"Error saving transcript: {e}")

    ctx.add_shutdown_callback(send_transcript)

    await session.start(
        room=ctx.room,
        agent=BlackKeyXAdvisor(investor_context=investor_context),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # Use telephony-optimized noise cancellation for SIP calls
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    # Get investor name for personalization
    investor_name = investor_context.get("name", "there")

    if is_outbound:
        # For outbound calls: introduce, confirm identity, ask permission
        await session.generate_reply(
            instructions=f"""Introduce yourself warmly as Sarah calling from Black Key Exchange.
Confirm you're speaking with {investor_name} by asking if this is them.
Then politely ask if they have a few minutes to discuss commercial real estate investment opportunities.
Keep it brief and natural - you're making a phone call, not reading a script."""
        )
    else:
        # For inbound calls: greet and ask how to help
        await session.generate_reply(
            instructions="Greet the caller warmly. Introduce yourself as Sarah from Black Key Exchange. Ask how you can help them today with their commercial real estate investment goals."
        )


if __name__ == "__main__":
    agents.cli.run_app(server)
