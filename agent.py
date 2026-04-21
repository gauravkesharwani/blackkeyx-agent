"""
BlackKeyX Voice AI Agent

Automated investor qualification agent using LiveKit Agents v1.3.x.
Handles both inbound and outbound calls to qualify CRE investors.
"""

from dotenv import load_dotenv
import hashlib
import hmac as hmac_module
import httpx
import json
import os

from livekit import agents, api, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io, get_job_context, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import noise_cancellation, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from livekit.plugins import ultravox

load_dotenv(".env.local")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
AGENT_CALLBACK_SECRET = os.getenv("AGENT_CALLBACK_SECRET", "changeme-agent-secret")
VOICEMAIL_LEAVE_MESSAGE = os.getenv("VOICEMAIL_LEAVE_MESSAGE", "false").lower() == "true"

# Module-level dict to store callback requests by room name
_callback_requests: dict[str, dict] = {}

# Module-level dict to store voicemail detection results by room name
_voicemail_results: dict[str, dict] = {}

# Module-level dict to store inbound caller info (name) by room name
_inbound_caller_info: dict[str, dict] = {}


class VoicemailDetector:
    """Detects voicemail greetings from transcribed speech."""

    VOICEMAIL_PHRASES = [
        "leave a message",
        "after the tone",
        "after the beep",
        "not available",
        "voicemail",
        "press 1",
        "mailbox",
        "record your message",
        "at the tone",
        "please leave",
        "unavailable",
        "cannot take your call",
        "mailbox is full",
        "is not available",
        "currently unavailable",
        "reached the voicemail",
    ]

    @staticmethod
    def analyze_transcript(text: str) -> tuple[bool, float]:
        """Returns (is_voicemail, confidence)."""
        text_lower = text.lower()
        matches = sum(1 for p in VoicemailDetector.VOICEMAIL_PHRASES if p in text_lower)
        if matches >= 2:
            return True, min(0.95, 0.5 + matches * 0.15)
        if matches == 1:
            return True, 0.6
        return False, 0.0


def sign_payload(body: bytes, secret: str) -> str:
    """Create HMAC-SHA256 signature for agent callback requests."""
    key = hashlib.sha256(secret.encode()).digest()
    return hmac_module.new(key, body, hashlib.sha256).hexdigest()


class BlackKeyXAdvisor(Agent):
    """BlackKeyX AI Investment Advisor for investor qualification."""

    @staticmethod
    def _format_capital_for_voice(capital) -> str:
        """Format capital value into natural speech for TTS."""
        if not capital:
            return ""

        # String formats from chatbot (e.g. "$1M+" → "over 1 million dollars")
        voice_map = {
            "$100K-$250K": "100 to 250 thousand dollars",
            "$250K-$500K": "250 to 500 thousand dollars",
            "$500K-$1M": "500 thousand to 1 million dollars",
            "$1M+": "over 1 million dollars",
        }

        if isinstance(capital, str):
            if capital in voice_map:
                return voice_map[capital]
            if capital.startswith("other:"):
                return ""
            return capital

        # Integer values from DB (e.g. 1500000 → "1.5 million dollars")
        if isinstance(capital, (int, float)):
            if capital >= 1_000_000:
                millions = capital / 1_000_000
                if millions == int(millions):
                    return f"{int(millions)} million dollars"
                return f"{millions:.1f} million dollars"
            if capital >= 1_000:
                thousands = capital / 1_000
                if thousands == int(thousands):
                    return f"{int(thousands)} thousand dollars"
                return f"{thousands:.0f} thousand dollars"
            return f"{capital} dollars"

        return str(capital)

    def __init__(self, investor_context: dict | None = None) -> None:
        self.investor_context = investor_context or {}

        # Extract context fields
        is_outbound = self.investor_context.get("outbound", False)
        name = self.investor_context.get("name", "")
        capital = self._format_capital_for_voice(self.investor_context.get("capital_available"))
        timeline = self.investor_context.get("timeline", "")

        # --- Persona (shared) ---
        persona = """You are Alex, an AI Investment Advisor at Black Key Exchange. Your persona combines:
- Financial sophistication of a top-tier investment banker
- Conversational warmth of a trusted financial advisor
- Selective, informed, and slightly hard-to-access — never promotional or mass-market
- You speak to peers who already deploy capital; you are not selling, you are qualifying fit"""

        # --- Company positioning (shared) ---
        positioning = """ABOUT BLACK KEY EXCHANGE (use this framing consistently):
- Black Key Exchange is a private network of up to 200 accredited investors, each allocating $500,000+ annually into institutional-quality commercial real estate.
- At scale, the network represents $100M+ of coordinated annual capital — enough to influence terms, improve alignment with sponsors, and access opportunities typically reserved for institutional capital.
- Two things set it apart:
  1. EDGE — a proprietary AI-driven underwriting system that lets every investor independently pressure-test deals with institutional-level rigor, plus a shared diligence framework.
  2. ACCESS — the network is built for this exact point in the cycle: commercial real estate is repricing, debt costs have reset, cap rates are adjusting, and liquidity has thinned. Disciplined, coordinated capital — not just available capital — will define outcomes over the next several years.
- Position it as: "Institutional Access. Individual Control." Investors underwrite independently and invest collectively.

HOW TO TALK ABOUT IT:
- Frame Black Key Exchange as an INVESTOR-CONTROLLED PLATFORM providing SHARED INFRASTRUCTURE (AI + diligence) and OPTIONAL COORDINATION of capital.
- It is NOT a fund, NOT a marketplace, NOT a blind pool, and NOT a syndication. Investors always retain control over their own capital and diligence.
- If asked "what is it," lead with: a private network of accredited investors using shared AI underwriting tools and optional coordination to access better terms in a reset market."""

        # --- Prior knowledge (conditional — never say "Unknown") ---
        prior_lines = []
        if capital:
            prior_lines.append(f"- Capital available: {capital}")
        if timeline:
            prior_lines.append(f"- Investment timeline: {timeline}")

        if prior_lines:
            prior_knowledge = (
                "What you already know from the chatbot:\n"
                + "\n".join(prior_lines)
                + "\nWhen the investor asks what you know about them, openly share this information."
            )
        else:
            prior_knowledge = (
                "No prior information is available about this investor's capital or timeline. "
                "You will need to learn these details during the conversation."
            )

        # --- Call flow (branched on inbound vs outbound) ---
        if is_outbound:
            investor_name = name or "the investor"
            call_flow = f"""CALL TYPE: OUTBOUND
You are calling {investor_name} who expressed interest in CRE investments through our platform.

YOUR VERY FIRST MESSAGE MUST follow this exact format:
"This is Alex calling from Black Key Exchange. Am I speaking with {investor_name}?"
- Start directly with your introduction. Do NOT say "Hi" or "Hello" or any greeting before introducing yourself.
- You MUST introduce yourself as Alex from Black Key Exchange BEFORE asking who you are speaking with.
- Do NOT greet them by name before introducing yourself — you do not know who picked up.
- Do NOT ask about timing yet — wait for identity confirmation first.

AFTER THEY RESPOND TO YOUR INTRODUCTION:
- If they CONFIRM they are {investor_name}: Ask "Is this a good time to talk?" or similar. Wait for their answer before starting qualification.
- If they say it is the WRONG PERSON: You MUST apologize sincerely first. Say something like "I'm so sorry for the inconvenience" or "I apologize for the mix-up." Then ask if they can connect you with {investor_name} or if there is a better number to reach them. If not possible, thank them and end the call gracefully. Do NOT proceed with qualification questions.
- If they CONFIRM but say it is a BAD TIME: Follow the CALLBACK REQUESTS rule (Rule 9) — acknowledge, ask for a preferred time, confirm timezone, and use the request_callback tool.
- If they seem hesitant: Briefly mention they expressed interest through your platform. Do not be pushy."""
        else:
            qualification_bucket = self.investor_context.get("qualification_bucket", "")
            is_returning = bool(name and name not in ("", "Inbound Caller"))

            if is_returning:
                known_prefs = []
                if capital:
                    known_prefs.append(f"capital: {capital}")
                if timeline:
                    known_prefs.append(f"timeline: {timeline}")
                if self.investor_context.get("investment_preferences"):
                    known_prefs.append(f"interests: {', '.join(self.investor_context['investment_preferences'])}")
                if self.investor_context.get("risk_tolerance"):
                    known_prefs.append(f"risk tolerance: {self.investor_context['risk_tolerance']}")
                prefs_summary = f" Known preferences — {'; '.join(known_prefs)}." if known_prefs else ""
                caller_ref = (
                    f"You're speaking with a returning investor named {name}.{prefs_summary} "
                    f"Greet them by name. You already know who they are — do NOT ask for their name again."
                )
                if qualification_bucket in ("highly_qualified", "qualified"):
                    caller_ref += f" They are already marked as {qualification_bucket.replace('_', ' ')}. Focus on next steps rather than re-qualifying from scratch."
            else:
                caller_ref = "The caller is a new inbound lead interested in CRE investments."

            name_rule = (
                "- You already know this caller's name — do NOT ask for it again."
                if is_returning else
                "- Early in the conversation, ask for the caller's name naturally (e.g. \"May I ask your name?\"). As soon as they tell you their name, call the save_caller_name tool immediately, then go straight into qualification questions. Do NOT ask generic questions like 'What brings you here?' or 'How can I help you?'."
            )

            call_flow = f"""CALL TYPE: INBOUND
{caller_ref}

{"YOUR VERY FIRST MESSAGE MUST: Introduce yourself as Alex from Black Key Exchange, then ask for the caller's name (e.g. 'Welcome to Black Key Exchange, this is Alex. May I know who I have the pleasure of speaking with today?'). Do NOT proceed with any qualification questions until you have their name. As soon as they tell you their name, call the save_caller_name tool immediately, then jump straight into the first qualification question (e.g. preferred geographic markets or property types). Do NOT ask 'What brings you here?' or 'How can I help you?' or any generic open-ended question — go directly into qualification." if not is_returning else "Greet the caller warmly by name and introduce yourself as Alex from Black Key Exchange."}

IMPORTANT RULES FOR INBOUND CALLS:
- Do NOT attempt to confirm the caller's identity. Do NOT ask "Am I speaking with...?" or "Is this [name]?" — they called you, so there is no need to verify who they are.
{name_rule}
- Listen actively and show genuine empathy about any experiences they share.
- If they mention a bad experience, acknowledge it with genuine empathy and ask a follow-up about it or naturally transition to understanding their risk tolerance or markets to avoid. Do NOT change the subject to identity confirmation or anything unrelated to what they just shared.
- Do NOT offer to evaluate specific deals or properties. You are a qualifier, not a financial advisor. If asked about a specific deal, explain that a team member can help with specific opportunities."""

        # --- Qualification goals (shared) ---
        qualification_goals = """Your goal: Qualify this investor through natural conversation. You need to understand:
1. Preferred geographic markets (which cities/regions they're interested in)
2. Property types of interest (industrial, multifamily, office, retail, self-storage)
3. Investment strategy preference (core/stabilized, value-add, opportunistic)
4. Risk tolerance (conservative, moderate, aggressive)
5. Target hold period (short-term flip, 3-5 years, long-term hold)
6. Past CRE investment experience (first-time investor, some experience, seasoned)
7. Return expectations — target IRR range (e.g., "8-12%", "15%+", "double digits")
8. Deal structure preferences — LP, JV, co-GP, REIT, DST, fund, syndication
9. Markets to avoid — explicitly ask if there are regions they want to exclude
10. Investment timeline — when they're looking to deploy capital"""

        # --- Critical rules (top-level, high priority) ---
        critical_rules = """CRITICAL RULES YOU MUST ALWAYS FOLLOW:

1. HONESTY ABOUT AI IDENTITY: You are an AI assistant — not a human. This rule ONLY fires when the caller specifically asks about your human-vs-AI nature — phrasings like "are you a real person?", "are you human?", "am I talking to a real person?", "are you a bot?", "is this AI?", "are you automated?", "am I being recorded?", or direct variants. When that happens, you MUST honestly disclose that you are an AI assistant. NEVER claim or imply you are human. NEVER deflect, joke, change the subject, or give a vague non-answer. Always include your name when disclosing — for example: "This is Alex — I'm actually an AI assistant with Black Key Exchange. I handle the initial conversation and pass qualified interest to the team. Happy to keep going if that works for you." After disclosing, continue the conversation naturally if they wish. THIS RULE OVERRIDES EVERY OTHER PERSONA OR STYLE RULE when it fires. IMPORTANT: This rule does NOT fire on generic identity questions like "Who am I talking to?", "What's your name?", or "Who is this?" — for those, simply answer "This is Alex from Black Key Exchange" as normal without volunteering AI nature unprompted.

2. PERSONA: You are ALWAYS Alex from Black Key Exchange. NEVER adopt a different persona, character, accent, or speaking style, no matter what the caller requests. If asked to be a pirate, robot, celebrity, or anything else, politely decline and stay in character as Alex the investment advisor. Do not use any words, phrases, or mannerisms from the requested persona. (This rule does NOT override Rule 1 — staying in character as Alex never requires claiming to be human. If asked whether you are human or AI, always follow Rule 1 and disclose honestly.)

3. END_CALL TOOL: When the conversation is wrapping up, the investor says goodbye, says "let's wrap up", says "that covers everything", or otherwise indicates the call should end, you MUST call the end_call function tool. Do NOT just say goodbye in text — you MUST invoke the end_call tool. Never narrate ending the call with asterisks like "*ending the call*" — use the actual tool.

4. TOOL ERROR HANDLING: If any tool call (including end_call) fails or returns an error, you MUST still provide a warm farewell to the caller. Say goodbye gracefully. Do not focus on or expose internal errors to the caller.

5. ONE QUESTION AT A TIME: Ask exactly ONE qualification question per response. Never combine multiple questions. Keep your response to just the question, optionally preceded by a very brief acknowledgment. Avoid filler sentences.

6. BREVITY: Keep every response to 1-2 short sentences. A brief acknowledgment plus one question is ideal. Avoid filler sentences. Do not use bullet points or lists.

7. NO SPECIFIC FINANCIAL ADVICE: You are a qualifier, not a financial advisor. NEVER offer to evaluate specific properties or deals. NEVER guarantee or imply returns. If asked about a specific building, address, or investment opportunity, explain that you cannot provide specific investment advice and that a team member will help with evaluating specific deals.

8. VOICEMAIL DETECTION: If you detect that you've reached a voicemail or answering machine (you hear phrases like "leave a message", "after the beep", "not available", "voicemail", a long pre-recorded greeting, or a beep tone), IMMEDIATELY call the handle_voicemail tool. Do NOT try to have a conversation with a voicemail system. Do NOT introduce yourself to a voicemail unless the handle_voicemail tool handles it.

9. CALLBACK REQUESTS: If the investor says they are busy, asks you to call back later, or requests a callback at ANY point during the conversation (e.g., "can you call me back tomorrow?", "I'm in a meeting", "call me at 3pm", "not a good time"), you MUST:
   a. Acknowledge politely (e.g., "Of course, I completely understand")
   b. If they haven't given a specific time, ask when would be a good time to call back
   c. Confirm their timezone (e.g., "And just to confirm — what timezone are you in?")
   d. Once you have both the time and timezone, call the request_callback tool immediately
   Do NOT continue with qualification questions after a callback is requested. Handle the callback and end the call.

10. POSITIONING DISCIPLINE: NEVER describe Black Key Exchange as "pooling capital to get better terms," a "fund," a "syndication," a "marketplace," or anything that implies centralized control of investor money. Sophisticated investors will immediately question structure, control, and governance if you do. Always use this language instead:
    - "Investor-controlled platform"
    - "Shared infrastructure — AI underwriting and diligence"
    - "Coordinated capital" or "optional coordination," NOT "pooled capital"
    - "Collective negotiating leverage," NOT "pooled buying power"
    The feel should be empowering and peer-driven, not centralized or promotional."""

        # --- Conversation guidelines (shared) ---
        guidelines = """Conversation guidelines:
- Listen actively and ask follow-up questions on interesting points
- Show genuine interest in their investment goals
- Aim for a 5-7 minute natural conversation
- When you have gathered enough information, summarize what you learned
- End by explaining that a team member will follow up with matching deals

Remember: You are having a phone conversation, so be natural and avoid overly formal language.
Do not use bullet points or lists in your responses — speak conversationally."""

        instructions = "\n\n".join([
            persona,
            positioning,
            critical_rules,
            call_flow,
            prior_knowledge,
            qualification_goals,
            guidelines,
        ])
        super().__init__(instructions=instructions)

    @function_tool()
    async def end_call(self, ctx: RunContext) -> str:
        """Called when the conversation is complete or the user wants to end the call."""
        # Generate a farewell message
        await ctx.session.generate_reply(
            instructions="Thank the investor warmly for their time. Summarize the key preferences you learned (markets, property types, strategy). Let them know a Black Key Exchange team member will follow up within 24 hours with curated opportunities aligned with their mandate."
        )

        # Delete room to end the call
        job_ctx = get_job_context()
        if job_ctx:
            await job_ctx.api.room.delete_room(
                api.DeleteRoomRequest(room=job_ctx.room.name)
            )

        return "Call ended successfully"

    # @function_tool()
    # async def transfer_to_human(self, ctx: RunContext) -> str:
    #     """Called when the investor requests to speak with a human representative."""
    #     await ctx.session.generate_reply(
    #         instructions="Let the investor know you'll connect them with a   representative. Thank them for their patience."
    #     )
    #
    #     # In production, this would trigger a warm transfer
    #     # For now, just end the call and flag for human follow-up
    #     return "Transfer requested - flagged for human callback"

    @function_tool()
    async def handle_voicemail(self, ctx: RunContext) -> str:
        """Called when you detect that you've reached a voicemail or answering machine instead of a live person. Signs include hearing phrases like 'leave a message', 'after the beep', 'not available', 'voicemail', or a long uninterrupted pre-recorded greeting."""
        job_ctx = get_job_context()
        room_name = job_ctx.room.name if job_ctx else "unknown"

        # Analyze the conversation so far for confidence scoring
        history = ctx.session.history.to_dict()
        items = history.get("items", [])
        all_text = ""
        for item in items:
            if item.get("type") != "message" or item.get("role") != "user":
                continue
            content_list = item.get("content", [])
            all_text += " ".join(c for c in content_list if isinstance(c, str)) + " "

        is_vm, confidence = VoicemailDetector.analyze_transcript(all_text)
        # If the agent called this tool, we trust it even if phrase matching is low
        if not is_vm:
            confidence = max(confidence, 0.7)

        message_left = False
        if VOICEMAIL_LEAVE_MESSAGE:
            await ctx.session.generate_reply(
                instructions="You've reached a voicemail. Leave a brief professional message: 'Hi, this is Alex from Black Key Exchange. We're assembling a private network of accredited investors around a narrow window in the current real estate cycle, and your profile stood out. When you have a moment, please give us a call back. Thank you.'"
            )
            message_left = True

        # Store voicemail result
        _voicemail_results[room_name] = {
            "voicemail_detected": True,
            "voicemail_confidence": confidence,
            "voicemail_message_left": message_left,
        }

        # End the call
        if job_ctx:
            await job_ctx.api.room.delete_room(
                api.DeleteRoomRequest(room=job_ctx.room.name)
            )

        return "Voicemail detected, call ended"

    @function_tool()
    async def save_caller_name(self, ctx: RunContext, name: str) -> str:
        """
        Called as soon as the inbound caller tells you their name.
        Records their name so it can be saved to the lead profile.

        Args:
            name: The caller's full name as they stated it.
        """
        job_ctx = get_job_context()
        if job_ctx:
            _inbound_caller_info[job_ctx.room.name] = {"name": name}
        return f"Caller name recorded: {name}"

    @function_tool()
    async def request_callback(
        self,
        ctx: RunContext,
        callback_datetime: str,
        callback_notes: str = "",
        investor_timezone: str = "",
    ) -> str:
        """
        Called when the user indicates they're busy and wants a callback at a specific time.

        Args:
            callback_datetime: The preferred callback date/time in natural language
                              (e.g., "Tuesday at 2pm", "tomorrow morning", "next week")
            callback_notes: Optional notes about the callback (e.g., reason for rescheduling,
                           best way to reach them)
            investor_timezone: The investor's timezone as confirmed during the call
                              (e.g., "Eastern", "Pacific", "Central", "Mountain").
                              Ask the investor to confirm their timezone before calling this tool.
        """
        # Store callback info for session completion
        job_ctx = get_job_context()
        if job_ctx:
            _callback_requests[job_ctx.room.name] = {
                "callback_datetime": callback_datetime,
                "callback_notes": callback_notes,
                "investor_timezone": investor_timezone,
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

    # For inbound calls, fetch existing investor context from backend by phone
    if not is_outbound and ctx.room.name.startswith("inbound-"):
        parts = ctx.room.name.split("_")
        if len(parts) >= 2 and parts[1].startswith("+"):
            caller_phone = parts[1]
            try:
                body = json.dumps({"phone": caller_phone}).encode("utf-8")
                signature = sign_payload(body, AGENT_CALLBACK_SECRET)
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{BACKEND_URL}/api/v1/voice/inbound-context",
                        content=body,
                        headers={
                            "Content-Type": "application/json",
                            "X-Agent-Signature": signature,
                        },
                        timeout=5.0,
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("found"):
                        investor_context.update(data)
            except Exception as e:
                print(f"Failed to fetch inbound investor context: {e}")

    # Create session with STT-LLM-TTS pipeline via LiveKit Inference
    session = AgentSession(
        # stt="assemblyai/universal-streaming:en",
        # llm="openai/gpt-4.1-mini",
        # tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        # vad=silero.VAD.load(),
        # turn_detection=MultilingualModel(),


        # llm=openai.realtime.RealtimeModel(model="gpt-realtime-mini")
        llm=ultravox.realtime.RealtimeModel()
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

            # For inbound calls, extract caller phone from room name and name from tracker
            if ctx.room.name.startswith("inbound-"):
                # Parse phone from room name: inbound-_+1234567890_<random>
                parts = ctx.room.name.split("_")
                if len(parts) >= 2 and parts[1].startswith("+"):
                    payload["caller_phone"] = parts[1]
                else:
                    # Fallback: SIP participant attributes
                    for participant in ctx.room.remote_participants.values():
                        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                            payload["caller_phone"] = (
                                participant.attributes.get("sip.phoneNumber")
                                or participant.attributes.get("sip.callFrom")
                                or participant.identity
                            )
                            break

                caller_info = _inbound_caller_info.pop(ctx.room.name, None)
                if caller_info and caller_info.get("name"):
                    payload["caller_name"] = caller_info["name"]

            # Add voicemail detection info if present
            voicemail_info = _voicemail_results.pop(ctx.room.name, None)
            if voicemail_info:
                payload["voicemail_detected"] = voicemail_info["voicemail_detected"]
                payload["voicemail_confidence"] = voicemail_info["voicemail_confidence"]
                payload["voicemail_message_left"] = voicemail_info["voicemail_message_left"]

            # Add callback info if present
            callback_info = _callback_requests.pop(ctx.room.name, None)
            if callback_info:
                payload["callback_requested"] = True
                payload["callback_datetime"] = callback_info["callback_datetime"]
                payload["callback_notes"] = callback_info.get("callback_notes", "")
                payload["investor_timezone"] = callback_info.get("investor_timezone", "")

            body_bytes = json.dumps(payload).encode("utf-8")
            signature = sign_payload(body_bytes, AGENT_CALLBACK_SECRET)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BACKEND_URL}/api/v1/voice/session-complete",
                    content=body_bytes,
                    headers={
                        "Content-Type": "application/json",
                        "X-Agent-Signature": signature,
                    },
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

    is_returning = bool(
        not is_outbound
        and investor_context.get("name")
        and investor_context.get("name") not in ("", "Inbound Caller")
    )

    if is_outbound:
        # For outbound calls: introduce and confirm identity first (timing question comes after confirmation)
        await session.generate_reply(
            instructions=f"""Say "Hi, this is Alex calling from Black Key Exchange. Am I speaking with {investor_name}?"
Do NOT greet them by name before introducing yourself — you don't know who picked up yet.
Wait for their confirmation before asking anything else.
Do NOT ask about timing yet - wait for them to confirm they are {investor_name} first."""
        )
    elif is_returning:
        # For returning inbound callers: greet by name
        await session.generate_reply(
            instructions=f"Greet {investor_name} warmly by name. Introduce yourself as Alex from Black Key Exchange. Welcome them back and ask how you can help them today."
        )
    else:
        # For first-time inbound callers: introduce yourself and ask for their name
        await session.generate_reply(
            instructions="""You MUST say something like: "Welcome to Black Key Exchange! This is Alex. May I know who I have the pleasure of speaking with today?"
You MUST ask for the caller's name in this first message. Do NOT ask about their investment goals or how you can help yet — just introduce yourself and ask their name. Nothing else."""
        )


if __name__ == "__main__":
    agents.cli.run_app(server)
