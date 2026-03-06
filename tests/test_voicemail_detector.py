"""Unit tests for VoicemailDetector.analyze_transcript() — Test IDs V-1 through V-7."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import VoicemailDetector


class TestVoicemailDetectorMultiplePhrases:
    """V-1: Two or more VM phrases → high confidence detection."""

    def test_classic_voicemail_greeting(self):
        text = "Hi, you've reached the voicemail of John. Please leave a message after the beep."
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is True
        assert confidence >= 0.8

    def test_carrier_voicemail(self):
        text = "The person you are trying to reach is not available. Please leave a message after the tone."
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is True
        assert confidence >= 0.8

    def test_three_phrase_match(self):
        text = "You've reached the voicemail box. I'm unavailable right now. Please leave a message after the beep."
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is True
        assert confidence >= 0.8

    def test_confidence_capped_at_095(self):
        text = (
            "voicemail mailbox leave a message after the tone after the beep "
            "not available unavailable please leave record your message"
        )
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is True
        assert confidence <= 0.95


class TestVoicemailDetectorSinglePhrase:
    """V-2: Single VM phrase → detection at 0.6 confidence."""

    def test_not_available(self):
        text = "I'm not available right now."
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is True
        assert confidence == 0.6

    def test_leave_a_message(self):
        # "Please leave a message" matches both "leave a message" and "please leave" → 2 matches
        text = "Please leave a message."
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is True
        assert confidence >= 0.6

    def test_unavailable(self):
        # "currently unavailable" matches both "unavailable" and "currently unavailable" → 2 matches
        text = "The subscriber you have called is currently unavailable."
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is True
        assert confidence >= 0.6

    def test_cannot_take_your_call(self):
        text = "Sorry, I cannot take your call right now."
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is True
        assert confidence == 0.6


class TestVoicemailDetectorNoMatch:
    """V-3: No VM phrases → not detected, confidence 0.0."""

    def test_normal_greeting(self):
        text = "Hello, who is this?"
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is False
        assert confidence == 0.0

    def test_human_conversation(self):
        text = "Yes, this is John speaking. How can I help you?"
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is False
        assert confidence == 0.0

    def test_empty_string(self):
        is_vm, confidence = VoicemailDetector.analyze_transcript("")
        assert is_vm is False
        assert confidence == 0.0

    def test_general_busy_response(self):
        text = "I'm in a meeting right now, can you call back?"
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is False
        assert confidence == 0.0


class TestVoicemailDetectorCaseInsensitive:
    """V-4: Case insensitive detection."""

    def test_uppercase(self):
        text = "PLEASE LEAVE A MESSAGE AFTER THE BEEP"
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is True
        assert confidence >= 0.8

    def test_mixed_case(self):
        text = "Leave A Message after The Tone"
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is True
        assert confidence >= 0.8

    def test_single_phrase_uppercase(self):
        text = "VOICEMAIL"
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is True
        assert confidence == 0.6


class TestVoicemailDetectorMailboxFull:
    """V-5: Mailbox full detection."""

    def test_mailbox_is_full(self):
        text = "The mailbox is full and cannot accept any messages at this time."
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is True
        # "mailbox is full" and "mailbox" both match
        assert confidence >= 0.6

    def test_voicemail_box_full(self):
        text = "This voicemail box is full. Goodbye."
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is True
        assert confidence >= 0.6


class TestVoicemailDetectorLongGreeting:
    """V-6: Long pre-recorded greeting with single VM phrase."""

    def test_long_greeting_single_phrase(self):
        text = (
            "Thank you for calling Acme Corporation. Our offices are located at "
            "123 Main Street. Our hours are Monday through Friday, 9am to 5pm. "
            "For sales press 1, for support press 2. If you know your party's "
            "extension, please dial it now."
        )
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is True
        assert confidence == 0.6  # Only "press 1" matches


class TestVoicemailDetectorFalsePositiveEdge:
    """V-7: Ambiguous — human mentions 'voicemail' in conversation."""

    def test_human_mentions_voicemail(self):
        """When a human says 'voicemail', single-phrase match triggers at 0.6.
        This is a known false-positive edge case — the agent's judgment
        (handle_voicemail tool call) should be the final arbiter."""
        text = "Hey, I just got your voicemail from yesterday."
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is True
        assert confidence == 0.6

    def test_human_discusses_voicemail_system(self):
        text = "Yeah my voicemail has been acting up, I didn't get your message."
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is True
        assert confidence == 0.6

    def test_not_triggered_without_keyword(self):
        """Confirming that talking ABOUT phone issues without VM keywords doesn't trigger."""
        text = "I missed your call earlier, my phone was on silent."
        is_vm, confidence = VoicemailDetector.analyze_transcript(text)
        assert is_vm is False
        assert confidence == 0.0
