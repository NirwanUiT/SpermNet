"""Tests for llm/chatbot.py — initialization and structure (no API calls)."""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_chatbot_import():
    """Verify the module imports without error."""
    from llm.chatbot import SpermAnalysisChatbot
    assert SpermAnalysisChatbot is not None


def test_chatbot_init_builds_prompt():
    """Init should build a system prompt even if no data files exist."""
    from llm.chatbot import SpermAnalysisChatbot
    bot = SpermAnalysisChatbot("99999")  # non-existent video
    assert bot.turn_count == 0
    assert len(bot._system_prompt) > 100, "System prompt should be non-trivial"
    assert "sperm" in bot._system_prompt.lower()


def test_chatbot_reset():
    """Reset should clear conversation history."""
    from llm.chatbot import SpermAnalysisChatbot
    bot = SpermAnalysisChatbot("99999")
    # Manually add a fake user message
    bot.messages.append({"role": "user", "content": "test"})
    assert bot.turn_count == 1
    bot.reset()
    assert bot.turn_count == 0


def test_chatbot_load_video():
    """load_video should accept string or list."""
    from llm.chatbot import SpermAnalysisChatbot
    bot = SpermAnalysisChatbot("99999")
    # Loading a different non-existent video should not crash
    bot.load_video("88888")
    assert bot.turn_count == 0  # reset on load


@patch("llm.chatbot.openai")
def test_chatbot_chat_calls_openai(mock_openai):
    """chat() should call the OpenAI API with correct structure."""
    from llm.chatbot import SpermAnalysisChatbot

    # Mock the API response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response

    bot = SpermAnalysisChatbot("99999")
    result = bot.chat("Hello")

    assert result == "Test response"
    assert bot.turn_count == 1  # 1 user turn
