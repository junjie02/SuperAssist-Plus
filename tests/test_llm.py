from superassist_plus.config import Settings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from superassist_plus.llm import MiniMaxCompatibleChatModel, OneSecondRetryChatModel, create_chat_model, is_minimax_model


def test_minimax_defaults_temperature_to_one() -> None:
    settings = Settings(
        SUPERASSIST_PLUS_MODEL="MiniMax-M2.7",
        SUPERASSIST_PLUS_API_KEY="secret",
        SUPERASSIST_PLUS_BASE_URL="https://api.minimaxi.com/v1",
    )

    model = create_chat_model(settings)

    assert model.temperature == 1.0
    assert isinstance(model, MiniMaxCompatibleChatModel)


def test_explicit_temperature_wins() -> None:
    settings = Settings(
        SUPERASSIST_PLUS_MODEL="MiniMax-M2.7",
        SUPERASSIST_PLUS_API_KEY="secret",
        SUPERASSIST_PLUS_BASE_URL="https://api.minimaxi.com/v1",
        SUPERASSIST_PLUS_TEMPERATURE=0.7,
    )

    model = create_chat_model(settings)

    assert model.temperature == 0.7


def test_minimax_detection_uses_model_or_base_url() -> None:
    assert is_minimax_model("MiniMax-M2.7")
    assert is_minimax_model("other-model", "https://api.minimaxi.com/v1")


def test_minimax_tool_binding_keeps_tools_and_adds_reasoning_split() -> None:
    model = MiniMaxCompatibleChatModel(
        model="MiniMax-M2.7",
        api_key="secret",
        base_url="https://api.minimaxi.com/v1",
        temperature=1.0,
    )
    bound = model.bind_tools(
        [{"type": "function", "function": {"name": "echo", "description": "Echo", "parameters": {"type": "object"}}}],
        tool_choice="auto",
    )

    payload = bound.bound._get_request_payload([HumanMessage(content="hello")], **bound.kwargs)

    assert payload["tools"]
    assert payload["tool_choice"] == "auto"
    assert payload["extra_body"]["reasoning_split"] is True


def test_minimax_payload_uses_max_tokens_for_compatibility() -> None:
    model = MiniMaxCompatibleChatModel(
        model="MiniMax-M2.7",
        api_key="secret",
        base_url="https://api.minimaxi.com/v1",
        temperature=1.0,
        max_tokens=128,
    )

    payload = model._get_request_payload([HumanMessage(content="hello")])

    assert payload["max_tokens"] == 128
    assert "max_completion_tokens" not in payload


def test_minimax_payload_strips_message_names_for_compatibility() -> None:
    model = MiniMaxCompatibleChatModel(
        model="MiniMax-M2.7",
        api_key="secret",
        base_url="https://api.minimaxi.com/v1",
        temperature=1.0,
    )

    payload = model._get_request_payload([HumanMessage(content="summary", name="summary")])

    assert payload["messages"] == [{"content": "summary", "role": "user"}]


def test_minimax_reasoning_details_are_preserved() -> None:
    model = MiniMaxCompatibleChatModel(
        model="MiniMax-M2.7",
        api_key="secret",
        base_url="https://api.minimaxi.com/v1",
        temperature=1.0,
    )
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "<think>reason</think>answer",
                    "reasoning_details": [{"text": "split reason"}],
                },
                "finish_reason": "stop",
            }
        ],
        "model": "MiniMax-M2.7",
    }

    result = model._create_chat_result(response)
    message = result.generations[0].message

    assert message.content == "answer"
    assert message.additional_kwargs["reasoning_content"] == "split reason\n\nreason"


def test_openai_compatible_model_retries_once_after_one_second(monkeypatch) -> None:
    calls = []
    sleeps = []

    def fake_parent_generate(self, messages, stop=None, run_manager=None, **kwargs):
        calls.append("call")
        if len(calls) == 1:
            raise RuntimeError("temporary")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    monkeypatch.setattr("superassist_plus.llm.ChatOpenAI._generate", fake_parent_generate)
    monkeypatch.setattr("superassist_plus.llm.time.sleep", sleeps.append)
    model = OneSecondRetryChatModel(model="gpt-test", api_key="secret")

    result = model._generate([HumanMessage(content="hello")])

    assert len(calls) == 2
    assert sleeps == [1]
    assert result.generations[0].message.content == "ok"
