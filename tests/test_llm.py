import json

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from superassist.config import Settings
from superassist.llm import (
    MiniMaxCompatibleChatModel,
    OneSecondRetryChatModel,
    create_chat_model,
    create_memory_model,
    is_minimax_model,
)


def test_minimax_defaults_temperature_to_one() -> None:
    settings = Settings(
        SUPERASSIST_MODEL="MiniMax-M2.7",
        SUPERASSIST_API_KEY="secret",
        SUPERASSIST_BASE_URL="https://api.minimaxi.com/v1",
    )

    model = create_chat_model(settings)

    assert model.temperature == 1.0
    assert isinstance(model, MiniMaxCompatibleChatModel)


def test_explicit_temperature_wins() -> None:
    settings = Settings(
        SUPERASSIST_MODEL="MiniMax-M2.7",
        SUPERASSIST_API_KEY="secret",
        SUPERASSIST_BASE_URL="https://api.minimaxi.com/v1",
        SUPERASSIST_TEMPERATURE=0.7,
    )

    model = create_chat_model(settings)

    assert model.temperature == 0.7


def test_gpt_5_6_uses_responses_api_with_reasoning_summary() -> None:
    settings = Settings(
        SUPERASSIST_MODEL="gpt-5.6-sol",
        SUPERASSIST_API_KEY="secret",
        SUPERASSIST_REASONING_EFFORT="high",
    )

    model = create_chat_model(settings)
    payload = model._get_request_payload([HumanMessage(content="hello")])

    assert model.use_responses_api is True
    assert model.use_previous_response_id is False
    assert payload["reasoning"] == {"effort": "high", "summary": "detailed"}
    assert "reasoning_effort" not in payload


def test_gpt_5_6_none_effort_does_not_request_reasoning_summary() -> None:
    settings = Settings(
        SUPERASSIST_MODEL="gpt-5.6-sol",
        SUPERASSIST_API_KEY="secret",
        SUPERASSIST_REASONING_EFFORT="none",
    )

    model = create_chat_model(settings)
    payload = model._get_request_payload([HumanMessage(content="hello")])

    assert payload["reasoning"] == {"effort": "none"}


def test_gpt_5_6_multimodal_message_becomes_responses_image_input() -> None:
    settings = Settings(
        SUPERASSIST_MODEL="gpt-5.6-sol",
        SUPERASSIST_API_KEY="secret",
    )
    model = create_chat_model(settings)
    content = [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,aW1hZ2U="},
        },
        {"type": "text", "text": "请讲解这道题"},
    ]

    payload = model._get_request_payload([HumanMessage(content=content)])

    user_input = payload["input"][0]
    assert user_input["role"] == "user"
    assert user_input["content"][0] == {
        "type": "input_image",
        "image_url": "data:image/jpeg;base64,aW1hZ2U=",
    }
    assert user_input["content"][1] == {"type": "input_text", "text": "请讲解这道题"}


def test_gpt_5_6_tool_followup_payload_still_contains_original_image() -> None:
    settings = Settings(
        SUPERASSIST_MODEL="gpt-5.6-sol",
        SUPERASSIST_API_KEY="secret",
    )
    model = create_chat_model(settings)
    image_url = "data:image/png;base64,aW1hZ2U="
    messages = [
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "请按 skill 分析图片"},
            ]
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"path": "/mnt/skills/public/example/SKILL.md"},
                    "id": "call_skill",
                }
            ],
        ),
        ToolMessage(
            content="# Example Skill\nFollow these steps.",
            tool_call_id="call_skill",
            name="read_file",
        ),
    ]

    payload = model._get_request_payload(messages)

    image_parts = [
        part
        for item in payload["input"]
        if isinstance(item, dict) and isinstance(item.get("content"), list)
        for part in item["content"]
        if isinstance(part, dict) and part.get("type") == "input_image"
    ]
    assert image_parts == [{"type": "input_image", "image_url": image_url}]
    assert any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in payload["input"]
    )


def test_memory_model_uses_independent_openai_compatible_settings() -> None:
    settings = Settings(
        SUPERASSIST_MODEL="gpt-5.6-sol",
        SUPERASSIST_API_KEY="main-secret",
        SUPERASSIST_BASE_URL="https://main.example/v1",
        SUPERASSIST_MEMORY_MODEL="deepseek-v4-flash",
        SUPERASSIST_MEMORY_API_KEY="memory-secret",
        SUPERASSIST_MEMORY_BASE_URL="https://memory.example/v1",
    )

    model = create_memory_model(settings)
    payload = model._get_request_payload([HumanMessage(content="update memory")])

    assert isinstance(model, OneSecondRetryChatModel)
    assert model.model_name == "deepseek-v4-flash"
    assert str(model.openai_api_base) == "https://memory.example/v1"
    assert model.openai_api_key.get_secret_value() == "memory-secret"
    assert payload["model"] == "deepseek-v4-flash"
    assert "reasoning" not in payload
    assert "reasoning_effort" not in payload


def test_model_input_payload_is_written_to_jsonl_log(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_MODEL="gpt-5.6-sol",
        SUPERASSIST_API_KEY="secret",
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_MODEL_INPUT_LOG_ENABLED=True,
    )

    model = create_chat_model(settings)
    payload = model._get_request_payload(
        [
            SystemMessage(content="stable system"),
            SystemMessage(content="<ShortMemory>summary</ShortMemory>"),
            SystemMessage(content="<TurnContext><LongTermMemory>{}</LongTermMemory></TurnContext>"),
            HumanMessage(content="inspect this input"),
        ]
    )

    log_path = tmp_path / "logs" / "model-input.jsonl"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["payload"] == payload
    assert records[0]["call_kind"] == "lead_agent"
    assert records[0]["estimated_input_tokens"] > 0
    components = records[0]["input_manifest"]["component_tokens"]
    assert components["static_system"] > 0
    assert components["short_memory"] > 0
    assert components["turn_context"] > 0
    assert components["current_user"] > 0
    sections = records[0]["input_manifest"]["sections"]
    assert sections["long_term_memory"] > 0
    assert sections["long_term_memory"] < sections["turn_context"]
    assert "secret" not in log_path.read_text(encoding="utf-8")


def test_memory_model_input_log_has_distinct_call_kind(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_API_KEY="main-secret",
        SUPERASSIST_MEMORY_MODEL="deepseek-v4-flash",
        SUPERASSIST_MEMORY_API_KEY="memory-secret",
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_MODEL_INPUT_LOG_ENABLED=True,
    )

    model = create_memory_model(settings, call_kind="memory_updater")
    model._get_request_payload([HumanMessage(content='<MemoryWriteInput format="json">{}</MemoryWriteInput>')])

    record = json.loads((tmp_path / "logs" / "model-input.jsonl").read_text(encoding="utf-8"))
    assert record["call_kind"] == "memory_updater"
    assert record["input_manifest"]["component_tokens"]["memory_write_input"] > 0


def test_model_input_log_omits_large_image_data_urls(tmp_path) -> None:
    settings = Settings(
        SUPERASSIST_MODEL="gpt-5.6-sol",
        SUPERASSIST_API_KEY="secret",
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_MODEL_INPUT_LOG_ENABLED=True,
    )
    model = create_chat_model(settings)
    image_url = "data:image/jpeg;base64," + ("A" * 1000)

    model._log_model_input({"input": [{"image_url": image_url}]})

    record = json.loads((tmp_path / "logs" / "model-input.jsonl").read_text(encoding="utf-8"))
    logged_image = record["payload"]["input"][0]["image_url"]
    assert logged_image["data_url_omitted"] is True
    assert logged_image["characters"] == len(image_url)
    assert len(logged_image["sha256"]) == 64


def test_openai_compatible_stream_preserves_reasoning_content() -> None:
    model = OneSecondRetryChatModel(model="gpt-reasoning", api_key="secret")
    generation = model._convert_chunk_to_generation_chunk(
        {
            "choices": [
                {
                    "delta": {"role": "assistant", "content": "", "reasoning_content": "分析中"},
                    "finish_reason": None,
                }
            ]
        },
        AIMessageChunk,
        None,
    )

    assert generation is not None
    assert generation.message.additional_kwargs["reasoning_content"] == "分析中"


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

    monkeypatch.setattr("superassist.llm.ChatOpenAI._generate", fake_parent_generate)
    monkeypatch.setattr("superassist.llm.time.sleep", sleeps.append)
    model = OneSecondRetryChatModel(model="gpt-test", api_key="secret")

    result = model._generate([HumanMessage(content="hello")])

    assert len(calls) == 2
    assert sleeps == [1]
    assert result.generations[0].message.content == "ok"
