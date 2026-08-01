import json

from langchain_core.messages import AIMessage

from superassist.config import Settings
from superassist.memory.service import MemoryService, MemoryWritePayload
from superassist.memory.writer import MEMORY_WRITER_PROMPT, MemoryWriter
from superassist.models import EdgeType, NodeType


class ExplodingModel:
    def invoke(self, *args, **kwargs):
        raise AssertionError("model should not be called")


class JsonModel:
    def __init__(self):
        self.calls = []

    def invoke(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return AIMessage(
            content='{"nodes":[{"ref":"c","type":"concept","title":"t","description":"durable memory"}],"edges":[]}'
        )


class OperationsJsonModel:
    def invoke(self, *args, **kwargs):
        return AIMessage(
            content="""
            {
              "reasoning": "Store durable response preference.",
              "operations": [
                {
                  "op": "ADD_NODE",
                  "node_type": "event",
                  "data": {
                    "ref": "current_event",
                    "title": "User preference conversation",
                    "description": "User expressed a preference for concise answers."
                  },
                  "reasoning": "This turn captured a user preference."
                },
                {
                  "op": "ADD_NODE",
                  "node_type": "concept",
                  "data": {
                    "ref": "concise_pref",
                    "title": "Prefers concise answers",
                    "description": "User prefers concise and direct answers.",
                    "importance": 0.7
                  },
                  "reasoning": "The user explicitly asked the assistant to remember this preference.",
                  "grounded_in": ["current_event"]
                },
                {
                  "op": "ADD_NODE",
                  "node_type": "intent",
                  "data": {
                    "ref": "answer_style_goal",
                    "title": "Answer style goal",
                    "description": "Keep future answers concise and direct."
                  },
                  "reasoning": "The preference implies a standing response-style goal.",
                  "grounded_in": ["concise_pref"]
                },
                {
                  "op": "ADD_EDGE",
                  "source_id": "concise_pref",
                  "target_id": "answer_style_goal",
                  "edge_type": "TRIGGERS",
                  "weight": 0.8
                }
              ],
              "symbolic_actions": []
            }
            """
        )


def make_service(tmp_path):
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
    )
    return MemoryService(settings=settings)


def make_payload(service: MemoryService) -> MemoryWritePayload:
    event_id, _ = service.prepare_turn("u", "t", "User likes concise direct answers.")
    return MemoryWritePayload(
        user_id="u",
        thread_id="t",
        event_id=event_id,
        user_message="User likes concise direct answers.",
        assistant_answer="Got it.",
        tool_events=[],
        memory_context={"immediate": [], "working": [], "background": [], "buffer": []},
    )


def test_memory_writer_is_deterministic_by_default(tmp_path) -> None:
    service = make_service(tmp_path)
    payload = make_payload(service)
    writer = MemoryWriter(service, ExplodingModel())

    result = writer.write(payload)

    # Fallback writer now creates 2 nodes: 1 event + 1 concept
    assert result["nodes"] == 2


def test_memory_writer_can_opt_into_llm_plan(tmp_path) -> None:
    service = make_service(tmp_path)
    payload = make_payload(service)
    model = JsonModel()
    writer = MemoryWriter(service, model, llm_enabled=True)

    result = writer.write(payload)

    assert result["nodes"] == 1
    assert "memory_context" in str(model.calls[0][0])


def test_memory_writer_llm_payload_omits_embeddings(tmp_path) -> None:
    service = make_service(tmp_path)
    event_id, _ = service.prepare_turn("u", "t", "Prepare an interview answer.")
    raw_context = {
        "immediate": [
            {
                "id": "node_1",
                "type": "concept",
                "title": "Interview prep",
                "description": "Useful context.",
                "user_id": "u",
                "importance": 0.7,
                "grounded_in": ["event_0"],
                "source": "test",
                "created_at": "2026-07-29T08:00:00+00:00",
                "updated_at": "2026-07-30T08:00:00+00:00",
                "access_count": 3,
                "embedding": [0.1, 0.2, 0.3],
                "reasoning": "internal",
                "metadata": {"thread_id": "t"},
            }
        ],
        "working": [],
        "background": [],
        "buffer": [],
    }
    assert "embedding" in str(raw_context)
    payload = MemoryWritePayload(
        user_id="u",
        thread_id="t",
        event_id=event_id,
        user_message="Prepare an interview answer.",
        user_message_created_at="2026-08-01T08:00:00+00:00",
        assistant_answer="Drafted.",
        assistant_message_created_at="2026-08-01T08:01:00+00:00",
        tool_events=[
            {"type": "tool_start", "tool": "task", "args": {"prompt": "private"}},
            {"type": "tool_result", "tool": "task", "status": "success", "content": "x" * 2000},
            {"type": "tool_result", "tool": "web_fetch", "status": "error", "content": "failed " + "y" * 1000},
        ],
        memory_context=raw_context,
    )
    model = JsonModel()
    writer = MemoryWriter(service, model, llm_enabled=True)

    writer.write(payload)

    human_message = model.calls[0][0][0][1]
    wrapped_input = human_message[1]
    sent_payload = json.loads(wrapped_input[wrapped_input.index("{") : wrapped_input.rindex("}") + 1])
    assert wrapped_input.startswith('<MemoryWriteInput format="json">')
    assert "embedding" not in wrapped_input
    assert "access_count" not in wrapped_input
    assert "reasoning" not in sent_payload["memory_context"]["immediate"][0]
    assert sent_payload["memory_context"]["immediate"][0] == {
        "tier": "",
        "id": "node_1",
        "type": "concept",
        "title": "Interview prep",
        "description": "Useful context.",
        "user_id": "u",
        "importance": 0.7,
        "grounded_in": ["event_0"],
        "source": "test",
        "created_at": "2026-07-29T08:00:00+00:00",
        "updated_at": "2026-07-30T08:00:00+00:00",
    }
    assert sent_payload["user_message"] == "Prepare an interview answer."
    assert sent_payload["user_message_created_at"] == "2026-08-01T08:00:00+00:00"
    assert sent_payload["assistant_message_created_at"] == "2026-08-01T08:01:00+00:00"
    assert sent_payload["tool_events"] == [
        {"name": "task", "status": "success", "error_summary": ""},
        {"name": "web_fetch", "status": "error", "error_summary": "failed " + "y" * 493 + "..."},
    ]
    assert "private" not in wrapped_input
    assert "x" * 100 not in wrapped_input


def test_memory_writer_prompt_uses_cognifold_update_plan() -> None:
    assert '"operations"' in MEMORY_WRITER_PROMPT
    assert "MERGE_NODES" in MEMORY_WRITER_PROMPT
    assert "grounded_in" in MEMORY_WRITER_PROMPT
    assert "GROUNDS" in MEMORY_WRITER_PROMPT
    assert "ADD_NODE" in MEMORY_WRITER_PROMPT
    assert '"operations":[]' in MEMORY_WRITER_PROMPT
    assert "user_message_created_at" in MEMORY_WRITER_PROMPT
    assert "OCCURRED_AT" in MEMORY_WRITER_PROMPT
    assert "普通对话不要创建 TIME" in MEMORY_WRITER_PROMPT


def test_memory_writer_fallback_returns_empty_plan_for_greeting(tmp_path) -> None:
    service = make_service(tmp_path)
    event_id, _ = service.prepare_turn("u", "t", "hello")
    payload = MemoryWritePayload(
        user_id="u",
        thread_id="t",
        event_id=event_id,
        user_message="hello",
        assistant_answer="Hello!",
        tool_events=[],
    )

    result = MemoryWriter(service, ExplodingModel()).write(payload)

    assert result["nodes"] == 0
    assert service.store.list_nodes("u") == []


def test_memory_writer_fallback_keeps_short_durable_preference(tmp_path) -> None:
    service = make_service(tmp_path)
    event_id, _ = service.prepare_turn("u", "t", "我喜欢蓝色")
    payload = MemoryWritePayload(
        user_id="u",
        thread_id="t",
        event_id=event_id,
        user_message="我喜欢蓝色",
        assistant_answer="记住了。",
        tool_events=[],
    )

    result = MemoryWriter(service, ExplodingModel()).write(payload)

    assert result["nodes"] == 2


def test_memory_writer_applies_operations_plan(tmp_path) -> None:
    service = make_service(tmp_path)
    payload = make_payload(service)
    writer = MemoryWriter(service, OperationsJsonModel(), llm_enabled=True)

    result = writer.write(payload)

    concepts = service.store.list_nodes("u", NodeType.CONCEPT)
    intents = service.store.list_nodes("u", NodeType.INTENT)
    edges = service.store.list_edges("u")
    # Now creates 3 nodes: 1 event + 1 concept + 1 intent
    assert result["nodes"] == 3
    assert concepts
    assert intents
    assert any(edge.target_id == concepts[0].id and edge.edge_type == EdgeType.GROUNDS for edge in edges)
    assert any(edge.source_id == concepts[0].id and edge.target_id == intents[0].id for edge in edges)


def test_memory_service_applies_update_and_merge_operations(tmp_path) -> None:
    service = make_service(tmp_path)
    payload = make_payload(service)
    first = service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Concise answers",
        description="User likes concise answers.",
        embedding=service.embed("User likes concise answers."),
        grounded_in=[payload.event_id],
    )
    second = service.store.add_node(
        user_id="u",
        node_type=NodeType.CONCEPT,
        title="Direct answers",
        description="User likes direct answers.",
        embedding=service.embed("User likes direct answers."),
        grounded_in=[payload.event_id],
    )
    plan = {
        "operations": [
            {
                "op": "UPDATE_NODE",
                "node_id": first.id,
                "data": {"importance": 0.8, "description": "User prefers concise direct answers."},
                "update_reasoning": "Explicitly reinforced by the current turn.",
            },
            {
                "op": "MERGE_NODES",
                "node_ids": [first.id, second.id],
                "merged_data": {
                    "title": "Prefers concise direct answers",
                    "description": "User prefers concise and direct answers.",
                },
                "reasoning": "The concepts are near-duplicates.",
            },
        ]
    }

    result = service.apply_structured_memory(payload, plan)

    concepts = service.store.list_nodes("u", NodeType.CONCEPT)
    assert result["updated"] == 1
    assert result["merged"] == 1
    assert len(concepts) == 1
    assert concepts[0].title == "Prefers concise direct answers"
