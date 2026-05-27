from langchain_core.messages import AIMessage

from superassist_plus.config import Settings
from superassist_plus.memory.service import MemoryService, MemoryWritePayload
from superassist_plus.memory.writer import MEMORY_WRITER_PROMPT, MemoryWriter
from superassist_plus.models import EdgeType, NodeType


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
        SUPERASSIST_PLUS_DATA_DIR=tmp_path,
        SUPERASSIST_PLUS_EMBEDDING_PROVIDER="hash",
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

    assert result["nodes"] == 1


def test_memory_writer_can_opt_into_llm_plan(tmp_path) -> None:
    service = make_service(tmp_path)
    payload = make_payload(service)
    model = JsonModel()
    writer = MemoryWriter(service, model, llm_enabled=True)

    result = writer.write(payload)

    assert result["nodes"] == 1
    assert "memory_context" in str(model.calls[0][0])


def test_memory_writer_prompt_uses_cognifold_update_plan() -> None:
    assert "UpdatePlan" in MEMORY_WRITER_PROMPT
    assert '"operations"' in MEMORY_WRITER_PROMPT
    assert "MERGE_NODES" in MEMORY_WRITER_PROMPT
    assert "grounded_in" in MEMORY_WRITER_PROMPT
    assert "GROUNDS (0.9)" in MEMORY_WRITER_PROMPT


def test_memory_writer_applies_operations_plan(tmp_path) -> None:
    service = make_service(tmp_path)
    payload = make_payload(service)
    writer = MemoryWriter(service, OperationsJsonModel(), llm_enabled=True)

    result = writer.write(payload)

    concepts = service.store.list_nodes("u", NodeType.CONCEPT)
    intents = service.store.list_nodes("u", NodeType.INTENT)
    edges = service.store.list_edges("u")
    assert result["nodes"] == 2
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
