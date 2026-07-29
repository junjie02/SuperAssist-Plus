import asyncio

from superassist.channels.store import WeComThreadStore
from superassist.channels.wecom_rpa import (
    VisualMessage,
    VisualRPAStateStore,
    VisualSnapshot,
    WeComRPAChannel,
    extract_triggered_prompt,
    split_reply,
)
from superassist.config import Settings


def _run(coro):
    return asyncio.run(coro)


def _settings(tmp_path, **overrides) -> Settings:
    values = {
        "SUPERASSIST_DATA_DIR": tmp_path,
        "SUPERASSIST_EMBEDDING_PROVIDER": "hash",
        "SUPERASSIST_WECOM_RPA_ALLOWED_GROUPS": "项目答疑群",
        "SUPERASSIST_WECOM_RPA_TRIGGER_PREFIXES": "@SuperAssist,小助手",
        "SUPERASSIST_WECOM_RPA_POLL_INTERVAL_SECONDS": 0.5,
    }
    values.update(overrides)
    return Settings(**values)


class FakeDriver:
    def __init__(self, snapshot: VisualSnapshot) -> None:
        self.snapshot = snapshot
        self.connected = False
        self.sent = []

    def connect(self) -> None:
        self.connected = True

    def read_snapshot(self) -> VisualSnapshot:
        return self.snapshot

    def send_text(self, expected_group: str, text: str) -> None:
        if self.snapshot.group_name != expected_group or not self.snapshot.is_external_group:
            raise RuntimeError("unsafe active chat")
        self.sent.append((expected_group, text))


class FakeEngineClient:
    def __init__(self) -> None:
        self.calls = []
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def stream_chat(self, **kwargs):
        self.calls.append(kwargs)
        yield {"type": "done", "answer": "群聊回答"}


def test_trigger_parser_and_reply_splitter() -> None:
    assert extract_triggered_prompt("@SuperAssist： Graph RAG 是什么？", ["@SuperAssist"]) == "Graph RAG 是什么？"
    assert extract_triggered_prompt("普通聊天", ["@SuperAssist"]) is None
    assert extract_triggered_prompt("@SuperAssist", ["@SuperAssist"]) is None
    assert split_reply("第一段。第二段。第三段。", 5) == ["第一段。", "第二段。", "第三段。"]


def test_visual_state_store_primes_and_persists_replay_guard(tmp_path) -> None:
    path = tmp_path / "state.json"
    message = VisualMessage("项目答疑群", "张三", "@SuperAssist 你好")
    store = VisualRPAStateStore(path)
    store.prime((message,))

    assert store.claim(message) is False
    assert VisualRPAStateStore(path).claim(message) is False


def test_rpa_channel_only_processes_allowed_external_group_with_wake_prefix(tmp_path) -> None:
    initial = VisualSnapshot("项目答疑群", True, ())
    driver = FakeDriver(initial)
    engine = FakeEngineClient()
    settings = _settings(tmp_path)
    channel = WeComRPAChannel(
        settings,
        driver=driver,
        engine_client=engine,
        thread_store=WeComThreadStore(tmp_path / "threads.json"),
        state_store=VisualRPAStateStore(tmp_path / "state.json"),
    )

    async def scenario() -> None:
        await channel.start()
        driver.snapshot = VisualSnapshot(
            "项目答疑群",
            True,
            (VisualMessage("", "张三", "@SuperAssist Graph RAG 是什么？"),),
        )
        await channel.poll_once()
        await channel.stop()

    _run(scenario())

    assert driver.connected is True
    assert engine.calls[0]["user_id"].startswith("wecom-rpa-group:")
    assert engine.calls[0]["message"] == "群成员 张三：Graph RAG 是什么？"
    assert engine.calls[0]["rag_mode"] is False
    assert driver.sent == [("项目答疑群", "群聊回答")]


def test_rpa_channel_hard_rejects_private_and_non_allowlisted_chats(tmp_path) -> None:
    driver = FakeDriver(VisualSnapshot("项目答疑群", True, ()))
    engine = FakeEngineClient()
    channel = WeComRPAChannel(
        _settings(tmp_path),
        driver=driver,
        engine_client=engine,
        thread_store=WeComThreadStore(tmp_path / "threads.json"),
        state_store=VisualRPAStateStore(tmp_path / "state.json"),
    )

    async def scenario() -> None:
        await channel.start()
        message = (VisualMessage("", "李四", "@SuperAssist 不应处理"),)
        driver.snapshot = VisualSnapshot("李四", False, message)
        await channel.poll_once()
        driver.snapshot = VisualSnapshot("其他群", True, message)
        await channel.poll_once()
        await channel.stop()

    _run(scenario())

    assert engine.calls == []
    assert driver.sent == []


def test_rpa_group_members_share_thread_and_rag_state(tmp_path) -> None:
    driver = FakeDriver(VisualSnapshot("项目答疑群", True, ()))
    engine = FakeEngineClient()
    channel = WeComRPAChannel(
        _settings(tmp_path),
        driver=driver,
        engine_client=engine,
        thread_store=WeComThreadStore(tmp_path / "threads.json"),
        state_store=VisualRPAStateStore(tmp_path / "state.json"),
    )

    async def scenario() -> None:
        await channel.start()
        driver.snapshot = VisualSnapshot(
            "项目答疑群",
            True,
            (VisualMessage("", "张三", "@SuperAssist /rag on"),),
        )
        await channel.poll_once()
        driver.snapshot = VisualSnapshot(
            "项目答疑群",
            True,
            (
                VisualMessage("", "张三", "@SuperAssist /rag on"),
                VisualMessage("", "李四", "小助手 根据资料回答"),
            ),
        )
        await channel.poll_once()
        await channel.stop()

    _run(scenario())

    assert driver.sent[0] == ("项目答疑群", "知识库 RAG 模式已开启。")
    assert engine.calls[0]["rag_mode"] is True
