from __future__ import annotations

import asyncio
from types import SimpleNamespace

from superassist.channels.feishu import FeishuCardImage, FeishuCardView, FeishuChannel, FeishuInboundMessage
from superassist.channels.feishu_documents import (
    FeishuDocumentPublisher,
    PublishedDocument,
    contains_math_formula,
    markdown_to_feishu_blocks,
)
from superassist.config import Settings


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _settings(tmp_path) -> Settings:
    return Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_API_KEY="",
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_FEISHU_APP_ID="",
        SUPERASSIST_FEISHU_APP_SECRET="",
    )


def _inbound() -> FeishuInboundMessage:
    return FeishuInboundMessage(
        chat_id="chat_1",
        message_id="msg_1",
        sender_open_id="ou_1",
        text="question",
        chat_type="group",
    )


def test_formula_detection_ignores_currency_and_code() -> None:
    assert contains_math_formula(r"由 \(x+1=2\) 可得答案")
    assert contains_math_formula(r"$$\frac{1}{2}$$")
    assert contains_math_formula(r"使用 \sqrt{x} 计算")
    assert not contains_math_formula("价格是 $100，优惠后 $80")
    assert not contains_math_formula("```python\nvalue = '$x^2$'\n```")
    assert not contains_math_formula("命令路径是 C:\\temp\\file.txt")


def test_markdown_conversion_creates_native_inline_equation() -> None:
    blocks = markdown_to_feishu_blocks("计算结果：$x^2 + y^2 = 1$。")

    elements = blocks[0].text.elements
    assert elements[0].text_run.content == "计算结果："
    assert elements[1].equation.content == "x^2 + y^2 = 1"
    assert elements[2].text_run.content == "。"

    unwrapped = markdown_to_feishu_blocks(r"使用 \sqrt{x} 计算。")
    assert unwrapped[0].text.elements[1].equation.content == r"\sqrt{x}"


def test_document_publisher_creates_blocks_and_grants_group_access() -> None:
    requests: dict[str, object] = {}

    class Response:
        code = 0
        msg = "ok"

        def __init__(self, data=None):
            self.data = data

        def success(self):
            return True

    class DocumentResource:
        def create(self, request):
            requests["document"] = request
            return Response(SimpleNamespace(document=SimpleNamespace(document_id="docx_1")))

    class ChildrenResource:
        def create(self, request):
            requests["children"] = request
            return Response()

    class PermissionResource:
        def create(self, request):
            requests["permission"] = request
            return Response()

    client = SimpleNamespace(
        docx=SimpleNamespace(
            v1=SimpleNamespace(
                document=DocumentResource(),
                document_block_children=ChildrenResource(),
            )
        ),
        drive=SimpleNamespace(v1=SimpleNamespace(permission_member=PermissionResource())),
    )
    publisher = FeishuDocumentPublisher(client, doc_url_base="https://tenant.feishu.cn/docx")

    result = publisher.publish(
        "答案是 $x=1$。",
        chat_id="chat_1",
        sender_open_id="ou_1",
        is_private=False,
        idempotency_key="msg_1",
    )

    assert result.url == "https://tenant.feishu.cn/docx/docx_1"
    permission = requests["permission"]
    assert permission.request_body.member_type == "openchat"
    assert permission.request_body.member_id == "chat_1"
    assert permission.request_body.perm == "view"
    assert requests["children"].request_body.children[0].text.elements[1].equation.content == "x=1"


def test_final_formula_routes_to_document_and_plain_link(tmp_path) -> None:
    async def go():
        channel = FeishuChannel(_settings(tmp_path))
        inbound = _inbound()
        channel._running_cards[inbound.message_id] = "temporary_card"
        calls: list[str] = []
        published_text = ""

        async def publish(_inbound, text):
            nonlocal published_text
            published_text = text
            calls.append("publish")
            return PublishedDocument("docx_1", "公式回答", "https://feishu.cn/docx/docx_1")

        async def send_link(_inbound, _document):
            calls.append("link")
            return "link_message"

        async def delete(message_id):
            calls.append(f"delete:{message_id}")

        async def send_image(_inbound, image_key):
            calls.append(f"image:{image_key}")
            return "image_message"

        async def unexpected(*_args, **_kwargs):
            raise AssertionError("formula answer must not be sent as a card")

        channel._publish_math_document = publish
        channel._send_document_link = send_link
        channel._delete_message_best_effort = delete
        channel._send_image_message = send_image
        channel._update_card = unexpected

        result = await channel._send_or_patch(
            inbound,
            FeishuCardView(
                answer="由 $$x=1$$ 得到答案。",
                images=(FeishuCardImage("示意图", "https://example.com/source", "img_1"),),
            ),
            final=True,
        )

        assert result == "link_message"
        assert calls == ["publish", "link", "delete:temporary_card", "image:img_1"]
        assert "[示意图](https://example.com/source)" in published_text
        assert inbound.message_id not in channel._running_cards

    _run(go())


def test_document_failure_falls_back_to_card(tmp_path) -> None:
    async def go():
        channel = FeishuChannel(_settings(tmp_path))
        inbound = _inbound()
        cards: list[str] = []

        async def publish(_inbound, _text):
            raise RuntimeError("permission denied")

        async def create_card(_chat_id, text):
            cards.append(text)
            return "card_1"

        channel._publish_math_document = publish
        channel._create_card = create_card

        result = await channel._send_or_patch(inbound, "答案是 $x=1$。", final=True)

        assert result == "card_1"
        assert cards == ["答案是 $x=1$。"]

    _run(go())


def test_math_document_is_cached_by_inbound_message_id(tmp_path) -> None:
    async def go():
        channel = FeishuChannel(_settings(tmp_path))
        calls = 0

        class Publisher:
            def publish(self, *_args, **_kwargs):
                nonlocal calls
                calls += 1
                return PublishedDocument("docx_1", "公式回答", "https://feishu.cn/docx/docx_1")

        channel._document_publisher = Publisher()
        first = await channel._publish_math_document(_inbound(), "答案 $x=1$")
        second = await channel._publish_math_document(_inbound(), "答案 $x=1$")

        assert first == second
        assert calls == 1

    _run(go())
