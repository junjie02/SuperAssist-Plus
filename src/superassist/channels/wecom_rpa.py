from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import signal
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from superassist.config import Settings, get_settings

from .ai_engine_client import AIEngineClient
from .store import WeComThreadStore
from .wecom import ENGINE_ERROR_MESSAGE, parse_rag_command

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualMessage:
    group_name: str
    sender: str
    text: str

    @property
    def fingerprint(self) -> str:
        value = f"{self.group_name}\0{self.sender}\0{self.text}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VisualSnapshot:
    group_name: str
    is_external_group: bool
    messages: tuple[VisualMessage, ...]


def extract_triggered_prompt(text: str, trigger_prefixes: list[str]) -> str | None:
    """Return the prompt only when the message starts with a configured wake prefix."""

    normalized = text.strip()
    for prefix in sorted(trigger_prefixes, key=len, reverse=True):
        if normalized.casefold().startswith(prefix.casefold()):
            prompt = normalized[len(prefix) :].lstrip(" \t\r\n:：,，。")
            return prompt or None
    return None


def split_reply(text: str, max_chars: int) -> list[str]:
    """Split long replies at natural boundaries accepted by the desktop client."""

    remaining = text.strip()
    if not remaining:
        return []
    chunks: list[str] = []
    boundaries = ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ")
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        split_at = max(window.rfind(boundary) + len(boundary) for boundary in boundaries)
        if split_at < max_chars // 2:
            split_at = max_chars
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


class VisualRPAStateStore:
    """Persistent replay guard for messages that remain visible after a restart."""

    def __init__(self, path: str | Path, *, ttl_seconds: float = 86400) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._seen = self._load()

    def prime(self, messages: tuple[VisualMessage, ...]) -> None:
        changed = False
        now = time.time()
        for message in messages:
            if message.fingerprint not in self._seen:
                self._seen[message.fingerprint] = now
                changed = True
        if self._prune(now) or changed:
            self._save()

    def claim(self, message: VisualMessage) -> bool:
        now = time.time()
        self._prune(now)
        if message.fingerprint in self._seen:
            return False
        self._seen[message.fingerprint] = now
        self._save()
        return True

    def _prune(self, now: float) -> bool:
        expired = [key for key, seen_at in self._seen.items() if seen_at < now - self.ttl_seconds]
        for key in expired:
            self._seen.pop(key, None)
        return bool(expired)

    def _load(self) -> dict[str, float]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(value, dict):
            return {}
        return {
            str(key): float(seen_at)
            for key, seen_at in value.items()
            if isinstance(seen_at, (int, float))
        }

    def _save(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.path.parent,
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            json.dump(self._seen, handle, ensure_ascii=False)
            temp_name = handle.name
        Path(temp_name).replace(self.path)


class WeComDesktopDriver:
    """Visual adapter for WeCom 5.x, whose chat controls are not exposed through UIA."""

    window_class = "WeWorkWindow"
    header_region = (0.315, 0.025, 0.84, 0.115)
    chat_region = (0.315, 0.12, 0.84, 0.76)

    def __init__(self) -> None:
        self._window: Any = None
        self._ocr: Any = None

    def connect(self) -> None:
        try:
            from pywinauto import Desktop
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "WeCom RPA dependencies are missing. In the CF environment run: python -m pip install -e ."
            ) from exc
        windows = Desktop(backend="win32").windows(class_name=self.window_class, visible_only=False)
        candidates = [window for window in windows if window.is_visible() and window.is_enabled()]
        if not candidates:
            raise RuntimeError("No visible WeCom main window was found. Open and sign in to WeCom first.")
        candidates.sort(
            key=lambda window: window.rectangle().width() * window.rectangle().height(),
            reverse=True,
        )
        self._window = candidates[0]
        if self._is_minimized():
            raise RuntimeError("The WeCom window is minimized. Restore it and keep the target group visible.")
        self._ocr = RapidOCR()

    def read_snapshot(self) -> VisualSnapshot:
        image = self._capture_bgr()
        height, width = image.shape[:2]
        hx1, hy1, hx2, hy2 = self._region_pixels(self.header_region, width, height)
        header_lines = self._ocr_lines(image[hy1:hy2, hx1:hx2])
        group_name = header_lines[0][0].strip() if header_lines else ""
        header_text = " ".join(text for text, _box in header_lines)
        is_external_group = "外部群" in header_text

        cx1, cy1, cx2, cy2 = self._region_pixels(self.chat_region, width, height)
        chat_image = image[cy1:cy2, cx1:cx2]
        messages = tuple(self._read_incoming_messages(chat_image))
        return VisualSnapshot(group_name, is_external_group, messages)

    def send_text(self, expected_group: str, text: str) -> None:
        image = self._capture_bgr()
        snapshot = self._snapshot_header(image)
        if not snapshot.is_external_group or snapshot.group_name != expected_group:
            raise RuntimeError(
                f"Active WeCom chat changed; expected external group {expected_group!r}, "
                f"got {snapshot.group_name!r}. Reply was not sent."
            )

        height, width = image.shape[:2]
        input_point = (int(width * 0.55), int(height * 0.85))
        self._window.click_input(coords=input_point)
        self._paste_text(text)
        time.sleep(0.15)

        updated = self._capture_bgr()
        send_point = self._find_send_button(updated)
        if send_point is None:
            self._clear_input()
            raise RuntimeError("Could not locate the WeCom send button; reply was not sent.")
        self._window.click_input(coords=send_point)

    def _capture_bgr(self) -> Any:
        if self._window is None:
            raise RuntimeError("WeCom desktop driver is not connected")
        if self._is_minimized():
            raise RuntimeError("The WeCom window is minimized; RPA is paused and will not send messages.")
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required by the WeCom RPA channel") from exc
        image = self._window.capture_as_image().convert("RGB")
        return np.asarray(image)[:, :, ::-1].copy()

    def _is_minimized(self) -> bool:
        rectangle = self._window.rectangle()
        return rectangle.left < -10000 or rectangle.top < -10000

    def _snapshot_header(self, image: Any) -> VisualSnapshot:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = self._region_pixels(self.header_region, width, height)
        lines = self._ocr_lines(image[y1:y2, x1:x2])
        group_name = lines[0][0].strip() if lines else ""
        return VisualSnapshot(
            group_name=group_name,
            is_external_group="外部群" in " ".join(text for text, _box in lines),
            messages=(),
        )

    def _read_incoming_messages(self, image: Any) -> list[VisualMessage]:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("opencv-python is required by the WeCom RPA channel") from exc

        lower = np.array([225, 225, 220], dtype=np.uint8)
        upper = np.array([240, 239, 236], dtype=np.uint8)
        mask = cv2.inRange(image, lower, upper)
        _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
        height, width = image.shape[:2]
        bubbles: list[tuple[int, int, int, int]] = []
        for x, y, bubble_width, bubble_height, area in stats[1:]:
            if x >= width * 0.55 or area < 180:
                continue
            if bubble_width < 16 or bubble_height < 18:
                continue
            if bubble_width > width * 0.75 or bubble_height > height * 0.65:
                continue
            bubbles.append((int(x), int(y), int(bubble_width), int(bubble_height)))

        messages: list[VisualMessage] = []
        for x, y, bubble_width, bubble_height in sorted(bubbles, key=lambda item: item[1]):
            padding = 7
            bubble = image[
                max(0, y - padding) : min(height, y + bubble_height + padding),
                max(0, x - padding) : min(width, x + bubble_width + padding),
            ]
            lines = self._ocr_lines(bubble)
            text = "\n".join(line for line, _box in lines).strip()
            if not text:
                continue
            sender_area = image[max(0, y - 42) : max(0, y - 2), max(0, x - 5) : min(width, x + 350)]
            sender_lines = self._ocr_lines(sender_area)
            sender = " ".join(line for line, _box in sender_lines).strip() or "unknown"
            messages.append(VisualMessage(group_name="", sender=sender, text=text))
        return messages

    def _ocr_lines(self, image: Any) -> list[tuple[str, list[list[float]]]]:
        if image.size == 0:
            return []
        result, _elapsed = self._ocr(image)
        lines = [
            (str(item[1]).strip(), item[0])
            for item in (result or [])
            if str(item[1]).strip() and float(item[2]) >= 0.55
        ]
        return sorted(lines, key=lambda item: (min(point[1] for point in item[1]), min(point[0] for point in item[1])))

    def _find_send_button(self, image: Any) -> tuple[int, int] | None:
        height, width = image.shape[:2]
        y_offset = int(height * 0.7)
        lines = self._ocr_lines(image[y_offset:, int(width * 0.315) : int(width * 0.84)])
        for text, box in reversed(lines):
            if "发送" not in text:
                continue
            x_offset = int(width * 0.315)
            x = int(sum(point[0] for point in box) / len(box)) + x_offset
            y = int(sum(point[1] for point in box) / len(box)) + y_offset
            return x, y
        return None

    @staticmethod
    def _region_pixels(region: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = region
        return int(width * x1), int(height * y1), int(width * x2), int(height * y2)

    @staticmethod
    def _paste_text(text: str) -> None:
        import win32clipboard
        from pywinauto.keyboard import send_keys

        for attempt in range(5):
            try:
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, text)
                win32clipboard.CloseClipboard()
                break
            except Exception:
                try:
                    win32clipboard.CloseClipboard()
                except Exception:
                    pass
                if attempt == 4:
                    raise
                time.sleep(0.1)
        send_keys("^v")

    @staticmethod
    def _clear_input() -> None:
        from pywinauto.keyboard import send_keys

        send_keys("^a{BACKSPACE}")


class WeComRPAChannel:
    """Group-only WeCom desktop RPA channel backed by the existing AI Engine."""

    def __init__(
        self,
        settings: Settings,
        *,
        driver: Any | None = None,
        engine_client: AIEngineClient | None = None,
        thread_store: WeComThreadStore | None = None,
        state_store: VisualRPAStateStore | None = None,
    ) -> None:
        self.settings = settings
        self.allowed_groups = settings.wecom_rpa_allowed_group_set
        self.trigger_prefixes = settings.wecom_rpa_trigger_prefix_list
        self.driver = driver or WeComDesktopDriver()
        self.engine_client = engine_client or AIEngineClient(settings.wecom_ai_engine_url)
        self.thread_store = thread_store or WeComThreadStore(settings.wecom_thread_store_path)
        self.state_store = state_store or VisualRPAStateStore(settings.wecom_rpa_state_path)
        self.user_id_mapping = settings.wecom_user_id_mapping
        self._running = False
        self._last_paused_context: tuple[str, bool] | None = None

    async def start(self) -> None:
        if not self.allowed_groups:
            raise RuntimeError("SUPERASSIST_WECOM_RPA_ALLOWED_GROUPS must contain at least one exact group name.")
        if not self.trigger_prefixes:
            raise RuntimeError("SUPERASSIST_WECOM_RPA_TRIGGER_PREFIXES must contain at least one wake prefix.")
        await asyncio.to_thread(self.driver.connect)
        await self.engine_client.start()
        initial = await asyncio.to_thread(self.driver.read_snapshot)
        self.state_store.prime(self._messages_with_group(initial))
        self._running = True
        logger.info(
            "WeCom desktop RPA started; groups=%s triggers=%s active=%s",
            sorted(self.allowed_groups),
            self.trigger_prefixes,
            initial.group_name or "<unrecognized>",
        )

    async def stop(self) -> None:
        self._running = False
        await self.engine_client.close()

    async def poll_once(self) -> None:
        snapshot = await asyncio.to_thread(self.driver.read_snapshot)
        if not snapshot.is_external_group or snapshot.group_name not in self.allowed_groups:
            context = (snapshot.group_name, snapshot.is_external_group)
            if context != self._last_paused_context:
                logger.warning(
                    "WeCom RPA paused: active chat=%r external_group=%s",
                    snapshot.group_name,
                    snapshot.is_external_group,
                )
                self._last_paused_context = context
            return
        self._last_paused_context = None

        for message in self._messages_with_group(snapshot):
            if not self.state_store.claim(message):
                continue
            prompt = extract_triggered_prompt(message.text, self.trigger_prefixes)
            if prompt is None:
                continue
            await self._handle_message(message, prompt)

    async def run_forever(self) -> None:
        await self.start()
        try:
            while self._running:
                try:
                    await self.poll_once()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("WeCom RPA polling failed")
                await asyncio.sleep(self.settings.wecom_rpa_poll_interval_seconds)
        finally:
            await self.stop()

    def _messages_with_group(self, snapshot: VisualSnapshot) -> tuple[VisualMessage, ...]:
        return tuple(
            VisualMessage(group_name=snapshot.group_name, sender=message.sender, text=message.text)
            for message in snapshot.messages
        )

    async def _handle_message(self, message: VisualMessage, prompt: str) -> None:
        chat_id = f"rpa:{message.group_name}"
        mapping_key = f"rpa:{message.group_name}"
        identity_hash = hashlib.sha256(message.group_name.encode("utf-8")).hexdigest()[:16]
        identity = self.user_id_mapping.get(mapping_key, f"wecom-rpa-group:{identity_hash}")
        thread_id, rag_mode = self.thread_store.resolve(
            chat_id=chat_id,
            sender_user_id=message.sender,
            user_id=identity,
            rag_mode_default=self.settings.wecom_rag_mode_default,
            scope_id="__group__",
        )

        rag_command = parse_rag_command(prompt)
        if rag_command is not None:
            if isinstance(rag_command, bool):
                self.thread_store.set_rag_mode(
                    chat_id=chat_id,
                    sender_user_id=message.sender,
                    enabled=rag_command,
                    scope_id="__group__",
                )
                answer = f"知识库 RAG 模式已{'开启' if rag_command else '关闭'}。"
            else:
                answer = f"当前知识库 RAG 模式：{'开启' if rag_mode else '关闭'}。"
            await self._send_reply(message.group_name, answer)
            return

        answer = ""
        try:
            async for event in self.engine_client.stream_chat(
                user_id=identity,
                message=f"群成员 {message.sender}：{prompt}",
                thread_id=thread_id,
                rag_mode=rag_mode,
            ):
                event_type = str(event.get("type") or "")
                if event_type == "done":
                    answer = str(event.get("answer") or "").strip()
                    break
                if event_type == "error":
                    raise RuntimeError(str(event.get("message") or "AI Engine chat failed"))
            await self._send_reply(message.group_name, answer or "本次请求没有生成有效回答，请重试。")
        except Exception:
            logger.exception("WeCom RPA agent request failed for group=%s", message.group_name)
            try:
                await self._send_reply(message.group_name, ENGINE_ERROR_MESSAGE)
            except Exception:
                logger.exception("WeCom RPA could not send the engine error message")

    async def _send_reply(self, group_name: str, text: str) -> None:
        for chunk in split_reply(text, self.settings.wecom_rpa_reply_max_chars):
            await asyncio.to_thread(self.driver.send_text, group_name, chunk)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    channel = WeComRPAChannel(get_settings())

    async def run() -> None:
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except (NotImplementedError, RuntimeError):
                pass
        task = asyncio.create_task(channel.run_forever())
        waiter = asyncio.create_task(stop_event.wait())
        done, _pending = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if waiter in done:
            task.cancel()
        if waiter in done:
            await asyncio.gather(task, return_exceptions=True)
        else:
            waiter.cancel()
            await task

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
