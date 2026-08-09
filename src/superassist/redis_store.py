from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from superassist.config import Settings, get_settings

logger = logging.getLogger(__name__)


class RedisRuntimeStore:
    """Optional Redis-backed runtime state with fail-open local fallbacks."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self.prefix = settings.redis_prefix.strip().strip(":") or "superassist"
        self._client = client
        self._failure_logged = False
        self._image_root = settings.data_dir / "cache" / "feishu-images"
        self._last_image_cleanup_at = 0.0
        if self._client is None and settings.redis_enabled:
            try:
                import redis

                self._client = redis.Redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=settings.redis_socket_timeout_seconds,
                    socket_timeout=settings.redis_socket_timeout_seconds,
                    health_check_interval=30,
                )
                self._client.ping()
            except Exception as exc:
                self._client = None
                if settings.redis_required:
                    raise RuntimeError(f"Redis is required but unavailable: {exc}") from exc
                logger.warning("Redis unavailable; using local/SQLite fallbacks: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def ping(self) -> bool:
        return bool(self._run(False, lambda: self._client.ping()))

    def key(self, *parts: object) -> str:
        rendered = [self.prefix, "v1"]
        rendered.extend(_safe_key_part(str(part)) for part in parts)
        return ":".join(rendered)

    def claim_once(self, namespace: str, identity: str, *, ttl_seconds: int) -> bool:
        if not self.enabled:
            return True
        return bool(
            self._run(
                True,
                lambda: self._client.set(
                    self.key(namespace, _digest(identity)),
                    "1",
                    nx=True,
                    ex=max(1, ttl_seconds),
                ),
            )
        )

    def release_claim(self, namespace: str, identity: str) -> None:
        self._run(None, lambda: self._client.delete(self.key(namespace, _digest(identity))))

    def set_json(self, namespace: str, identity: str, value: Any, *, ttl_seconds: int) -> bool:
        if not self.enabled:
            return False
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        return bool(
            self._run(
                False,
                lambda: self._client.set(
                    self.key(namespace, _digest(identity)),
                    payload,
                    ex=max(1, ttl_seconds),
                ),
            )
        )

    def get_json(self, namespace: str, identity: str) -> Any | None:
        if not self.enabled:
            return None
        raw = self._run(None, lambda: self._client.get(self.key(namespace, _digest(identity))))
        if not raw:
            return None
        try:
            return json.loads(str(raw))
        except json.JSONDecodeError:
            return None

    def delete(self, namespace: str, identity: str) -> None:
        if self.enabled:
            self._run(None, lambda: self._client.delete(self.key(namespace, _digest(identity))))

    @contextmanager
    def lock(self, namespace: str, identity: str, *, ttl_seconds: int = 900) -> Iterator[bool]:
        if not self.enabled:
            yield True
            return
        key = self.key("lock", namespace, _digest(identity))
        token = uuid4().hex
        acquired = bool(
            self._run(False, lambda: self._client.set(key, token, nx=True, ex=max(1, ttl_seconds)))
        )
        try:
            yield acquired
        finally:
            if acquired:
                script = (
                    "if redis.call('get', KEYS[1]) == ARGV[1] then "
                    "return redis.call('del', KEYS[1]) else return 0 end"
                )
                self._run(None, lambda: self._client.eval(script, 1, key, token))

    def touch_activation(self, scope_key: str, *, started: bool = False, ttl_seconds: int = 60) -> None:
        if not self.enabled:
            return
        key = self.key("feishu", "activation", _digest(scope_key))
        now = time.time()

        def operation() -> Any:
            pipe = self._client.pipeline()
            if started:
                pipe.hsetnx(key, "started_at", now)
            pipe.hset(key, mapping={"last_message_at": now})
            pipe.expire(key, max(1, ttl_seconds))
            return pipe.execute()

        self._run(None, operation)

    def activation_times(self, scope_key: str) -> tuple[float, float] | None:
        if not self.enabled:
            return None
        raw = self._run(None, lambda: self._client.hgetall(self.key("feishu", "activation", _digest(scope_key))))
        if not isinstance(raw, dict) or not raw:
            return None
        try:
            return float(raw.get("started_at") or 0), float(raw.get("last_message_at") or 0)
        except (TypeError, ValueError):
            return None

    def clear_activation(self, scope_key: str) -> None:
        if self.enabled:
            self._run(
                None,
                lambda: self._client.delete(self.key("feishu", "activation", _digest(scope_key))),
            )

    def save_skill_activations(self, thread_id: str, activations: dict[str, float], *, ttl_seconds: int) -> None:
        self.set_json("skills", thread_id, activations, ttl_seconds=ttl_seconds)

    def load_skill_activations(self, thread_id: str) -> dict[str, float]:
        value = self.get_json("skills", thread_id)
        if not isinstance(value, dict):
            return {}
        result: dict[str, float] = {}
        for name, activated_at in value.items():
            try:
                result[str(name)] = float(activated_at)
            except (TypeError, ValueError):
                continue
        return result

    def save_short_memory(self, thread_id: str, value: dict[str, Any]) -> None:
        self.set_json(
            "short-memory",
            thread_id,
            value,
            ttl_seconds=self.settings.redis_short_memory_ttl_seconds,
        )

    def load_short_memory(self, thread_id: str) -> dict[str, Any] | None:
        value = self.get_json("short-memory", thread_id)
        return value if isinstance(value, dict) else None

    def memory_version(self, user_id: str) -> int:
        if not self.enabled:
            return 0
        raw = self._run(None, lambda: self._client.get(self.key("memory", "version", _digest(user_id))))
        try:
            return int(raw or 0)
        except (TypeError, ValueError):
            return 0

    def bump_memory_version(self, user_id: str) -> None:
        if self.enabled:
            self._run(None, lambda: self._client.incr(self.key("memory", "version", _digest(user_id))))

    def load_recall(self, user_id: str, query: str) -> dict[str, Any] | None:
        identity = f"{user_id}:{self.memory_version(user_id)}:{_digest(query)}"
        value = self.get_json("memory:recall", identity)
        return value if isinstance(value, dict) else None

    def save_recall(self, user_id: str, query: str, value: dict[str, Any]) -> None:
        identity = f"{user_id}:{self.memory_version(user_id)}:{_digest(query)}"
        self.set_json(
            "memory:recall",
            identity,
            value,
            ttl_seconds=self.settings.redis_recall_ttl_seconds,
        )

    def save_task(self, task: dict[str, Any]) -> None:
        if not self.enabled or not task.get("task_id"):
            return
        task_id = str(task["task_id"])
        payload_key = self.key("subagent", "task", _digest(task_id))
        order_key = self.key("subagent", "tasks")
        payload = json.dumps(task, ensure_ascii=False, separators=(",", ":"), default=str)

        def operation() -> Any:
            pipe = self._client.pipeline()
            pipe.set(payload_key, payload, ex=self.settings.redis_task_ttl_seconds)
            pipe.zadd(order_key, {task_id: time.time()})
            pipe.zremrangebyrank(order_key, 0, -1001)
            pipe.expire(order_key, self.settings.redis_task_ttl_seconds)
            return pipe.execute()

        self._run(None, operation)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        value = self._run(
            None,
            lambda: self._client.get(self.key("subagent", "task", _digest(task_id))),
        ) if self.enabled else None
        if not value:
            return None
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def list_tasks(self, limit: int) -> list[dict[str, Any]] | None:
        if not self.enabled:
            return None
        task_ids = self._run(
            None,
            lambda: self._client.zrevrange(self.key("subagent", "tasks"), 0, max(0, limit - 1)),
        )
        if not isinstance(task_ids, list):
            return None
        return [item for task_id in task_ids if (item := self.get_task(str(task_id))) is not None]

    def delete_task(self, task_id: str) -> bool:
        if not self.enabled:
            return False
        result = self._run(
            0,
            lambda: self._client.delete(self.key("subagent", "task", _digest(task_id))),
        )
        self._run(None, lambda: self._client.zrem(self.key("subagent", "tasks"), task_id))
        return bool(result)

    def schedule_memory_job(self, job_id: str, payload: dict[str, Any], *, due_at: float) -> None:
        if not self.enabled:
            return
        payload_key = self.key("memory", "job", _digest(job_id))
        due_key = self.key("memory", "due")
        value = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)

        def operation() -> Any:
            pipe = self._client.pipeline()
            pipe.set(payload_key, value, ex=max(86400, self.settings.redis_task_ttl_seconds))
            pipe.zadd(due_key, {job_id: due_at})
            return pipe.execute()

        self._run(None, operation)

    def claim_memory_jobs(self, *, force: bool, limit: int = 50, lease_seconds: int = 900) -> list[str]:
        if not self.enabled:
            return []
        now = time.time()
        cutoff = "+inf" if force else str(now)
        script = """
local expired = redis.call('ZRANGEBYSCORE', KEYS[2], '-inf', ARGV[1])
for _, id in ipairs(expired) do
  redis.call('ZREM', KEYS[2], id)
  redis.call('ZADD', KEYS[1], ARGV[1], id)
end
local ids = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[2], 'LIMIT', 0, ARGV[3])
for _, id in ipairs(ids) do
  redis.call('ZREM', KEYS[1], id)
  redis.call('ZADD', KEYS[2], ARGV[4], id)
end
return ids
"""
        value = self._run(
            [],
            lambda: self._client.eval(
                script,
                2,
                self.key("memory", "due"),
                self.key("memory", "processing"),
                now,
                cutoff,
                max(1, limit),
                now + max(30, lease_seconds),
            ),
        )
        return [str(item) for item in value] if isinstance(value, list) else []

    def get_memory_job(self, job_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        raw = self._run(None, lambda: self._client.get(self.key("memory", "job", _digest(job_id))))
        if not raw:
            return None
        try:
            value = json.loads(str(raw))
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    def finish_memory_job(self, job_id: str) -> None:
        if not self.enabled:
            return

        def operation() -> Any:
            pipe = self._client.pipeline()
            pipe.delete(self.key("memory", "job", _digest(job_id)))
            pipe.zrem(self.key("memory", "processing"), job_id)
            pipe.zrem(self.key("memory", "due"), job_id)
            return pipe.execute()

        self._run(None, operation)

    def reschedule_memory_job(self, job_id: str, *, due_at: float) -> None:
        if not self.enabled:
            return

        def operation() -> Any:
            pipe = self._client.pipeline()
            pipe.zrem(self.key("memory", "processing"), job_id)
            pipe.zadd(self.key("memory", "due"), {job_id: due_at})
            return pipe.execute()

        self._run(None, operation)

    def next_memory_job_due_at(self) -> float | None:
        if not self.enabled:
            return None
        value = self._run(
            None,
            lambda: self._client.zrange(self.key("memory", "due"), 0, 0, withscores=True),
        )
        if not isinstance(value, list) or not value:
            return None
        try:
            return float(value[0][1])
        except (IndexError, TypeError, ValueError):
            return None

    def set_card_state(self, message_id: str, *, card_id: str, last_text: str = "") -> None:
        self.set_json(
            "feishu:card",
            message_id,
            {"card_id": card_id, "last_text": last_text},
            ttl_seconds=3600,
        )

    def get_card_state(self, message_id: str) -> dict[str, str] | None:
        value = self.get_json("feishu:card", message_id)
        if not isinstance(value, dict):
            return None
        return {"card_id": str(value.get("card_id") or ""), "last_text": str(value.get("last_text") or "")}

    def save_image_context(self, scope_key: str, payloads: list[tuple[bytes, str]], *, ttl_seconds: int) -> None:
        if not self.enabled:
            return
        self._image_root.mkdir(parents=True, exist_ok=True)
        refs: list[dict[str, str]] = []
        for data, mime_type in payloads:
            digest = hashlib.sha256(data).hexdigest()
            path = self._image_root / f"{digest}.bin"
            if not path.exists():
                temporary = path.with_suffix(f".{os.getpid()}.tmp")
                temporary.write_bytes(data)
                try:
                    temporary.replace(path)
                except FileExistsError:
                    temporary.unlink(missing_ok=True)
            else:
                path.touch()
            refs.append({"path": str(path), "mime_type": mime_type})
        self.set_json("feishu:image-context", scope_key, refs, ttl_seconds=ttl_seconds)
        self._cleanup_image_cache(ttl_seconds)

    def load_image_context(self, scope_key: str) -> list[tuple[bytes, str]]:
        value = self.get_json("feishu:image-context", scope_key)
        if not isinstance(value, list):
            return []
        payloads: list[tuple[bytes, str]] = []
        root = self._image_root.resolve()
        for item in value:
            if not isinstance(item, dict):
                continue
            path = Path(str(item.get("path") or ""))
            try:
                resolved = path.resolve()
                resolved.relative_to(root)
                payloads.append((resolved.read_bytes(), str(item.get("mime_type") or "image/jpeg")))
            except (OSError, ValueError):
                continue
        return payloads

    def allow_request(self, identity: str) -> tuple[bool, int]:
        limit = self.settings.api_rate_limit_per_minute
        if not self.enabled or limit <= 0:
            return True, 0
        bucket = int(time.time() // 60)
        key = self.key("rate", _digest(identity), bucket)

        def operation() -> Any:
            pipe = self._client.pipeline()
            pipe.incr(key)
            pipe.expire(key, 120)
            return pipe.execute()

        result = self._run(None, operation)
        if not isinstance(result, list) or not result:
            return True, 0
        count = int(result[0])
        return count <= limit, count

    def _cleanup_image_cache(self, ttl_seconds: int) -> None:
        now = time.time()
        if now - self._last_image_cleanup_at < 3600:
            return
        self._last_image_cleanup_at = now
        cutoff = now - max(3600, ttl_seconds * 2)
        try:
            for path in self._image_root.glob("*.bin"):
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Unable to clean expired Feishu Redis image references", exc_info=True)

    def _run(self, default: Any, operation: Any) -> Any:
        if not self.enabled:
            return default
        try:
            result = operation()
            self._failure_logged = False
            return result
        except Exception as exc:
            if self.settings.redis_required:
                raise
            if not self._failure_logged:
                logger.warning("Redis operation failed; falling back to local state: %s", exc)
                self._failure_logged = True
            return default


def _safe_key_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return cleaned[:80] or "empty"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


@lru_cache(maxsize=16)
def _cached_store(
    enabled: bool,
    url: str,
    prefix: str,
    required: bool,
    socket_timeout: float,
    data_dir: str,
    task_ttl: int,
    short_ttl: int,
    recall_ttl: int,
    rate_limit: int,
) -> RedisRuntimeStore:
    settings = Settings(
        SUPERASSIST_REDIS_ENABLED=enabled,
        SUPERASSIST_REDIS_URL=url,
        SUPERASSIST_REDIS_PREFIX=prefix,
        SUPERASSIST_REDIS_REQUIRED=required,
        SUPERASSIST_REDIS_SOCKET_TIMEOUT_SECONDS=socket_timeout,
        SUPERASSIST_DATA_DIR=Path(data_dir),
        SUPERASSIST_REDIS_TASK_TTL_SECONDS=task_ttl,
        SUPERASSIST_REDIS_SHORT_MEMORY_TTL_SECONDS=short_ttl,
        SUPERASSIST_REDIS_RECALL_TTL_SECONDS=recall_ttl,
        SUPERASSIST_API_RATE_LIMIT_PER_MINUTE=rate_limit,
    )
    return RedisRuntimeStore(settings)


def get_redis_store(settings: Settings | None = None) -> RedisRuntimeStore:
    settings = settings or get_settings()
    return _cached_store(
        settings.redis_enabled,
        settings.redis_url,
        settings.redis_prefix,
        settings.redis_required,
        settings.redis_socket_timeout_seconds,
        str(settings.data_dir.resolve()),
        settings.redis_task_ttl_seconds,
        settings.redis_short_memory_ttl_seconds,
        settings.redis_recall_ttl_seconds,
        settings.api_rate_limit_per_minute,
    )


__all__ = ["RedisRuntimeStore", "get_redis_store"]
