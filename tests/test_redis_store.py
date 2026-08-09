from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

from superassist.config import Settings
from superassist.memory.service import MemoryWritePayload
from superassist.memory.writer import MemoryWriteQueue
from superassist.redis_store import RedisRuntimeStore


class FakePipeline:
    def __init__(self, client: FakeRedis) -> None:
        self.client = client
        self.operations = []

    def __getattr__(self, name):
        def enqueue(*args, **kwargs):
            self.operations.append((name, args, kwargs))
            return self

        return enqueue

    def execute(self):
        return [getattr(self.client, name)(*args, **kwargs) for name, args, kwargs in self.operations]


class FakeRedis:
    def __init__(self) -> None:
        self.values = {}
        self.hashes = {}
        self.sorted_sets = {}

    def ping(self):
        return True

    def pipeline(self):
        return FakePipeline(self)

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return False
        self.values[key] = str(value)
        return True

    def get(self, key):
        return self.values.get(key)

    def delete(self, *keys):
        deleted = 0
        for key in keys:
            deleted += int(key in self.values or key in self.hashes or key in self.sorted_sets)
            self.values.pop(key, None)
            self.hashes.pop(key, None)
            self.sorted_sets.pop(key, None)
        return deleted

    def expire(self, key, seconds):
        return bool(key in self.values or key in self.hashes or key in self.sorted_sets)

    def incr(self, key):
        value = int(self.values.get(key, 0)) + 1
        self.values[key] = str(value)
        return value

    def hsetnx(self, key, field, value):
        target = self.hashes.setdefault(key, {})
        if field in target:
            return 0
        target[field] = str(value)
        return 1

    def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update({name: str(value) for name, value in mapping.items()})
        return len(mapping)

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def zadd(self, key, mapping):
        self.sorted_sets.setdefault(key, {}).update({str(name): float(score) for name, score in mapping.items()})
        return len(mapping)

    def zrem(self, key, *members):
        target = self.sorted_sets.setdefault(key, {})
        deleted = 0
        for member in members:
            deleted += int(str(member) in target)
            target.pop(str(member), None)
        return deleted

    def zremrangebyrank(self, key, start, stop):
        ordered = self._ordered(key)
        selected = ordered[start : stop + 1 if stop >= 0 else stop or None]
        return self.zrem(key, *selected)

    def zrevrange(self, key, start, stop):
        ordered = list(reversed(self._ordered(key)))
        return ordered[start : stop + 1]

    def zrange(self, key, start, stop, withscores=False):
        ordered = self._ordered(key)
        selected = ordered[start : stop + 1]
        if withscores:
            return [(item, self.sorted_sets[key][item]) for item in selected]
        return selected

    def eval(self, script, number_of_keys, *args):
        keys = list(args[:number_of_keys])
        argv = list(args[number_of_keys:])
        if number_of_keys == 1:
            if self.get(keys[0]) == str(argv[0]):
                return self.delete(keys[0])
            return 0
        due_key, processing_key = keys
        now, cutoff, limit, lease_until = float(argv[0]), argv[1], int(argv[2]), float(argv[3])
        for job_id, score in list(self.sorted_sets.get(processing_key, {}).items()):
            if score <= now:
                self.zrem(processing_key, job_id)
                self.zadd(due_key, {job_id: now})
        max_score = float("inf") if cutoff == "+inf" else float(cutoff)
        due = [
            job_id
            for job_id in self._ordered(due_key)
            if self.sorted_sets[due_key][job_id] <= max_score
        ][:limit]
        for job_id in due:
            self.zrem(due_key, job_id)
            self.zadd(processing_key, {job_id: lease_until})
        return due

    def _ordered(self, key):
        return [
            item[0]
            for item in sorted(self.sorted_sets.get(key, {}).items(), key=lambda item: (item[1], item[0]))
        ]


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_REDIS_ENABLED=True,
        SUPERASSIST_REDIS_PREFIX="test",
        SUPERASSIST_API_RATE_LIMIT_PER_MINUTE=2,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
        **overrides,
    )


def test_redis_runtime_state_supports_ttl_caches_and_rate_limit(tmp_path) -> None:
    store = RedisRuntimeStore(_settings(tmp_path), client=FakeRedis())

    assert store.claim_once("message", "m1", ttl_seconds=60) is True
    assert store.claim_once("message", "m1", ttl_seconds=60) is False
    store.save_skill_activations("t1", {"skill-a": 123.0}, ttl_seconds=300)
    store.save_short_memory("t1", {"summary": "done", "records": []})
    store.save_recall("u1", "hello", {"read_recall": {}})

    assert store.load_skill_activations("t1") == {"skill-a": 123.0}
    assert store.load_short_memory("t1")["summary"] == "done"
    assert store.load_recall("u1", "hello") == {"read_recall": {}}
    store.bump_memory_version("u1")
    assert store.load_recall("u1", "hello") is None
    assert [store.allow_request("u1")[0] for _ in range(3)] == [True, True, False]


def test_redis_runtime_state_persists_tasks_cards_and_image_references(tmp_path) -> None:
    store = RedisRuntimeStore(_settings(tmp_path), client=FakeRedis())
    task = {"task_id": "task-1", "status": "completed", "result": "ok"}

    store.save_task(task)
    store.set_card_state("message-1", card_id="card-1", last_text="working")
    store.save_image_context("group:1", [(b"image-data", "image/png")], ttl_seconds=180)

    assert store.get_task("task-1") == task
    assert store.list_tasks(10) == [task]
    assert store.get_card_state("message-1") == {"card_id": "card-1", "last_text": "working"}
    assert store.load_image_context("group:1") == [(b"image-data", "image/png")]
    assert store.delete_task("task-1") is True


def test_memory_write_queue_uses_redis_claim_and_sql_completion_ledger(tmp_path, monkeypatch) -> None:
    redis_store = RedisRuntimeStore(_settings(tmp_path), client=FakeRedis())
    ledger = FakeMemoryJobLedger()
    service = SimpleNamespace(settings=_settings(tmp_path), store=ledger)
    writer = SimpleNamespace(service=service, write=lambda payload: ledger.written.append(payload))
    monkeypatch.setattr("superassist.memory.writer.get_redis_store", lambda settings: redis_store)
    queue = MemoryWriteQueue(writer, debounce_seconds=3600)
    payload = MemoryWritePayload(
        user_id="u1",
        thread_id="t1",
        event_id="event-1",
        user_message="hello",
        assistant_answer="hi",
        tool_events=[],
    )

    queue.add(payload)
    queue.flush()

    assert len(ledger.written) == 1
    assert asdict(ledger.written[0]) == asdict(payload)
    assert ledger.finished == [(next(iter(ledger.jobs)), "")]


class FakeMemoryJobLedger:
    def __init__(self) -> None:
        self.jobs = {}
        self.written = []
        self.finished = []

    def enqueue_memory_job(self, job_id, user_id, payload):
        self.jobs[job_id] = {"payload": payload, "attempts": 0, "status": "pending"}
        return True

    def list_memory_jobs(self):
        return []

    def requeue_stale_memory_jobs(self, stale_before):
        return 0

    def mark_memory_job_running(self, job_id):
        self.jobs[job_id]["attempts"] += 1
        self.jobs[job_id]["status"] = "running"
        return self.jobs[job_id]["attempts"]

    def finish_memory_job(self, job_id, *, error=""):
        self.jobs[job_id]["status"] = "failed" if error else "completed"
        self.finished.append((job_id, error))
