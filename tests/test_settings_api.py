from dotenv import dotenv_values
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superassist.config import Settings
from superassist.ui.settings import register_settings_routes


def _client(settings: Settings, env_path) -> TestClient:
    app = FastAPI()
    register_settings_routes(app, settings, env_path)
    return TestClient(app)


def _update_payload(data: dict) -> dict:
    return {
        "memory": data["memory"],
        "feishu": {
            "app_id": data["feishu"]["app_id"],
            "domain": data["feishu"]["domain"],
            "allowed_open_ids": data["feishu"]["allowed_open_ids"],
            "mention_only": data["feishu"]["mention_only"],
        },
    }


def test_settings_api_masks_secret_and_persists_runtime_updates(tmp_path) -> None:
    env_path = tmp_path / ".env"
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_FEISHU_APP_SECRET="existing-secret",
    )
    client = _client(settings, env_path)

    initial = client.get("/internal/settings")

    assert initial.status_code == 200
    initial_data = initial.json()
    assert initial_data["feishu"]["app_secret_configured"] is True
    assert "app_secret" not in initial_data["feishu"]
    assert "existing-secret" not in initial.text

    payload = _update_payload(initial_data)
    payload["memory"]["top_k"] = 30
    payload["memory"]["candidate_pool_size"] = 180
    payload["feishu"]["app_id"] = "cli_test"
    updated = client.put("/internal/settings", json=payload)

    assert updated.status_code == 200
    assert updated.json()["memory"]["top_k"] == 30
    assert updated.json()["meta"]["feishu_restart_required"] is True
    assert settings.memory_top_k == 30
    assert settings.feishu_app_id == "cli_test"
    assert settings.feishu_app_secret == "existing-secret"
    persisted = dotenv_values(env_path)
    assert persisted["SUPERASSIST_MEMORY_TOP_K"] == "30"
    assert persisted["SUPERASSIST_MEMORY_CANDIDATE_POOL_SIZE"] == "180"
    assert persisted["SUPERASSIST_FEISHU_APP_ID"] == "cli_test"
    assert "SUPERASSIST_FEISHU_APP_SECRET" not in persisted


def test_settings_api_can_clear_secret_and_rejects_invalid_relationships(tmp_path) -> None:
    env_path = tmp_path / ".env"
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_FEISHU_APP_SECRET="existing-secret",
    )
    client = _client(settings, env_path)
    payload = _update_payload(client.get("/internal/settings").json())
    payload["feishu"]["app_secret"] = ""

    cleared = client.put("/internal/settings", json=payload)

    assert cleared.status_code == 200
    assert cleared.json()["feishu"]["app_secret_configured"] is False
    assert settings.feishu_app_secret == ""
    assert dotenv_values(env_path)["SUPERASSIST_FEISHU_APP_SECRET"] == ""

    invalid = _update_payload(cleared.json())
    invalid["memory"]["top_k"] = 200
    invalid["memory"]["candidate_pool_size"] = 100
    rejected = client.put("/internal/settings", json=invalid)

    assert rejected.status_code == 422
    assert settings.memory_top_k != 200
