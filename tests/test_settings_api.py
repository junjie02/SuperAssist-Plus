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
        "skills": data["skills"],
        "feishu": {
            "app_id": data["feishu"]["app_id"],
            "domain": data["feishu"]["domain"],
            "allowed_open_ids": data["feishu"]["allowed_open_ids"],
            "mention_only": data["feishu"]["mention_only"],
            "active_session_seconds": data["feishu"]["active_session_seconds"],
            "activation_debounce_seconds": data["feishu"]["activation_debounce_seconds"],
            "activation_max_wait_seconds": data["feishu"]["activation_max_wait_seconds"],
            "max_images_per_activation": data["feishu"]["max_images_per_activation"],
        },
        "wecom": {
            "bot_id": data["wecom"]["bot_id"],
            "allowed_user_ids": data["wecom"]["allowed_user_ids"],
            "user_id_map": data["wecom"]["user_id_map"],
            "rag_mode_default": data["wecom"]["rag_mode_default"],
            "max_concurrent": data["wecom"]["max_concurrent"],
            "stream_interval_ms": data["wecom"]["stream_interval_ms"],
            "ai_engine_url": data["wecom"]["ai_engine_url"],
            "rpa_allowed_groups": data["wecom"]["rpa_allowed_groups"],
            "rpa_trigger_prefixes": data["wecom"]["rpa_trigger_prefixes"],
            "rpa_poll_interval_seconds": data["wecom"]["rpa_poll_interval_seconds"],
            "rpa_reply_max_chars": data["wecom"]["rpa_reply_max_chars"],
        },
    }


def test_settings_api_masks_secret_and_persists_runtime_updates(tmp_path) -> None:
    env_path = tmp_path / ".env"
    settings = Settings(
        SUPERASSIST_DATA_DIR=tmp_path,
        SUPERASSIST_EMBEDDING_PROVIDER="hash",
        SUPERASSIST_FEISHU_APP_SECRET="existing-secret",
        SUPERASSIST_WECOM_BOT_SECRET="wecom-secret",
    )
    client = _client(settings, env_path)

    initial = client.get("/internal/settings")

    assert initial.status_code == 200
    initial_data = initial.json()
    assert initial_data["feishu"]["app_secret_configured"] is True
    assert "app_secret" not in initial_data["feishu"]
    assert "existing-secret" not in initial.text
    assert initial_data["wecom"]["bot_secret_configured"] is True
    assert "bot_secret" not in initial_data["wecom"]
    assert "wecom-secret" not in initial.text

    payload = _update_payload(initial_data)
    payload["memory"]["top_k"] = 30
    payload["memory"]["candidate_pool_size"] = 180
    payload["skills"]["active_ttl_seconds"] = 420
    payload["feishu"]["app_id"] = "cli_test"
    payload["feishu"]["active_session_seconds"] = 240
    payload["wecom"]["bot_id"] = "bot_test"
    payload["wecom"]["rag_mode_default"] = True
    payload["wecom"]["user_id_map"] = '{"zhangsan":"user_123"}'
    payload["wecom"]["rpa_allowed_groups"] = "项目答疑群"
    payload["wecom"]["rpa_trigger_prefixes"] = "@SuperAssist"
    updated = client.put("/internal/settings", json=payload)

    assert updated.status_code == 200
    assert updated.json()["memory"]["top_k"] == 30
    assert updated.json()["meta"]["feishu_restart_required"] is True
    assert updated.json()["meta"]["wecom_restart_required"] is True
    assert settings.memory_top_k == 30
    assert settings.skill_active_ttl_seconds == 420
    assert settings.feishu_app_id == "cli_test"
    assert settings.feishu_active_session_seconds == 240
    assert settings.feishu_app_secret == "existing-secret"
    assert settings.wecom_bot_id == "bot_test"
    assert settings.wecom_bot_secret == "wecom-secret"
    assert settings.wecom_rag_mode_default is True
    assert settings.wecom_user_id_mapping == {"zhangsan": "user_123"}
    assert settings.wecom_rpa_allowed_group_set == {"项目答疑群"}
    assert settings.wecom_rpa_trigger_prefix_list == ["@SuperAssist"]
    persisted = dotenv_values(env_path)
    assert persisted["SUPERASSIST_MEMORY_TOP_K"] == "30"
    assert persisted["SUPERASSIST_MEMORY_CANDIDATE_POOL_SIZE"] == "180"
    assert persisted["SUPERASSIST_SKILL_ACTIVE_TTL_SECONDS"] == "420"
    assert persisted["SUPERASSIST_FEISHU_APP_ID"] == "cli_test"
    assert persisted["SUPERASSIST_FEISHU_ACTIVE_SESSION_SECONDS"] == "240"
    assert "SUPERASSIST_FEISHU_APP_SECRET" not in persisted
    assert persisted["SUPERASSIST_WECOM_BOT_ID"] == "bot_test"
    assert persisted["SUPERASSIST_WECOM_RAG_MODE_DEFAULT"] == "true"
    assert persisted["SUPERASSIST_WECOM_RPA_ALLOWED_GROUPS"] == "项目答疑群"
    assert "SUPERASSIST_WECOM_BOT_SECRET" not in persisted


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
    payload["wecom"]["bot_secret"] = ""

    cleared = client.put("/internal/settings", json=payload)

    assert cleared.status_code == 200
    assert cleared.json()["feishu"]["app_secret_configured"] is False
    assert settings.feishu_app_secret == ""
    assert dotenv_values(env_path)["SUPERASSIST_FEISHU_APP_SECRET"] == ""
    assert cleared.json()["wecom"]["bot_secret_configured"] is False
    assert settings.wecom_bot_secret == ""
    assert dotenv_values(env_path)["SUPERASSIST_WECOM_BOT_SECRET"] == ""

    invalid = _update_payload(cleared.json())
    invalid["memory"]["top_k"] = 200
    invalid["memory"]["candidate_pool_size"] = 100
    rejected = client.put("/internal/settings", json=invalid)

    assert rejected.status_code == 422
    assert settings.memory_top_k != 200

    invalid_mapping = _update_payload(cleared.json())
    invalid_mapping["wecom"]["user_id_map"] = "not-json"

    mapping_rejected = client.put("/internal/settings", json=invalid_mapping)

    assert mapping_rejected.status_code == 422
