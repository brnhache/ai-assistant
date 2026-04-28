from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_chat_missing_auth() -> None:
    r = client.post(
        "/api/v1/chat",
        json={
            "tenant_id": "t1",
            "user_id": 1,
            "capabilities": ["chat"],
            "message": "hello",
        },
    )
    assert r.status_code == 401


def test_chat_wrong_capability() -> None:
    r = client.post(
        "/api/v1/chat",
        headers={"Authorization": "Bearer test-inbound-token"},
        json={
            "tenant_id": "t1",
            "user_id": 1,
            "capabilities": ["briefing"],
            "message": "hello",
        },
    )
    assert r.status_code == 403


def test_chat_no_llm_key_returns_503() -> None:
    r = client.post(
        "/api/v1/chat",
        headers={"Authorization": "Bearer test-inbound-token"},
        json={
            "tenant_id": "t1",
            "user_id": 1,
            "capabilities": ["chat"],
            "message": "List equipment",
        },
    )
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "ANTHROPIC_API_KEY" in detail or "OPENAI_API_KEY" in detail
