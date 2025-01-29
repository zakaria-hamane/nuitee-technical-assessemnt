from fastapi.testclient import TestClient
from src.api.endpoints import app
import json

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["version"] == "1.0.0"


def test_process_rooms():
    test_data = {
        "reference_rooms": [
            {"room_name": "Deluxe King Room with City View"}
        ],
        "supplier_rooms": [
            {"room_name": "King Deluxe City View Room"}
        ]
    }
    response = client.post("/process_rooms", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "results" in data
    assert len(data["results"]) > 0


def test_match_rooms():
    test_data = {
        "reference_rooms": [
            {"room_id": "ref1", "room_name": "Deluxe King Room with City View"},
            {"room_id": "ref2", "room_name": "Standard Twin Room"}
        ],
        "supplier_rooms": [
            {"room_id": "sup1", "room_name": "King Deluxe City View"},
            {"room_id": "sup2", "room_name": "Twin Standard Room"}
        ]
    }

    response = client.post("/match_rooms", json=test_data)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert "matches" in data
    assert "unmatched_reference" in data
    assert "unmatched_supplier" in data

    # Additional assertions for the matching results
    matches = data["matches"]
    assert isinstance(matches, list)
    assert all(
        isinstance(match, dict) and
        "reference_room" in match and
        "supplier_room" in match and
        "similarity_score" in match
        for match in matches
    )