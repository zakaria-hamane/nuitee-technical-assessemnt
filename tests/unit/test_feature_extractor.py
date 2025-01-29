import pytest
from unittest.mock import Mock, patch, MagicMock, call
import json
from src.preprocessing.feature_extractor import FeatureExtractor


@pytest.fixture
def feature_extractor():
    return FeatureExtractor()


def test_extract_features(monkeypatch):
    """Test successful feature extraction"""
    mock_response = {
        "bedType": "king",
        "roomClass": "deluxe",
        "viewType": "city",
        "boardType": "room-only",
        "accessibility": "standard",
        "occupancy": "2"
    }

    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(mock_response)
            )
        )
    ]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    def mock_openai(*args, **kwargs):
        return mock_client

    monkeypatch.setattr("src.preprocessing.feature_extractor.OpenAI", mock_openai)

    extractor = FeatureExtractor()
    input_text = "deluxe king room city view"
    features = extractor.extract_features(input_text)

    # Verify the API call
    assert mock_client.chat.completions.create.call_count == 1
    actual_call = mock_client.chat.completions.create.call_args
    assert actual_call.kwargs['model'] == "gpt-4"
    assert actual_call.kwargs['temperature'] == 0
    assert len(actual_call.kwargs['messages']) == 2
    assert actual_call.kwargs['messages'][0]['role'] == "system"
    assert actual_call.kwargs['messages'][1]['role'] == "user"

    # Verify the extracted features
    assert isinstance(features, dict)
    assert features.get("bedType") == "king"
    assert features.get("roomClass") == "deluxe"
    assert features.get("viewType") == "city"
    assert features.get("boardType") == "room-only"
    assert features.get("accessibility") == "standard"
    assert features.get("occupancy") == "2"


def test_extract_features_error_handling(monkeypatch):
    """Test error handling in feature extraction"""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")

    def mock_openai(*args, **kwargs):
        return mock_client

    monkeypatch.setattr("src.preprocessing.feature_extractor.OpenAI", mock_openai)

    extractor = FeatureExtractor()
    default_features = extractor._get_default_features()

    features = extractor.extract_features(None)
    assert features == default_features, "Should return default features for None input"

    features = extractor.extract_features("")
    assert features == default_features, "Should return default features for empty string"

    features = extractor.extract_features("test")
    assert features == default_features, "Should return default features on API error"


def test_extract_features_invalid_json(monkeypatch):
    """Test handling of invalid JSON response"""
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(
            message=MagicMock(
                content="Invalid JSON"
            )
        )
    ]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    def mock_openai(*args, **kwargs):
        return mock_client

    monkeypatch.setattr("src.preprocessing.feature_extractor.OpenAI", mock_openai)

    extractor = FeatureExtractor()
    default_features = extractor._get_default_features()

    features = extractor.extract_features("test")
    assert features == default_features, "Should return default features for invalid JSON"