import pytest
import pandas as pd
from src.preprocessing.cleaner import RoomNameProcessor
from src.preprocessing.feature_extractor import FeatureExtractor
from src.matching.similarity import RoomMatcher
from src.evaluation.metrics import RoomMatchEvaluator


def test_full_pipeline():
    # Test data
    test_data = {
        'reference_rooms': [
            {'room_name': 'Deluxe King Room with City View'},
            {'room_name': 'Standard Twin Room'}
        ],
        'supplier_rooms': [
            {'room_name': 'King Deluxe City View'},
            {'room_name': 'Twin Standard Room'}
        ]
    }

    # Initialize components
    processor = RoomNameProcessor()
    feature_extractor = FeatureExtractor()
    matcher = RoomMatcher()

    # Process reference rooms
    processed_ref = [
        processor.process_room_name(room['room_name'])
        for room in test_data['reference_rooms']
    ]

    # Process supplier rooms
    processed_supp = [
        processor.process_room_name(room['room_name'])
        for room in test_data['supplier_rooms']
    ]

    # Extract features
    for room in processed_ref + processed_supp:
        room['features'] = feature_extractor.extract_features(room['cleaned_name'])

    # Calculate similarities
    similarities = []
    for ref, supp in zip(processed_ref, processed_supp):
        sim_score = matcher.calculate_similarity(ref, supp)
        similarities.append(sim_score)

    assert len(similarities) == len(test_data['reference_rooms'])
    assert all(0 <= score <= 1 for score in similarities)


# In tests/integration/test_pipeline.py, add:

def test_end_to_end_matching():
    # Test data
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

    # Process and match rooms
    processor = RoomNameProcessor()
    feature_extractor = FeatureExtractor()
    matcher = RoomMatcher()

    processed_ref = [
        {**processor.process_room_name(room["room_name"]),
         'features': feature_extractor.extract_features(room["room_name"]),
         'id': room["room_id"]}
        for room in test_data["reference_rooms"]
    ]

    processed_supp = [
        {**processor.process_room_name(room["room_name"]),
         'features': feature_extractor.extract_features(room["room_name"]),
         'id': room["room_id"]}
        for room in test_data["supplier_rooms"]
    ]

    results = matcher.match_rooms(processed_ref, processed_supp)

    # Assertions
    assert len(results["matched_pairs"]) > 0
    assert all(m["similarity_score"] >= 0.8 for m in results["matched_pairs"])