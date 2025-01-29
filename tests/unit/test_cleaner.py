import pytest
from src.preprocessing.cleaner import RoomNameProcessor


@pytest.fixture
def processor():
    return RoomNameProcessor()


def test_basic_clean():
    processor = RoomNameProcessor()
    test_cases = [
        ("Deluxe King Room!", "deluxe king room"),
        ("2 BED SUITE", "two bed suite"),
        ("Room-with-View", "room with view"),
    ]

    for input_text, expected in test_cases:
        result = processor.basic_clean(input_text)
        assert result == expected, f"Failed for input: {input_text}"


def test_normalize_abbreviations():
    processor = RoomNameProcessor()
    test_cases = [
        ("dlx rm ro", "deluxe room room-only"),
        ("std rm bb", "std room breakfast-included"),
    ]

    for input_text, expected in test_cases:
        cleaned = processor.basic_clean(input_text)
        result = processor.normalize_abbreviations(cleaned)
        assert result == expected, f"Failed for input: {input_text}"


def test_number_conversion():
    processor = RoomNameProcessor()
    test_cases = [
        ("2 BED SUITE", "two bed suite"),
        ("3 BEDROOM APT", "three bedroom apt"),
        ("4 PERSON ROOM", "four person room"),
        ("10 BED DORM", "ten bed dorm"),
    ]

    for input_text, expected in test_cases:
        result = processor.basic_clean(input_text)
        assert result == expected, f"Failed for input: {input_text}"