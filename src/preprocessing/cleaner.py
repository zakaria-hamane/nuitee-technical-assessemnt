import re
from typing import Dict
from word2number import w2n
import inflect


class RoomNameProcessor:
    def __init__(self):
        self.abbreviations = {
            'rm': 'room',
            'ro': 'room-only',
            'nr': 'non-refundable',
            'ada': 'accessible',
            'bb': 'breakfast-included',
            'hb': 'half-board',
            'fb': 'full-board',
            'ai': 'all-inclusive',
            'dlx': 'deluxe',
            'std': 'std'
        }

        self.abbreviations.update({
            'superior': 'sup',
            'executive': 'exec',
            'standard': 'std',
            'deluxe': 'dlx',
            'apartment': 'apt',
            'bedroom': 'bdrm'
        })

        self.room_types = {
            'single': 'single',
            'double': 'double',
            'twin': 'twin',
            'triple': 'triple',
            'quad': 'quadruple',
            'suite': 'suite',
            'studio': 'studio',
            'apartment': 'apartment'
        }
        self.p = inflect.engine()  # Initialize inflect engine for number conversion

    def _convert_number(self, text: str) -> str:
        """Convert numeric strings to words, handling special characters."""
        try:
            # Handle special number characters
            special_numbers = {
                '①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5',
                '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⑩': '10'
            }

            # Replace special number characters
            if text in special_numbers:
                text = special_numbers[text]

            if text.isdigit():
                return self.p.number_to_words(int(text))
            return text
        except Exception as e:
            logger.error(f"Error converting number: {e}, text: {text}")
            return text

    def basic_clean(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters except hyphens
        text = re.sub(r'[^\w\s-]', ' ', text)

        # Split into words and process each word
        words = []
        for word in text.split():
            # Convert numbers to words
            if word.isdigit():
                word = self._convert_number(word)
            words.append(word)

        # Join words and normalize spaces
        text = ' '.join(words)
        text = re.sub(r'\s+', ' ', text).strip()

        # Handle hyphenated words
        text = text.replace('-', ' ')

        return text

    def normalize_abbreviations(self, text: str) -> str:
        """Replace abbreviations with their full forms."""
        words = text.split()
        return ' '.join(self.abbreviations.get(word, word) for word in words)

    def process_room_name(self, room_name: str) -> Dict:
        """Process room name through the complete pipeline."""
        cleaned_text = self.basic_clean(room_name)
        normalized_text = self.normalize_abbreviations(cleaned_text)
        return {
            'original_name': room_name,
            'cleaned_name': normalized_text
        }