import json
import logging
from typing import Dict, List
from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)


class FeatureExtractor:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def extract_features(self, cleaned_text: str) -> Dict:
        """Extract features from cleaned room description text."""
        if not cleaned_text:
            return self._get_default_features()

        try:
            system_prompt = (
                "You are a hotel room feature extractor. "
                "Extract features from room descriptions and return them in a specific JSON format. "
                "Always return valid JSON with all required fields, using 'unknown' when information is not available."
            )

            user_prompt = (
                f'Extract features from this room description: "{cleaned_text}"\n\n'
                'Return a JSON object with exactly these fields and allowed values:\n'
                '{\n'
                '    "bedType": ["single", "twin", "double", "queen", "king", "unknown"],\n'
                '    "roomClass": ["standard", "deluxe", "superior", "suite", "executive", "unknown"],\n'
                '    "viewType": ["city", "ocean", "garden", "mountain", "none", "unknown"],\n'
                '    "boardType": ["room-only", "breakfast", "half-board", "full-board", "unknown"],\n'
                '    "accessibility": ["accessible", "standard", "unknown"],\n'
                '    "occupancy": ["1", "2", "3", "4", "unknown"]\n'
                '}\n\n'
                'Rules:\n'
                '1. Use lowercase values only\n'
                '2. Choose the closest matching value from the allowed options\n'
                '3. Use "unknown" when information is not clear\n'
                '4. Return ONLY the JSON object, no other text'
            )

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )

            try:
                content = response.choices[0].message.content.strip()
                json_str = self._extract_json(content)
                features = json.loads(json_str)

                # Normalize and validate features
                normalized_features = {}
                for key in ["bedType", "roomClass", "viewType", "boardType", "accessibility", "occupancy"]:
                    value = str(features.get(key, "unknown")).lower()
                    normalized_features[key] = value if value != "none" else "unknown"

                return normalized_features

            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                logger.error(f"Error parsing API response: {e}, content: {content}")
                return self._get_default_features()

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return self._get_default_features()

    def _get_default_features(self) -> Dict:
        """Return default features when extraction fails."""
        return {
            "bedType": "unknown",
            "roomClass": "standard",
            "viewType": "unknown",
            "boardType": "room-only",
            "accessibility": "standard",
            "occupancy": "2"
        }

    def _extract_json(self, text: str) -> str:
        """Extract JSON object from text that might contain additional content."""
        try:
            # Find the first '{' and last '}'
            start = text.find('{')
            end = text.rfind('}')

            if start != -1 and end != -1:
                return text[start:end + 1]
            return text
        except Exception:
            return text

    def _normalize_value(self, value: str, valid_values: List[str], default: str) -> str:
        """Normalize a value to the closest match from valid values."""
        value = str(value).lower()
        if value in valid_values:
            return value
        return default