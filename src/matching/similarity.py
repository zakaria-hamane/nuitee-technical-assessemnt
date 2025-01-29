import json
from typing import List, Dict, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class RoomMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Add bigrams
            stop_words='english',  # Remove stop words
            max_features=1000
        )

    def calculate_similarity(self, ref_room: Dict, supplier_room: Dict) -> float:
        # Calculate feature similarity
        feature_sim = self._calculate_feature_similarity(
            self._ensure_dict(ref_room['features']),
            self._ensure_dict(supplier_room['features'])
        )

        # Calculate text similarity
        text_sim = self._calculate_text_similarity(
            ref_room['cleaned_name'],
            supplier_room['cleaned_name']
        )

        # Calculate name token overlap
        ref_tokens = set(ref_room['cleaned_name'].split())
        supp_tokens = set(supplier_room['cleaned_name'].split())
        overlap_sim = len(ref_tokens & supp_tokens) / len(ref_tokens | supp_tokens)

        # Weighted combination
        return (0.4 * feature_sim +
                0.3 * text_sim +
                0.3 * overlap_sim)

    def _ensure_dict(self, features: Union[str, Dict]) -> Dict:
        """Ensure features are in dictionary format"""
        if isinstance(features, str):
            try:
                return json.loads(features)
            except json.JSONDecodeError:
                return {}
        return features if isinstance(features, dict) else {}

    def _calculate_partial_similarity(self, value1: str, value2: str) -> float:
        """Calculate partial similarity between two feature values"""
        # Define similar values
        similar_values = {
            'double': ['queen', 'full'],
            'king': ['california king', 'super king'],
            'twin': ['single'],
            'deluxe': ['luxury', 'premium'],
            'suite': ['apartment', 'studio'],
            'standard': ['classic', 'basic']
        }

        if value1 == value2:
            return 1.0

        # Check if values are similar
        for main_value, similar in similar_values.items():
            if (value1 == main_value and value2 in similar) or (value2 == main_value and value1 in similar):
                return 0.8

        return 0.0

    def _calculate_feature_similarity(self, ref_features: Dict, supp_features: Dict) -> float:
        weights = {
            'bedType': 0.3,
            'roomClass': 0.3,
            'viewType': 0.15,
            'boardType': 0.1,
            'accessibility': 0.05,
            'occupancy': 0.1
        }

        score = 0
        total_weight = 0

        for feature, weight in weights.items():
            ref_value = ref_features.get(feature, 'unknown')
            supp_value = supp_features.get(feature, 'unknown')

            if ref_value == 'unknown' or supp_value == 'unknown':
                continue

            if ref_value == supp_value:
                score += weight
            else:
                # Check for partial matches
                partial_score = self._get_partial_match_score(feature, ref_value, supp_value)
                score += weight * partial_score

            total_weight += weight

        return score / total_weight if total_weight > 0 else 0

    def _get_partial_match_score(self, feature: str, value1: str, value2: str) -> float:
        """Get partial match score for feature values"""
        if feature == 'bedType':
            bed_similarities = {
                'double': {'queen': 0.8, 'full': 0.9},
                'king': {'california king': 0.9, 'super king': 0.9},
                'twin': {'single': 0.9},
            }
            for main_type, similars in bed_similarities.items():
                if (value1 == main_type and value2 in similars) or (value2 == main_type and value1 in similars):
                    return similars.get(value2 if value1 == main_type else value1, 0)

        elif feature == 'roomClass':
            class_similarities = {
                'deluxe': {'luxury': 0.8, 'premium': 0.8},
                'standard': {'classic': 0.9, 'basic': 0.8},
                'suite': {'apartment': 0.7, 'studio': 0.6}
            }
            for main_class, similars in class_similarities.items():
                if (value1 == main_class and value2 in similars) or (value2 == main_class and value1 in similars):
                    return similars.get(value2 if value1 == main_class else value1, 0)

        return 0.0

    def _calculate_text_similarity(self, ref_text: str, supp_text: str) -> float:
        # Normalize text
        ref_text = ' '.join(sorted(ref_text.split()))
        supp_text = ' '.join(sorted(supp_text.split()))

        # Calculate TF-IDF similarity
        tfidf_matrix = self.vectorizer.fit_transform([ref_text, supp_text])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # Calculate Jaccard similarity
        ref_words = set(ref_text.split())
        supp_words = set(supp_text.split())
        jaccard_sim = len(ref_words & supp_words) / len(ref_words | supp_words)

        # Combine similarities
        return 0.7 * cosine_sim + 0.3 * jaccard_sim

    def _compare_feature(self, ref_features: Dict, supp_features: Dict, feature: str, weight: float) -> float:
        if feature not in ref_features or feature not in supp_features:
            return 0.0
        return weight if ref_features[feature] == supp_features[feature] else 0.0

    def match_rooms(self, reference_rooms: List[Dict], supplier_rooms: List[Dict], threshold: float = 0.8) -> Dict:
        matches = []
        matched_supplier_ids = set()

        for ref_room in reference_rooms:
            best_match = None
            best_score = 0

            for supp_room in supplier_rooms:
                if supp_room['id'] in matched_supplier_ids:
                    continue

                score = self.calculate_similarity(ref_room, supp_room)
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = supp_room

            if best_match:
                matches.append({
                    'reference_room': ref_room,
                    'supplier_room': best_match,
                    'similarity_score': best_score
                })
                matched_supplier_ids.add(best_match['id'])

        unmatched_ref = [r for r in reference_rooms if not any(m['reference_room']['id'] == r['id'] for m in matches)]
        unmatched_supp = [s for s in supplier_rooms if s['id'] not in matched_supplier_ids]

        return {
            "matched_pairs": matches,
            "unmatched_reference": unmatched_ref,
            "unmatched_supplier": unmatched_supp
        }