from typing import List, Dict
import openai
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd


class RoomMatchEvaluator:
    def __init__(self, api_key: str):
        self.openai_client = openai.OpenAI(api_key=api_key)

    def generate_ground_truth(self, ref_room: Dict, supplier_room: Dict) -> int:
        prompt = f"""
        As an expert in hotel room matching, determine if these two room descriptions refer to the same room type.
        Consider all aspects including room type, bed configuration, view, and amenities.

        Reference Room: {ref_room['original_name']}
        Supplier Room: {supplier_room['original_name']}

        Cleaned Reference: {ref_room['cleaned_name']}
        Cleaned Supplier: {supplier_room['cleaned_name']}

        Rules for matching:
        1. Room type/class must match (e.g., deluxe, standard, suite)
        2. Bed configuration must match (e.g., king, twin, double)
        3. View type should match if specified
        4. Board type should match if specified
        5. Accessibility features should match if specified

        Respond with only:
        1 (if rooms match)
        0 (if rooms don't match)
        """

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",  # Use the most capable model for evaluation
            messages=[
                {"role": "system", "content": "You are a hotel room matching expert. Respond only with 1 or 0."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1,
            temperature=0
        )

        return int(response.choices[0].message.content.strip())

    def evaluate_matches(self, predictions: List[int], ground_truth: List[int]) -> Dict:
        """
        Evaluate matching performance with detailed metrics
        """
        try:
            metrics = {
                'precision': precision_score(ground_truth, predictions),
                'recall': recall_score(ground_truth, predictions),
                'f1': f1_score(ground_truth, predictions),
                'total_samples': len(predictions),
                'true_positives': sum((p == 1 and g == 1) for p, g in zip(predictions, ground_truth)),
                'false_positives': sum((p == 1 and g == 0) for p, g in zip(predictions, ground_truth)),
                'false_negatives': sum((p == 0 and g == 1) for p, g in zip(predictions, ground_truth)),
                'true_negatives': sum((p == 0 and g == 0) for p, g in zip(predictions, ground_truth))
            }
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise