import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple
from src.preprocessing.cleaner import RoomNameProcessor
from src.preprocessing.feature_extractor import FeatureExtractor
from src.matching.similarity import RoomMatcher
from src.evaluation.metrics import RoomMatchEvaluator
from config import settings  # Add this import
import logging

logger = logging.getLogger(__name__)


class BenchmarkGenerator:
    def __init__(self, api_key: str):
        self.processor = RoomNameProcessor()
        self.feature_extractor = FeatureExtractor()
        self.matcher = RoomMatcher()
        self.evaluator = RoomMatchEvaluator(api_key)

    def generate_benchmark_dataset(
            self,
            reference_df: pd.DataFrame,
            supplier_df: pd.DataFrame,
            sample_size: int = 100,
            noise_level: float = 0.2  # 20% noise
    ) -> pd.DataFrame:
        """Generate a balanced benchmark dataset with controlled noise."""
        try:
            logger.info(f"Generating balanced benchmark dataset with {noise_level * 100}% noise...")

            # Base room types
            room_types = ["Single", "Double", "Twin", "King", "Queen", "Suite", "Studio", "Deluxe", "Superior",
                          "Standard"]
            bed_types = ["Single Bed", "Double Bed", "Twin Beds", "King Bed", "Queen Bed"]
            views = ["City View", "Ocean View", "Garden View", "Mountain View", "Pool View"]
            amenities = ["Balcony", "Terrace", "Kitchen", "Lounge Access", "Private Bathroom"]
            board_types = ["Room Only", "Breakfast Included", "Half Board", "All Inclusive"]

            def generate_room_name():
                """Generate a random room name with variations."""
                components = []

                # Add room class sometimes
                if random.random() < 0.7:
                    components.append(random.choice(["Standard", "Deluxe", "Superior", "Executive", "Premium"]))

                # Add main room type
                components.append(random.choice(room_types))

                # Add bed type sometimes
                if random.random() < 0.6:
                    components.append(random.choice(bed_types))

                # Add view sometimes
                if random.random() < 0.4:
                    components.append(random.choice(views))

                # Add amenity sometimes
                if random.random() < 0.3:
                    components.append(random.choice(amenities))

                # Add board type sometimes
                if random.random() < 0.2:
                    components.append(random.choice(board_types))

                return " - ".join(components)

            def add_noise(name: str, noise_level: float) -> str:
                """Add noise to room names."""
                if random.random() > noise_level:
                    return name

                noise_types = [
                    lambda x: x.replace("Double", "Twin"),  # Swap bed types
                    lambda x: x.replace("King", "Queen"),
                    lambda x: x.replace("City View", "Ocean View"),  # Swap views
                    lambda x: x.replace("Deluxe", "Superior"),  # Swap room classes
                    lambda x: x.replace("Standard", "Basic"),
                    lambda x: x + " (Renovated)",  # Add extra info
                    lambda x: x + " (Non-Smoking)",
                    lambda x: x.replace(" - ", " "),  # Change separators
                    lambda x: x.replace(" ", "_"),
                    lambda x: x.lower(),  # Change case
                    lambda x: x.upper(),
                ]

                # Apply random noise
                noise_func = random.choice(noise_types)
                return noise_func(name)

            # Generate matching pairs
            matching_data = []
            for i in range(sample_size // 2):
                base_name = generate_room_name()
                ref_name = add_noise(base_name, noise_level / 2)  # Less noise for reference
                supp_name = add_noise(base_name, noise_level)  # More noise for supplier

                matching_data.append({
                    'reference_room': ref_name,
                    'supplier_room': supp_name,
                    'reference_cleaned': self.processor.process_room_name(ref_name)['cleaned_name'],
                    'supplier_cleaned': self.processor.process_room_name(supp_name)['cleaned_name'],
                    'ground_truth': 1
                })

            # Generate non-matching pairs
            non_matching_data = []
            for i in range(sample_size // 2):
                ref_name = generate_room_name()
                supp_name = generate_room_name()  # Generate completely different name

                # Add noise to both names
                ref_name = add_noise(ref_name, noise_level / 2)
                supp_name = add_noise(supp_name, noise_level)

                non_matching_data.append({
                    'reference_room': ref_name,
                    'supplier_room': supp_name,
                    'reference_cleaned': self.processor.process_room_name(ref_name)['cleaned_name'],
                    'supplier_cleaned': self.processor.process_room_name(supp_name)['cleaned_name'],
                    'ground_truth': 0
                })

            # Add some ambiguous cases
            ambiguous_count = int(sample_size * 0.1)  # 10% ambiguous cases
            ambiguous_data = []
            for i in range(ambiguous_count):
                base_name = generate_room_name()
                ref_name = add_noise(base_name, noise_level * 2)  # Extra noise
                supp_name = add_noise(base_name, noise_level * 2)

                # Randomly assign ground truth for ambiguous cases
                ground_truth = random.choice([0, 1])

                ambiguous_data.append({
                    'reference_room': ref_name,
                    'supplier_room': supp_name,
                    'reference_cleaned': self.processor.process_room_name(ref_name)['cleaned_name'],
                    'supplier_cleaned': self.processor.process_room_name(supp_name)['cleaned_name'],
                    'ground_truth': ground_truth
                })

            # Combine all data
            all_data = matching_data + non_matching_data + ambiguous_data
            random.shuffle(all_data)

            # Create DataFrame
            benchmark_df = pd.DataFrame(all_data)

            # Log statistics
            match_count = sum(benchmark_df['ground_truth'])
            non_match_count = len(benchmark_df) - match_count

            logger.info(f"Generated benchmark dataset with:")
            logger.info(f"- Total pairs: {len(benchmark_df)}")
            logger.info(f"- Matching pairs: {match_count}")
            logger.info(f"- Non-matching pairs: {non_match_count}")
            logger.info(f"- Ambiguous pairs: {ambiguous_count}")

            return benchmark_df

        except Exception as e:
            logger.error(f"Error generating benchmark dataset: {e}")
            raise

    def analyze_matching_decisions(self, benchmark_df: pd.DataFrame) -> Dict:
        """Analyze matching decisions with better error handling."""
        analysis = {
            'matches': [],
            'mismatches': [],
            'similarity_distribution': [],
            'errors': []
        }

        total = len(benchmark_df)
        for idx, row in benchmark_df.iterrows():
            try:
                if idx % 10 == 0:
                    logger.info(f"Processing {idx}/{total} pairs...")

                ref_dict = {
                    'cleaned_name': row['reference_cleaned'],
                    'features': self.feature_extractor.extract_features(row['reference_cleaned'])
                }
                supp_dict = {
                    'cleaned_name': row['supplier_cleaned'],
                    'features': self.feature_extractor.extract_features(row['supplier_cleaned'])
                }

                similarity = self.matcher.calculate_similarity(ref_dict, supp_dict)
                match_decision = 1 if similarity >= settings.SIMILARITY_THRESHOLD else 0

                details = {
                    'reference_room': row['reference_room'],
                    'supplier_room': row['supplier_room'],
                    'similarity_score': similarity,
                    'ground_truth': row['ground_truth'],
                    'features_ref': ref_dict['features'],
                    'features_supp': supp_dict['features']
                }

                if match_decision != row['ground_truth']:
                    analysis['mismatches'].append(details)
                else:
                    analysis['matches'].append(details)

                analysis['similarity_distribution'].append(similarity)

            except Exception as e:
                logger.error(f"Error processing pair {idx}: {str(e)}")
                analysis['errors'].append({
                    'index': idx,
                    'reference_room': row['reference_room'],
                    'supplier_room': row['supplier_room'],
                    'error': str(e)
                })

        # Log summary statistics
        logger.info(f"\nAnalysis Summary:")
        logger.info(f"Total pairs processed: {total}")
        logger.info(f"Successful matches: {len(analysis['matches'])}")
        logger.info(f"Mismatches: {len(analysis['mismatches'])}")
        logger.info(f"Errors: {len(analysis['errors'])}")

        if analysis['similarity_distribution']:
            similarities = analysis['similarity_distribution']
            logger.info(f"\nSimilarity Scores:")
            logger.info(f"Mean: {np.mean(similarities):.3f}")
            logger.info(f"Median: {np.median(similarities):.3f}")
            logger.info(f"Min: {min(similarities):.3f}")
            logger.info(f"Max: {max(similarities):.3f}")

        return analysis

    def evaluate_matching_performance(self, benchmark_df: pd.DataFrame) -> Dict:
        try:
            # Perform analysis
            analysis = self.analyze_matching_decisions(benchmark_df)

            # Log detailed analysis
            logger.info("\nMatching Analysis:")
            logger.info(f"Total pairs: {len(benchmark_df)}")
            logger.info(f"Number of matches: {len(analysis['matches'])}")
            logger.info(f"Number of mismatches: {len(analysis['mismatches'])}")

            # Calculate similarity statistics
            similarities = analysis['similarity_distribution']
            logger.info(f"\nSimilarity Score Statistics:")
            logger.info(f"Mean similarity: {np.mean(similarities):.3f}")
            logger.info(f"Median similarity: {np.median(similarities):.3f}")
            logger.info(f"Min similarity: {min(similarities):.3f}")
            logger.info(f"Max similarity: {max(similarities):.3f}")

            # Log some example mismatches
            if analysis['mismatches']:
                logger.info("\nExample Mismatches:")
                for mismatch in analysis['mismatches'][:3]:
                    logger.info(f"\nReference: {mismatch['reference_room']}")
                    logger.info(f"Supplier: {mismatch['supplier_room']}")
                    logger.info(f"Similarity: {mismatch['similarity_score']:.3f}")
                    logger.info(f"Ground Truth: {mismatch['ground_truth']}")
                    logger.info(f"Reference Features: {mismatch['features_ref']}")
                    logger.info(f"Supplier Features: {mismatch['features_supp']}")

            # Calculate metrics with different thresholds
            thresholds = np.arange(0.5, 0.9, 0.1)
            best_metrics = None
            best_f1 = 0
            best_threshold = None

            for threshold in thresholds:
                predictions = [1 if score >= threshold else 0
                               for score in analysis['similarity_distribution']]

                metrics = self.evaluator.evaluate_matches(
                    predictions,
                    benchmark_df['ground_truth'].tolist()
                )

                logger.info(f"\nThreshold {threshold:.1f}:")
                logger.info(f"F1: {metrics['f1']:.3f}")
                logger.info(f"Precision: {metrics['precision']:.3f}")
                logger.info(f"Recall: {metrics['recall']:.3f}")

                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_metrics = metrics
                    best_threshold = threshold

            best_metrics.update({
                'total_pairs': len(benchmark_df),
                'matched_pairs': len(analysis['matches']),
                'ground_truth_matches': sum(benchmark_df['ground_truth']),
                'best_threshold': best_threshold,
                'mean_similarity': float(np.mean(similarities)),
                'median_similarity': float(np.median(similarities))
            })

            return best_metrics

        except Exception as e:
            logger.error(f"Error evaluating matching performance: {e}")
            raise


def analyze_mismatches(self, benchmark_df: pd.DataFrame) -> Dict:
    """Analyze false positives and false negatives"""
    analysis = {
        'false_negatives': [],
        'similarity_scores': []
    }

    for idx, row in benchmark_df.iterrows():
        ref_dict = {
            'cleaned_name': row['reference_cleaned'],
            'features': self.feature_extractor.extract_features(row['reference_cleaned'])
        }
        supp_dict = {
            'cleaned_name': row['supplier_cleaned'],
            'features': self.feature_extractor.extract_features(row['supplier_cleaned'])
        }

        similarity = self.matcher.calculate_similarity(ref_dict, supp_dict)

        # If ground truth is match but we didn't catch it
        if row['ground_truth'] == 1 and similarity < settings.SIMILARITY_THRESHOLD:
            analysis['false_negatives'].append({
                'reference_room': row['reference_room'],
                'supplier_room': row['supplier_room'],
                'similarity_score': similarity,
                'features_ref': ref_dict['features'],
                'features_supp': supp_dict['features']
            })

        analysis['similarity_scores'].append(similarity)

    return analysis


def cross_validate(self, reference_df: pd.DataFrame, supplier_df: pd.DataFrame,
                   n_splits: int = 5) -> Dict:
    """Perform cross-validation to find optimal parameters"""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_list = []

    for train_idx, test_idx in kf.split(reference_df):
        ref_train = reference_df.iloc[train_idx]
        ref_test = reference_df.iloc[test_idx]

        # Generate and evaluate benchmark dataset
        benchmark_df = self.generate_benchmark_dataset(ref_test, supplier_df)
        metrics = self.evaluate_matching_performance(benchmark_df)
        metrics_list.append(metrics)

    return {
        'mean_precision': np.mean([m['precision'] for m in metrics_list]),
        'mean_recall': np.mean([m['recall'] for m in metrics_list]),
        'mean_f1': np.mean([m['f1'] for m in metrics_list])
    }