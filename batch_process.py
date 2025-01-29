import pandas as pd
from pathlib import Path
import time
from src.preprocessing.cleaner import RoomNameProcessor
from src.preprocessing.feature_extractor import FeatureExtractor
from src.matching.similarity import RoomMatcher
from src.utils.logger import setup_logger
from src.utils.benchmark_generator import BenchmarkGenerator
from config import settings

logger = setup_logger("batch_processor")


def process_batch(
        reference_file: str,
        supplier_file: str,
        output_dir: str,
        generate_benchmark: bool = True,
        benchmark_size: int = 100,
        noise_level: float = 0.2,
        max_retries: int = 3
) -> None:
    """Process batch with retries and better error handling."""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define file paths
        processed_file = output_dir / "processed_rooms.csv"
        benchmark_file = output_dir / "benchmark_dataset.csv"
        metrics_file = output_dir / "matching_metrics.csv"

        # Process rooms if processed file doesn't exist
        if not processed_file.exists():
            logger.info("Loading data files...")
            ref_df = pd.read_excel(reference_file)
            supp_df = pd.read_csv(supplier_file)

            processor = RoomNameProcessor()
            feature_extractor = FeatureExtractor()
            matcher = RoomMatcher()

            logger.info("Processing reference rooms...")
            ref_df['processed'] = ref_df['room_name'].apply(processor.process_room_name)

            logger.info("Processing supplier rooms...")
            supp_df['processed'] = supp_df['supplier_room_name'].apply(
                processor.process_room_name
            )

            # Save processed data
            results_df = pd.concat([ref_df, supp_df])
            results_df.to_csv(processed_file, index=False)
            logger.info(f"Processed data saved to {processed_file}")
        else:
            logger.info("Loading processed data from existing file...")
            results_df = pd.read_csv(processed_file)
            ref_df = results_df[results_df['room_name'].notna()]
            supp_df = results_df[results_df['supplier_room_name'].notna()]

        if generate_benchmark:
            if not benchmark_file.exists():
                logger.info("Generating new benchmark dataset...")
                for attempt in range(max_retries):
                    try:
                        benchmark_generator = BenchmarkGenerator(settings.OPENAI_API_KEY)
                        benchmark_df = benchmark_generator.generate_benchmark_dataset(
                            ref_df, supp_df,
                            sample_size=benchmark_size,
                            noise_level=noise_level
                        )

                        benchmark_df.to_csv(benchmark_file, index=False)
                        logger.info(f"Benchmark dataset saved to {benchmark_file}")
                        break

                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                            time.sleep(5)  # Wait before retry
                        else:
                            raise
            else:
                logger.info("Loading existing benchmark dataset...")
                benchmark_df = pd.read_csv(benchmark_file)

            # Always evaluate performance
            logger.info("Evaluating matching performance...")
            benchmark_generator = BenchmarkGenerator(settings.OPENAI_API_KEY)
            metrics = benchmark_generator.evaluate_matching_performance(benchmark_df)
            logger.info(f"Matching performance metrics: {metrics}")

            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(metrics_file, index=False)
            logger.info(f"Performance metrics saved to {metrics_file}")

    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise


def tune_parameters(benchmark_file: str = "data/processed/benchmark_dataset.csv"):
    """Tune similarity threshold parameters using existing benchmark dataset."""
    try:
        # Load benchmark dataset
        benchmark_df = pd.read_csv(benchmark_file)
        benchmark_generator = BenchmarkGenerator(settings.OPENAI_API_KEY)

        # Test different thresholds
        thresholds = [0.5, 0.6, 0.7, 0.8]
        results = {}

        logger.info("Tuning similarity threshold...")
        for threshold in thresholds:
            settings.SIMILARITY_THRESHOLD = threshold
            metrics = benchmark_generator.evaluate_matching_performance(benchmark_df)
            results[threshold] = metrics

            logger.info(f"\nThreshold: {threshold}")
            logger.info(f"F1: {metrics['f1']:.3f}")
            logger.info(f"Precision: {metrics['precision']:.3f}")
            logger.info(f"Recall: {metrics['recall']:.3f}")

        # Find best threshold
        best_threshold = max(results.items(), key=lambda x: x[1]['f1'])[0]
        logger.info(f"\nBest threshold: {best_threshold}")
        logger.info(f"Best F1: {results[best_threshold]['f1']:.3f}")

        return results

    except Exception as e:
        logger.error(f"Parameter tuning failed: {str(e)}")
        raise


if __name__ == "__main__":
    process_batch(
        reference_file="data/raw/referance_rooms.xlsx",
        supplier_file="data/raw/supplier_rooms.csv",
        output_dir="data/processed",
        generate_benchmark=True,
        benchmark_size=100,
        noise_level=0.2  # 20% noise
    )

    # Tune parameters using existing benchmark
    tune_parameters()