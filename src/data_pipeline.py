import logging
from pathlib import Path

from src.export import export_to_parquet
from src.ingest import ingest_operations
from src.transform import transform_operations

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RAW_OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "operations.parquet"

DEFAULT_PROCESSED_OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "operations_clean.parquet"
)


def run_pipeline(
    raw_output_file: Path = DEFAULT_RAW_OUTPUT_FILE,
    processed_output_file: Path = DEFAULT_PROCESSED_OUTPUT_FILE,
) -> list[dict]:
    """Run the complete operation ingestion pipeline."""

    logger.info("Starting ingestion pipeline")

    raw_operations = ingest_operations()

    export_to_parquet(
        operations=raw_operations,
        output_file=raw_output_file,
    )

    cleaned_operations = transform_operations(raw_operations)

    export_to_parquet(
        operations=cleaned_operations,
        output_file=processed_output_file,
    )

    logger.info(
        "Pipeline completed: %d raw records and %d cleaned records",
        len(raw_operations),
        len(cleaned_operations),
    )

    return cleaned_operations


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s | %(levelname)s | " "%(name)s | %(message)s"),
    )

    run_pipeline()


if __name__ == "__main__":
    main()
