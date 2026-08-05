import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


OPERATION_SCHEMA = pa.schema(
    [
        ("service", pa.string()),
        ("service_name", pa.string()),
        ("api_version", pa.string()),
        ("operation", pa.string()),
        ("description", pa.string()),
    ]
)


def export_to_parquet(
    operations: list[dict],
    output_file: Path,
) -> None:
    """Export operation records to a Parquet file."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pa.Table.from_pylist(
        operations,
        schema=OPERATION_SCHEMA,
    )

    pq.write_table(
        table,
        output_file,
        compression="snappy",
    )

    logger.info(
        "Exported %d operation records to %s",
        len(operations),
        output_file,
    )
