import html
import logging
import re

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = {
    "service",
    "service_name",
    "api_version",
    "operation",
    "description",
}


def clean_description(
    description: str | None,
) -> str:
    """Remove HTML and normalise whitespace in a description."""

    if not description:
        return ""

    decoded_description = html.unescape(description)

    plain_text = BeautifulSoup(
        decoded_description,
        "html.parser",
    ).get_text(separator=" ")

    plain_text = re.sub(
        r"\s+",
        " ",
        plain_text,
    )

    plain_text = re.sub(
        r"\s+([.,!?;:])",
        r"\1",
        plain_text,
    )

    return plain_text.strip()


def transform_operations(
    operations: list[dict],
) -> list[dict]:
    """Clean and validate extracted AWS operation records."""

    transformed_operations = []
    seen_operations = set()

    for operation in operations:
        missing_columns = REQUIRED_COLUMNS - operation.keys()

        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        transformed_operation = {
            "service": operation["service"].strip().lower(),
            "service_name": operation["service_name"].strip(),
            "api_version": operation["api_version"].strip(),
            "operation": operation["operation"].strip(),
            "description": clean_description(operation["description"]),
        }

        if not transformed_operation["description"]:
            logger.warning(
                "Removed operation with empty description: %s.%s",
                transformed_operation["service"],
                transformed_operation["operation"],
            )
            continue

        operation_key = (
            transformed_operation["service"],
            transformed_operation["operation"],
        )

        if operation_key in seen_operations:
            logger.warning(
                "Removed duplicate operation: %s.%s",
                *operation_key,
            )
            continue

        seen_operations.add(operation_key)
        transformed_operations.append(transformed_operation)

    logger.info(
        "Transformed %d records into %d cleaned records",
        len(operations),
        len(transformed_operations),
    )

    return transformed_operations
