import logging

from botocore.loaders import create_loader

logger = logging.getLogger(__name__)


SELECTED_SERVICES = [
    # Compute
    "ec2",
    "lambda",
    # Storage
    "s3",
    "efs",
    # Databases
    "rds",
    "dynamodb",
    # Networking
    "cloudfront",
   # Security
    "iam",
    "secretsmanager",
    "wafv2",
    "shield",
    "guardduty",
    "inspector2",
    "securityhub",
    # Monitoring and governance
    "cloudwatch",
    "logs",
    "cloudtrail",
    "config",
    # Application integration
    "sns",
    "events",
    "stepfunctions",
    # Containers
    "ecs",
    "eks",
    "ecr",
    # Analytics
    "athena",
    "glue",
]


def load_service_model(
    loader,
    service_name: str,
) -> dict:
    """Load the Botocore service model for an AWS service."""

    service_model = loader.load_service_model(
        service_name,
        "service-2",
    )

    metadata = service_model["metadata"]

    logger.info(
        "Loaded %s (%s), API version %s: %d operations",
        service_name,
        metadata["serviceFullName"],
        metadata["apiVersion"],
        len(service_model["operations"]),
    )

    return service_model


def extract_operations(
    service_name: str,
    service_model: dict,
) -> list[dict]:
    """Convert a Botocore service model into operation records."""

    operations = []
    metadata = service_model["metadata"]

    for operation_name, operation_data in service_model["operations"].items():
        operation_record = {
            "service": service_name,
            "service_name": metadata["serviceFullName"],
            "api_version": metadata["apiVersion"],
            "operation": operation_name,
            "description": operation_data.get(
                "documentation",
                "",
            ),
        }

        operations.append(operation_record)

        logger.debug(
            "Extracted operation %s.%s",
            service_name,
            operation_name,
        )

    return operations


def ingest_operations(
    service_names: list[str] | None = None,
) -> list[dict]:
    """Load and extract operations for all requested AWS services."""

    if service_names is None:
        service_names = SELECTED_SERVICES

    loader = create_loader()
    dataset_records = []

    for service_name in service_names:
        service_model = load_service_model(
            loader=loader,
            service_name=service_name,
        )

        service_records = extract_operations(
            service_name=service_name,
            service_model=service_model,
        )

        dataset_records.extend(service_records)

    logger.info(
        "Extracted %d operation records from %d services",
        len(dataset_records),
        len(service_names),
    )

    return dataset_records
