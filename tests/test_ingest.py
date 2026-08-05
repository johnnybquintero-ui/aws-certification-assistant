from src.ingest import load_service_model, extract_operations

import pyarrow.parquet as pq

from botocore.loaders import create_loader


def test_load_service_model_returns_dict():
    loader = create_loader()
    service_name = "s3"
    result = load_service_model(loader, service_name)
    assert isinstance(result, dict)


def test_load_service_model_contains_expected_keys():
    loader = create_loader()

    result = load_service_model(loader, "s3")

    expected_keys = [
        "metadata",
        "operations",
        "shapes",
        "documentation",
    ]

    for key in expected_keys:
        assert key in result


def test_extract_operations_returns_a_list(fake_service_model, fake_service_name):
    result = extract_operations(
        service_name=fake_service_name,
        service_model=fake_service_model,
    )

    assert isinstance(result, list)


def test_extract_operations_returns_one_record_per_operation(
    fake_service_model,
    fake_service_name,
):
    result = extract_operations(
        service_name=fake_service_name,
        service_model=fake_service_model,
    )

    assert len(result) == len(fake_service_model["operations"])


def test_extract_operations_contains_expected_keys(
    fake_service_model, fake_service_name
):

    result = extract_operations(
        service_name=fake_service_name,
        service_model=fake_service_model,
    )

    expected_keys = [
        "service",
        "service_name",
        "api_version",
        "operation",
        "description",
    ]

    assert all(all(key in record for key in expected_keys) for record in result)


def test_extract_operations_contains_correct_values(
    fake_service_model,
    fake_service_name,
):
    result = extract_operations(
        service_name=fake_service_name,
        service_model=fake_service_model,
    )

    for record in result:
        assert record["service"] == fake_service_name
        assert (
            record["service_name"] == fake_service_model["metadata"]["serviceFullName"]
        )
        assert record["api_version"] == fake_service_model["metadata"]["apiVersion"]
        assert record["operation"] in fake_service_model["operations"]
        assert record["description"] == fake_service_model["operations"][
            record["operation"]
        ].get("documentation", "")


def test_extract_operations_contains_correct_values(
    fake_service_model,
    fake_service_name,
):
    result = extract_operations(
        service_name=fake_service_name,
        service_model=fake_service_model,
    )

    expected = [
        {
            "service": fake_service_name,
            "service_name": "Example Storage Service",
            "api_version": "2026-01-01",
            "operation": "CreateBucket",
            "description": "<p>Creates a bucket.</p>",
        },
        {
            "service": fake_service_name,
            "service_name": "Example Storage Service",
            "api_version": "2026-01-01",
            "operation": "DeleteBucket",
            "description": "<p>Deletes a bucket.</p>",
        },
    ]

    assert result == expected


def test_extract_operations_uses_empty_string_when_documentation_missing(
    fake_service_model,
    fake_service_name,
):
    fake_service_model["operations"]["CreateBucket"].pop("documentation")

    result = extract_operations(
        service_name=fake_service_name,
        service_model=fake_service_model,
    )

    create_bucket_record = next(
        record for record in result if record["operation"] == "CreateBucket"
    )

    assert create_bucket_record["description"] == ""


def test_extract_operations_returns_empty_list_when_no_operations(
    fake_service_model,
    fake_service_name,
):
    fake_service_model["operations"] = {}

    result = extract_operations(
        service_name=fake_service_name,
        service_model=fake_service_model,
    )

    assert result == []
