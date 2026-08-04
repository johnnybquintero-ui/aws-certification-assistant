import pytest


@pytest.fixture
def fake_service_model():
    return {
        "metadata": {
            "serviceFullName": "Example Storage Service",
            "apiVersion": "2026-01-01",
        },
        "operations": {
            "CreateBucket": {
                "documentation": "<p>Creates a bucket.</p>",
            },
            "DeleteBucket": {
                "documentation": "<p>Deletes a bucket.</p>",
            },
        },
    }

@pytest.fixture
def fake_service_name():
    return "example"

@pytest.fixture
def fake_operations(fake_service_model, fake_service_name):
    return [
        {
            "service": fake_service_name,
            "service_name": fake_service_model["metadata"]["serviceFullName"],
            "api_version": fake_service_model["metadata"]["apiVersion"],
            "operation": operation_name,
            "description": operation_data.get("documentation", ""),
        }
        for operation_name, operation_data in fake_service_model["operations"].items()
    ]