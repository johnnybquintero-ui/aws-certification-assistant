import pickle
from pathlib import Path

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


class AlwaysS3Model:
    def predict(self, descriptions):
        return ["s3"]


@pytest.fixture
def fake_model_path(tmp_path: Path) -> Path:
    model_path = tmp_path / "test_model.pkl"

    with model_path.open("wb") as model_file:
        pickle.dump(AlwaysS3Model(), model_file)

    return model_path


class FakeClassifier:
    def __init__(self):
        self.received_inputs = []

    def classify_with_confidence(self, description):
        # Records what the interface passed to the classifier.
        self.received_inputs.append(description)

        # Returns predictable fake probabilities.
        return [
            ("s3", 0.80),
            ("glue", 0.15),
            ("rds", 0.05),
        ]


@pytest.fixture
def fake_classifier():
    return FakeClassifier()
