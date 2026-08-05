import pytest

from src.classifier import AWSServiceClassifier


def test_classify_returns_model_prediction(fake_model_path):
    classifier = AWSServiceClassifier(fake_model_path)

    result = classifier.classify("Creates an AWS bucket.")

    assert result == "s3"