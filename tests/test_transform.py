import pytest

from src.transform import (
    clean_description,
    transform_operations,
)


def test_clean_description_removes_html():
    description = "<p>Creates an <b>AWS bucket</b>.</p>"

    result = clean_description(description)

    assert result == "Creates an AWS bucket."


def test_clean_description_decodes_html_entities():
    description = "Creates buckets &amp; objects."

    result = clean_description(description)

    assert result == "Creates buckets & objects."


def test_clean_description_normalises_whitespace():
    description = "Creates   a\n\nnew\tbucket."

    result = clean_description(description)

    assert result == "Creates a new bucket."


def test_clean_description_returns_empty_string_for_none():
    result = clean_description(None)

    assert result == ""


def test_transform_operations_standardises_values(
    fake_operations,
):
    fake_operations[0]["service"] = " S3 "
    fake_operations[0]["operation"] = " CreateBucket "
    fake_operations[0]["description"] = "<p>Creates a bucket.</p>"

    result = transform_operations(fake_operations)

    assert result[0]["service"] == "s3"
    assert result[0]["operation"] == "CreateBucket"
    assert result[0]["description"] == "Creates a bucket."


def test_transform_operations_removes_duplicates(
    fake_operations,
):
    duplicate_operations = [
        fake_operations[0],
        fake_operations[0].copy(),
    ]

    result = transform_operations(duplicate_operations)

    assert len(result) == 1


def test_transform_operations_raises_for_missing_columns(
    fake_operations,
):
    del fake_operations[0]["description"]

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        transform_operations(fake_operations)
