import pandas as pd
import pytest

from src.train_model import (
    load_data,
    prepare_training_data,
    save_model,
    split_data,
    train_classifier,
    evaluate_classifier,
)

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


def test_load_data_returns_parquet_dataframe(tmp_path):
    expected_dataframe = pd.DataFrame(
        {
            "service": ["s3", "lambda"],
            "operation": ["PutObject", "Invoke"],
            "description": [
                "Adds an object to a bucket.",
                "Invokes a function.",
            ],
        }
    )

    data_path = tmp_path / "operations_clean.parquet"
    expected_dataframe.to_parquet(data_path)

    result = load_data(data_path)

    pd.testing.assert_frame_equal(
        result,
        expected_dataframe,
    )

def test_load_data_raises_error_when_column_missing(
    tmp_path,
):
    dataframe = pd.DataFrame(
        {
            "service": ["s3"],
            "operation": ["PutObject"],
        }
    )

    data_path = tmp_path / "invalid.parquet"
    dataframe.to_parquet(data_path)

    with pytest.raises(
        ValueError,
        match="Missing required columns.*description",
    ):
        load_data(data_path)

def test_prepare_training_data_returns_features_and_labels():
    dataframe = pd.DataFrame(
        {
            "service": ["s3", "lambda"],
            "operation": ["PutObject", "Invoke"],
            "description": [
                "Adds an object to a bucket.",
                "Invokes a function.",
            ],
        }
    )

    X, y = prepare_training_data(dataframe)

    assert X.tolist() == [
        "Adds an object to a bucket.",
        "Invokes a function.",
    ]

    assert y.tolist() == [
        "s3",
        "lambda",
    ]

def test_split_data_creates_reproducible_stratified_split():
    X = pd.Series(
        [
            "s3 text 1",
            "s3 text 2",
            "s3 text 3",
            "s3 text 4",
            "lambda text 1",
            "lambda text 2",
            "lambda text 3",
            "lambda text 4",
            "rds text 1",
            "rds text 2",
            "rds text 3",
            "rds text 4",
        ]
    )

    y = pd.Series(
        ["s3"] * 4
        + ["lambda"] * 4
        + ["rds"] * 4
    )

    first_split = split_data(X, y)
    second_split = split_data(X, y)

    X_train, X_test, y_train, y_test = first_split

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    assert set(y_train.unique()) == {"s3", "lambda", "rds"}
    assert set(y_test.unique()) == {"s3", "lambda", "rds"}

    for first_result, second_result in zip(
        first_split,
        second_split,
    ):
        pd.testing.assert_series_equal(
            first_result,
            second_result,
        )

def test_train_classifier_returns_fitted_pipeline():
    X_train = pd.Series(
        [
            "Upload an object to an S3 bucket",
            "Download an object from a bucket",
            "Store files in object storage",
            "Invoke a Lambda function",
            "Run serverless function code",
            "Configure a function runtime",
        ]
    )

    y_train = pd.Series(
        [
            "s3",
            "s3",
            "s3",
            "lambda",
            "lambda",
            "lambda",
        ]
    )

    model = train_classifier(X_train, y_train)

    predictions = model.predict(
        pd.Series(
            [
                "Upload a file to a bucket",
                "Invoke serverless function code",
            ]
        )
    )

    #The model returned is a scikit-learn Pipeline object.
    assert isinstance(model, Pipeline)

    check_is_fitted(
        model.named_steps["vectoriser"]
    )
    check_is_fitted(
        model.named_steps["classifier"]
    )

    predictions = model.predict(
        pd.Series(
            [
                "Upload a file to a bucket",
                "Invoke serverless function code",
            ]
        )
    )
    #One prediction was returned for each of the two input descriptions.
    assert len(predictions) == 2
    #The predictions are valid service names from the training data.
    assert set(predictions).issubset(
        {"s3", "lambda"}
    )

def test_evaluate_classifier_returns_expected_metrics():
    X_train = pd.Series(
        [
            "Upload an object to an S3 bucket",
            "Download an object from a bucket",
            "Store files in object storage",
            "Invoke a Lambda function",
            "Run serverless function code",
            "Configure a function runtime",
        ]
    )

    y_train = pd.Series(
        [
            "s3",
            "s3",
            "s3",
            "lambda",
            "lambda",
            "lambda",
        ]
    )

    X_test = pd.Series(
        [
            "Upload an object to an S3 bucket",
            "Invoke a Lambda function",
        ]
    )

    y_test = pd.Series(
        [
            "s3",
            "lambda",
        ]
    )

    model = train_classifier(X_train, y_train)

    metrics = evaluate_classifier(
        model,
        X_test,
        y_test,
    )

    assert isinstance(metrics, dict)

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    }

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1_score"] == pytest.approx(1.0)

def test_save_model_creates_output_file_and_correct_type(tmp_path):
    model = train_classifier(
        pd.Series(
            [
                "Upload an object to a bucket",
                "Download an object from a bucket",
                "Invoke a Lambda function",
                "Configure a function runtime",
            ]
        ),
        pd.Series(
            [
                "s3",
                "s3",
                "lambda",
                "lambda",
            ]
        ),
    )

    output_file = tmp_path / "trained_model.pkl"

    save_model(model, output_file)

    assert output_file.exists()
    assert output_file.is_file()
    assert output_file.suffix == ".pkl"