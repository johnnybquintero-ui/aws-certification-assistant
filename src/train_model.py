import argparse
import logging
import pickle
from pathlib import Path
from time import perf_counter

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt


DEFAULT_DATA_SOURCE = Path(
    "data/processed/operations_clean.parquet"
)
DEFAULT_MODEL_OUTPUT = Path(
    "models/aws_service_classifier.pkl"
)

# Set a random number so that our results are the 
# same every time we run our code. 
RANDOM_SEED = 42

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an AWS service text classifier."
    )

    parser.add_argument(
        "--data-source",
        type=Path,
        default=DEFAULT_DATA_SOURCE,
        help="Local path for the cleaned Parquet dataset.",
    )

    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_OUTPUT,
        help="Location for the saved trained model.",
    )

    return parser.parse_args()

def load_data(data_path: Path) -> pd.DataFrame:
    """Load the cleaned AWS operations dataset."""

    # Validate that the required columns are present in the dataset
    # operation is unused in the model training, 
    # but we still want to keep it in the dataset for future reference.
    required_columns = {
        "service",
        "operation",
        "description",
    }

    dataframe = pd.read_parquet(data_path)

    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    logger.info(
        "Loaded %d cleaned records from %s",
        len(dataframe),
        data_path,
    )

    return dataframe

def prepare_training_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Separate model inputs from target labels."""

    X = dataframe["description"]
    y = dataframe["service"]

    logger.info(
        "Prepared %d samples across %d service classes",
        len(X),
        y.nunique(),
    )

    return X, y


def split_data(
    X: pd.Series,
    y: pd.Series,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Split text samples and labels into training and test sets."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    logger.info(
        "Split %d samples into %d training and %d test samples",
        len(X),
        len(X_train),
        len(X_test),
    )

    return X_train, X_test, y_train, y_test

def train_classifier(
    X_train: pd.Series,
    y_train: pd.Series,
) -> Pipeline:
    """Train a text classifier for AWS service classification."""

    pipeline = Pipeline(
        steps=[
            (
                "vectoriser",
                TfidfVectorizer(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    logger.info(
        "Training with %d samples across %d service classes",
        len(X_train),
        y_train.nunique(),
    )

    pipeline.fit(X_train, y_train)

    logger.info("Classifier training completed")

    return pipeline

def evaluate_classifier(
    model: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate the classifier using unseen test data."""

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }

    logger.info("Evaluation metrics: %s", metrics)

    logger.info(
        "Classification report:\n%s",
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        ),
    )

    labels = model.named_steps["classifier"].classes_

    normalised_matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
        normalize="true",
    )

    figure, axis = plt.subplots(
        figsize=(18, 16),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=normalised_matrix,
        display_labels=labels,
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        xticks_rotation=90,
        values_format=".0%",
        colorbar=False,
    )

    axis.set_title(
        "AWS Service Classifier Confusion Matrix"
    )

    figure.tight_layout()
    figure.savefig(
        "models/confusion_matrix.png",
        dpi=200,
    )

    plt.close(figure)

    logger.info(
        "Saved confusion matrix visualisation to "
        "models/confusion_matrix.png"
    )

    return metrics

def save_model(
    model: Pipeline,
    model_path: Path,
) -> None:
    """Save the fitted classification pipeline."""

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with model_path.open("wb") as file:
        pickle.dump(model, file)

    logger.info("Saved trained model to %s", model_path)

def main() -> None:
    """Run the complete model-training pipeline."""

    args = parse_arguments()
    start_time = perf_counter()

    logger.info("Starting AWS service classifier training")

    dataframe = load_data(args.data_source)
    X, y = prepare_training_data(dataframe)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_classifier(
        X_train,
        y_train,
    )

    evaluate_classifier(
        model,
        X_test,
        y_test,
    )

    save_model(
        model,
        args.model_output,
    )

    logger.info(
        "Training pipeline completed in %.2f seconds",
        perf_counter() - start_time,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    main()