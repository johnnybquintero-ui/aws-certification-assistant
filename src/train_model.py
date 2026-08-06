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
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

DEFAULT_DATA_SOURCE = Path("data/processed/operations_clean.parquet")
# Introduce a higher weight for the intent examples to help the classifier
DEFAULT_INTENTS_SOURCE = Path("data/service_intents.csv")
DEFAULT_MODEL_OUTPUT = Path("models/aws_service_classifier.pkl")

INTENT_WEIGHT = 3

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

    parser.add_argument(
        "--intents-source",
        type=Path,
        default=DEFAULT_INTENTS_SOURCE,
        help="Local path for the service-intent training dataset.",
    )

    parser.add_argument(
        "--include-intents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include service-intent examples in the training data.",
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


def load_intent_data(intents_path: Path) -> pd.DataFrame:
    """Load and validate the service-intent training dataset."""

    required_columns = {
        "service",
        "description",
    }

    dataframe = pd.read_csv(intents_path)

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Missing intent columns: {sorted(missing_columns)}")

    dataframe = dataframe[["service", "description"]].copy()

    dataframe["service"] = dataframe["service"].str.strip()
    dataframe["description"] = dataframe["description"].str.strip()

    if dataframe[["service", "description"]].isna().any().any():
        raise ValueError("Intent data contains missing values.")

    empty_columns = (dataframe[["service", "description"]] == "").any()

    if empty_columns.any():
        raise ValueError(
            "Intent data contains empty values in: "
            f"{empty_columns[empty_columns].index.tolist()}"
        )

    logger.info(
        "Loaded %d service intents across %d classes from %s",
        len(dataframe),
        dataframe["service"].nunique(),
        intents_path,
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
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
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
    confusion_matrix_path: Path,
    matrix_title: str,
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

    axis.set_title(matrix_title)

    confusion_matrix_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()
    figure.savefig(confusion_matrix_path)
    plt.close(figure)

    logger.info(
        "Saved confusion matrix visualisation to %s",
        confusion_matrix_path,
    )

    return metrics


def add_intents_to_training_data(
    X_train: pd.Series,
    y_train: pd.Series,
    intent_X_train: pd.Series,
    intent_y_train: pd.Series,
    weight: int = INTENT_WEIGHT,
) -> tuple[pd.Series, pd.Series]:
    """Add weighted intent examples to the Botocore training data."""

    if weight < 1:
        raise ValueError("Intent weight must be at least 1.")

    weighted_intent_X = pd.concat(
        [intent_X_train.reset_index(drop=True)] * weight,
        ignore_index=True,
    )

    weighted_intent_y = pd.concat(
        [intent_y_train.reset_index(drop=True)] * weight,
        ignore_index=True,
    )

    combined_X_train = pd.concat(
        [
            X_train.reset_index(drop=True),
            weighted_intent_X,
        ],
        ignore_index=True,
    )

    combined_y_train = pd.concat(
        [
            y_train.reset_index(drop=True),
            weighted_intent_y,
        ],
        ignore_index=True,
    )

    logger.info(
        "Added %d weighted intent samples to the training data",
        len(weighted_intent_X),
    )

    return combined_X_train, combined_y_train


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

    # Load both datasets. The intent data is needed as the shared test set
    # even when intents are not included in training.
    operations_dataframe = load_data(args.data_source)
    intents_dataframe = load_intent_data(args.intents_source)

    unknown_services = set(intents_dataframe["service"]) - set(
        operations_dataframe["service"]
    )

    if unknown_services:
        raise ValueError(
            "Intent data contains unknown services: " f"{sorted(unknown_services)}"
        )

    # Split the Botocore operation descriptions.
    X, y = prepare_training_data(operations_dataframe)

    (
        X_train,
        X_botocore_test,
        y_train,
        y_botocore_test,
    ) = split_data(X, y)

    # Separately split the user-style intent examples.
    intent_X, intent_y = prepare_training_data(intents_dataframe)

    (
        intent_X_train,
        intent_X_test,
        intent_y_train,
        intent_y_test,
    ) = split_data(intent_X, intent_y)

    logger.info(
        "Reserved %d intent samples for evaluation",
        len(intent_X_test),
    )

    if args.include_intents:
        # Add only the intent training split. The intent test split remains
        # unseen and can therefore be used for evaluation.
        X_train, y_train = add_intents_to_training_data(
            X_train,
            y_train,
            intent_X_train,
            intent_y_train,
        )

        variant = "post_intents"
        training_stage = "Post-Intents Training"

    else:
        logger.info(
            "Training without service-intent examples. "
            "Held-out intents will still be used for evaluation."
        )

        variant = "pre_intents"
        training_stage = "Pre-Intents Training"

    # Optional debug to validate training examples (currently configured to WAF)
    logger.debug(
        "Final training rows by service:\n%s",
        y_train.value_counts().sort_index().to_string(),
    )

    if args.include_intents:
        wafv2_training_examples = intent_X_train.loc[intent_y_train == "wafv2"]

        logger.debug(
            "WAF intent examples included in the training split:\n%s",
            wafv2_training_examples.to_string(index=False),
        )

    intent_matrix_title = (
        "AWS Service Classifier — Intent Test Set " f"({training_stage})"
    )

    botocore_matrix_title = (
        "AWS Service Classifier — Botocore Test Set " f"({training_stage})"
    )

    intent_matrix_path = (
        args.model_output.parent / f"confusion_matrix_intents_{variant}.png"
    )

    botocore_matrix_path = (
        args.model_output.parent / f"confusion_matrix_botocore_{variant}.png"
    )

    model = train_classifier(
        X_train,
        y_train,
    )

    # Evaluate performance on unseen user-style requests.
    evaluate_classifier(
        model,
        intent_X_test,
        intent_y_test,
        confusion_matrix_path=intent_matrix_path,
        matrix_title=intent_matrix_title,
    )

    # Evaluate performance on unseen Botocore documentation.
    evaluate_classifier(
        model,
        X_botocore_test,
        y_botocore_test,
        confusion_matrix_path=botocore_matrix_path,
        matrix_title=botocore_matrix_title,
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
        format=("%(asctime)s | %(levelname)s | " "%(name)s | %(message)s"),
    )

    main()
