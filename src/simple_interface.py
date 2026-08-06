import logging
from pathlib import Path

from src.classifier import AWSServiceClassifier

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "aws_service_classifier.pkl"

EXIT_COMMANDS = {"exit", "quit"}


def run_interface(classifier: AWSServiceClassifier) -> None:
    print("AWS Service Classifier")
    print("Describe an AWS resource operation or requirement.")
    print("Type 'exit' or 'quit' to close the program.")

    while True:
        user_input = input("\nEnter a description: ")

        if user_input.strip().lower() in EXIT_COMMANDS:
            print("Goodbye!")
            break

        try:
            predictions = classifier.classify_with_confidence(user_input)
        except ValueError as error:
            print(f"[Error] {error}")
            continue

        best_service, best_probability = predictions[0]

        print(f"[Result] {best_service} " f"({best_probability:.1%} confidence)")

        print("[Other possibilities]")

        for service, probability in predictions[1:]:
            print(f"- {service}: {probability:.1%}")


def main() -> None:
    try:
        classifier = AWSServiceClassifier(MODEL_PATH)
    except FileNotFoundError:
        logger.error("Classifier model not found at %s", MODEL_PATH)
        print(
            "[Error] Trained model not found. " "Run 'python -m src.train_model' first."
        )
        return

    run_interface(classifier)


if __name__ == "__main__":
    main()
