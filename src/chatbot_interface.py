from pathlib import Path

from src.classifier import AWSServiceClassifier
from src.language_model import LanguageModel

from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "aws_service_classifier.pkl"

EXIT_COMMANDS = {"exit", "quit"}

DEBUG = False

class AWSChatbot:
    def __init__(
        self,
        classifier: AWSServiceClassifier,
        language_model: LanguageModel,
    ) -> None:
        """Store the classifier and language model used by the chatbot."""

        self.classifier = classifier
        self.language_model = language_model

    def respond(self, user_input: str) -> str:
        """Classify and respond to a user request."""

        if not user_input.strip():
            raise ValueError("Please enter an AWS requirement.")

        predictions = self.classifier.classify_with_confidence(user_input)

        if DEBUG:
            print(f"\n[DEBUG] Classifier input: {user_input}")
            print(f"[DEBUG] Predictions: {predictions}")

        return self.language_model.generate_reply(
            original_input=user_input,
            predictions=predictions,
        )


def run_interface(chatbot: AWSChatbot) -> None:
    """Run the interactive command-line interface."""

    print("Assistant: Hi! Describe what you need to do in AWS.")
    print("Type 'exit' or 'quit' to finish.")

    while True:
        user_input = input("\nYou: ")

        if user_input.strip().lower() in EXIT_COMMANDS:
            print("Assistant: Goodbye!")
            break

        try:
            reply = chatbot.respond(user_input)
        except ValueError as error:
            print(f"Assistant: {error}")
            continue

        print(f"Assistant: {reply}")


def main() -> None:
    """Load the models and start the interface."""

    classifier = AWSServiceClassifier(MODEL_PATH)
    language_model = LanguageModel()

    chatbot = AWSChatbot(
        classifier=classifier,
        language_model=language_model,
    )

    run_interface(chatbot)


if __name__ == "__main__":
    main()
