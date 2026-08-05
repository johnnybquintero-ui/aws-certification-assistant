from pathlib import Path

from src.classifier import AWSServiceClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "aws_service_classifier.pkl"

EXIT_COMMANDS = {"exit", "quit"}


class AWSChatbot:
    def __init__(self, classifier: AWSServiceClassifier) -> None:
        """Store the classifier used by the chatbot."""

        self.classifier = classifier

    def respond(self, user_input: str) -> str:
        """Classify the user's original request."""

        cleaned_input = " ".join(user_input.split())

        if not cleaned_input:
            raise ValueError("Input must not be empty.")

        predictions = self.classifier.classify_with_confidence(cleaned_input)

        if not predictions:
            raise ValueError("The classifier returned no predictions.")

        print(f"\n[DEBUG] Predictions: {predictions[:3]}")

        top_service, confidence = predictions[0]

        return (
            f"The classifier suggests {top_service.upper()} "
            f"with {confidence:.1%} confidence."
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
    """Load the classifier and start the interface."""

    classifier = AWSServiceClassifier(MODEL_PATH)
    chatbot = AWSChatbot(classifier=classifier)

    run_interface(chatbot)


if __name__ == "__main__":
    main()
