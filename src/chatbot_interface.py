import logging
from pathlib import Path

from src.classifier import AWSServiceClassifier
from src.language_model import LanguageModel

from transformers.utils import logging as transformers_logging

from src.chatbot_logging import configure_chatbot_logging

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "aws_service_classifier.pkl"

EXIT_COMMANDS = {"exit", "quit"}

DEBUG = False
MINIMUM_CONFIDENCE = 0.12
MINIMUM_MARGIN = 0.03

class AWSChatbot:
    def __init__(
        self,
        classifier: AWSServiceClassifier,
        language_model: LanguageModel,
        logger: logging.Logger,
    ) -> None:
        """Store the classifier and language model used by the chatbot."""

        self.classifier = classifier
        self.language_model = language_model
        self.logger = logger

    def respond(self, user_input: str) -> str:
        """Classify and respond to a user request."""

        if not user_input.strip():
            self.logger.warning(
                "status=input_rejected | user_input=%r",
                user_input,
            )
            raise ValueError("Please enter an AWS requirement.")

        predictions = self.classifier.classify_with_confidence(user_input)

        self.logger.debug(
            "Classifier input=%r | predictions=%r",
            user_input,
            predictions,
        )

        # Check the strength of the classification before calling the LLM.
        top_service, top_confidence = predictions[0]
        second_service, second_confidence = predictions[1]

        confidence_margin = top_confidence - second_confidence

        if (
            top_confidence < MINIMUM_CONFIDENCE
            or confidence_margin < MINIMUM_MARGIN
        ):
            reply = (
                f"I'm currently deciding between {top_service.upper()} and "
                f"{second_service.upper()}, but I don't have enough confidence "
                "to recommend one yet. Could you describe the workload or "
                "resource involved and the specific outcome you need?"
            )
            status = "clarification_required"

        else:
            reply = self.language_model.generate_reply(
                original_input=user_input,
                predictions=predictions,
            )
            status = "completed"

        self.logger.info(
            "status=%s | user_input=%r | predictions=%r | reply=%r",
            status,
            user_input,
            predictions,
            reply,
        )

        return reply

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

    logger = configure_chatbot_logging()

    classifier = AWSServiceClassifier(MODEL_PATH)
    language_model = LanguageModel()

    chatbot = AWSChatbot(
        classifier=classifier,
        language_model=language_model,
        logger=logger,
    )

    run_interface(chatbot)


if __name__ == "__main__":
    main()
