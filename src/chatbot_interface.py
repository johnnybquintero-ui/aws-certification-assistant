import logging
from pathlib import Path

from src.classifier import AWSServiceClassifier
from src.language_model import LanguageModel
from src.chatbot_logging import configure_chatbot_logging
from src.rag import (
    chunk_documents,
    create_vector_store,
    load_documents,
    load_embedding_model,
    retrieve_relevant_chunks,
)

from transformers.utils import logging as transformers_logging

from sentence_transformers import SentenceTransformer

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "aws_service_classifier.pkl"

# Directory containing the CLF-C02 knowledge-base text files.
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "data" / "knowledge_base" / "clf_c02"

EXIT_COMMANDS = {"exit", "quit"}

DEBUG = False
MINIMUM_CONFIDENCE = 0.12
MINIMUM_MARGIN = 0.03


def is_service_recommendation(user_input: str) -> bool:
    """Return whether the user appears to be requesting an AWS service."""

    recommendation_markers = (
        "i need",
        "which service",
        "what service",
        "recommend",
        "what should i use",
        "best service",
    )

    normalised_input = user_input.lower()

    return any(marker in normalised_input for marker in recommendation_markers)


class AWSChatbot:
    def __init__(
        self,
        classifier: AWSServiceClassifier,
        language_model: LanguageModel,
        logger: logging.Logger,
        embedding_model: SentenceTransformer,
        vector_store: list[tuple],
    ) -> None:
        """Store the classifier and language model used by the chatbot."""

        self.classifier = classifier
        self.language_model = language_model
        self.logger = logger
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def respond(self, user_input: str) -> str:
        """Classify the input, retrieve context and generate a reply."""

        # Reject empty or whitespace-only input.
        if not user_input.strip():
            self.logger.warning(
                "status=input_rejected | user_input=%r",
                user_input,
            )

            raise ValueError("Please enter an AWS requirement.")

        # Decide whether this input is asking for an AWS service
        # recommendation or asking a general/exam question.
        recommendation_request = is_service_recommendation(user_input)

        # Exam questions do not use the service classifier, so they
        # keep an empty predictions list.
        predictions = []

        # Assume that clarification is not required unless the
        # classifier results prove otherwise.
        clarification_required = False

        # Only service-recommendation requests should use the classifier.
        if recommendation_request:
            predictions = self.classifier.classify_with_confidence(user_input)

            # These values can only be extracted when classifier
            # predictions exist. This is why they belong inside
            # the recommendation_request branch.
            top_service, top_confidence = predictions[0]
            second_service, second_confidence = predictions[1]

            # Work out how far ahead the strongest prediction is.
            confidence_margin = top_confidence - second_confidence

            # Decide whether the classifier is confident enough to
            # make a recommendation.
            clarification_required = (
                top_confidence < MINIMUM_CONFIDENCE
                or confidence_margin < MINIMUM_MARGIN
            )

        # This is safe for both routes:
        #
        # Service request: predictions contains classifier results.
        # Exam question: predictions is an empty list.
        self.logger.debug(
            "Classifier input=%r | predictions=%r",
            user_input,
            predictions,
        )

        # If a service request is too uncertain, ask for more detail.
        # The RAG retriever and language model are not needed here.
        if clarification_required:
            reply = (
                f"I'm currently deciding between "
                f"{top_service.upper()} and "
                f"{second_service.upper()}, but I don't have enough "
                "confidence to recommend one yet. Could you describe "
                "the workload or resource involved and the specific "
                "outcome you need?"
            )

            status = "clarification_required"

        else:
            # Both confident service requests and exam questions use RAG.
            retrieved = retrieve_relevant_chunks(
                query=user_input,
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                top_n=5,
            )

            # Remove the similarity scores and combine the retrieved
            # document chunks into one context string for the model.
            context = "\n".join(f"- {chunk}" for chunk, _ in retrieved)

            # Log each retrieved chunk and its similarity score.
            for chunk, score in retrieved:
                self.logger.debug(
                    "Retrieved chunk=%r | similarity=%.3f",
                    chunk,
                    score,
                )

            self.logger.info(
                "RAG context passed to language model | " "chunks=%d | characters=%d",
                len(retrieved),
                len(context),
            )

            self.logger.debug(
                "RAG context preview=%r",
                context[:500],
            )

            # For an exam question:
            #   predictions=[]
            #   show_classifier=False
            #
            # For a service request:
            #   predictions contains the ranked services
            #   show_classifier=True
            reply = self.language_model.generate_reply(
                original_input=user_input,
                predictions=predictions,
                context=context,
                show_classifier=recommendation_request,
            )

            status = "completed"

        # Record the final result for both completed responses
        # and clarification responses.
        self.logger.info(
            "status=%s | user_input=%r | " "predictions=%r | reply=%r",
            status,
            user_input,
            predictions,
            reply,
        )

        return reply


def run_interface(chatbot: AWSChatbot) -> None:
    """Run the interactive command-line interface."""

    print(
        "Assistant: Hi! Ask me which AWS service suits your needs, "
        "or ask me a question about the AWS Certified Cloud Practitioner exam."
    )
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
    """Load all models, build the RAG store and start the interface."""

    # Configure the application's logger.
    logger = configure_chatbot_logging()

    # Load the previously trained AWS service classifier.
    classifier = AWSServiceClassifier(MODEL_PATH)

    # Load the language model that writes the final response.
    language_model = LanguageModel()

    # Load the six AWS certification text documents.
    documents = load_documents(KNOWLEDGE_BASE_PATH)

    # Split the complete documents into smaller overlapping chunks.
    chunks = chunk_documents(
        documents=documents,
    )

    # Load the model that converts text into embedding vectors.
    embedding_model = load_embedding_model()

    # Embed every document chunk and store the results in memory.
    #
    # The resulting structure is:
    # [
    #     (embedding, chunk_text),
    #     (embedding, chunk_text),
    # ]
    vector_store = create_vector_store(
        chunks=chunks,
        model=embedding_model,
    )

    # Give the chatbot everything it needs for classification,
    # retrieval and final response generation.
    chatbot = AWSChatbot(
        classifier=classifier,
        language_model=language_model,
        logger=logger,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    # Start accepting prompts from the user.
    run_interface(chatbot)


if __name__ == "__main__":
    main()
