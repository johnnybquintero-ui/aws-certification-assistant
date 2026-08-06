from unittest.mock import MagicMock

import src.chatbot_interface as chatbot_interface
from src.chatbot_interface import AWSChatbot, is_service_recommendation


def test_is_service_recommendation_distinguishes_request_types():
    service_request = "I need a service that provides secure storage"
    exam_question = (
        "What are the main resources in scope for the " "Cloud Practitioner exam?"
    )

    assert is_service_recommendation(service_request) is True
    assert is_service_recommendation(exam_question) is False


def test_service_request_calls_classifier_and_returns_model_response(
    monkeypatch,
):
    user_input = "I need a managed MySQL database."
    predictions = [
        ("rds", 0.85),
        ("dynamodb", 0.10),
    ]

    classifier = MagicMock()
    classifier.classify_with_confidence.return_value = predictions

    language_model = MagicMock()
    language_model.generate_reply.return_value = "Use Amazon RDS."

    monkeypatch.setattr(
        chatbot_interface,
        "retrieve_relevant_chunks",
        MagicMock(
            return_value=[
                (
                    "Amazon RDS provides managed relational databases.",
                    0.90,
                )
            ]
        ),
    )

    chatbot = AWSChatbot(
        classifier=classifier,
        language_model=language_model,
        logger=MagicMock(),
        embedding_model=MagicMock(),
        vector_store=[],
    )

    result = chatbot.respond(user_input)

    assert result == "Use Amazon RDS."

    classifier.classify_with_confidence.assert_called_once_with(user_input)

    language_model.generate_reply.assert_called_once_with(
        original_input=user_input,
        predictions=predictions,
        context="- Amazon RDS provides managed relational databases.",
        show_classifier=True,
    )


def test_exam_question_bypasses_classifier(monkeypatch):
    user_input = "What security topics are covered in the " "Cloud Practitioner exam?"

    classifier = MagicMock()

    language_model = MagicMock()
    language_model.generate_reply.return_value = (
        "The exam covers several security topics."
    )

    monkeypatch.setattr(
        chatbot_interface,
        "retrieve_relevant_chunks",
        MagicMock(return_value=[("The exam includes security and compliance.", 0.90)]),
    )

    chatbot = AWSChatbot(
        classifier=classifier,
        language_model=language_model,
        logger=MagicMock(),
        embedding_model=MagicMock(),
        vector_store=[],
    )

    result = chatbot.respond(user_input)

    assert result == "The exam covers several security topics."
    classifier.classify_with_confidence.assert_not_called()

    language_model.generate_reply.assert_called_once_with(
        original_input=user_input,
        predictions=[],
        context="- The exam includes security and compliance.",
        show_classifier=False,
    )
