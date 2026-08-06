import logging
import re

from src.chatbot_logging import configure_chatbot_logging


def test_configure_chatbot_logging_has_expected_attributes(tmp_path):
    log_path = tmp_path / "chatbot_history.log"

    logger = configure_chatbot_logging(log_path=log_path)

    assert logger.name == "src.chatbot_logging"
    assert logger.propagate is False
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.FileHandler)

    logger.info("Chatbot test message")

    log_contents = log_path.read_text(encoding="utf-8")

    # Contains a timestamp such as 2026-08-06 10:42:18.
    assert re.search(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
        log_contents,
    )

    assert "INFO" in log_contents
    assert "Chatbot test message" in log_contents
