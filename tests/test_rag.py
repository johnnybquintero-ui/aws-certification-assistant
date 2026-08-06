from unittest.mock import MagicMock, patch

import pytest

from src.rag import (
    EMBEDDING_MODEL_NAME,
    chunk_documents,
    chunk_text,
    create_vector_store,
    load_embedding_model,
)


def test_short_text_creates_one_chunk():
    """Text shorter than the chunk size should remain as one chunk."""

    text = "AWS WAF filters incoming web requests."

    result = chunk_text(
        text=text,
        chunk_size=100,
        overlap=20,
    )

    assert result == ["AWS WAF filters incoming web requests."]


def test_long_text_creates_overlapping_chunks():
    """Neighbouring chunks should contain the requested overlap."""

    # Create 150 predictable words:
    # word1, word2, ... word150
    text = " ".join(f"word{number}" for number in range(1, 151))

    result = chunk_text(
        text=text,
        chunk_size=100,
        overlap=20,
    )

    # Chunk one contains words 1–100.
    # Chunk two contains words 81–150.
    assert len(result) == 2
    assert len(result[0].split()) == 100
    assert len(result[1].split()) == 70

    # The final 20 words of the first chunk should also appear
    # at the beginning of the second chunk.
    assert result[0].split()[-20:] == result[1].split()[:20]


def test_empty_text_returns_empty_list():
    """An empty document should not produce any chunks."""

    result = chunk_text("")

    assert result == []


@pytest.mark.parametrize(
    "chunk_size, overlap, expected_message",
    [
        (0, 0, "chunk_size must be greater than zero"),
        (100, -1, "overlap cannot be negative"),
        (100, 100, "overlap must be smaller than chunk_size"),
        (100, 101, "overlap must be smaller than chunk_size"),
    ],
)
def test_chunk_text_rejects_invalid_settings(
    chunk_size,
    overlap,
    expected_message,
):
    """Invalid chunk settings should raise a helpful error."""

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        chunk_text(
            text="Example AWS document",
            chunk_size=chunk_size,
            overlap=overlap,
        )


def test_chunk_documents_preserves_source_and_chunk_number():
    """Each chunk should retain its source file and position."""

    documents = [
        {
            "source": "security.txt",
            "text": "one two three four five six",
        }
    ]

    result = chunk_documents(
        documents=documents,
        chunk_size=4,
        overlap=1,
    )

    assert result == [
        {
            "source": "security.txt",
            "chunk_number": 1,
            "text": "one two three four",
        },
        {
            "source": "security.txt",
            "chunk_number": 2,
            "text": "four five six",
        },
    ]


def test_create_vector_store_returns_embedding_text_tuples():
    """Each chunk should be stored as (embedding, chunk_text)."""

    chunks = [
        {
            "source": "security.txt",
            "chunk_number": 1,
            "text": "AWS WAF filters web requests.",
        },
        {
            "source": "security.txt",
            "chunk_number": 2,
            "text": "AWS Shield protects against DDoS attacks.",
        },
    ]

    # Use a fake model so the test does not load or run the real
    # SentenceTransformer model.
    mock_model = MagicMock()

    # Return a predictable fake embedding for each chunk.
    mock_model.encode.side_effect = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    result = create_vector_store(
        chunks=chunks,
        model=mock_model,
    )

    # Confirm that the result is the required in-memory vector store:
    # [(embedding, chunk_text), ...]
    assert result == [
        (
            [0.1, 0.2, 0.3],
            "AWS WAF filters web requests.",
        ),
        (
            [0.4, 0.5, 0.6],
            "AWS Shield protects against DDoS attacks.",
        ),
    ]

    # Confirm that every chunk was embedded.
    assert mock_model.encode.call_count == 2

    # Confirm that normalised embeddings were requested.
    mock_model.encode.assert_any_call(
        "AWS WAF filters web requests.",
        normalize_embeddings=True,
    )

    mock_model.encode.assert_any_call(
        "AWS Shield protects against DDoS attacks.",
        normalize_embeddings=True,
    )


def test_empty_chunks_create_empty_vector_store():
    """No chunks should produce an empty vector store."""

    mock_model = MagicMock()

    result = create_vector_store(
        chunks=[],
        model=mock_model,
    )

    assert result == []

    # The embedding model should not be called unnecessarily.
    mock_model.encode.assert_not_called()


@patch("src.rag.SentenceTransformer")
def test_load_embedding_model_uses_expected_model(
    mock_sentence_transformer,
):
    """The configured SentenceTransformer model should be loaded."""

    fake_model = MagicMock()
    mock_sentence_transformer.return_value = fake_model

    result = load_embedding_model()

    mock_sentence_transformer.assert_called_once_with(EMBEDDING_MODEL_NAME)

    assert result is fake_model
