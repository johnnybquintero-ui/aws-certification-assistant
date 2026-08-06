from sentence_transformers import SentenceTransformer

from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

# The pretrained model used to convert text into numerical embeddings.
# We must use the same model for document chunks and future user prompts.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents(
    knowledge_base_path: Path,
) -> list[dict[str, str]]:
    """Load every text document from the RAG knowledge base."""

    # Make sure the supplied knowledge-base directory exists.
    if not knowledge_base_path.exists():
        raise FileNotFoundError(
            f"Knowledge-base directory not found: {knowledge_base_path}"
        )

    # This list will contain all the loaded documents.
    documents = []

    # Find every .txt file in the directory.
    #
    # sorted() ensures the files are always loaded in a predictable order.
    for document_path in sorted(knowledge_base_path.glob("*.txt")):
        # Read the complete contents of the current text file.
        document_text = document_path.read_text(encoding="utf-8")

        # Store the filename and its contents together.
        #
        # This matches the structure expected by chunk_documents():
        #
        # {
        #     "source": "clf_c02_exam_overview.txt",
        #     "text": "The document contents..."
        # }
        documents.append(
            {
                "source": document_path.name,
                "text": document_text,
            }
        )

    # Raise a helpful error if the directory contains no text files.
    if not documents:
        raise FileNotFoundError(f"No .txt documents found in: {knowledge_base_path}")

    return documents


def chunk_text(
    text: str,
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[str]:
    """Split text into overlapping word-based chunks."""

    # Prevent invalid chunk sizes.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    # An overlap cannot contain a negative number of words.
    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    # The overlap must be smaller than the complete chunk.
    # Otherwise, the loop would never move forward.
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Split the complete document into individual words.
    words = text.split()

    # There is nothing to chunk if the document is empty.
    if not words:
        return []

    # This list will contain the completed text chunks.
    chunks = []

    # Start the first chunk at the first word.
    start = 0

    while start < len(words):

        # Calculate where the current chunk should end.
        end = start + chunk_size

        # Select the words belonging to this chunk.
        chunk_words = words[start:end]

        # Join the individual words back into one string.
        chunk = " ".join(chunk_words)

        # Save the completed chunk.
        chunks.append(chunk)

        # Stop when this chunk contains the end of the document.
        # This prevents an unnecessary chunk containing only overlap.
        if end >= len(words):
            break

        # Move forwards by the chunk size, but move backwards by the
        # overlap so some words appear in both neighbouring chunks.
        start = end - overlap

    return chunks


def chunk_documents(
    documents: list[dict[str, str]],
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[dict[str, str | int]]:
    """Split every loaded document and preserve its source information."""

    # This will contain chunks from every document.
    all_chunks = []

    # Process each document returned by the document loader.
    for document in documents:

        # Split the current document's text into smaller chunks.
        document_chunks = chunk_text(
            text=document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        # Add each chunk to the complete collection.
        for chunk_number, chunk in enumerate(
            document_chunks,
            start=1,
        ):
            all_chunks.append(
                {
                    # Preserve the filename so we know where the
                    # retrieved information originally came from.
                    "source": document["source"],
                    # Record the chunk's position within its document.
                    "chunk_number": chunk_number,
                    # Store the actual text that will be embedded.
                    "text": chunk,
                }
            )

    return all_chunks


def load_embedding_model() -> SentenceTransformer:
    """Load and return the SentenceTransformer embedding model."""

    # The model will be downloaded the first time this runs.
    # After that, Hugging Face normally loads it from its local cache.
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return model


def create_vector_store(
    chunks: list[dict[str, str | int]],
    model: SentenceTransformer,
) -> list[tuple]:
    """Generate an embedding for each chunk and store them in memory."""

    # This list is our simple in-memory vector store.
    vector_store = []

    # Process one chunk at a time so each step is easy to understand.
    for chunk in chunks:

        # Extract the text from the chunk dictionary.
        chunk_text = str(chunk["text"])

        # Convert the text into a numerical embedding.
        #
        # Normalisation gives the vector a length of one, which makes
        # cosine-similarity comparisons easier during retrieval.
        embedding = model.encode(
            chunk_text,
            normalize_embeddings=True,
        )

        # Store each embedding alongside its corresponding chunk text:
        # (embedding, chunk_text)
        vector_store.append(
            (
                embedding,
                chunk_text,
            )
        )

    return vector_store


def retrieve_relevant_chunks(
    query: str,
    vector_store: list[tuple],
    embedding_model: SentenceTransformer,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Return the document chunks most relevant to the query."""

    if not query.strip():
        raise ValueError("Query must not be empty.")

    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")

    if not vector_store:
        return []

    # Embed the query using the same model used for the documents.
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
    )

    scored_chunks = []

    # The vector store contains:
    # (embedding, chunk_text)
    for chunk_embedding, chunk_text in vector_store:
        similarity_score = cosine_similarity(
            [query_embedding],
            [chunk_embedding],
        )[
            0
        ][0]

        # (chunk_text, similarity_score)
        scored_chunks.append(
            (
                chunk_text,
                float(similarity_score),
            )
        )

    # The highest similarity should appear first.
    scored_chunks.sort(
        key=lambda result: result[1],
        reverse=True,
    )

    return scored_chunks[:top_n]
