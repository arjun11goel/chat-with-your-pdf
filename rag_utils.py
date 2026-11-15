"""Embedding, ChromaDB retrieval, and Groq generation for Chat With Your PDF."""

from __future__ import annotations

from typing import Any

import chromadb
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"  # Carried forward from Week 3.
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
TOP_K = 3
MIN_SIMILARITY = 0.25  # Week 3 heuristic; this is not a confidence probability.
ABSTENTION_ANSWER = "I don't know based on the provided context."


@st.cache_resource(show_spinner=False)
def load_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> SentenceTransformer:
    """Load one CPU model per Streamlit process and reuse it across sessions."""
    return SentenceTransformer(model_name, device="cpu")


@st.cache_resource(show_spinner=False)
def get_chroma_client() -> Any:
    """Return an in-memory Chroma client; no paid or hosted vector DB is used."""
    return chromadb.Client()


def delete_collection(collection_name: str | None) -> None:
    """Delete an upload's collection when the user resets or replaces a PDF."""
    if not collection_name:
        return
    client = get_chroma_client()
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        # The collection may already have disappeared after a session restart.
        pass


def index_pdf_chunks(
    collection_name: str,
    chunks: list[dict[str, Any]],
) -> int:
    """Embed chunks and add text plus page metadata to a session collection."""
    if not chunks:
        raise ValueError("There are no text chunks to index.")

    client = get_chroma_client()
    delete_collection(collection_name)
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    model = load_embedding_model()
    documents = [chunk["text"] for chunk in chunks]
    vectors = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )

    metadatas = []
    for chunk in chunks:
        page_numbers = chunk["page_numbers"]
        metadatas.append(
            {
                "chunk_index": int(chunk["chunk_index"]),
                "pages": ", ".join(str(page) for page in page_numbers),
                "page_start": int(min(page_numbers)),
                "page_end": int(max(page_numbers)),
                "word_start": int(chunk["word_start"]),
                "word_end": int(chunk["word_end"]),
            }
        )

    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=documents,
        embeddings=vectors.tolist(),
        metadatas=metadatas,
    )
    return collection.count()


def retrieve_relevant_chunks(
    collection_name: str,
    question: str,
    top_k: int = TOP_K,
    min_similarity: float = MIN_SIMILARITY,
) -> list[dict[str, Any]]:
    """Retrieve up to top_k passages and apply the Week 3 similarity gate."""
    clean_question = question.strip()
    if not clean_question:
        return []

    collection = get_chroma_client().get_collection(name=collection_name)
    count = collection.count()
    if count == 0:
        return []

    model = load_embedding_model()
    question_vector = model.encode(
        [clean_question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    result = collection.query(
        query_embeddings=[question_vector.tolist()],
        n_results=min(max(1, top_k), count),
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]

    retrieved: list[dict[str, Any]] = []
    for chunk_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        # Chroma's cosine distance is 1 - cosine similarity for this collection.
        similarity = 1.0 - float(distance)
        if similarity < min_similarity:
            continue

        retrieved.append(
            {
                "id": chunk_id,
                "text": document,
                "metadata": metadata or {},
                "similarity": similarity,
            }
        )

    return retrieved


def generate_grounded_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    history: list[dict[str, str]],
    api_key: str,
    model_name: str = DEFAULT_GROQ_MODEL,
) -> str:
    """Call Groq with retrieved PDF passages as the only factual evidence."""
    if not retrieved_chunks:
        return ABSTENTION_ANSWER
    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured.")

    system_prompt = (
        "You are a careful assistant that answers questions about one uploaded PDF. "
        "The PDF excerpts in the current user message are the ONLY factual evidence "
        "you may use. Do not use outside knowledge or previous assistant answers as "
        "evidence. Conversation history may only help resolve references such as "
        "'it' or 'that policy'; it is not a source of facts. If the excerpts do not "
        "contain enough information to answer, reply exactly: "
        f"{ABSTENTION_ANSWER} Do not guess or fill gaps. For every substantive "
        "claim, include a citation using the page labels provided with the excerpt, "
        "such as [Page 2] or [Pages 2, 3]. Never invent a page number."
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]

    # Keep a small recent window for conversational continuity. The system
    # instruction above explicitly prevents old answers from becoming evidence.
    for item in history[-6:]:
        role = item.get("role")
        content = item.get("content", "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    excerpt_blocks = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})
        pages = metadata.get("pages", "unknown page")
        excerpt_blocks.append(
            f"[Excerpt {index} | PDF page(s): {pages} | "
            f"Cosine similarity: {chunk['similarity']:.3f}]\n{chunk['text']}"
        )

    current_user_message = (
        f"Question: {question.strip()}\n\n"
        "Retrieved PDF excerpts (use only these as factual evidence):\n\n"
        + "\n\n---\n\n".join(excerpt_blocks)
    )
    messages.append({"role": "user", "content": current_user_message})

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=1000,
    )

    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        return ABSTENTION_ANSWER
    return answer.strip()
