"""Streamlit application: upload a PDF and ask grounded questions about it."""

from __future__ import annotations

import hashlib
import os
import uuid
from typing import Any

import streamlit as st

from pdf_utils import PDFProcessingError, chunk_pdf_pages, extract_pdf_pages
from rag_utils import (
    ABSTENTION_ANSWER,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_GROQ_MODEL,
    EMBEDDING_MODEL_NAME,
    MIN_SIMILARITY,
    TOP_K,
    delete_collection,
    generate_grounded_answer,
    index_pdf_chunks,
    load_embedding_model,
    retrieve_relevant_chunks,
)


st.set_page_config(
    page_title="Chat With Your PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def read_setting(name: str, default: str | None = None) -> str | None:
    """Read Streamlit secrets first, with environment variables as a fallback."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = os.getenv(name, default)
    if value is None:
        return default
    return str(value)


def initialize_session_state() -> None:
    defaults: dict[str, Any] = {
        "session_id": uuid.uuid4().hex,
        "uploader_version": 0,
        "current_upload_hash": None,
        "pdf_filename": None,
        "collection_name": None,
        "chunk_count": 0,
        "page_count": 0,
        "messages": [],
        "processing_error": None,
        "is_ready": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_chat_and_pdf() -> None:
    delete_collection(st.session_state.get("collection_name"))
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.uploader_version += 1
    st.session_state.current_upload_hash = None
    st.session_state.pdf_filename = None
    st.session_state.collection_name = None
    st.session_state.chunk_count = 0
    st.session_state.page_count = 0
    st.session_state.messages = []
    st.session_state.processing_error = None
    st.session_state.is_ready = False


def source_page_label(metadata: dict[str, Any]) -> str:
    pages = metadata.get("pages", "")
    if not pages:
        return "Page not available"
    page_numbers = [part.strip() for part in str(pages).split(",") if part.strip()]
    if len(page_numbers) == 1:
        return f"Page {page_numbers[0]}"
    return f"Pages {', '.join(page_numbers)}"


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"PDF sources ({len(sources)})", expanded=False):
        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata", {})
            label = source_page_label(metadata)
            score = float(source.get("similarity", 0.0))
            st.markdown(
                f"**Source {index} · {label} · cosine similarity {score:.3f}**"
            )
            st.write(source.get("text", ""))
            if index < len(sources):
                st.divider()


def process_uploaded_pdf(uploaded_file: Any) -> None:
    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if file_hash == st.session_state.current_upload_hash:
        return

    # Replacing a PDF starts a fresh chat and removes the old session collection.
    delete_collection(st.session_state.get("collection_name"))
    st.session_state.current_upload_hash = file_hash
    st.session_state.pdf_filename = uploaded_file.name
    st.session_state.collection_name = None
    st.session_state.chunk_count = 0
    st.session_state.page_count = 0
    st.session_state.messages = []
    st.session_state.processing_error = None
    st.session_state.is_ready = False

    try:
        with st.spinner("Reading the PDF, creating overlapping chunks, and indexing it..."):
            pages = extract_pdf_pages(pdf_bytes)
            chunks = chunk_pdf_pages(
                pages,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )
            if not chunks:
                raise PDFProcessingError(
                    "No usable text chunks could be created from this PDF."
                )

            # Load the same Week 2/3 embedding model on CPU. Streamlit caches it
            # once per app process so separate sessions do not each load a copy.
            load_embedding_model()

            collection_name = (
                f"pdf-{st.session_state.session_id[:12]}-{file_hash[:12]}"
            )
            indexed_count = index_pdf_chunks(collection_name, chunks)

        st.session_state.collection_name = collection_name
        st.session_state.chunk_count = indexed_count
        st.session_state.page_count = len(pages)
        st.session_state.is_ready = True

    except PDFProcessingError as exc:
        st.session_state.processing_error = str(exc)
    except Exception as exc:
        # Keep the app usable and provide a practical error without exposing
        # credentials or a full traceback in the public demo.
        st.session_state.processing_error = (
            "The PDF could not be indexed. Check the file and try again. "
            f"Technical detail: {type(exc).__name__}: {exc}"
        )


def main() -> None:
    initialize_session_state()

    groq_api_key = read_setting("GROQ_API_KEY")
    groq_model = read_setting("GROQ_MODEL", DEFAULT_GROQ_MODEL) or DEFAULT_GROQ_MODEL

    with st.sidebar:
        st.title("📄 Chat With Your PDF")
        st.caption("PromptForge · Week 4 Final Project")
        st.markdown("### How to use")
        st.markdown(
            "1. Upload one text-based PDF.\n"
            "2. Wait while its text is extracted and indexed.\n"
            "3. Ask questions in the chat.\n"
            "4. Expand **PDF sources** to inspect the supporting passages."
        )
        st.divider()
        st.markdown("### Models and retrieval")
        st.caption(f"LLM: Groq · `{groq_model}`")
        st.caption(f"Embeddings: `{EMBEDDING_MODEL_NAME}` (CPU)")
        st.caption(f"Vector store: ChromaDB (session-scoped, in-memory)")
        st.caption(f"Chunking: {CHUNK_SIZE} words · {CHUNK_OVERLAP}-word overlap")
        st.caption(f"Retrieval: top {TOP_K} · minimum similarity {MIN_SIMILARITY:.2f}")
        st.divider()
        if st.button("New Chat / Upload New PDF", use_container_width=True):
            reset_chat_and_pdf()
            st.rerun()
        if not groq_api_key:
            st.warning(
                "GROQ_API_KEY is not configured. Add it to "
                "`.streamlit/secrets.toml` locally or to the app's Streamlit "
                "Cloud Secrets before asking questions."
            )
        st.caption(
            "Answers are restricted to retrieved PDF passages. "
            "The assistant may say it does not know when the PDF lacks evidence."
        )

    st.title("Chat With Your PDF")
    st.write(
        "Upload a PDF and ask questions about its contents. "
        "Answers include the PDF pages and excerpts used for grounding."
    )

    uploader_key = f"pdf_uploader_{st.session_state.uploader_version}"
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key=uploader_key,
        help="Upload a text-based PDF. Scanned/image-only PDFs are not OCR-processed.",
    )

    if uploaded_file is None:
        if not st.session_state.is_ready:
            st.info("Upload a PDF to start a new document chat.")
    else:
        process_uploaded_pdf(uploaded_file)

    if st.session_state.processing_error:
        st.error(st.session_state.processing_error)

    if st.session_state.is_ready:
        st.success(
            f"Indexed **{st.session_state.pdf_filename}** — "
            f"{st.session_state.page_count} text-containing pages, "
            f"{st.session_state.chunk_count} chunks."
        )

    # Re-render persisted messages on every Streamlit rerun.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    if st.session_state.is_ready:
        if not groq_api_key:
            st.chat_input(
                "Add GROQ_API_KEY to enable questions...",
                disabled=True,
            )
        else:
            question = st.chat_input("Ask a question about your PDF...")
            if question is not None:
                clean_question = question.strip()
                if not clean_question:
                    st.warning("Please enter a question before submitting.")
                else:
                    st.session_state.messages.append(
                        {"role": "user", "content": clean_question}
                    )
                    with st.chat_message("user"):
                        st.markdown(clean_question)

                    with st.chat_message("assistant"):
                        with st.spinner("Retrieving relevant PDF passages and generating an answer..."):
                            try:
                                retrieved = retrieve_relevant_chunks(
                                    st.session_state.collection_name,
                                    clean_question,
                                    top_k=TOP_K,
                                    min_similarity=MIN_SIMILARITY,
                                )
                                if not retrieved:
                                    answer = ABSTENTION_ANSWER
                                else:
                                    history = st.session_state.messages[:-1]
                                    answer = generate_grounded_answer(
                                        question=clean_question,
                                        retrieved_chunks=retrieved,
                                        history=history,
                                        api_key=groq_api_key,
                                        model_name=groq_model,
                                    )
                            except Exception as exc:
                                answer = (
                                    "I couldn't complete that request. Check the "
                                    "Groq API key, model setting, and connection, "
                                    "then try again."
                                )
                                st.error(
                                    f"Request error: {type(exc).__name__}: {exc}"
                                )
                                retrieved = []

                        st.markdown(answer)
                        render_sources(retrieved)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": retrieved,
                        }
                    )

    st.divider()
    st.caption(
        "Built for PromptForge Week 4 using Streamlit, Groq, "
        "Sentence Transformers, ChromaDB, and pypdf."
    )


if __name__ == "__main__":
    main()
