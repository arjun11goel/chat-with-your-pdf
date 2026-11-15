"""PDF text extraction and Week 3-compatible word chunking utilities."""

from __future__ import annotations

import io
import re
from typing import Any

from pypdf import PdfReader


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be read or does not contain usable text."""


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """Extract text page by page so retrieved chunks can cite PDF page numbers.

    Scanned/image-only PDFs normally produce no extractable text with pypdf.
    OCR is intentionally not added because the Week 4 brief specifies pypdf
    or pdfplumber extraction and asks the app to handle PDFs with no text.
    """
    if not pdf_bytes:
        raise PDFProcessingError("The uploaded PDF is empty.")

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
    except Exception as exc:  # pypdf exposes several parser-specific errors
        raise PDFProcessingError(
            "The file could not be read as a valid PDF. Please try another PDF."
        ) from exc

    if reader.is_encrypted:
        try:
            decrypt_result = reader.decrypt("")
        except Exception as exc:
            raise PDFProcessingError(
                "This PDF is password-protected. Upload an unlocked PDF."
            ) from exc
        if decrypt_result == 0:
            raise PDFProcessingError(
                "This PDF is password-protected. Upload an unlocked PDF."
            )

    extracted_pages: list[dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            raise PDFProcessingError(
                f"Text extraction failed on PDF page {page_number}."
            ) from exc

        # Normalize whitespace without changing the extracted words.
        clean_text = re.sub(r"\s+", " ", raw_text).strip()
        if clean_text:
            extracted_pages.append(
                {"page_number": page_number, "text": clean_text}
            )

    if not extracted_pages:
        raise PDFProcessingError(
            "No selectable text was found. This PDF may be scanned or image-only. "
            "Please upload a text-based PDF."
        )

    return extracted_pages


def chunk_pdf_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 250,
    overlap: int = 50,
) -> list[dict[str, Any]]:
    """Split page text into overlapping word chunks, retaining source pages.

    Defaults deliberately match the Week 3 RAG project: 250 words per chunk
    and 50 words of overlap. Chunks may cross page boundaries; each chunk
    therefore records every PDF page represented in its text.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    all_words: list[str] = []
    word_pages: list[int] = []

    for page in pages:
        page_number = int(page["page_number"])
        page_text = re.sub(r"\s+", " ", str(page.get("text", ""))).strip()
        page_words = page_text.split()
        all_words.extend(page_words)
        word_pages.extend([page_number] * len(page_words))

    if not all_words:
        return []

    chunks: list[dict[str, Any]] = []
    step = chunk_size - overlap
    start = 0

    while start < len(all_words):
        end = min(start + chunk_size, len(all_words))
        chunk_words = all_words[start:end]
        page_numbers = sorted(set(word_pages[start:end]))
        chunk_index = len(chunks)

        chunks.append(
            {
                "id": f"chunk_{chunk_index:04d}",
                "text": " ".join(chunk_words),
                "chunk_index": chunk_index,
                "page_numbers": page_numbers,
                "word_start": start,
                "word_end": end,
            }
        )

        if end == len(all_words):
            break
        start += step

    return chunks
